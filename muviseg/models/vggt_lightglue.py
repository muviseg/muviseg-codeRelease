"""
SegVGGT — VGGT Aggregator backbone + LightGlue-style segment matcher.

Replaces the frozen MASt3R backbone with a frozen VGGT Aggregator while
reusing the same LightGlue matching head (attention layers + DoubleSoftmax).

Key differences vs SegMASt3RLGv2 (lightglue_v2.py):
  - Backbone: VGGT Aggregator (alternating frame+global attention) instead
    of MASt3R decoder.
  - Descriptor dim: 2048 (concatenated frame+global tokens) instead of 24.
  - Image normalization: dataset provides [-1, 1]; extract_desc converts
    to [0, 1] since the Aggregator normalizes internally with ResNet mean/std.
  - Input format: [B, 2, 3, H, W] stacked pair (vs two separate view dicts).
  - Optimal image size: 518×518 (patch_size=14 → 37×37 patch grid).
  - Precomputed mode is strongly recommended (online VGGT is expensive).

Architecture:
  dsc (B, M, 2048)
    ↓ MLP(2048→256→128) + GELU + LayerNorm
  x  (B, M, 128)
    ↓ N_LAYERS × [Pre-LN self-attn + cross-attn + FFN(4×)]
  x' (B, M, 128)
    ↓ DoubleSoftmaxMatcher
  log_mutual (B, M, N),  match0 (B, M),  match1 (B, N)

Trainable params (proj_dim=128, 3 layers, 4 heads): ~800K
  (same as LGv2 — only the projector input dim differs)

NOTE: This file is self-contained — it does NOT import from lightglue_v2.py
to avoid pulling in segmast3r submodule dependencies.  The attention layer
and matcher are defined inline (identical to lightglue_v2 versions).
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from muviseg.paths import setup_vggt_path


# ---------------------------------------------------------------------------
# masked_average_pooling — inline (no segmast3r dependency)
# ---------------------------------------------------------------------------

def masked_average_pooling(feat: Tensor, masks: Tensor) -> Tensor:
    """
    Args:
        feat:  (B, D, H, W) dense feature map
        masks: (B, M, H, W) binary/float masks
    Returns:
        (B, D, M) per-segment descriptors
    """
    B, D, H, W = feat.shape
    M = masks.shape[1]
    feat_flat = feat.view(B, D, H * W)                    # (B, D, HW)
    masks_flat = masks.view(B, M, H * W).float()           # (B, M, HW)
    area = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, M, 1)
    return torch.bmm(feat_flat, masks_flat.transpose(1, 2)) / area.transpose(1, 2)


# ---------------------------------------------------------------------------
# MLP Projector (2048 → mid → proj_dim)
# ---------------------------------------------------------------------------

class DescriptorProjector(nn.Module):
    """Two-layer MLP: in_dim → mid_dim → out_dim, with GELU + LayerNorm."""

    def __init__(self, in_dim: int = 2048, mid_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, out_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Attention layer — identical to lightglue_v2.SegmentAttentionLayerV2
# Inlined to avoid importing lightglue_v2 (which pulls segmast3r deps).
# ---------------------------------------------------------------------------

class SegmentAttentionLayerV2(nn.Module):
    """
    Pre-LN self + cross attention block with configurable FFN expansion.

    Both images share all weights — equivariant to swap img0 ↔ img1.
    Uses F.scaled_dot_product_attention → FlashAttention on Ampere+.
    """

    def __init__(self, dim: int = 128, num_heads: int = 4, ffn_expansion: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim      = dim
        self.heads    = num_heads
        self.head_dim = dim // num_heads

        # Self-attention (Pre-LN)
        self.self_norm = nn.LayerNorm(dim)
        self.self_qkv  = nn.Linear(dim, 3 * dim, bias=False)
        self.self_out  = nn.Linear(dim, dim, bias=False)

        # Cross-attention (Pre-LN on both query and context)
        self.cross_norm     = nn.LayerNorm(dim)
        self.cross_norm_ctx = nn.LayerNorm(dim)
        self.cross_q    = nn.Linear(dim, dim, bias=False)
        self.cross_kv   = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_out  = nn.Linear(dim, dim, bias=False)

        # Feed-forward (Pre-LN)
        ffn_dim = dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _self_attn(self, x: Tensor) -> Tensor:
        B, M, D = x.shape
        qkv = self.self_qkv(self.self_norm(x))
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.self_out(out.transpose(1, 2).reshape(B, M, D))

    def _cross_attn(self, x: Tensor, ctx: Tensor) -> Tensor:
        B, M, D = x.shape
        N = ctx.shape[1]
        Q    = self.cross_q(self.cross_norm(x))
        K, V = self.cross_kv(self.cross_norm_ctx(ctx)).chunk(2, dim=-1)
        Q = Q.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.cross_out(out.transpose(1, 2).reshape(B, M, D))

    def forward(self, x0: Tensor, x1: Tensor):
        """(B,M,D), (B,N,D) → refined (B,M,D), (B,N,D)"""
        x0 = x0 + self._self_attn(x0)
        x1 = x1 + self._self_attn(x1)
        x0 = x0 + self._cross_attn(x0, x1)
        x1 = x1 + self._cross_attn(x1, x0)
        x0 = x0 + self.ffn(x0)
        x1 = x1 + self.ffn(x1)
        return x0, x1


# ---------------------------------------------------------------------------
# DoubleSoftmax Matcher — inline implementation
# Identical to src.models.mast3r_segfeat.double_softmax_matcher but
# inlined to avoid segmast3r submodule dependency.
# ---------------------------------------------------------------------------

class DoubleSoftmaxMatcher(nn.Module):
    """
    Double-softmax mutual matching + matchability MLP.

    Computes log(softmax_row * softmax_col) as mutual agreement score,
    plus per-segment matchability logits via a small MLP.
    """

    def __init__(self, desc_dim: int = 128):
        super().__init__()
        self.matchability = nn.Sequential(
            nn.Linear(desc_dim, desc_dim),
            nn.ReLU(),
            nn.Linear(desc_dim, 1),
        )
        for m in self.matchability.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, desc0: Tensor, desc1: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            desc0: (B, D, M) refined segment descriptors img0
            desc1: (B, D, N) refined segment descriptors img1
        Returns:
            log_mutual: (B, M, N) log mutual agreement scores
            match0:     (B, M)    matchability logits for img0
            match1:     (B, N)    matchability logits for img1
        """
        # Similarity matrix
        sim = torch.einsum("bdm,bdn->bmn", desc0, desc1)  # (B, M, N)
        D = desc0.shape[1]
        sim = sim / D ** 0.5

        # Double softmax → mutual agreement
        log_mutual = F.log_softmax(sim, dim=-1) + F.log_softmax(sim, dim=-2)

        # Matchability logits
        match0 = self.matchability(desc0.transpose(1, 2)).squeeze(-1)  # (B, M)
        match1 = self.matchability(desc1.transpose(1, 2)).squeeze(-1)  # (B, N)

        return log_mutual, match0, match1


# ---------------------------------------------------------------------------
# VGGT Aggregator
# ---------------------------------------------------------------------------
setup_vggt_path()
from vggt.models.aggregator import Aggregator  # noqa: E402


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SegVGGT(nn.Module):
    """
    VGGT Aggregator backbone + LightGlue-style matching head.

    The Aggregator is frozen; only the projector, attention layers, and
    matcher are trained.  Forward interface is identical to SegMASt3RLGv2
    so the trainer works without modification.
    """

    DESC_DIM = 2048  # concatenated frame + global token dim
    # TODO: multi-scale version using layers [5, 11, 17, 23] → concat → 8192

    def __init__(
        self,
        vggt_ckpt: str,
        layer_idx: int = 23,
        proj_dim: int = 128,
        proj_mid_dim: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        use_grad_checkpoint: bool = False,
        deep_supervision: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        self.layer_idx           = layer_idx
        self.proj_dim            = proj_dim
        self.use_grad_checkpoint = use_grad_checkpoint
        self.deep_supervision    = deep_supervision

        # Frozen VGGT Aggregator
        self.aggregator = self._load_aggregator(vggt_ckpt, device)
        self._freeze(self.aggregator)

        # MLP projector: 2048 → proj_mid_dim → proj_dim
        self.proj = DescriptorProjector(
            in_dim=self.DESC_DIM, mid_dim=proj_mid_dim, out_dim=proj_dim
        )

        # Attention refinement layers
        self.attn_layers = nn.ModuleList([
            SegmentAttentionLayerV2(
                dim=proj_dim, num_heads=n_heads, ffn_expansion=ffn_expansion
            )
            for _ in range(n_layers)
        ])

        # Matching head
        self.matcher = DoubleSoftmaxMatcher(desc_dim=proj_dim)

    @staticmethod
    def _load_aggregator(ckpt_path: str, device: str) -> Aggregator:
        """
        Load VGGT Aggregator with default params and frozen weights.

        Args:
            ckpt_path: local .pt file OR HuggingFace model id
                       (e.g. "facebook/VGGT-1B").
            device: target device for loading.
        """
        aggregator = Aggregator()

        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        else:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=ckpt_path, filename="model.pt")
            state = torch.load(local, map_location=device, weights_only=True)

        # Extract aggregator keys from full VGGT checkpoint
        prefix = "aggregator."
        agg_state = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix)
        }
        if not agg_state:
            # Checkpoint may already be aggregator-only
            agg_state = state

        aggregator.load_state_dict(agg_state, strict=True)
        aggregator.eval()
        return aggregator

    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)

    def extract_desc(
        self, img0: Tensor, img1: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Extract dense per-pixel descriptors from VGGT Aggregator.

        The Aggregator uses alternating frame + global attention, so
        descriptors for each frame are cross-view aware (like MASt3R).

        Args:
            img0: (B, 3, H, W) in [-1, 1] range (dataset convention).
                  Converted to [0, 1] internally — Aggregator normalizes
                  with ResNet mean/std on its own.
            img1: (B, 3, H, W) in [-1, 1] range.
        Returns:
            desc0: (B, 2048, H_p, W_p)  where H_p, W_p = nearest multiple of 14
            desc1: (B, 2048, H_p, W_p)
        """
        # Dataset returns [-1, 1] (MASt3R convention) → convert to [0, 1]
        img0 = (img0 + 1.0) * 0.5
        img1 = (img1 + 1.0) * 0.5

        # Resize to nearest multiple of patch_size=14 (518 = 14×37, VGGT native)
        B, _, H, W = img0.shape
        H_new = round(H / 14) * 14  # 512 → 518 (37 patches)
        W_new = round(W / 14) * 14
        if (H_new, W_new) != (H, W):
            img0 = F.interpolate(img0, (H_new, W_new), mode="bilinear", align_corners=False)
            img1 = F.interpolate(img1, (H_new, W_new), mode="bilinear", align_corners=False)
        H_p, W_p = H_new // 14, W_new // 14

        # Stack pair: (B, 2, 3, H, W)
        images = torch.stack([img0, img1], dim=1)

        with torch.no_grad():
            aggregated_tokens_list, ps_idx = self.aggregator(images)

        # Take tokens from selected layer: (B, 2, P, 2048)
        tokens = aggregated_tokens_list[self.layer_idx]

        # Extract patch tokens (skip camera + register tokens)
        patch_tokens = tokens[:, :, ps_idx:]  # (B, 2, num_patches, 2048)

        # Reshape to spatial grid: (B, 2, H_p, W_p, 2048) → split views
        patch_tokens = patch_tokens[:, :, :H_p * W_p]  # trim if needed
        patch_tokens = patch_tokens.view(B, 2, H_p, W_p, self.DESC_DIM)
        patch_tokens = patch_tokens.permute(0, 1, 4, 2, 3)  # (B, 2, 2048, H_p, W_p)

        desc0 = patch_tokens[:, 0].contiguous()  # (B, 2048, H_p, W_p)
        desc1 = patch_tokens[:, 1].contiguous()
        return desc0, desc1

    def forward(
        self,
        img0=None, img1=None,
        masks0=None, masks1=None,
        dsc0_pre=None, dsc1_pre=None,
    ):
        """
        Same interface as SegMASt3RLGv2.forward().

        Returns (when deep_supervision=False):
            log_mutual  (B, M, N)       log mutual agreement scores
            match0      (B, M)          matchability logits img0
            match1      (B, N)          matchability logits img1
            dsc0        (B, 2048, M)    raw VGGT segment descriptors
            dsc1        (B, 2048, N)    raw VGGT segment descriptors

        Returns (when deep_supervision=True):
            layer_outputs  list of (log_mutual, match0, match1) per layer
            dsc0, dsc1     same as above
        """
        proj_dtype = next(self.proj.parameters()).dtype

        if dsc0_pre is not None:
            # Precomputed path: skip backbone + pooling
            dsc0 = dsc0_pre.transpose(1, 2).to(dtype=proj_dtype)  # (B, 2048, M)
            dsc1 = dsc1_pre.transpose(1, 2).to(dtype=proj_dtype)
        else:
            # Online path: VGGT extraction + masked pooling
            feat0, feat1 = self.extract_desc(img0, img1)  # (B, 2048, H_p, W_p)
            _, _, dH, dW = feat0.shape
            if masks0.shape[-2:] != (dH, dW):
                masks0 = F.interpolate(masks0.float(), (dH, dW), mode="nearest")
                masks1 = F.interpolate(masks1.float(), (dH, dW), mode="nearest")
            dsc0 = masked_average_pooling(feat0, masks0.float())  # (B, 2048, M)
            dsc1 = masked_average_pooling(feat1, masks1.float())

        # MLP projection: (B, 2048, M) → (B, M, 2048) → (B, M, proj_dim)
        x0 = self.proj(dsc0.transpose(1, 2))
        x1 = self.proj(dsc1.transpose(1, 2))

        # Attention refinement
        layer_outputs = []
        for layer in self.attn_layers:
            if self.use_grad_checkpoint and self.training:
                x0, x1 = checkpoint(layer, x0, x1, use_reentrant=False)
            else:
                x0, x1 = layer(x0, x1)

            if self.deep_supervision:
                lm, m0, m1 = self.matcher(
                    x0.transpose(1, 2), x1.transpose(1, 2)
                )
                layer_outputs.append((lm, m0, m1))

        if not self.deep_supervision:
            log_mutual, match0, match1 = self.matcher(
                x0.transpose(1, 2), x1.transpose(1, 2)
            )
            return log_mutual, match0, match1, dsc0, dsc1
        else:
            return layer_outputs, dsc0, dsc1


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def lightglue_loss(
    log_mutual: Tensor,     # (B, M_max, N_max)
    match0: Tensor,         # (B, M_max)   matchability logits img0
    match1: Tensor,         # (B, N_max)   matchability logits img1
    seg_corr_list: list,    # list[B] of (K, 2) GT pairs
    masks0_list: list,      # list[B] of (M_b, H, W) — for actual M per sample
    masks1_list: list,
    lambda_match: float = 1.0,
) -> Tensor:
    """
    Per sample:
      - Matched pair (i,j):  -log_mutual[i,j] + BCE(matchability, 1)
      - Unmatched segment:   BCE(matchability, 0)
    """
    total = log_mutual.new_zeros(1)
    B = log_mutual.shape[0]

    for b in range(B):
        M = masks0_list[b].shape[0]
        N = masks1_list[b].shape[0]
        corr = seg_corr_list[b].to(log_mutual.device)

        lm = log_mutual[b, :M, :N]
        m0 = match0[b, :M]
        m1 = match1[b, :N]

        gt0 = torch.zeros(M, device=log_mutual.device)
        gt1 = torch.zeros(N, device=log_mutual.device)

        match_loss = lm.new_zeros(1)

        if corr.shape[0] > 0:
            valid = (corr[:, 0] < M) & (corr[:, 1] < N)
            corr_v = corr[valid]
            if corr_v.shape[0] > 0:
                i_idx = corr_v[:, 0]
                j_idx = corr_v[:, 1]
                match_loss = -lm[i_idx, j_idx].mean()
                gt0[i_idx] = 1.0
                gt1[j_idx] = 1.0

        bce0 = F.binary_cross_entropy_with_logits(m0, gt0)
        bce1 = F.binary_cross_entropy_with_logits(m1, gt1)

        total = total + match_loss + lambda_match * (bce0 + bce1)

    return total / B


def lightglue_loss_deep(
    layer_outputs: list,
    seg_corr_list: list,
    masks0_list: list,
    masks1_list: list,
    lambda_match: float = 1.0,
    layer_weights: list = None,
) -> Tensor:
    """Deep supervision: weighted sum of lightglue_loss at every layer."""
    n = len(layer_outputs)
    if layer_weights is None:
        layer_weights = [(i + 1) / n for i in range(n)]

    total = layer_outputs[0][0].new_zeros(1)
    for (lm, m0, m1), w in zip(layer_outputs, layer_weights):
        total = total + w * lightglue_loss(
            lm, m0, m1, seg_corr_list, masks0_list, masks1_list, lambda_match
        )
    return total


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_matching_metrics_lg(
    log_mutual: Tensor,     # (B, M_max, N_max)
    match0: Tensor,         # (B, M_max) matchability logits
    seg_corr_list: list,
    masks0_list: list,
    masks1_list: list,
) -> dict:
    """
    matching_accuracy, recall_at_1, recall_at_5, auprc,
    mean_gt_logprob, dustbin_rate.
    """
    total_correct = 0
    total_correct_top5 = 0
    total_gt = 0
    total_logprob = 0.0
    total_segs = 0
    total_unmatch = 0
    all_scores = []
    all_labels = []

    for b in range(log_mutual.shape[0]):
        M = masks0_list[b].shape[0]
        N = masks1_list[b].shape[0]
        corr = seg_corr_list[b]
        lm = log_mutual[b, :M, :N]
        m0_b = match0[b, :M]

        if corr.shape[0] > 0:
            pred_j = lm.argmax(dim=1)
            _, topk_j = lm.topk(min(5, N), dim=1)

            gt_map = {}
            for k in range(corr.shape[0]):
                i0, j0 = corr[k, 0].item(), corr[k, 1].item()
                if i0 < M and j0 < N:
                    total_correct += int(pred_j[i0].item() == j0)
                    total_correct_top5 += int(j0 in topk_j[i0].tolist())
                    total_gt += 1
                    total_logprob += lm[i0, j0].item()
                    gt_map[i0] = j0

            for i0, j0 in gt_map.items():
                row_scores = lm[i0]
                for j in range(N):
                    all_scores.append(row_scores[j].item())
                    all_labels.append(1.0 if j == j0 else 0.0)

        total_segs += M
        total_unmatch += (m0_b < 0).sum().item()

    auprc = 0.0
    if all_scores:
        from muviseg.training.utils import _compute_auprc
        auprc = _compute_auprc(torch.tensor(all_scores), torch.tensor(all_labels))

    return {
        "matching_accuracy": total_correct / max(total_gt, 1),
        "recall_at_1": total_correct / max(total_gt, 1),
        "recall_at_5": total_correct_top5 / max(total_gt, 1),
        "auprc": auprc,
        "mean_gt_logprob": total_logprob / max(total_gt, 1),
        "dustbin_rate": total_unmatch / max(total_segs, 1),
    }
