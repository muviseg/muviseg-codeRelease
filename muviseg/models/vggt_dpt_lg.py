"""
SegVGGT-DPT — VGGT Aggregator + DPT-inspired multi-layer fusion + LightGlue matcher.

Extracts tokens from multiple Aggregator layers (default [5, 11, 17, 23]),
fuses them with DPT-style progressive bottom-up refinement, then matches
segments with LightGlue-style attention + DoubleSoftmax.

Key improvements over single-layer SegVGGT (vggt_lightglue.py):
  - Multi-scale features: texture (layer 5) to semantics (layer 23)
  - DPT-style spatial fusion via conv3×3 RCU blocks before masked pooling
  - Simpler projector (fusion_dim → proj_dim) since heavy lifting is in fusion

Inspired by DPT (Ranftl et al., "Vision Transformers for Dense Prediction",
https://arxiv.org/abs/2103.13413). Adaptations for segment matching:
  - No spatial resample (all Aggregator layers have same H/14 × W/14 grid)
  - No readout projection (no CLS token in VGGT; camera/register info
    already diffused into patch tokens through 24 layers of attention)
  - Pooling by segment masks instead of dense per-pixel output

Alternative projection strategies (not implemented, for future experiments):
  A. [current] Single conv1×1(2048 → D_hat) per layer — standard DPT approach
  B. Two-step conv: conv1×1(2048→512) → ReLU → conv1×1(512→D_hat)
  C. Larger D_hat (512 instead of 256) — preserves more info, doubles RCU params

Architecture (online mode):
  VGGT Aggregator (frozen):
    img0, img1 → tokens from layers [5, 11, 17, 23]
                  each (B, 2, 2048, H_p, W_p)

  DPT Fusion (trainable, per view):
    Layer 23: conv1×1(2048→256) → RCU →                    feat
    Layer 17: conv1×1(2048→256) → RCU → (+) feat → RCU →   feat
    Layer 11: conv1×1(2048→256) → RCU → (+) feat → RCU →   feat
    Layer 5:  conv1×1(2048→256) → RCU → (+) feat → RCU →   feat
                                                             ↓
                                                (B, 256, H_p, W_p) per view
    masked_average_pooling → (B, 256, M)
    ↓ Linear(256→128) + LayerNorm
    ↓ 3× SegmentAttentionLayerV2 (self + cross + FFN 4×)
    ↓ DoubleSoftmaxMatcher
    log_mutual (B, M, N),  match0 (B, M),  match1 (B, N)

Trainable params (~10.7M):
  4× conv1×1(2048→256)           ~2.1M
  7× RCU (256-ch, 2× conv3×3)   ~8.3M
  Projector Linear(256→128)      ~33K
  3× attention layers (128-dim)  ~260K
  Matcher MLP                    ~17K

Online-only: precomputed descriptors from vggt_lightglue are per-segment
(M, 2048), but DPT fusion needs spatial maps (2048, H_p, W_p) before pooling.
Use arch="vggt" for precomputed mode.

Accelerate notes:
  - Aggregator params have requires_grad=False → DDP skips gradient sync
  - extract_desc runs under torch.no_grad() → no activation memory for backbone
  - Fusion convs run in mixed precision (bf16) via accelerate autocast
  - Model created on CPU, accelerator.prepare() handles DDP + device placement
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
    feat_flat = feat.view(B, D, H * W)
    masks_flat = masks.view(B, M, H * W).float()
    area = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1)
    return torch.bmm(feat_flat, masks_flat.transpose(1, 2)) / area.transpose(1, 2)


# ---------------------------------------------------------------------------
# DPT-inspired Fusion modules
# ---------------------------------------------------------------------------

class ResidualConvUnit(nn.Module):
    """
    Residual Conv Unit from DPT (Ranftl et al.).

    x → ReLU → Conv3×3 → ReLU → Conv3×3 → (+x) → out

    No BatchNorm — following DPT for regression-like tasks.
    Identity skip (dims don't change).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        for m in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual


class FusionBlock(nn.Module):
    """
    One level of DPT-style progressive fusion.

    incoming → RCU → (+prev_fused) → RCU → out
    First level (prev_fused=None): incoming → RCU → RCU → out
    """

    def __init__(self, dim: int):
        super().__init__()
        self.rcu1 = ResidualConvUnit(dim)
        self.rcu2 = ResidualConvUnit(dim)

    def forward(self, incoming: Tensor, prev_fused: Tensor | None = None) -> Tensor:
        x = self.rcu1(incoming)
        if prev_fused is not None:
            x = x + prev_fused
        x = self.rcu2(x)
        return x


class MultiLayerFusion(nn.Module):
    """
    DPT-inspired multi-layer fusion.

    Takes features from N Aggregator layers (ordered deep → shallow),
    projects each to D_hat via conv1×1, then progressively fuses with
    element-wise add + Residual Conv Units.

    Args:
        in_dim:   raw token dim from Aggregator (2048)
        d_hat:    fusion output dim (256)
        n_layers: number of Aggregator layers to fuse (4)
    """

    def __init__(self, in_dim: int = 2048, d_hat: int = 256, n_layers: int = 4):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(in_dim, d_hat, kernel_size=1, bias=True)
            for _ in range(n_layers)
        ])
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(d_hat) for _ in range(n_layers)
        ])

        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(self, layer_features: list[Tensor]) -> Tensor:
        """
        Args:
            layer_features: list of N tensors, ordered deep → shallow.
                Each tensor: (B, in_dim, H, W)
        Returns:
            (B, d_hat, H, W) fused feature map
        """
        fused = None
        for proj, block, feat in zip(
            self.projections, self.fusion_blocks, layer_features
        ):
            projected = proj(feat)
            fused = block(projected, fused)
        return fused


# ---------------------------------------------------------------------------
# Projector (fusion_dim → proj_dim)
# ---------------------------------------------------------------------------

class DescriptorProjector(nn.Module):
    """Simple linear projection + LayerNorm after fusion."""

    def __init__(self, in_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.linear(x))


# ---------------------------------------------------------------------------
# Attention layer — identical to vggt_lightglue.SegmentAttentionLayerV2
# Inlined to keep this file self-contained.
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
        self.dim = dim
        self.heads = num_heads
        self.head_dim = dim // num_heads

        # Self-attention (Pre-LN)
        self.self_norm = nn.LayerNorm(dim)
        self.self_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.self_out = nn.Linear(dim, dim, bias=False)

        # Cross-attention (Pre-LN on both query and context)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_norm_ctx = nn.LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_out = nn.Linear(dim, dim, bias=False)

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
        Q = self.cross_q(self.cross_norm(x))
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
# Joint multi-frame segment attention
# ---------------------------------------------------------------------------

class JointSegmentAttention(nn.Module):
    """
    Self-attention over concatenated segment descriptors from all N frames.

    Mirrors VGGT Aggregator's global attention pattern at the segment level:
    all segments from all frames attend to each other in a single pass,
    enabling transitive matching (A↔C through shared context with B).

    Learnable frame embeddings let the model distinguish which frame each
    segment belongs to (up to max_frames).
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        ffn_expansion: int = 4,
        max_frames: int = 16,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.heads = num_heads
        self.head_dim = dim // num_heads

        # Learnable frame position embeddings
        self.frame_embed = nn.Embedding(max_frames, dim)

        # Self-attention (Pre-LN)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

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

    def forward(
        self,
        segments: list[Tensor],
        frame_embeds_added: bool = False,
    ) -> list[Tensor]:
        """
        Args:
            segments: list of N tensors, each (B, M_k, D).
            frame_embeds_added: if True, skip adding frame embeddings
                (caller already added them before the first layer).
        Returns:
            list of N tensors (B, M_k, D), same shapes as input.
        """
        B = segments[0].shape[0]
        N = len(segments)

        # Record split sizes and concatenate
        sizes = [s.shape[1] for s in segments]

        if not frame_embeds_added:
            # Add frame embeddings only on first layer call
            parts = [s + self.frame_embed.weight[v] for v, s in enumerate(segments)]
            x = torch.cat(parts, dim=1)
        else:
            x = torch.cat(segments, dim=1)

        L = x.shape[1]

        # Self-attention
        normed = self.norm(x)
        qkv = self.qkv(normed)
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, self.dim)
        x = x + self.out_proj(attn_out)

        # FFN
        x = x + self.ffn(x)

        # Split back into per-frame tensors
        return list(x.split(sizes, dim=1))


# ---------------------------------------------------------------------------
# DoubleSoftmax Matcher — inline implementation
# ---------------------------------------------------------------------------

class DoubleSoftmaxMatcher(nn.Module):
    """
    Double-softmax mutual matching + matchability MLP.

    Computes log(softmax_row * softmax_col) as mutual agreement score,
    plus per-segment matchability logits via a small MLP.
    """

    def __init__(self, desc_dim: int = 128, matchability_bias: float = 0.0,
                 temperature_init: float = 1.0):
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

        # Bias the final matchability logit: positive = more likely to match,
        # negative = more likely dustbin.  Default 0.0 = neutral.
        with torch.no_grad():
            self.matchability[-1].bias.fill_(matchability_bias)

        # Learnable temperature for similarity scaling (clamped ≤ 1.0).
        # tau < 1 → sharper softmax. Clamping prevents the model from
        # smoothing scores to trivially reduce BCE loss.
        self.log_tau = nn.Parameter(torch.tensor(float(temperature_init)).log())

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
        sim = torch.einsum("bdm,bdn->bmn", desc0, desc1)
        D = desc0.shape[1]
        tau = self.log_tau.exp().clamp(max=1.0)
        sim = sim / (D ** 0.5 * tau)

        log_mutual = F.log_softmax(sim, dim=-1) + F.log_softmax(sim, dim=-2)

        match0 = self.matchability(desc0.transpose(1, 2)).squeeze(-1)
        match1 = self.matchability(desc1.transpose(1, 2)).squeeze(-1)

        return log_mutual, match0, match1


# ---------------------------------------------------------------------------
# VGGT Aggregator
# ---------------------------------------------------------------------------
setup_vggt_path()
from vggt.models.aggregator import Aggregator  # noqa: E402


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SegVGGTDPT(nn.Module):
    """
    VGGT Aggregator + DPT-inspired multi-layer fusion + LightGlue matcher.

    The Aggregator is frozen; fusion, projector, attention layers, and
    matcher are trained.  Online-only (no precomputed descriptor support).

    Forward interface is identical to SegVGGT / SegMASt3RLGv2 so the
    trainer works without modification.
    """

    AGG_DIM = 2048  # concatenated frame + global token dim per Aggregator layer

    def __init__(
        self,
        vggt_ckpt: str,
        layer_indices: tuple[int, ...] = (5, 11, 17, 23),
        fusion_dim: int = 256,
        proj_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        use_grad_checkpoint: bool = False,
        deep_supervision: bool = False,
        matchability_bias: float = 0.0,
        temperature_init: float = 1.0,
        device: str = "cpu",
    ):
        super().__init__()

        # Store sorted deep → shallow for fusion order
        self.layer_indices = sorted(layer_indices, reverse=True)
        self.fusion_dim = fusion_dim
        self.proj_dim = proj_dim
        self.use_grad_checkpoint = use_grad_checkpoint
        self.deep_supervision = deep_supervision

        # Frozen VGGT Aggregator
        self.aggregator = self._load_aggregator(vggt_ckpt, device)
        self._freeze(self.aggregator)

        # DPT-inspired multi-layer fusion
        self.fusion = MultiLayerFusion(
            in_dim=self.AGG_DIM,
            d_hat=fusion_dim,
            n_layers=len(self.layer_indices),
        )

        # Projector: fusion_dim → proj_dim
        self.proj = DescriptorProjector(in_dim=fusion_dim, out_dim=proj_dim)

        # Attention refinement layers
        self.attn_layers = nn.ModuleList([
            SegmentAttentionLayerV2(
                dim=proj_dim, num_heads=n_heads, ffn_expansion=ffn_expansion
            )
            for _ in range(n_layers)
        ])

        # Matching head
        self.matcher = DoubleSoftmaxMatcher(
            desc_dim=proj_dim,
            matchability_bias=matchability_bias,
            temperature_init=temperature_init,
        )

    @staticmethod
    def _load_aggregator(ckpt_path: str, device: str) -> Aggregator:
        """Load VGGT Aggregator with frozen weights."""
        aggregator = Aggregator()

        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        else:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=ckpt_path, filename="model.pt")
            state = torch.load(local, map_location=device, weights_only=True)

        prefix = "aggregator."
        agg_state = {
            k[len(prefix):]: v
            for k, v in state.items()
            if k.startswith(prefix)
        }
        if not agg_state:
            agg_state = state

        aggregator.load_state_dict(agg_state, strict=True)
        aggregator.eval()
        return aggregator

    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)

    def extract_multi_layer(
        self, img0: Tensor, img1: Tensor,
    ) -> tuple[list[Tensor], list[Tensor], int, int]:
        """
        Extract dense patch features from multiple Aggregator layers.

        Args:
            img0: (B, 3, H, W) in [-1, 1] range (dataset convention).
            img1: (B, 3, H, W) in [-1, 1] range.
        Returns:
            feats_0: list of N tensors (B, 2048, H_p, W_p), deep → shallow
            feats_1: list of N tensors (B, 2048, H_p, W_p), deep → shallow
            H_p: patch grid height
            W_p: patch grid width
        """
        # [-1, 1] → [0, 1]
        img0 = (img0 + 1.0) * 0.5
        img1 = (img1 + 1.0) * 0.5

        # Resize to nearest multiple of patch_size=14
        B, _, H, W = img0.shape
        H_new = round(H / 14) * 14
        W_new = round(W / 14) * 14
        if (H_new, W_new) != (H, W):
            img0 = F.interpolate(
                img0, (H_new, W_new), mode="bilinear", align_corners=False
            )
            img1 = F.interpolate(
                img1, (H_new, W_new), mode="bilinear", align_corners=False
            )
        H_p, W_p = H_new // 14, W_new // 14

        # Stack pair: (B, 2, 3, H, W)
        images = torch.stack([img0, img1], dim=1)

        with torch.no_grad():
            aggregated_tokens_list, ps_idx = self.aggregator(images)

        feats_0 = []
        feats_1 = []
        for layer_idx in self.layer_indices:  # deep → shallow
            tokens = aggregated_tokens_list[layer_idx]  # (B, 2, P, 2048)
            patch_tokens = tokens[:, :, ps_idx:]
            patch_tokens = patch_tokens[:, :, :H_p * W_p]
            patch_tokens = patch_tokens.view(B, 2, H_p, W_p, self.AGG_DIM)
            patch_tokens = patch_tokens.permute(0, 1, 4, 2, 3)
            feats_0.append(patch_tokens[:, 0].contiguous())
            feats_1.append(patch_tokens[:, 1].contiguous())

        return feats_0, feats_1, H_p, W_p

    # ------------------------------------------------------------------
    # Multi-frame extraction (variable N)
    # ------------------------------------------------------------------

    def extract_multi_layer_nframes(
        self, images: Tensor,
    ) -> tuple[list[list[Tensor]], int, int]:
        """
        Extract dense patch features from multiple Aggregator layers for N frames.

        Args:
            images: (B, N, 3, H, W) in [-1, 1] range (dataset convention).
        Returns:
            all_feats: list of N items, each a list of len(layer_indices) tensors
                       (B, 2048, H_p, W_p), ordered deep → shallow.
            H_p: patch grid height
            W_p: patch grid width
        """
        B, N = images.shape[:2]

        # [-1, 1] → [0, 1]
        images = (images + 1.0) * 0.5

        # Resize to nearest multiple of patch_size=14
        H, W = images.shape[-2:]
        H_new = round(H / 14) * 14
        W_new = round(W / 14) * 14
        if (H_new, W_new) != (H, W):
            images = images.view(B * N, 3, H, W)
            images = F.interpolate(
                images, (H_new, W_new), mode="bilinear", align_corners=False
            )
            images = images.view(B, N, 3, H_new, W_new)
        H_p, W_p = H_new // 14, W_new // 14

        with torch.no_grad():
            aggregated_tokens_list, ps_idx = self.aggregator(images)

        # all_feats[v] = list of layer features for view v
        all_feats = [[] for _ in range(N)]
        for layer_idx in self.layer_indices:  # deep → shallow
            tokens = aggregated_tokens_list[layer_idx]  # (B, N, P, 2048)
            patch_tokens = tokens[:, :, ps_idx:]
            patch_tokens = patch_tokens[:, :, :H_p * W_p]
            patch_tokens = patch_tokens.view(B, N, H_p, W_p, self.AGG_DIM)
            patch_tokens = patch_tokens.permute(0, 1, 4, 2, 3)  # (B, N, 2048, H_p, W_p)
            for v in range(N):
                all_feats[v].append(patch_tokens[:, v].contiguous())

        return all_feats, H_p, W_p

    def forward_multiframe(
        self,
        images: Tensor,
        masks_list: list[Tensor],
        pair_indices: list[tuple[int, int]],
    ) -> dict[tuple[int, int], tuple[Tensor, Tensor, Tensor]]:
        """
        Multi-frame forward: N frames through backbone, pairwise matching head.

        Args:
            images: (B, N, 3, H, W) in [-1, 1] range.
            masks_list: list of N tensors, each (B, M_k, H, W).
            pair_indices: list of (a, b) frame-index pairs to match.
        Returns:
            dict mapping (a, b) → (log_mutual, match0, match1)
        """
        B, N = images.shape[:2]

        # 1. Extract multi-layer features for all N frames
        all_feats, H_p, W_p = self.extract_multi_layer_nframes(images)

        # 2. DPT fusion per view (shared weights)
        fused = [self.fusion(all_feats[v]) for v in range(N)]

        # 3. Masked pooling + projection per view
        proj_dtype = next(self.proj.parameters()).dtype
        projs = []
        for v in range(N):
            m = masks_list[v]
            if m.shape[-2:] != (H_p, W_p):
                m = F.interpolate(m.float(), (H_p, W_p), mode="nearest")
            dsc = masked_average_pooling(fused[v], m.float())  # (B, fusion_dim, M_k)
            proj = self.proj(dsc.transpose(1, 2).to(dtype=proj_dtype))  # (B, M_k, proj_dim)
            projs.append(proj)

        # 4. Pairwise matching for each requested pair
        results = {}
        for (a, b) in pair_indices:
            x0, x1 = projs[a].clone(), projs[b].clone()

            for layer in self.attn_layers:
                if self.use_grad_checkpoint and self.training:
                    x0, x1 = checkpoint(layer, x0, x1, use_reentrant=False)
                else:
                    x0, x1 = layer(x0, x1)

            log_mutual, m0, m1 = self.matcher(
                x0.transpose(1, 2), x1.transpose(1, 2)
            )
            results[(a, b)] = (log_mutual, m0, m1)

        return results

    # ------------------------------------------------------------------
    # Original pairwise forward (backward compatible)
    # ------------------------------------------------------------------

    def forward(
        self,
        img0: Tensor, img1: Tensor,
        masks0: Tensor, masks1: Tensor,
    ):
        """
        Online-only forward pass with DPT fusion.

        Returns (when deep_supervision=False):
            log_mutual  (B, M, N)
            match0      (B, M)
            match1      (B, N)
            dsc0        (B, fusion_dim, M)   fused segment descriptors
            dsc1        (B, fusion_dim, N)

        Returns (when deep_supervision=True):
            layer_outputs  list of (log_mutual, match0, match1) per layer
            dsc0, dsc1     same as above
        """
        # 1. Extract multi-layer features from frozen Aggregator
        feats_0, feats_1, H_p, W_p = self.extract_multi_layer(img0, img1)

        # 2. DPT fusion per view → (B, fusion_dim, H_p, W_p)
        fused_0 = self.fusion(feats_0)
        fused_1 = self.fusion(feats_1)

        # 3. Interpolate masks to match patch grid
        if masks0.shape[-2:] != (H_p, W_p):
            masks0 = F.interpolate(masks0.float(), (H_p, W_p), mode="nearest")
            masks1 = F.interpolate(masks1.float(), (H_p, W_p), mode="nearest")

        # 4. Masked average pooling → per-segment descriptors
        dsc0 = masked_average_pooling(fused_0, masks0.float())  # (B, fusion_dim, M)
        dsc1 = masked_average_pooling(fused_1, masks1.float())

        # 5. Project fusion_dim → proj_dim
        proj_dtype = next(self.proj.parameters()).dtype
        x0 = self.proj(dsc0.transpose(1, 2).to(dtype=proj_dtype))  # (B, M, proj_dim)
        x1 = self.proj(dsc1.transpose(1, 2).to(dtype=proj_dtype))

        # 6. Attention refinement
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

        # 7. Final matching
        if not self.deep_supervision:
            log_mutual, match0, match1 = self.matcher(
                x0.transpose(1, 2), x1.transpose(1, 2)
            )
            return log_mutual, match0, match1, dsc0, dsc1
        else:
            return layer_outputs, dsc0, dsc1


# ---------------------------------------------------------------------------
# Joint multi-frame variant (separate arch: vggt_dpt_joint)
# ---------------------------------------------------------------------------

class SegVGGTDPTJoint(SegVGGTDPT):
    """
    SegVGGT-DPT with joint multi-frame segment attention.

    Replaces per-pair attention refinement with a single self-attention pass
    over all segments from all N frames.  Pairwise scores are extracted
    afterwards via DoubleSoftmaxMatcher.

    Inherits backbone, fusion, projector, and matcher from SegVGGTDPT.
    Adds joint_attn_layers (JointSegmentAttention) as the only trainable
    attention component — attn_layers (pairwise) are not used.
    """

    def __init__(
        self,
        vggt_ckpt: str,
        layer_indices: tuple[int, ...] = (5, 11, 17, 23),
        fusion_dim: int = 256,
        proj_dim: int = 128,
        n_joint_layers: int = 3,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        max_frames: int = 16,
        use_grad_checkpoint: bool = False,
        matchability_bias: float = 0.0,
        temperature_init: float = 1.0,
        device: str = "cpu",
    ):
        # Parent creates aggregator, fusion, proj, matcher.
        # n_layers=0 → empty attn_layers (not used in joint mode).
        super().__init__(
            vggt_ckpt=vggt_ckpt,
            layer_indices=layer_indices,
            fusion_dim=fusion_dim,
            proj_dim=proj_dim,
            n_layers=0,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            use_grad_checkpoint=use_grad_checkpoint,
            deep_supervision=False,
            matchability_bias=matchability_bias,
            temperature_init=temperature_init,
            device=device,
        )

        self.joint_attn_layers = nn.ModuleList([
            JointSegmentAttention(
                dim=proj_dim, num_heads=n_heads,
                ffn_expansion=ffn_expansion, max_frames=max_frames,
            )
            for _ in range(n_joint_layers)
        ])

    def forward_multiframe(
        self,
        images: Tensor,
        masks_list: list[Tensor],
        pair_indices: list[tuple[int, int]],
    ) -> dict[tuple[int, int], tuple[Tensor, Tensor, Tensor]]:
        """
        Joint multi-frame forward.

        Pipeline:
          1. Frozen VGGT Aggregator → multi-layer features for N frames
          2. DPT fusion per view → spatial feature maps
          3. Masked pooling + projection → per-segment descriptors
          4. Joint self-attention over ALL segments from ALL frames (one pass)
          5. DoubleSoftmax pairwise scoring for requested pairs

        Args:
            images: (B, N, 3, H, W) in [-1, 1] range.
            masks_list: list of N tensors, each (B, M_k, H, W).
            pair_indices: list of (a, b) frame-index pairs to match.
        Returns:
            dict mapping (a, b) → (log_mutual, match0, match1)
        """
        B, N = images.shape[:2]

        # 1. Extract multi-layer features for all N frames
        all_feats, H_p, W_p = self.extract_multi_layer_nframes(images)

        # 2. DPT fusion per view (shared weights)
        fused = [self.fusion(all_feats[v]) for v in range(N)]

        # 3. Masked pooling + projection per view
        proj_dtype = next(self.proj.parameters()).dtype
        projs = []
        for v in range(N):
            m = masks_list[v]
            if m.shape[-2:] != (H_p, W_p):
                m = F.interpolate(m.float(), (H_p, W_p), mode="nearest")
            dsc = masked_average_pooling(fused[v], m.float())
            proj = self.proj(dsc.transpose(1, 2).to(dtype=proj_dtype))
            projs.append(proj)

        # 4. Joint self-attention: all segments attend to all segments
        for i, layer in enumerate(self.joint_attn_layers):
            # Frame embeddings added only on first layer
            skip_embed = (i > 0)
            if self.use_grad_checkpoint and self.training:
                projs = checkpoint(layer, projs, skip_embed, use_reentrant=False)
            else:
                projs = layer(projs, frame_embeds_added=skip_embed)

        # 5. Pairwise scoring from joint representations
        results = {}
        for (a, b) in pair_indices:
            log_mutual, m0, m1 = self.matcher(
                projs[a].transpose(1, 2), projs[b].transpose(1, 2)
            )
            results[(a, b)] = (log_mutual, m0, m1)

        return results


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def lightglue_loss(
    log_mutual: Tensor,
    match0: Tensor,
    match1: Tensor,
    seg_corr_list: list,
    masks0_list: list,
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


def _lightglue_loss_single(
    lm: Tensor, m0: Tensor, m1: Tensor,
    corr: Tensor, M: int, N: int,
    lambda_match: float = 1.0,
) -> Tensor:
    """
    Single-sample lightglue loss (no batch dim).

    Args:
        lm: (M_pad, N_pad) log mutual agreement scores
        m0: (M_pad,) matchability logits
        m1: (N_pad,) matchability logits
        corr: (K, 2) GT correspondences
        M: actual number of segments in view 0
        N: actual number of segments in view 1
        lambda_match: BCE weight
    """
    lm = lm[:M, :N]
    m0 = m0[:M]
    m1 = m1[:N]

    gt0 = torch.zeros(M, device=lm.device)
    gt1 = torch.zeros(N, device=lm.device)

    match_loss = lm.new_zeros(1)

    if corr.shape[0] > 0:
        corr = corr.to(lm.device)
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

    return match_loss + lambda_match * (bce0 + bce1)


def multiframe_lightglue_loss(
    pair_results: dict[tuple[int, int], tuple[Tensor, Tensor, Tensor]],
    pair_corrs_batch: list[dict],
    masks_batch: list[list[Tensor]],
    lambda_match: float = 1.0,
) -> Tensor:
    """
    Multi-frame loss with per-tuple normalization.

    Each batch sample (tuple) contributes equally to the loss,
    regardless of how many GT pairs it has.

    Args:
        pair_results: {(a,b): (log_mutual, match0, match1)} from forward_multiframe
        pair_corrs_batch: list[B] of dicts {(a,b): corr_tensor}
        masks_batch: list[B] of list[N] of mask tensors
        lambda_match: BCE weight
    """
    B = len(pair_corrs_batch)
    # Get device from first result tensor
    device = next(iter(pair_results.values()))[0].device
    total = torch.zeros(1, device=device)

    for b in range(B):
        tuple_loss = torch.zeros(1, device=device)
        n_pairs_b = 0

        for (a, ab_b), (lm, m0, m1) in pair_results.items():
            corr = pair_corrs_batch[b].get((a, ab_b))
            if corr is None:
                continue

            M = masks_batch[b][a].shape[0]
            N = masks_batch[b][ab_b].shape[0]

            tuple_loss = tuple_loss + _lightglue_loss_single(
                lm[b], m0[b], m1[b], corr, M, N, lambda_match
            )
            n_pairs_b += 1

        total = total + tuple_loss / max(n_pairs_b, 1)

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
    log_mutual: Tensor,
    match0: Tensor,
    seg_corr_list: list,
    masks0_list: list,
    masks1_list: list,
) -> dict:
    """
    matching_accuracy, recall_at_1, recall_at_5, auprc,
    mean_gt_logprob, dustbin_rate.

    Error decomposition (rates are fractions of total_segs):
      false_dustbin_rate — model → dustbin, but GT match exists (conservatism)
      wrong_match_rate   — model → match, GT match exists, but picked wrong j
      false_match_rate   — model → match, but no GT match (should be dustbin)
    """
    total_correct = 0
    total_correct_top5 = 0
    total_gt = 0
    total_logprob = 0.0
    total_segs = 0
    total_unmatch = 0
    all_scores = []
    all_labels = []

    # Error decomposition counters
    total_false_dustbin = 0   # has GT match, predicted dustbin
    total_wrong_match = 0     # has GT match, predicted match, but wrong
    total_false_match = 0     # no GT match, predicted match

    for b in range(log_mutual.shape[0]):
        M = masks0_list[b].shape[0]
        N = masks1_list[b].shape[0]
        corr = seg_corr_list[b]
        lm = log_mutual[b, :M, :N]
        m0_b = match0[b, :M]

        # Build GT map: which segments in view0 have a GT match
        gt_map = {}
        if corr.shape[0] > 0:
            pred_j = lm.argmax(dim=1)
            _, topk_j = lm.topk(min(5, N), dim=1)

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

        # Error decomposition per segment
        for i in range(M):
            predicted_dustbin = m0_b[i].item() < 0
            has_gt = i in gt_map

            if has_gt and predicted_dustbin:
                total_false_dustbin += 1
            elif has_gt and not predicted_dustbin:
                pred = lm.argmax(dim=1)[i].item() if corr.shape[0] > 0 else -1
                if pred != gt_map[i]:
                    total_wrong_match += 1
            elif not has_gt and not predicted_dustbin:
                total_false_match += 1

        total_segs += M
        total_unmatch += (m0_b < 0).sum().item()

    auprc = 0.0
    if all_scores:
        from muviseg.training.utils import _compute_auprc
        auprc = _compute_auprc(
            torch.tensor(all_scores), torch.tensor(all_labels)
        )

    return {
        "matching_accuracy": total_correct / max(total_gt, 1),
        "recall_at_1": total_correct / max(total_gt, 1),
        "recall_at_5": total_correct_top5 / max(total_gt, 1),
        "auprc": auprc,
        "mean_gt_logprob": total_logprob / max(total_gt, 1),
        "dustbin_rate": total_unmatch / max(total_segs, 1),
        "false_dustbin_rate": total_false_dustbin / max(total_segs, 1),
        "wrong_match_rate": total_wrong_match / max(total_segs, 1),
        "false_match_rate": total_false_match / max(total_segs, 1),
    }
