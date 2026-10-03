"""
SegMASt3RLG — LightGlue-style architecture for segment matching.

Architecture vs original SegMASt3R (sinkhorn):
  Before:  dsc0, dsc1 → [Sinkhorn OT] → log_P
  Now:     dsc0, dsc1 → [proj] → [L × self+cross attention] → [double-softmax] → log_mutual + matchability

Trainable parameters:
  - Input projection:      24 → proj_dim               (~3K)
  - L attention layers:    self+cross+FFN per layer     (~150K × L)
  - DoubleSoftmaxMatcher:  matchability MLP             (~4K)
  Total (L=3, dim=128):    ~460K  (vs 1 scalar in sinkhorn version)

Optional deep supervision: compute loss at every attention layer output,
not just the last one. Enables faster convergence (see LightGlue paper).

Optional gradient checkpointing: trades compute for memory in attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from muviseg.paths import setup_segmast3r_path
setup_segmast3r_path()

from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling
from src.models.mast3r_segfeat.segment_attention import SegmentAttentionLayer
from src.models.mast3r_segfeat.double_softmax_matcher import DoubleSoftmaxMatcher


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SegMASt3RLG(nn.Module):
    """
    Args:
        mast3r_ckpt:         path to MASt3R checkpoint
        proj_dim:            project MASt3R 24-dim descriptors to this dimension
        n_layers:            number of self+cross attention blocks
        n_heads:             attention heads per block
        use_grad_checkpoint: gradient checkpointing for attention layers (saves VRAM)
        deep_supervision:    if True, forward returns scores from every layer
        device:              device for backbone loading
    """

    DESC_DIM = 24  # MASt3R output descriptor dim

    def __init__(
        self,
        mast3r_ckpt: str,
        proj_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        use_grad_checkpoint: bool = False,
        deep_supervision: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        self.proj_dim           = proj_dim
        self.use_grad_checkpoint = use_grad_checkpoint
        self.deep_supervision   = deep_supervision

        # Frozen MASt3R backbone
        self.backbone = self._load_backbone(mast3r_ckpt, device)
        self._freeze(self.backbone)

        # Input projection: MASt3R 24-dim → proj_dim
        self.proj = nn.Linear(self.DESC_DIM, proj_dim)
        nn.init.xavier_uniform_(self.proj.weight)

        # Refinement layers (self + cross attention)
        self.attn_layers = nn.ModuleList([
            SegmentAttentionLayer(dim=proj_dim, num_heads=n_heads)
            for _ in range(n_layers)
        ])

        # Matching head
        self.matcher = DoubleSoftmaxMatcher(desc_dim=proj_dim)

    # ------------------------------------------------------------------

    @staticmethod
    def _load_backbone(ckpt_path: str, device: str):
        from mast3r_src.mast3r.model import load_model
        model = load_model(ckpt_path, device=device, verbose=True)
        model.eval()
        return model

    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------

    def extract_desc(
        self, img0: torch.Tensor, img1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract dense per-pixel descriptors from MASt3R using cross-pair.

        Args:
            img0: (B, 3, H, W) normalized [-1, 1]
            img1: (B, 3, H, W) normalized [-1, 1]
        Returns:
            desc0: (B, 24, H, W)
            desc1: (B, 24, H, W)
        """
        B = img0.shape[0]
        view0 = {
            "img": img0,
            "instance": [f"0_{i}" for i in range(B)],
        }
        view1 = {
            "img": img1,
            "instance": [f"1_{i}" for i in range(B)],
        }
        with torch.no_grad():
            pred0, pred1 = self.backbone(view0, view1)
        desc0 = pred0["desc"].permute(0, 3, 1, 2).contiguous()
        desc1 = pred1["desc"].permute(0, 3, 1, 2).contiguous()
        return desc0, desc1

    # ------------------------------------------------------------------

    def forward(
        self,
        img0: torch.Tensor,              # (B, 3, H, W)   — None if precomputed
        img1: torch.Tensor,              # (B, 3, H, W)   — None if precomputed
        masks0: torch.Tensor = None,     # (B, M, H, W)   — None if precomputed
        masks1: torch.Tensor = None,     # (B, N, H, W)   — None if precomputed
        dsc0_pre: torch.Tensor = None,   # (B, M, 24) precomputed pooled descriptors
        dsc1_pre: torch.Tensor = None,   # (B, N, 24) precomputed pooled descriptors
    ):
        """
        Returns (when deep_supervision=False):
            log_mutual  (B, M, N)   log mutual agreement scores
            match0      (B, M)      matchability logits img0
            match1      (B, N)      matchability logits img1
            dsc0        (B, 24, M)  raw MASt3R segment descriptors img0
            dsc1        (B, 24, N)  raw MASt3R segment descriptors img1

        Returns (when deep_supervision=True):
            layer_outputs  list of (log_mutual, match0, match1) per layer
            dsc0, dsc1     same as above

        Precomputed mode (dsc0_pre/dsc1_pre provided):
            Skip backbone forward and masked average pooling.
            dsc0_pre/dsc1_pre: (B, M, 24) float — output of pooling, pre-padded.
        """
        if dsc0_pre is not None:
            # ── Precomputed path: skip backbone + pooling ──────────────
            # dsc0_pre: (B, M, 24) → (B, 24, M) to match pooling output convention
            dsc0 = dsc0_pre.transpose(1, 2).to(dtype=self.proj.weight.dtype)  # (B, 24, M)
            dsc1 = dsc1_pre.transpose(1, 2).to(dtype=self.proj.weight.dtype)  # (B, 24, N)
        else:
            # ── Extract frozen backbone features ──────────────────────────
            feat0, feat1 = self.extract_desc(img0, img1)   # (B, 24, H, W) each

            # Resize masks to descriptor grid if needed
            _, _, dH, dW = feat0.shape
            if masks0.shape[-2:] != (dH, dW):
                masks0 = F.interpolate(masks0.float(), (dH, dW), mode="nearest")
                masks1 = F.interpolate(masks1.float(), (dH, dW), mode="nearest")

            # ── Masked average pooling → segment descriptors ──────────────
            dsc0 = masked_average_pooling(feat0, masks0.float())  # (B, 24, M)
            dsc1 = masked_average_pooling(feat1, masks1.float())  # (B, 24, N)

        # ── Project to attention dim ───────────────────────────────────
        # (B, 24, M) → (B, M, 24) → linear → (B, M, proj_dim)
        x0 = self.proj(dsc0.transpose(1, 2))   # (B, M, proj_dim)
        x1 = self.proj(dsc1.transpose(1, 2))   # (B, N, proj_dim)

        # ── Attention refinement ───────────────────────────────────────
        layer_outputs = []
        for layer in self.attn_layers:
            if self.use_grad_checkpoint and self.training:
                x0, x1 = checkpoint(layer, x0, x1, use_reentrant=False)
            else:
                x0, x1 = layer(x0, x1)

            if self.deep_supervision:
                # Compute matching scores at this layer for deep supervision loss
                lm, m0, m1 = self.matcher(
                    x0.transpose(1, 2),   # (B, proj_dim, M)
                    x1.transpose(1, 2),   # (B, proj_dim, N)
                )
                layer_outputs.append((lm, m0, m1))

        # ── Final matching head ────────────────────────────────────────
        if not self.deep_supervision:
            log_mutual, match0, match1 = self.matcher(
                x0.transpose(1, 2),
                x1.transpose(1, 2),
            )
            return log_mutual, match0, match1, dsc0, dsc1
        else:
            # Last layer output is authoritative; earlier layers for deep supervision
            return layer_outputs, dsc0, dsc1


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def lightglue_loss(
    log_mutual: torch.Tensor,   # (B, M_max, N_max)
    match0: torch.Tensor,       # (B, M_max)   matchability logits img0
    match1: torch.Tensor,       # (B, N_max)   matchability logits img1
    seg_corr_list: list,        # list[B] of (K, 2) GT pairs
    masks0_list: list,          # list[B] of (M_b, H, W) — for actual M per sample
    masks1_list: list,
    lambda_match: float = 1.0,  # weight for matchability BCE loss
) -> torch.Tensor:
    """
    For each sample:
      - Matched pair (i,j):   -log_mutual[i,j]              (maximize mutual agreement)
                            + BCE(matchability0[i], 1)      (i is matchable)
                            + BCE(matchability1[j], 1)      (j is matchable)
      - Unmatched i:          BCE(matchability0[i], 0)       (i has no match)
      - Unmatched j:          BCE(matchability1[j], 0)       (j has no match)
    """
    total = log_mutual.new_zeros(1)
    B = log_mutual.shape[0]

    for b in range(B):
        M = masks0_list[b].shape[0]
        N = masks1_list[b].shape[0]
        corr = seg_corr_list[b].to(log_mutual.device)

        lm = log_mutual[b, :M, :N]    # (M, N)
        m0 = match0[b, :M]            # (M,) logits
        m1 = match1[b, :N]            # (N,) logits

        # GT matchability targets (0 = no match, 1 = has match)
        gt0 = torch.zeros(M, device=log_mutual.device)
        gt1 = torch.zeros(N, device=log_mutual.device)

        match_loss = lm.new_zeros(1)
        n_matched = 0

        if corr.shape[0] > 0:
            valid = (corr[:, 0] < M) & (corr[:, 1] < N)
            corr_v = corr[valid]
            if corr_v.shape[0] > 0:
                i_idx = corr_v[:, 0]
                j_idx = corr_v[:, 1]
                # Maximize mutual agreement on GT pairs
                match_loss = -lm[i_idx, j_idx].mean()
                n_matched   = corr_v.shape[0]
                # Mark these segments as matchable
                gt0[i_idx] = 1.0
                gt1[j_idx] = 1.0

        # Matchability BCE — supervises which segments have a pair
        bce0 = F.binary_cross_entropy_with_logits(m0, gt0)
        bce1 = F.binary_cross_entropy_with_logits(m1, gt1)

        sample_loss = match_loss + lambda_match * (bce0 + bce1)
        total = total + sample_loss

    return total / B


def lightglue_loss_deep(
    layer_outputs: list,        # [(log_mutual, match0, match1), ...]  one per layer
    seg_corr_list: list,
    masks0_list: list,
    masks1_list: list,
    lambda_match: float = 1.0,
    layer_weights: list = None, # per-layer loss weights (default: uniform)
) -> torch.Tensor:
    """
    Deep supervision: compute lightglue_loss at every attention layer output
    and take a weighted sum. Earlier layers get lower weight.
    """
    n = len(layer_outputs)
    if layer_weights is None:
        # Linear ramp: last layer has highest weight
        layer_weights = [(i + 1) / n for i in range(n)]

    total = layer_outputs[0][0].new_zeros(1)
    for (lm, m0, m1), w in zip(layer_outputs, layer_weights):
        total = total + w * lightglue_loss(lm, m0, m1, seg_corr_list,
                                           masks0_list, masks1_list, lambda_match)
    return total


# ---------------------------------------------------------------------------
# Metrics (LightGlue-compatible, no dustbin)
# ---------------------------------------------------------------------------

def compute_matching_metrics_lg(
    log_mutual: torch.Tensor,   # (B, M_max, N_max)
    match0: torch.Tensor,       # (B, M_max) matchability logits
    seg_corr_list: list,
    masks0_list: list,
    masks1_list: list,
) -> dict:
    """
    matching_accuracy — fraction of GT pairs where row-wise argmax is correct
    recall_at_1       — same as matching_accuracy
    recall_at_5       — fraction of GT pairs where correct target is in top-5
    auprc             — Area Under Precision-Recall Curve
    mean_gt_logprob   — mean log_mutual[i,j] for GT pairs
    dustbin_rate      — fraction of segments predicted as unmatched (matchability < 0)
    """
    total_correct = 0
    total_correct_top5 = 0
    total_gt      = 0
    total_logprob = 0.0
    total_segs    = 0
    total_unmatch = 0
    all_scores = []
    all_labels = []

    for b in range(log_mutual.shape[0]):
        M    = masks0_list[b].shape[0]
        N    = masks1_list[b].shape[0]
        corr = seg_corr_list[b]
        lm   = log_mutual[b, :M, :N]    # (M, N)
        m0_b = match0[b, :M]            # (M,) logits

        if corr.shape[0] > 0:
            pred_j = lm.argmax(dim=1)   # (M,)
            _, topk_j = lm.topk(min(5, N), dim=1)  # (M, min(5,N))

            gt_map = {}
            for k in range(corr.shape[0]):
                i0, j0 = corr[k, 0].item(), corr[k, 1].item()
                if i0 < M and j0 < N:
                    total_correct += int(pred_j[i0].item() == j0)
                    total_correct_top5 += int(j0 in topk_j[i0].tolist())
                    total_gt      += 1
                    total_logprob += lm[i0, j0].item()
                    gt_map[i0] = j0

            # AUPRC: for each matched row, collect scores
            for i0, j0 in gt_map.items():
                row_scores = lm[i0]  # (N,)
                for j in range(N):
                    all_scores.append(row_scores[j].item())
                    all_labels.append(1.0 if j == j0 else 0.0)

        total_segs    += M
        total_unmatch += (m0_b < 0).sum().item()   # sigmoid < 0.5 → predicted unmatched

    auprc = 0.0
    if all_scores:
        from muviseg.training.utils import _compute_auprc
        auprc = _compute_auprc(torch.tensor(all_scores), torch.tensor(all_labels))

    return {
        "matching_accuracy": total_correct / max(total_gt, 1),
        "recall_at_1":       total_correct / max(total_gt, 1),
        "recall_at_5":       total_correct_top5 / max(total_gt, 1),
        "auprc":             auprc,
        "mean_gt_logprob":   total_logprob / max(total_gt, 1),
        "dustbin_rate":      total_unmatch / max(total_segs, 1),
    }
