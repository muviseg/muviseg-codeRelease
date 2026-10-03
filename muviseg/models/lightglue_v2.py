"""
SegMASt3RLGv2 — improved LightGlue-style segment matcher.

Changes vs v1 (lightglue.py):
  1. MLP projector: Linear(24→64) + GELU + LayerNorm + Linear(64→dim)
     v1 used a single Linear with no nonlinearity.  The MLP gives the model
     a nonlinear adapter before attention — useful when the input space (24-dim
     MASt3R descriptors) is much smaller than the working dim (128).

  2. FFN expansion 4× instead of 2×.
     Standard transformer FFN uses 4× hidden dim.  v1 used 2× (half the
     capacity).  At dim=128: FFN goes 128→512→128 instead of 128→256→128.

  3. SegmentAttentionLayerV2 is self-contained in this file — no changes to
     segment_attention.py, so both models can coexist.

Architecture:
  dsc (B, M, 24)
    ↓ MLP(24→64→dim) + GELU + LayerNorm
  x  (B, M, dim)
    ↓ N_LAYERS × [Pre-LN self-attn + residual | Pre-LN cross-attn + residual
                  | Pre-LN FFN(4×) + residual]
  x' (B, M, dim)
    ↓ DoubleSoftmaxMatcher
  log_mutual (B, M, N),  match0 (B, M),  match1 (B, N)

Trainable params (dim=128, 3 layers, 4 heads):  ~800K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from muviseg.paths import setup_segmast3r_path
setup_segmast3r_path()

from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling
from src.models.mast3r_segfeat.double_softmax_matcher import DoubleSoftmaxMatcher


# ---------------------------------------------------------------------------
# MLP Projector
# ---------------------------------------------------------------------------

class DescriptorProjector(nn.Module):
    """
    Two-layer MLP: in_dim → mid_dim → out_dim, with GELU + LayerNorm.

    Replaces the single Linear projection in v1.  Gives the model a nonlinear
    adapter between MASt3R's compressed descriptors and the attention working
    space — important when in_dim << out_dim.
    """

    def __init__(self, in_dim: int = 24, mid_dim: int = 64, out_dim: int = 128):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Attention layer (4× FFN, symmetric cross-attn normalization)
# ---------------------------------------------------------------------------

class SegmentAttentionLayerV2(nn.Module):
    """
    Pre-LN self + cross attention block with 4× FFN expansion.

    Changes vs SegmentAttentionLayer (v1):
      - ffn_dim = dim * ffn_expansion (default 4, was 2 in v1)
      - cross_norm_ctx: LayerNorm on context before K/V projection
        (v1 only normalized the query side)

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

        # Feed-forward (Pre-LN, shared between img0 and img1)
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

    def _self_attn(self, x: torch.Tensor) -> torch.Tensor:
        B, M, D = x.shape
        qkv = self.self_qkv(self.self_norm(x))
        Q, K, V = qkv.chunk(3, dim=-1)
        Q = Q.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.self_out(out.transpose(1, 2).reshape(B, M, D))

    def _cross_attn(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, M, D = x.shape
        N = ctx.shape[1]
        Q    = self.cross_q(self.cross_norm(x))
        K, V = self.cross_kv(self.cross_norm_ctx(ctx)).chunk(2, dim=-1)
        Q = Q.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.cross_out(out.transpose(1, 2).reshape(B, M, D))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        """(B,M,D), (B,N,D) → refined (B,M,D), (B,N,D)"""
        x0 = x0 + self._self_attn(x0)
        x1 = x1 + self._self_attn(x1)
        x0 = x0 + self._cross_attn(x0, x1)
        x1 = x1 + self._cross_attn(x1, x0)
        x0 = x0 + self.ffn(x0)
        x1 = x1 + self.ffn(x1)
        return x0, x1


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SegMASt3RLGv2(nn.Module):
    """
    SegMASt3R + LightGlue-style matching head v2.

    Drop-in replacement for SegMASt3RLG.  Identical forward interface.
    """

    DESC_DIM = 24  # MASt3R output descriptor dim

    def __init__(
        self,
        mast3r_ckpt: str,
        proj_dim: int = 128,
        proj_mid_dim: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_expansion: int = 4,
        use_grad_checkpoint: bool = False,
        deep_supervision: bool = False,
        device: str = "cpu",
    ):
        super().__init__()

        self.proj_dim            = proj_dim
        self.use_grad_checkpoint = use_grad_checkpoint
        self.deep_supervision    = deep_supervision

        # Frozen MASt3R backbone
        self.backbone = self._load_backbone(mast3r_ckpt, device)
        self._freeze(self.backbone)

        # MLP projector: 24 → proj_mid_dim → proj_dim  (nonlinear adapter)
        self.proj = DescriptorProjector(
            in_dim=self.DESC_DIM, mid_dim=proj_mid_dim, out_dim=proj_dim
        )

        # Attention refinement layers
        self.attn_layers = nn.ModuleList([
            SegmentAttentionLayerV2(dim=proj_dim, num_heads=n_heads,
                                    ffn_expansion=ffn_expansion)
            for _ in range(n_layers)
        ])

        # Matching head (same as v1)
        self.matcher = DoubleSoftmaxMatcher(desc_dim=proj_dim)

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

    def extract_desc(
        self, img0: torch.Tensor, img1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract dense per-pixel descriptors from MASt3R using cross-pair.

        MASt3R decoder uses cross-attention between view0 and view1.
        Both views must be different images for meaningful descriptors.

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

    def forward(
        self,
        img0=None, img1=None,
        masks0=None, masks1=None,
        dsc0_pre=None, dsc1_pre=None,
    ):
        """
        Same interface as SegMASt3RLG.forward().

        Returns (when deep_supervision=False):
            log_mutual  (B, M, N)   log mutual agreement scores
            match0      (B, M)      matchability logits img0
            match1      (B, N)      matchability logits img1
            dsc0        (B, 24, M)  raw MASt3R segment descriptors
            dsc1        (B, 24, N)  raw MASt3R segment descriptors

        Returns (when deep_supervision=True):
            layer_outputs  list of (log_mutual, match0, match1) per layer
            dsc0, dsc1     same as above
        """
        proj_dtype = next(self.proj.parameters()).dtype

        if dsc0_pre is not None:
            # Precomputed path: skip backbone + pooling
            dsc0 = dsc0_pre.transpose(1, 2).to(dtype=proj_dtype)   # (B, 24, M)
            dsc1 = dsc1_pre.transpose(1, 2).to(dtype=proj_dtype)
        else:
            # Online path: cross-pair backbone extraction
            feat0, feat1 = self.extract_desc(img0, img1)   # (B, 24, H, W) each
            _, _, dH, dW = feat0.shape
            if masks0.shape[-2:] != (dH, dW):
                masks0 = F.interpolate(masks0.float(), (dH, dW), mode="nearest")
                masks1 = F.interpolate(masks1.float(), (dH, dW), mode="nearest")
            dsc0 = masked_average_pooling(feat0, masks0.float())    # (B, 24, M)
            dsc1 = masked_average_pooling(feat1, masks1.float())

        # MLP projection: (B, 24, M) → (B, M, 24) → (B, M, proj_dim)
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


# Re-export loss/metrics from v1 — identical interface, no duplication.
from muviseg.models.lightglue import (  # noqa: E402
    lightglue_loss,
    lightglue_loss_deep,
    compute_matching_metrics_lg,
)
