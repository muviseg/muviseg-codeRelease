"""
Segment-level self + cross attention layer for segment matching (v1).

Used by SegMASt3RLG (lightglue.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentAttentionLayer(nn.Module):
    """
    Self + cross attention block with 2x FFN expansion.

    Both images share all weights — equivariant to swap img0 <-> img1.
    Uses F.scaled_dot_product_attention -> FlashAttention on Ampere+.
    """

    def __init__(self, dim: int = 128, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.heads = num_heads
        self.head_dim = dim // num_heads

        # Self-attention
        self.self_norm = nn.LayerNorm(dim)
        self.self_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.self_out = nn.Linear(dim, dim, bias=False)

        # Cross-attention
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_q = nn.Linear(dim, dim, bias=False)
        self.cross_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_out = nn.Linear(dim, dim, bias=False)

        # Feed-forward (2x expansion)
        ffn_dim = dim * 2
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
        Q = self.cross_q(self.cross_norm(x))
        K, V = self.cross_kv(ctx).chunk(2, dim=-1)
        Q = Q.view(B, M, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V)
        return self.cross_out(out.transpose(1, 2).reshape(B, M, D))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor):
        """(B,M,D), (B,N,D) -> refined (B,M,D), (B,N,D)"""
        x0 = x0 + self._self_attn(x0)
        x1 = x1 + self._self_attn(x1)
        x0 = x0 + self._cross_attn(x0, x1)
        x1 = x1 + self._cross_attn(x1, x0)
        x0 = x0 + self.ffn(x0)
        x1 = x1 + self.ffn(x1)
        return x0, x1
