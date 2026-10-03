"""
DoubleSoftmax mutual matching + matchability head for segment matching.

Used by SegMASt3RLG and SegMASt3RLGv2 (LightGlue-style architectures).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
        sim = torch.einsum("bdm,bdn->bmn", desc0, desc1)
        D = desc0.shape[1]
        sim = sim / D ** 0.5

        log_mutual = F.log_softmax(sim, dim=-1) + F.log_softmax(sim, dim=-2)

        match0 = self.matchability(desc0.transpose(1, 2)).squeeze(-1)
        match1 = self.matchability(desc1.transpose(1, 2)).squeeze(-1)

        return log_mutual, match0, match1
