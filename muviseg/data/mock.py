"""
Synthetic dataset for --mock smoke tests.

No external dependencies (no MASt3R, no submodule imports).
Produces the same dict schema as ScanNetPPSegDataset so the rest of
the pipeline (trainer, validator, collate) works unchanged.
"""

import random

import torch
from torch.utils.data import Dataset


class MockDataset(Dataset):
    def __init__(self, n: int = 128, img_size: int = 64, desc_dim: int = 24):
        self.n = n
        self.img_size = img_size
        self.desc_dim = desc_dim

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        S = self.img_size
        n_masks0 = random.randint(5, 10)
        n_masks1 = random.randint(5, 10)

        masks0 = (torch.rand(n_masks0, S, S) > 0.7).to(torch.uint8)
        masks1 = (torch.rand(n_masks1, S, S) > 0.7).to(torch.uint8)

        # Random correspondences (subset of valid pairs)
        n_corr = random.randint(1, min(n_masks0, n_masks1))
        idxs0 = random.sample(range(n_masks0), n_corr)
        idxs1 = random.sample(range(n_masks1), n_corr)
        seg_corr = torch.tensor(list(zip(idxs0, idxs1)), dtype=torch.long)

        return {
            "img0": torch.randn(3, S, S),
            "img1": torch.randn(3, S, S),
            "masks0": masks0,
            "masks1": masks1,
            "seg_corr": seg_corr,
            "valid": True,
            "scene": f"mock_scene_{idx % 4}",
            "name_i": f"mock_{idx:04d}_i",
            "name_j": f"mock_{idx:04d}_j",
        }


def mock_collate(batch):
    return {
        "img0": torch.stack([b["img0"] for b in batch]),
        "img1": torch.stack([b["img1"] for b in batch]),
        "masks0": [b["masks0"] for b in batch],
        "masks1": [b["masks1"] for b in batch],
        "seg_corr": [b["seg_corr"] for b in batch],
        "valid": torch.tensor([b["valid"] for b in batch]),
        "scene": [b["scene"] for b in batch],
        "name_i": [b["name_i"] for b in batch],
        "name_j": [b["name_j"] for b in batch],
    }
