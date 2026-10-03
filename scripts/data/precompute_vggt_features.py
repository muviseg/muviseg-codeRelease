"""
Precompute per-pair segment descriptors using VGGT Aggregator.

VGGT Aggregator uses alternating frame + global attention, so descriptors
for img0 depend on which img1 it's paired with (same cross-view dependency
as MASt3R). Per-image precompute is invalid — only per-pair works.

Output layout:
    <output_root>/<scene>/<name_i>__<name_j>.pt
    Contains: {"dsc0": (M0, 2048) float16, "dsc1": (M1, 2048) float16}

WARNING: VGGT descriptors are 2048-dim (vs MASt3R 24-dim) → ~85× larger.
    Estimated disk: ~3.5M pairs × ~1.7MB ≈ 6 TB (fp16).
    For a subset of 100K pairs: ~170 GB.
    Always store as fp16 to halve disk usage.

The Aggregator normalizes images internally with ResNet mean/std.
Input images must be in [0, 1] range — do NOT use MASt3R's [-1, 1].

Usage (multi-GPU via Accelerate):
    accelerate launch --num_processes=2 \\
        -m training.scripts.precompute_vggt_features \\
        --config training/config/segvggt_train.yaml \\
        --output /mnt/vol1/datasets/ScanNet++/vggt_pair_dsc \\
        --vggt_ckpt third_party/vggt_weights.pt

Resume-safe: already-computed .pt files are skipped automatically.
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from muviseg.paths import setup_segmast3r_path, setup_vggt_path

# masked_average_pooling: try segmast3r submodule, inline fallback
try:
    setup_segmast3r_path()
    from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling
except ImportError:
    def masked_average_pooling(feat, masks):
        B, D, H, W = feat.shape
        M = masks.shape[1]
        feat_flat = feat.view(B, D, H * W)
        masks_flat = masks.view(B, M, H * W).float()
        area = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1)
        return torch.bmm(feat_flat, masks_flat.transpose(1, 2)) / area.transpose(1, 2)

setup_vggt_path()
from vggt.models.aggregator import Aggregator  # noqa: E402

MAX_MASKS = 100
DESC_DIM = 2048
VGGT_IMG_SIZE = 518  # optimal for patch_size=14 → 37×37 grid


# ---------------------------------------------------------------------------
# Dataset — loads image pairs + masks (no correspondences needed)
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """Minimal dataset for precompute: (img0, img1, masks0, masks1) per pair."""

    def __init__(self, metadata_path, processed_root, masks_root, pairs_root,
                 target_size=VGGT_IMG_SIZE, max_masks=MAX_MASKS, output_root=None):
        from muviseg.data.dataset import _load_image, _decode_rles_batched, _resize_masks
        self._load_image = _load_image
        self._decode_rles = _decode_rles_batched
        self._resize_masks = _resize_masks

        self.processed_root = Path(processed_root)
        self.masks_root = Path(masks_root)
        self.pairs_root = Path(pairs_root)
        self.output_root = Path(output_root) if output_root else None
        self.target_size = target_size
        self.max_masks = max_masks

        meta = np.load(metadata_path, allow_pickle=True)
        self.scenes = meta["scenes"]
        self.sceneids = meta["sceneids"]
        self.images = meta["images"]
        raw_pairs = meta["pairs"][:, :2].astype(int)

        # Filter to pairs that have correspondence files
        all_pairs = []
        for i, j in raw_pairs:
            scene = self.scenes[self.sceneids[i]]
            pair_pkl = self.pairs_root / scene / f"{self.images[i]}__{self.images[j]}.pkl"
            if pair_pkl.exists():
                all_pairs.append((int(i), int(j)))

        # Skip already-computed pairs
        if self.output_root:
            self.pairs = []
            for i, j in all_pairs:
                scene = self.scenes[self.sceneids[i]]
                out_path = (
                    self.output_root / scene / f"{self.images[i]}__{self.images[j]}.pt"
                )
                if not out_path.exists():
                    self.pairs.append((i, j))
            n_skip = len(all_pairs) - len(self.pairs)
        else:
            self.pairs = all_pairs
            n_skip = 0

        # Image path cache
        self._img_cache = {}
        unique_ids = set()
        for i, j in self.pairs:
            unique_ids.add(i)
            unique_ids.add(j)
        for idx in unique_ids:
            scene = self.scenes[self.sceneids[idx]]
            name = self.images[idx]
            base = self.processed_root / scene / "images" / name
            for ext in (".jpg", ".JPG", ".jpeg", ".png"):
                p = base.with_suffix(ext)
                if p.exists():
                    self._img_cache[idx] = p
                    break

        print(
            f"PairDataset: {len(self.pairs):,} pairs to compute "
            f"({n_skip:,} already done, {len(all_pairs):,} total)"
        )

    def __len__(self):
        return len(self.pairs)

    def _mask_pkl(self, idx):
        scene = self.scenes[self.sceneids[idx]]
        return self.masks_root / scene / f"{self.images[idx]}.pkl"

    def __getitem__(self, index):
        idx_i, idx_j = self.pairs[index]
        scene = self.scenes[self.sceneids[idx_i]]
        name_i = self.images[idx_i]
        name_j = self.images[idx_j]

        # Load images — _load_image returns [-1, 1] normalized for MASt3R.
        # We need [0, 1] for VGGT, so undo: img = (img + 1) / 2
        img0 = self._load_image(
            self._img_cache[idx_i], self.target_size, "square"
        )
        img1 = self._load_image(
            self._img_cache[idx_j], self.target_size, "square"
        )
        img0 = (img0 + 1.0) / 2.0  # [-1,1] → [0,1]
        img1 = (img1 + 1.0) / 2.0

        with open(self._mask_pkl(idx_i), "rb") as f:
            rles0 = pickle.load(f)["mask_coco_rles_resized"]
        with open(self._mask_pkl(idx_j), "rb") as f:
            rles1 = pickle.load(f)["mask_coco_rles_resized"]

        _, H, W = img0.shape

        if rles0:
            masks0 = self._decode_rles(rles0[:self.max_masks])
            if masks0.shape[-2:] != (H, W):
                masks0 = self._resize_masks(masks0, (H, W))
        else:
            masks0 = torch.zeros(0, H, W, dtype=torch.uint8)

        if rles1:
            masks1 = self._decode_rles(rles1[:self.max_masks])
            if masks1.shape[-2:] != (H, W):
                masks1 = self._resize_masks(masks1, (H, W))
        else:
            masks1 = torch.zeros(0, H, W, dtype=torch.uint8)

        return {
            "img0": img0,
            "img1": img1,
            "masks0": masks0,
            "masks1": masks1,
            "scene": scene,
            "name_i": name_i,
            "name_j": name_j,
        }


def _collate(batch):
    return {
        "img0": torch.stack([b["img0"] for b in batch]),
        "img1": torch.stack([b["img1"] for b in batch]),
        "masks0": [b["masks0"] for b in batch],
        "masks1": [b["masks1"] for b in batch],
        "scene": [b["scene"] for b in batch],
        "name_i": [b["name_i"] for b in batch],
        "name_j": [b["name_j"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Aggregator wrapper
# ---------------------------------------------------------------------------

class AggregatorExtractor(torch.nn.Module):
    """Loads VGGT Aggregator, extracts cross-view patch tokens."""

    def __init__(self, ckpt_path, layer_idx=23, device="cpu"):
        super().__init__()
        import os

        self.layer_idx = layer_idx
        self.aggregator = Aggregator()

        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        else:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(repo_id=ckpt_path, filename="model.pt")
            state = torch.load(local, map_location=device, weights_only=True)

        prefix = "aggregator."
        agg_state = {
            k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)
        }
        if not agg_state:
            agg_state = state

        self.aggregator.load_state_dict(agg_state, strict=True)
        self.aggregator.eval()
        for p in self.aggregator.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, img0, img1):
        """
        img0, img1: (B, 3, H, W) in [0, 1] range.
        Returns: desc0 (B, 2048, H_p, W_p), desc1 (B, 2048, H_p, W_p)
        """
        B, _, H, W = img0.shape
        H_p, W_p = H // 14, W // 14

        # Stack pair: (B, 2, 3, H, W)
        images = torch.stack([img0, img1], dim=1)

        aggregated_tokens_list, ps_idx = self.aggregator(images)

        # Take tokens from selected layer: (B, 2, P, 2048)
        tokens = aggregated_tokens_list[self.layer_idx]

        # Extract patch tokens (skip camera + register tokens)
        patch_tokens = tokens[:, :, ps_idx:]  # (B, 2, num_patches, 2048)
        patch_tokens = patch_tokens[:, :, :H_p * W_p]
        patch_tokens = patch_tokens.view(B, 2, H_p, W_p, DESC_DIM)
        patch_tokens = patch_tokens.permute(0, 1, 4, 2, 3)  # (B, 2, 2048, H_p, W_p)

        desc0 = patch_tokens[:, 0].contiguous()
        desc1 = patch_tokens[:, 1].contiguous()
        return desc0, desc1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute per-pair VGGT segment descriptors."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output", required=True, help="Output root for per-pair .pt files"
    )
    parser.add_argument(
        "--vggt_ckpt", default=None,
        help="VGGT checkpoint path (overrides config MODEL.VGGT_CKPT)"
    )
    parser.add_argument(
        "--layer_idx", type=int, default=None,
        help="Aggregator layer index (overrides config MODEL.VGGT_LAYER_IDX)"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print estimate and exit without computing"
    )
    args = parser.parse_args()

    from muviseg.training.utils import load_cfg
    cfg = load_cfg(args.config)
    output_root = Path(args.output)

    vggt_ckpt = args.vggt_ckpt or cfg.MODEL.VGGT_CKPT
    layer_idx = args.layer_idx if args.layer_idx is not None else cfg.MODEL.VGGT_LAYER_IDX

    ds = PairDataset(
        metadata_path=cfg.DATASET.METADATA_PATH,
        processed_root=cfg.DATASET.DATA_ROOT,
        masks_root=cfg.DATASET.SEGDATA_ROOT,
        pairs_root=cfg.DATASET.PAIRS_ROOT,
        target_size=VGGT_IMG_SIZE,
        max_masks=MAX_MASKS,
        output_root=output_root,
    )

    if args.dry_run:
        avg_masks = 60
        total_kb = len(ds) * 2 * avg_masks * DESC_DIM * 2 / 1e3  # fp16
        print(f"Pairs to compute: {len(ds):,}")
        print(f"Estimated disk: ~{total_kb/1e6:.1f} GB (fp16)")
        print("--dry_run: exiting without computing.")
        return

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate,
        pin_memory=True,
        prefetch_factor=4,
    )

    model = AggregatorExtractor(vggt_ckpt, layer_idx=layer_idx, device="cpu")
    model, loader = accelerator.prepare(model, loader)

    if accelerator.is_main_process:
        print(f"Output: {output_root}")
        print(f"Pairs: {len(ds):,}")
        print(f"Batch size: {args.batch_size} x {accelerator.num_processes} GPUs")
        print(f"VGGT layer: {layer_idx}, desc dim: {DESC_DIM}")

    t0 = time.time()
    n_saved = 0

    for batch in tqdm(
        loader,
        desc="Precompute VGGT",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    ):
        img0 = batch["img0"]
        img1 = batch["img1"]

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            desc0, desc1 = model(img0, img1)  # (B, 2048, H_p, W_p)

        _, _, dH, dW = desc0.shape

        for b in range(desc0.shape[0]):
            scene = batch["scene"][b]
            name_i = batch["name_i"][b]
            name_j = batch["name_j"][b]

            out_dir = output_root / scene
            out_path = out_dir / f"{name_i}__{name_j}.pt"

            if out_path.exists():
                continue

            m0 = batch["masks0"][b].to(device).float()
            m1 = batch["masks1"][b].to(device).float()

            if m0.shape[-2:] != (dH, dW) and m0.shape[0] > 0:
                m0 = F.interpolate(
                    m0.unsqueeze(0), (dH, dW), mode="nearest"
                ).squeeze(0)
            if m1.shape[-2:] != (dH, dW) and m1.shape[0] > 0:
                m1 = F.interpolate(
                    m1.unsqueeze(0), (dH, dW), mode="nearest"
                ).squeeze(0)

            d0 = desc0[b:b + 1]
            d1 = desc1[b:b + 1]

            if m0.shape[0] > 0:
                dsc0 = masked_average_pooling(
                    d0, m0.unsqueeze(0)
                ).squeeze(0).T  # (M0, 2048)
            else:
                dsc0 = torch.zeros(0, DESC_DIM, device=device)

            if m1.shape[0] > 0:
                dsc1 = masked_average_pooling(
                    d1, m1.unsqueeze(0)
                ).squeeze(0).T  # (M1, 2048)
            else:
                dsc1 = torch.zeros(0, DESC_DIM, device=device)

            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "dsc0": dsc0.half().cpu(),
                "dsc1": dsc1.half().cpu(),
            }, out_path)
            n_saved += 1

    elapsed = time.time() - t0
    saved_t = torch.tensor([n_saved], device=device)
    saved_total = accelerator.gather(saved_t).sum().item()

    if accelerator.is_main_process:
        print(f"\nDone in {elapsed / 60:.1f} min")
        print(f"Saved: {saved_total:,} pairs")
        print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
