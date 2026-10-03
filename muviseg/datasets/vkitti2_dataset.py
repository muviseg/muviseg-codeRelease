"""
Virtual KITTI 2 Dataset Loader for Segment Matching Evaluation

Loads RGB images and trackID instance masks from Virtual KITTI 2 for
segment matching evaluation. Ground truth correspondences are based on
trackID matching (same object across frames).

Dataset structure:
  {data_root}/vkitti_rgb/{scene}/{variant}/frames/rgb/Camera_0/rgb_{idx:05d}.jpg
  {data_root}/vkitti_instanceSegmentation/{scene}/{variant}/frames/
              instanceSegmentation/Camera_0/instancegt_{idx:05d}.png

Instance mask encoding: pixel_value = trackID + 1
  background: pixel_value=1, trackID=0 (filtered out)
  objects:    pixel_value>=2, trackID>=1
"""

import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
from pathlib import Path
from typing import List, Dict, Tuple, Optional


# Normalize to [-1, 1] range (same as replica_dataset.py)
ImgNorm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def load_vkitti2_trackid_mask(mask_path: str) -> Tuple[torch.Tensor, List[int]]:
    """
    Load Virtual KITTI 2 instance mask and convert to binary masks.

    Encoding: pixel_value = trackID + 1, so background has pixel_value=1 (trackID=0).

    Args:
        mask_path: Path to instancegt PNG file (uint16)

    Returns:
        masks: (M, H, W) binary uint8 mask tensor, one per unique object trackID
        track_ids: List of trackIDs (> 0) corresponding to each mask
    """
    mask_img = Image.open(mask_path)
    mask_array = np.array(mask_img).astype(np.int32)
    # Convert pixel value → trackID (background = 0)
    track_id_img = mask_array - 1

    # Get unique trackIDs, filter background (trackID = 0)
    unique_ids = np.unique(track_id_img)
    unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        H, W = track_id_img.shape
        return torch.zeros((0, H, W), dtype=torch.uint8), []

    masks = []
    track_ids = []
    for tid in unique_ids:
        mask = (track_id_img == tid).astype(np.uint8)
        masks.append(mask)
        track_ids.append(int(tid))

    return torch.from_numpy(np.stack(masks, axis=0)), track_ids


def resize_masks(masks: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if masks.shape[0] == 0:
        return torch.zeros((0, size[0], size[1]), dtype=masks.dtype)
    return torch.stack(
        [
            TF.resize(
                mask.unsqueeze(0), size, interpolation=TF.InterpolationMode.NEAREST
            ).squeeze(0)
            for mask in masks
        ]
    )


class VKitti2SegmentMatchDataset(Dataset):
    """
    Dataset for Virtual KITTI 2 segment matching evaluation.

    Each pair has keys: idx0, idx1, angle, pose_bin, scene, variant.

    Supports optional multi-frame tuple mode: when n_frames > 2, adds N-2
    context frames sampled deterministically in a temporal window around
    (idx0+idx1)//2 with the given step. Context frames exist only to feed
    joint attention — metrics and GT are still computed on the query pair
    (idx0, idx1) only.
    """

    def __init__(
        self,
        data_root: str,
        pairs: List[Dict],
        target_h: int = 336,
        target_w: int = 512,
        m_prime: Optional[int] = None,
        n_frames: int = 2,
        context_step: int = 5,
    ):
        self.data_root = Path(data_root)
        self.pairs = pairs
        self.target_h = target_h
        self.target_w = target_w
        self.m_prime = m_prime
        self.n_frames = n_frames
        self.context_step = context_step
        # Cache of per-(scene, variant) sorted frame indices, built lazily.
        self._scene_frames: Dict[Tuple[str, str], List[int]] = {}
        self._short_scene_warned = 0

    def __len__(self):
        return len(self.pairs)

    def _scene_frame_list(self, scene: str, variant: str) -> List[int]:
        """Return sorted list of available RGB frame indices for (scene, variant)."""
        key = (scene, variant)
        if key in self._scene_frames:
            return self._scene_frames[key]
        rgb_dir = (
            self.data_root / "vkitti_rgb" / scene / variant
            / "frames" / "rgb" / "Camera_0"
        )
        indices = []
        if rgb_dir.is_dir():
            for p in rgb_dir.glob("rgb_*.jpg"):
                try:
                    indices.append(int(p.stem.split("_")[1]))
                except (ValueError, IndexError):
                    continue
        indices.sort()
        self._scene_frames[key] = indices
        return indices

    def _sample_context(
        self, scene_frames: List[int], idx0: int, idx1: int, n_context: int
    ) -> List[int]:
        """Deterministic temporal-window context sampling (no randomness)."""
        if n_context <= 0:
            return []
        step = self.context_step
        mid = (idx0 + idx1) // 2
        available = set(scene_frames) - {idx0, idx1}
        picked: List[int] = []
        # Walk outward from mid: ±step, ±2·step, ...
        for k in range(1, n_context * 4 + 1):
            for offset in (-k * step, k * step):
                cand = mid + offset
                if cand in available and cand not in picked:
                    picked.append(cand)
                    if len(picked) == n_context:
                        return picked
        # Fallback: scene has too few frames — pad by reusing idx0
        if self._short_scene_warned < 10:
            print(f"  [warn] short scene '{scene_frames[:3] if scene_frames else []}...': "
                  f"only {len(picked)} context frames for query ({idx0}, {idx1}); "
                  f"padding with idx0={idx0}")
            self._short_scene_warned += 1
        while len(picked) < n_context:
            picked.append(idx0)
        return picked

    def _rgb_path(self, scene: str, variant: str, idx: int) -> Path:
        return (
            self.data_root
            / "vkitti_rgb" / scene / variant
            / "frames" / "rgb" / "Camera_0"
            / f"rgb_{idx:05d}.jpg"
        )

    def _mask_path(self, scene: str, variant: str, idx: int) -> Path:
        return (
            self.data_root
            / "vkitti_instanceSegmentation" / scene / variant
            / "frames" / "instanceSegmentation" / "Camera_0"
            / f"instancegt_{idx:05d}.png"
        )

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        img = exif_transpose(img)
        img = img.resize((self.target_w, self.target_h), Image.BILINEAR)
        return ImgNorm(img)

    def _load_masks(self, path: Path) -> Tuple[torch.Tensor, List[int]]:
        masks, track_ids = load_vkitti2_trackid_mask(str(path))

        if masks.shape[0] > 0:
            masks = resize_masks(masks, (self.target_h, self.target_w))

        if self.m_prime is not None and masks.shape[0] > self.m_prime:
            masks = masks[:self.m_prime]
            track_ids = track_ids[:self.m_prime]

        return masks.to(torch.uint8), track_ids

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]
        scene = pair['scene']
        variant = pair['variant']
        idx0 = pair['idx0']
        idx1 = pair['idx1']

        # Query frames (always first two in multi-frame mode)
        img0 = self._load_image(self._rgb_path(scene, variant, idx0))
        img1 = self._load_image(self._rgb_path(scene, variant, idx1))
        masks0, instance_ids0 = self._load_masks(self._mask_path(scene, variant, idx0))
        masks1, instance_ids1 = self._load_masks(self._mask_path(scene, variant, idx1))

        base = {
            'img0': img0, 'img1': img1,
            'masks0': masks0, 'masks1': masks1,
            'instance_ids0': instance_ids0, 'instance_ids1': instance_ids1,
            'pose_bin': pair['pose_bin'],
            'angle': pair['angle'],
            'scene': scene, 'variant': variant,
            'idx0': idx0, 'idx1': idx1,
        }

        if self.n_frames < 2:
            return base

        # Multi-frame mode: load N-2 context frames
        scene_frames = self._scene_frame_list(scene, variant)
        ctx_indices = self._sample_context(
            scene_frames, idx0, idx1, self.n_frames - 2
        )
        images = [img0, img1]
        masks = [masks0, masks1]
        instance_ids = [instance_ids0, instance_ids1]
        frame_indices = [idx0, idx1]
        for cidx in ctx_indices:
            images.append(self._load_image(self._rgb_path(scene, variant, cidx)))
            m, iid = self._load_masks(self._mask_path(scene, variant, cidx))
            masks.append(m)
            instance_ids.append(iid)
            frame_indices.append(cidx)

        base.update({
            'images': images,                # list[N] of (3, H, W) tensors
            'masks': masks,                  # list[N] of (M_v, H, W)
            'instance_ids': instance_ids,    # list[N] of list[int]
            'frame_indices': frame_indices,  # list[N] of int
        })
        return base


def collate_fn(batch: List[Dict]) -> Dict:
    img0 = torch.stack([item['img0'] for item in batch])
    img1 = torch.stack([item['img1'] for item in batch])
    masks0 = [item['masks0'] for item in batch]
    masks1 = [item['masks1'] for item in batch]
    instance_ids0 = [item['instance_ids0'] for item in batch]
    instance_ids1 = [item['instance_ids1'] for item in batch]
    return {
        'img0': img0,
        'img1': img1,
        'masks0': masks0,
        'masks1': masks1,
        'instance_ids0': instance_ids0,
        'instance_ids1': instance_ids1,
        'pose_bin': [item['pose_bin'] for item in batch],
        'angle': [item['angle'] for item in batch],
        'scene': [item['scene'] for item in batch],
        'variant': [item['variant'] for item in batch],
        'idx0': [item['idx0'] for item in batch],
        'idx1': [item['idx1'] for item in batch],
    }


def collate_fn_tuple(batch: List[Dict]) -> Dict:
    """
    Collate function for multi-frame tuple mode.

    Stacks per-view images into (B, N, 3, H, W); masks and instance_ids are
    kept as nested lists (list[B] of list[N] of tensor/list). Query-level
    fields (img0, masks0, instance_ids0, ...) are still provided so
    downstream GT / metrics code (which operates only on the query pair)
    works unchanged.
    """
    N = len(batch[0]['images'])
    # (B, N, 3, H, W)
    images = torch.stack([torch.stack(item['images'], dim=0) for item in batch], dim=0)
    masks = [item['masks'] for item in batch]
    instance_ids = [item['instance_ids'] for item in batch]
    frame_indices = [item['frame_indices'] for item in batch]

    return {
        'images': images,
        'masks': masks,
        'instance_ids': instance_ids,
        'frame_indices': frame_indices,
        # Query-level views for GT / metrics (unchanged from pairwise path)
        'img0': torch.stack([item['img0'] for item in batch]),
        'img1': torch.stack([item['img1'] for item in batch]),
        'masks0': [item['masks0'] for item in batch],
        'masks1': [item['masks1'] for item in batch],
        'instance_ids0': [item['instance_ids0'] for item in batch],
        'instance_ids1': [item['instance_ids1'] for item in batch],
        'pose_bin': [item['pose_bin'] for item in batch],
        'angle': [item['angle'] for item in batch],
        'scene': [item['scene'] for item in batch],
        'variant': [item['variant'] for item in batch],
        'idx0': [item['idx0'] for item in batch],
        'idx1': [item['idx1'] for item in batch],
        'n_frames': N,
    }


def load_pairs_from_json(json_path: str) -> List[Dict]:
    with open(json_path, 'r') as f:
        return json.load(f)
