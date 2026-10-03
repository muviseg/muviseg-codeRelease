"""
Replica Dataset Loader for Segment Matching Evaluation

Loads RGB images, instance masks, and camera poses from Replica dataset
for segment matching evaluation.
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

from muviseg.evaluation.ground_truth_generator import instance_img_to_binary_masks, load_instance_mask


# Normalize to [-1, 1] range (same as paired_data_interface.py)
ImgNorm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def resize_to_target(pil_img: Image.Image, target_h: int, target_w: int) -> Image.Image:
    """
    Resize image to target size.

    Args:
        pil_img: PIL Image
        target_h: Target height
        target_w: Target width

    Returns:
        Resized PIL Image
    """
    return pil_img.resize((target_w, target_h), Image.BILINEAR)


def resize_masks(masks: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Resize all masks (M, H, W) to (M, H_new, W_new) using nearest neighbor.

    Args:
        masks: (M, H, W) binary mask tensor
        size: (H_new, W_new) target size

    Returns:
        Resized masks (M, H_new, W_new)
    """
    if masks.shape[0] == 0:
        # Handle empty masks
        return torch.zeros((0, size[0], size[1]), dtype=masks.dtype)

    return torch.stack(
        [
            TF.resize(
                mask.unsqueeze(0), size, interpolation=TF.InterpolationMode.NEAREST
            ).squeeze(0)
            for mask in masks
        ]
    )


class ReplicaSegmentMatchDataset(Dataset):
    """
    Dataset for Replica segment matching evaluation.

    Loads RGB images and instance segmentation masks for image pairs,
    preprocesses them for model inference.
    """

    def __init__(
        self,
        data_root: str,
        instance_mask_root: str,
        pairs: List[Dict],
        target_h: int = 336,
        target_w: int = 512,
        m_prime: Optional[int] = None,
        n_frames: int = 2,
        context_step: int = 5,
    ):
        """
        Args:
            data_root: Root directory containing scene folders (e.g., /data)
            instance_mask_root: Root directory for instance masks
            pairs: List of pair dictionaries with keys:
                   'scene', 'idx0', 'idx1', 'pose_bin', 'angle'
            target_h: Target height for resizing
            target_w: Target width for resizing
            m_prime: Maximum number of masks to keep per image (None = keep all)
            n_frames: Number of frames per sample. If > 2, adds N-2 temporal
                context frames from the same scene (deterministic).
            context_step: Frame-index step for temporal context sampling.
        """
        self.data_root = Path(data_root)
        self.instance_mask_root = Path(instance_mask_root)
        self.pairs = pairs
        self.target_h = target_h
        self.target_w = target_w
        self.m_prime = m_prime
        self.n_frames = n_frames
        self.context_step = context_step
        self._scene_frames: Dict[str, List[int]] = {}
        self._short_scene_warned = 0

    def __len__(self):
        return len(self.pairs)

    def _scene_frame_list(self, scene: str) -> List[int]:
        """Return sorted list of available RGB frame indices for `scene`."""
        if scene in self._scene_frames:
            return self._scene_frames[scene]
        rgb_dir = self.data_root / scene / "Sequence_1" / "rgb"
        indices = []
        if rgb_dir.is_dir():
            for p in rgb_dir.glob("rgb_*.png"):
                try:
                    indices.append(int(p.stem.split("_")[1]))
                except (ValueError, IndexError):
                    continue
        indices.sort()
        self._scene_frames[scene] = indices
        return indices

    def _sample_context(
        self, scene_frames: List[int], idx0: int, idx1: int, n_context: int
    ) -> List[int]:
        if n_context <= 0:
            return []
        step = self.context_step
        mid = (idx0 + idx1) // 2
        available = set(scene_frames) - {idx0, idx1}
        picked: List[int] = []
        for k in range(1, n_context * 4 + 1):
            for offset in (-k * step, k * step):
                cand = mid + offset
                if cand in available and cand not in picked:
                    picked.append(cand)
                    if len(picked) == n_context:
                        return picked
        if self._short_scene_warned < 10:
            print(f"  [warn] short replica scene: only {len(picked)} context frames "
                  f"for query ({idx0}, {idx1}); padding with idx0={idx0}")
            self._short_scene_warned += 1
        while len(picked) < n_context:
            picked.append(idx0)
        return picked

    def _get_rgb_path(self, scene: str, idx: int) -> Path:
        """Get path to RGB image."""
        return self.data_root / scene / "Sequence_1" / "rgb" / f"rgb_{idx}.png"

    def _get_instance_mask_path(self, scene: str, idx: int) -> Path:
        """Get path to instance mask."""
        return self.instance_mask_root / scene / "Sequence_1" / "semantic_instance" / f"semantic_instance_{idx}.png"

    def _load_image(self, path: Path) -> torch.Tensor:
        """
        Load and preprocess RGB image.

        Args:
            path: Path to image file

        Returns:
            Normalized image tensor (3, H, W) in [-1, 1] range
        """
        img = Image.open(path).convert("RGB")
        img = exif_transpose(img)
        img = resize_to_target(img, self.target_h, self.target_w)
        img_tensor = ImgNorm(img)  # (3, H, W)
        return img_tensor

    def _load_masks(self, path: Path, size_hw: Tuple[int, int]) -> Tuple[torch.Tensor, List[int]]:
        """
        Load and preprocess instance masks.

        Args:
            path: Path to instance mask PNG file
            size_hw: (H, W) target size for masks

        Returns:
            masks: (M, H, W) binary mask tensor
            instance_ids: List of instance IDs corresponding to each mask
        """
        # Load instance mask image
        instance_img = load_instance_mask(str(path))

        # Convert to binary masks and get instance IDs
        masks, instance_ids = instance_img_to_binary_masks(instance_img, filter_background=True)

        # Resize masks to target size
        if masks.shape[0] > 0:
            masks = resize_masks(masks, size_hw)

        # Limit number of masks if m_prime is set
        if self.m_prime is not None and masks.shape[0] > self.m_prime:
            # Keep first m_prime masks
            masks = masks[:self.m_prime]
            instance_ids = instance_ids[:self.m_prime]

        return masks.to(torch.uint8), instance_ids

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a pair of images and masks.

        Args:
            idx: Index in the pairs list

        Returns:
            Dictionary with keys:
                - img0: (3, H, W) normalized image tensor
                - img1: (3, H, W) normalized image tensor
                - masks0: (M, H, W) binary mask tensor
                - masks1: (N, H, W) binary mask tensor
                - instance_ids0: List of instance IDs for masks0
                - instance_ids1: List of instance IDs for masks1
                - pose_bin: Pose bin index
                - angle: Rotation angle in degrees
                - scene: Scene name
                - idx0: Frame index 0
                - idx1: Frame index 1
        """
        pair = self.pairs[idx]

        scene = pair['scene']
        idx0 = pair['idx0']
        idx1 = pair['idx1']
        pose_bin = pair['pose_bin']
        angle = pair['angle']

        # Load RGB images
        img0_path = self._get_rgb_path(scene, idx0)
        img1_path = self._get_rgb_path(scene, idx1)

        img0 = self._load_image(img0_path)
        img1 = self._load_image(img1_path)

        # Load instance masks
        mask0_path = self._get_instance_mask_path(scene, idx0)
        mask1_path = self._get_instance_mask_path(scene, idx1)

        masks0, instance_ids0 = self._load_masks(mask0_path, (self.target_h, self.target_w))
        masks1, instance_ids1 = self._load_masks(mask1_path, (self.target_h, self.target_w))

        base = {
            'img0': img0,
            'img1': img1,
            'masks0': masks0,
            'masks1': masks1,
            'instance_ids0': instance_ids0,
            'instance_ids1': instance_ids1,
            'pose_bin': pose_bin,
            'angle': angle,
            'scene': scene,
            'idx0': idx0,
            'idx1': idx1
        }

        if self.n_frames < 2:
            return base

        scene_frames = self._scene_frame_list(scene)
        ctx_indices = self._sample_context(
            scene_frames, idx0, idx1, self.n_frames - 2
        )
        images = [img0, img1]
        masks = [masks0, masks1]
        instance_ids = [instance_ids0, instance_ids1]
        frame_indices = [idx0, idx1]
        size_hw = (self.target_h, self.target_w)
        for cidx in ctx_indices:
            images.append(self._load_image(self._get_rgb_path(scene, cidx)))
            m, iid = self._load_masks(self._get_instance_mask_path(scene, cidx), size_hw)
            masks.append(m)
            instance_ids.append(iid)
            frame_indices.append(cidx)

        base.update({
            'images': images,
            'masks': masks,
            'instance_ids': instance_ids,
            'frame_indices': frame_indices,
        })
        return base


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching.

    Since different images may have different numbers of masks,
    we return lists instead of stacked tensors for masks and instance_ids.

    Args:
        batch: List of sample dictionaries from __getitem__

    Returns:
        Batched dictionary with:
            - img0, img1: (B, 3, H, W) stacked tensors
            - masks0, masks1: Lists of (M_i, H, W) tensors
            - instance_ids0, instance_ids1: Lists of instance ID lists
            - pose_bin, angle, scene, idx0, idx1: Lists
    """
    # Stack images
    img0 = torch.stack([item['img0'] for item in batch])
    img1 = torch.stack([item['img1'] for item in batch])

    # Keep masks as lists (variable number per image)
    masks0 = [item['masks0'] for item in batch]
    masks1 = [item['masks1'] for item in batch]

    # Keep instance IDs as lists
    instance_ids0 = [item['instance_ids0'] for item in batch]
    instance_ids1 = [item['instance_ids1'] for item in batch]

    # Keep metadata as lists
    pose_bin = [item['pose_bin'] for item in batch]
    angle = [item['angle'] for item in batch]
    scene = [item['scene'] for item in batch]
    idx0 = [item['idx0'] for item in batch]
    idx1 = [item['idx1'] for item in batch]

    return {
        'img0': img0,
        'img1': img1,
        'masks0': masks0,
        'masks1': masks1,
        'instance_ids0': instance_ids0,
        'instance_ids1': instance_ids1,
        'pose_bin': pose_bin,
        'angle': angle,
        'scene': scene,
        'idx0': idx0,
        'idx1': idx1
    }


def collate_fn_tuple(batch: List[Dict]) -> Dict:
    """
    Multi-frame tuple collate. See VKitti2SegmentMatchDataset.collate_fn_tuple
    for layout: stacks images into (B, N, 3, H, W); keeps masks / ids as lists.
    """
    N = len(batch[0]['images'])
    images = torch.stack([torch.stack(item['images'], dim=0) for item in batch], dim=0)
    return {
        'images': images,
        'masks': [item['masks'] for item in batch],
        'instance_ids': [item['instance_ids'] for item in batch],
        'frame_indices': [item['frame_indices'] for item in batch],
        'img0': torch.stack([item['img0'] for item in batch]),
        'img1': torch.stack([item['img1'] for item in batch]),
        'masks0': [item['masks0'] for item in batch],
        'masks1': [item['masks1'] for item in batch],
        'instance_ids0': [item['instance_ids0'] for item in batch],
        'instance_ids1': [item['instance_ids1'] for item in batch],
        'pose_bin': [item['pose_bin'] for item in batch],
        'angle': [item['angle'] for item in batch],
        'scene': [item['scene'] for item in batch],
        'idx0': [item['idx0'] for item in batch],
        'idx1': [item['idx1'] for item in batch],
        'n_frames': N,
    }


def load_pairs_from_json(json_path: str) -> List[Dict]:
    """
    Load image pairs from JSON file.

    Args:
        json_path: Path to JSON file with pair information

    Returns:
        List of pair dictionaries
    """
    with open(json_path, 'r') as f:
        pairs = json.load(f)
    return pairs


if __name__ == "__main__":
    # Quick test
    print("Testing Replica dataset loader...")

    # Load pairs (assuming pairs file exists)
    pairs_file = "pairs_replica_3200.json"
    if Path(pairs_file).exists():
        pairs = load_pairs_from_json(pairs_file)
        print(f"Loaded {len(pairs)} pairs from {pairs_file}")

        # Create dataset
        dataset = ReplicaSegmentMatchDataset(
            data_root="/data",
            instance_mask_root="/data/Replica_Instance_Segmentation",
            pairs=pairs[:10],  # Test with first 10 pairs
            target_h=336,
            target_w=512,
            m_prime=50
        )

        print(f"Dataset size: {len(dataset)}")

        # Test loading one sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Scene: {sample['scene']}")
        print(f"  Indices: {sample['idx0']} -> {sample['idx1']}")
        print(f"  Pose bin: {sample['pose_bin']}, angle: {sample['angle']:.2f}°")
        print(f"  Image 0 shape: {sample['img0'].shape}")
        print(f"  Image 1 shape: {sample['img1'].shape}")
        print(f"  Masks 0 shape: {sample['masks0'].shape}, num instances: {len(sample['instance_ids0'])}")
        print(f"  Masks 1 shape: {sample['masks1'].shape}, num instances: {len(sample['instance_ids1'])}")
        print(f"  Instance IDs 0: {sample['instance_ids0'][:5]}...")
        print(f"  Instance IDs 1: {sample['instance_ids1'][:5]}...")

        # Test batching
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"\nBatch test:")
        print(f"  img0 shape: {batch['img0'].shape}")
        print(f"  img1 shape: {batch['img1'].shape}")
        print(f"  masks0 length: {len(batch['masks0'])}, first shape: {batch['masks0'][0].shape}")
        print(f"  masks1 length: {len(batch['masks1'])}, first shape: {batch['masks1'][0].shape}")

    else:
        print(f"Pairs file {pairs_file} not found. Run sample_pairs.py first.")

        # Test with dummy data
        print("\nTesting with dummy pair...")
        dummy_pairs = [{
            'scene': 'office_0',
            'idx0': 0,
            'idx1': 50,
            'pose_bin': 0,
            'angle': 30.0
        }]

        dataset = ReplicaSegmentMatchDataset(
            data_root="/data",
            instance_mask_root="/data/Replica_Instance_Segmentation",
            pairs=dummy_pairs,
            target_h=336,
            target_w=512
        )

        sample = dataset[0]
        print(f"  Scene: {sample['scene']}")
        print(f"  Image 0 shape: {sample['img0'].shape}")
        print(f"  Masks 0 shape: {sample['masks0'].shape}")
        print(f"  Instance IDs 0: {sample['instance_ids0']}")
