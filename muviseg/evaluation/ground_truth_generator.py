"""
Ground Truth Correspondence Generator for Replica Instance Masks

This module provides utilities to generate binary correspondence matrices
from instance segmentation masks based on matching instance IDs.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Tuple, List


def load_instance_mask(mask_path: str) -> np.ndarray:
    """
    Load instance mask from PNG file.

    Args:
        mask_path: Path to uint16 instance mask PNG file

    Returns:
        Instance mask as numpy array (H, W) with uint16 instance IDs
    """
    img = Image.open(mask_path)
    # Ensure we're reading as uint16
    mask = np.array(img, dtype=np.uint16)
    return mask


def instance_img_to_binary_masks(instance_img: np.ndarray, filter_background: bool = True) -> Tuple[torch.Tensor, List[int]]:
    """
    Convert instance segmentation image to binary masks.

    Args:
        instance_img: (H, W) array with instance IDs
        filter_background: If True, exclude instance ID 0 (background)

    Returns:
        masks: (M, H, W) binary mask tensor, where M is number of instances
        instance_ids: List of instance IDs corresponding to each mask
    """
    # Get unique instance IDs
    unique_ids = np.unique(instance_img)

    # Filter background if requested
    if filter_background:
        unique_ids = unique_ids[unique_ids > 0]

    if len(unique_ids) == 0:
        # Return empty masks if no instances
        H, W = instance_img.shape
        return torch.zeros((0, H, W), dtype=torch.uint8), []

    # Create binary masks for each instance
    masks = []
    instance_ids = []

    for inst_id in unique_ids:
        mask = (instance_img == inst_id).astype(np.uint8)
        masks.append(mask)
        instance_ids.append(int(inst_id))

    masks_tensor = torch.from_numpy(np.stack(masks, axis=0))

    return masks_tensor, instance_ids


def generate_instance_correspondences(
    instance_ids0: List[int],
    instance_ids1: List[int]
) -> np.ndarray:
    """
    Generate binary correspondence matrix based on instance ID matching.

    Args:
        instance_ids0: List of instance IDs in frame 0 (length M)
        instance_ids1: List of instance IDs in frame 1 (length N)

    Returns:
        Binary matrix (M, N) where gt[i, j] = 1 if instance_ids0[i] == instance_ids1[j]
    """
    M = len(instance_ids0)
    N = len(instance_ids1)

    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.uint8)

    # Create correspondence matrix
    gt = np.zeros((M, N), dtype=np.uint8)

    for i, id0 in enumerate(instance_ids0):
        for j, id1 in enumerate(instance_ids1):
            if id0 == id1 and id0 > 0:  # Match if same ID and not background
                gt[i, j] = 1

    return gt


def generate_correspondences_from_masks(
    mask_path0: str,
    mask_path1: str
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Load instance masks and generate correspondence matrix.

    Args:
        mask_path0: Path to instance mask for frame 0
        mask_path1: Path to instance mask for frame 1

    Returns:
        gt_matrix: Binary correspondence matrix (M, N)
        instance_ids0: List of instance IDs in frame 0
        instance_ids1: List of instance IDs in frame 1
    """
    # Load instance masks
    inst_img0 = load_instance_mask(mask_path0)
    inst_img1 = load_instance_mask(mask_path1)

    # Convert to binary masks and get instance IDs
    _, instance_ids0 = instance_img_to_binary_masks(inst_img0, filter_background=True)
    _, instance_ids1 = instance_img_to_binary_masks(inst_img1, filter_background=True)

    # Generate correspondence matrix
    gt_matrix = generate_instance_correspondences(instance_ids0, instance_ids1)

    return gt_matrix, instance_ids0, instance_ids1


def batch_generate_correspondences(
    instance_ids0_batch: List[List[int]],
    instance_ids1_batch: List[List[int]]
) -> List[np.ndarray]:
    """
    Generate correspondence matrices for a batch of instance ID lists.

    Args:
        instance_ids0_batch: List of instance ID lists for frame 0 (batch size B)
        instance_ids1_batch: List of instance ID lists for frame 1 (batch size B)

    Returns:
        List of binary correspondence matrices, one per batch element
    """
    batch_size = len(instance_ids0_batch)
    assert len(instance_ids1_batch) == batch_size

    gt_matrices = []
    for i in range(batch_size):
        gt = generate_instance_correspondences(
            instance_ids0_batch[i],
            instance_ids1_batch[i]
        )
        gt_matrices.append(gt)

    return gt_matrices


if __name__ == "__main__":
    # Quick test
    print("Testing ground truth generator...")

    # Test with synthetic data
    instance_ids0 = [1, 2, 3, 5]
    instance_ids1 = [2, 3, 4, 5, 6]

    gt = generate_instance_correspondences(instance_ids0, instance_ids1)
    print(f"Instance IDs 0: {instance_ids0}")
    print(f"Instance IDs 1: {instance_ids1}")
    print(f"Ground truth correspondence matrix:\n{gt}")
    print(f"Shape: {gt.shape}")

    # Test with real Replica data if available
    test_mask_path = "/data/Replica_Instance_Segmentation/office_0/Sequence_1/semantic_instance/semantic_instance_0.png"
    if Path(test_mask_path).exists():
        print(f"\nTesting with real Replica data...")
        inst_img = load_instance_mask(test_mask_path)
        print(f"Loaded instance mask shape: {inst_img.shape}")
        print(f"Instance mask dtype: {inst_img.dtype}")
        print(f"Unique instance IDs: {np.unique(inst_img)}")

        masks, ids = instance_img_to_binary_masks(inst_img)
        print(f"Number of instances (excluding background): {len(ids)}")
        print(f"Binary masks shape: {masks.shape}")
        print(f"Instance IDs: {ids}")
