"""
Stratified Pair Sampling for Virtual KITTI 2 Dataset Evaluation

Samples image pairs across different pose bins (rotation angles) for evaluation.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


def load_extrinsics(extrinsic_file: str) -> List[np.ndarray]:
    """
    Load camera extrinsics from Virtual KITTI 2 extrinsic.txt file.

    Args:
        extrinsic_file: Path to extrinsic.txt file

    Returns:
        List of 4x4 extrinsic matrices, one per frame
    """
    extrinsics = []

    with open(extrinsic_file, 'r') as f:
        # Skip header line
        next(f)

        for line in f:
            parts = line.strip().split()
            if len(parts) < 18:
                continue

            frame_idx = int(parts[0])
            camera_id = int(parts[1])

            # Only use Camera_0 (left camera)
            if camera_id != 0:
                continue

            # Parse 4x4 matrix (16 values starting from index 2)
            matrix_values = list(map(float, parts[2:18]))
            extrinsic = np.array(matrix_values).reshape(4, 4)

            extrinsics.append(extrinsic)

    return extrinsics


def compute_rotation_angle(extrinsic1: np.ndarray, extrinsic2: np.ndarray) -> float:
    """
    Compute rotation angle between two camera extrinsics using geodesic distance on SO(3).

    Args:
        extrinsic1: 4x4 extrinsic matrix
        extrinsic2: 4x4 extrinsic matrix

    Returns:
        Rotation angle in degrees [0, 180]
    """
    # Extract rotation matrices
    R1 = extrinsic1[:3, :3]
    R2 = extrinsic2[:3, :3]

    # Compute relative rotation
    R_rel = R1.T @ R2

    # Compute angle from trace
    trace = np.trace(R_rel)
    cos_angle = np.clip((trace - 1) / 2, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def assign_pose_bin(angle: float, bins: List[Tuple[float, float]]) -> int:
    """
    Assign rotation angle to a pose bin.

    Args:
        angle: Rotation angle in degrees
        bins: List of (min, max) tuples defining bin ranges

    Returns:
        Bin index, or -1 if angle doesn't fit in any bin
    """
    for i, (min_angle, max_angle) in enumerate(bins):
        if min_angle <= angle < max_angle:
            return i
    # Handle edge case for 180 degrees
    if angle == 180.0 and bins[-1][1] == 180:
        return len(bins) - 1
    return -1


def sample_pairs_for_scene_variant(
    scene: str,
    variant: str,
    data_root: Path,
    num_pairs_per_bin: int,
    temporal_gap_min: int,
    temporal_gap_max: int,
    pose_bins: List[Tuple[float, float]],
    random_seed: int = 42
) -> List[Dict]:
    """
    Sample image pairs for a single scene variant across pose bins.

    Args:
        scene: Scene name (e.g., 'Scene01')
        variant: Variant name (e.g., 'clone', 'fog')
        data_root: Root directory of Virtual KITTI 2
        num_pairs_per_bin: Number of pairs to sample per pose bin
        temporal_gap_min: Minimum temporal gap between frame indices
        temporal_gap_max: Maximum temporal gap between frame indices
        pose_bins: List of (min, max) tuples for pose bins
        random_seed: Random seed for reproducibility

    Returns:
        List of pair dictionaries with keys: idx0, idx1, pose_bin, angle
    """
    # Load extrinsics
    extrinsic_file = data_root / "vkitti_textgt" / scene / variant / "extrinsic.txt"

    if not extrinsic_file.exists():
        print(f"  WARNING: Extrinsic file not found: {extrinsic_file}")
        return []

    extrinsics = load_extrinsics(str(extrinsic_file))
    num_frames = len(extrinsics)

    print(f"  {scene}/{variant}: {num_frames} frames")

    if num_frames < temporal_gap_min + 1:
        print(f"  WARNING: Not enough frames for temporal gap {temporal_gap_min}")
        return []

    # Initialize random state
    rng = np.random.RandomState(random_seed)

    # Collect candidate pairs for each bin
    candidates_by_bin = [[] for _ in range(len(pose_bins))]

    # Generate all valid pairs with temporal gap in [min, max]
    for i in range(num_frames):
        # Upper limit: min(i + temporal_gap_max + 1, num_frames)
        max_j = min(i + temporal_gap_max + 1, num_frames) if temporal_gap_max > 0 else num_frames

        for j in range(i + temporal_gap_min + 1, max_j):
            # Compute rotation angle
            angle = compute_rotation_angle(extrinsics[i], extrinsics[j])

            # Assign to bin
            bin_idx = assign_pose_bin(angle, pose_bins)
            if bin_idx >= 0:
                candidates_by_bin[bin_idx].append({
                    'idx0': i,
                    'idx1': j,
                    'angle': angle
                })

    # Sample from each bin
    sampled_pairs = []
    for bin_idx, candidates in enumerate(candidates_by_bin):
        num_candidates = len(candidates)

        if num_candidates == 0:
            continue

        # Sample with replacement if not enough candidates
        if num_candidates < num_pairs_per_bin:
            print(f"    Bin {pose_bins[bin_idx]}: {num_candidates} candidates (sampling with replacement)")
            sampled_indices = rng.choice(num_candidates, size=num_pairs_per_bin, replace=True)
        else:
            sampled_indices = rng.choice(num_candidates, size=num_pairs_per_bin, replace=False)

        for idx in sampled_indices:
            pair = candidates[idx].copy()
            pair['pose_bin'] = bin_idx
            pair['scene'] = scene
            pair['variant'] = variant
            sampled_pairs.append(pair)

    return sampled_pairs


def sample_vkitti2_pairs(
    data_root: str,
    scenes: List[str],
    variants: List[str],
    num_pairs_per_variant: int,
    temporal_gap_min: int,
    temporal_gap_max: int,
    pose_bins: List[Tuple[float, float]],
    output_file: str,
    random_seed: int = 42
):
    """
    Sample image pairs across Virtual KITTI 2 scenes and variants.

    Args:
        data_root: Root directory of Virtual KITTI 2
        scenes: List of scene names (e.g., ['Scene01', 'Scene02'])
        variants: List of variant names (e.g., ['clone', 'fog'])
        num_pairs_per_variant: Total pairs per scene variant (distributed across bins)
        temporal_gap_min: Minimum temporal gap between frames
        temporal_gap_max: Maximum temporal gap between frames (0 = no limit)
        pose_bins: List of (min, max) tuples for pose bins
        output_file: Path to output JSON file
        random_seed: Random seed for reproducibility
    """
    data_root = Path(data_root)
    num_bins = len(pose_bins)
    num_pairs_per_bin = num_pairs_per_variant // num_bins

    print(f"Sampling {num_pairs_per_variant} pairs per variant ({num_pairs_per_bin} per bin)")
    print(f"Scenes: {scenes}")
    print(f"Variants: {variants}")
    print(f"Pose bins: {pose_bins}")
    if temporal_gap_max > 0:
        print(f"Temporal gap: [{temporal_gap_min}, {temporal_gap_max}] frames")
    else:
        print(f"Temporal gap: >= {temporal_gap_min} frames (no upper limit)")
    print(f"Random seed: {random_seed}\n")

    all_pairs = []

    for scene in tqdm(scenes, desc="Processing scenes"):
        for variant in variants:
            # Sample pairs for this scene variant
            pairs = sample_pairs_for_scene_variant(
                scene,
                variant,
                data_root,
                num_pairs_per_bin,
                temporal_gap_min,
                temporal_gap_max,
                pose_bins,
                random_seed
            )

            all_pairs.extend(pairs)

    # Save to JSON
    print(f"\nTotal pairs sampled: {len(all_pairs)}")

    # Print distribution by bin
    bin_counts = [0] * num_bins
    for pair in all_pairs:
        bin_counts[pair['pose_bin']] += 1

    print("\nDistribution by pose bin:")
    for i, (bin_range, count) in enumerate(zip(pose_bins, bin_counts)):
        print(f"  Bin {i} ({bin_range[0]:.0f}-{bin_range[1]:.0f}°): {count} pairs")

    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\nSaved pairs to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample image pairs for Virtual KITTI 2 evaluation")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/mnt/vol3/datasets/virtual-KITTI-2",
        help="Root directory of Virtual KITTI 2"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pairs_vkitti2.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num_pairs_per_variant",
        type=int,
        default=100,
        help="Number of pairs to sample per scene variant"
    )
    parser.add_argument(
        "--temporal_gap",
        type=int,
        default=10,
        help="Minimum temporal gap between frames"
    )
    parser.add_argument(
        "--temporal_gap_max",
        type=int,
        default=0,
        help="Maximum temporal gap between frames (0 = no limit)"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=['Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20'],
        help="List of scene names"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=['clone'],  # Start with clone only
        help="List of variants (clone, fog, rain, morning, overcast, sunset, 15-deg-left, etc.)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Define pose bins (optimized for driving dataset with forward motion)
    pose_bins = [
        (0, 20),
        (20, 40),
        (40, 60),
        (60, 90)
    ]

    sample_vkitti2_pairs(
        data_root=args.data_root,
        scenes=args.scenes,
        variants=args.variants,
        num_pairs_per_variant=args.num_pairs_per_variant,
        temporal_gap_min=args.temporal_gap,
        temporal_gap_max=args.temporal_gap_max,
        pose_bins=pose_bins,
        output_file=args.output,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
