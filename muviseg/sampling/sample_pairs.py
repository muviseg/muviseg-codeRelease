"""
Stratified Pair Sampling for Replica Dataset Evaluation

Samples image pairs across different pose bins (rotation angles) for evaluation.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


def load_poses(traj_file: str) -> np.ndarray:
    """
    Load camera poses from Replica trajectory file.

    Args:
        traj_file: Path to traj_w_c.txt file

    Returns:
        Array of 4x4 transformation matrices, shape (N, 4, 4)
    """
    poses = []
    with open(traj_file, 'r') as f:
        for line in f:
            # Each line has 16 values (4x4 matrix in row-major order)
            values = list(map(float, line.strip().split()))
            if len(values) == 16:
                pose = np.array(values).reshape(4, 4)
                poses.append(pose)

    return np.array(poses)


def compute_rotation_angle(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """
    Compute rotation angle between two poses using geodesic distance on SO(3).

    Args:
        pose1: 4x4 transformation matrix
        pose2: 4x4 transformation matrix

    Returns:
        Rotation angle in degrees [0, 180]
    """
    # Extract rotation matrices
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]

    # Compute relative rotation
    R_rel = R1.T @ R2

    # Compute angle from trace
    # trace(R) = 1 + 2*cos(theta)
    # theta = arccos((trace(R) - 1) / 2)
    trace = np.trace(R_rel)

    # Clamp to avoid numerical issues with arccos
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


def sample_pairs_for_scene(
    scene_path: Path,
    num_pairs_per_bin: int,
    temporal_gap_min: int,
    temporal_gap_max: int = 0,
    pose_bins: List[Tuple[float, float]] = None,
    random_seed: int = 42
) -> List[Dict]:
    """
    Sample image pairs for a single scene across pose bins.

    Args:
        scene_path: Path to scene directory (e.g., /data/office_0)
        num_pairs_per_bin: Number of pairs to sample per pose bin
        temporal_gap_min: Minimum temporal gap between frame indices
        temporal_gap_max: Maximum temporal gap between frame indices (0 = no limit)
        pose_bins: List of (min, max) tuples for pose bins
        random_seed: Random seed for reproducibility

    Returns:
        List of pair dictionaries with keys: idx0, idx1, pose_bin, angle
    """
    # Load poses
    traj_file = scene_path / "Sequence_1" / "traj_w_c.txt"
    poses = load_poses(str(traj_file))
    num_frames = len(poses)

    print(f"  Loaded {num_frames} poses from {scene_path.name}")

    # Initialize random state
    rng = np.random.RandomState(random_seed)

    # Collect candidate pairs for each bin
    candidates_by_bin = [[] for _ in range(len(pose_bins))]

    # Generate all valid pairs with temporal gap in [min, max]
    print(f"  Generating candidate pairs...")
    for i in range(num_frames):
        # Upper limit: min(i + temporal_gap_max + 1, num_frames)
        max_j = min(i + temporal_gap_max + 1, num_frames) if temporal_gap_max > 0 else num_frames

        for j in range(i + temporal_gap_min + 1, max_j):
            # Compute rotation angle
            angle = compute_rotation_angle(poses[i], poses[j])

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
        print(f"  Bin {pose_bins[bin_idx]}: {num_candidates} candidates")

        if num_candidates == 0:
            print(f"    WARNING: No candidates for bin {pose_bins[bin_idx]}")
            continue

        # Sample with replacement if not enough candidates
        if num_candidates < num_pairs_per_bin:
            print(f"    WARNING: Only {num_candidates} candidates, sampling with replacement")
            sampled_indices = rng.choice(num_candidates, size=num_pairs_per_bin, replace=True)
        else:
            sampled_indices = rng.choice(num_candidates, size=num_pairs_per_bin, replace=False)

        for idx in sampled_indices:
            pair = candidates[idx].copy()
            pair['pose_bin'] = bin_idx
            sampled_pairs.append(pair)

    return sampled_pairs


def sample_replica_pairs(
    data_root: str,
    scenes: List[str],
    num_pairs_per_scene: int,
    temporal_gap_min: int,
    temporal_gap_max: int = 0,
    pose_bins: List[Tuple[float, float]] = None,
    output_file: str = "pairs_replica.json",
    random_seed: int = 42
):
    """
    Sample image pairs across all Replica scenes.

    Args:
        data_root: Root directory containing scene folders
        scenes: List of scene names (e.g., ['office_0', 'office_1', ...])
        num_pairs_per_scene: Total pairs per scene (distributed across bins)
        temporal_gap_min: Minimum temporal gap between frames
        temporal_gap_max: Maximum temporal gap between frames (0 = no limit)
        pose_bins: List of (min, max) tuples for pose bins
        output_file: Path to output JSON file
        random_seed: Random seed for reproducibility
    """
    data_root = Path(data_root)
    num_bins = len(pose_bins)
    num_pairs_per_bin = num_pairs_per_scene // num_bins

    print(f"Sampling {num_pairs_per_scene} pairs per scene ({num_pairs_per_bin} per bin)")
    print(f"Pose bins: {pose_bins}")
    if temporal_gap_max > 0:
        print(f"Temporal gap: [{temporal_gap_min}, {temporal_gap_max}] frames")
    else:
        print(f"Temporal gap: >= {temporal_gap_min} frames (no upper limit)")
    print(f"Random seed: {random_seed}\n")

    all_pairs = []

    for scene in tqdm(scenes, desc="Sampling scenes"):
        scene_path = data_root / scene

        if not scene_path.exists():
            print(f"WARNING: Scene {scene} not found at {scene_path}")
            continue

        # Sample pairs for this scene
        pairs = sample_pairs_for_scene(
            scene_path,
            num_pairs_per_bin,
            temporal_gap_min,
            temporal_gap_max,
            pose_bins,
            random_seed
        )

        # Add scene name to each pair
        for pair in pairs:
            pair['scene'] = scene

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
    parser = argparse.ArgumentParser(description="Sample image pairs for Replica evaluation")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data",
        help="Root directory containing scene folders"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pairs_replica_3200.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--num_pairs_per_scene",
        type=int,
        default=400,
        help="Number of pairs to sample per scene"
    )
    parser.add_argument(
        "--temporal_gap",
        type=int,
        default=5,
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
        default=['office_0', 'office_1', 'office_2', 'office_3', 'office_4',
                 'room_0', 'room_1', 'room_2'],
        help="List of scene names"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Define pose bins
    pose_bins = [
        (0, 45),
        (45, 90),
        (90, 135),
        (135, 180)
    ]

    sample_replica_pairs(
        data_root=args.data_root,
        scenes=args.scenes,
        num_pairs_per_scene=args.num_pairs_per_scene,
        temporal_gap_min=args.temporal_gap,
        temporal_gap_max=args.temporal_gap_max,
        pose_bins=pose_bins,
        output_file=args.output,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
