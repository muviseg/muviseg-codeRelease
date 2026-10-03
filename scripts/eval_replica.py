"""
Main Evaluation Script for SegMASt3R Table 2 Reproduction on Replica Dataset

Orchestrates the full evaluation pipeline:
1. Load configuration and pre-sampled image pairs
2. Initialize model from checkpoint
3. Run inference on all pairs
4. Compute metrics (AUPRC, R@1, R@5) grouped by pose bin
5. Print results in Table 2 format
6. Save detailed results to JSON
"""

import sys
from pathlib import Path

# Add evaluation/ directory to path for local imports
_eval_dir = Path(__file__).parent.parent
if str(_eval_dir) not in sys.path:
    sys.path.insert(0, str(_eval_dir))

import argparse
import json
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from muviseg.datasets.replica_dataset import (
    ReplicaSegmentMatchDataset,
    collate_fn,
    collate_fn_tuple,
    load_pairs_from_json,
)
from muviseg.evaluation.model_infer import (
    MASt3RSegFeatInfer,
    SegMASt3RLGv2Infer,
    SegVGGTDPTInfer,
    SegVGGTDPTJointInfer,
    SegVGGTInfer,
    pad_masks_to_batch,
)
from muviseg.evaluation.ground_truth_generator import generate_instance_correspondences
from muviseg.config.paths import resolve_config_paths
from muviseg.evaluation.eval_metrics import compute_metrics, aggregate_metrics_by_bin, print_table2_format
from muviseg.evaluation.vis_utils import visualize_match_pair, visualize_match_tuple, visualize_score_heatmap


def load_config(config_path: str) -> dict:
    """Load a YAML config and resolve its paths against the repository root."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return resolve_config_paths(config)


def setup_model(cfg: dict, device: torch.device):
    """
    Initialize model and load checkpoint.

    Args:
        cfg: Configuration dictionary
        device: Device to load model on

    Returns:
        Initialized model ready for inference
    """
    print("Initializing model...")
    arch = cfg['MODEL'].get('ARCH', 'mast3r').lower()

    if arch == 'segvggt_dpt':
        model = SegVGGTDPTInfer(cfg)
        model.prepare(device)
        print("SegVGGTDPT loaded successfully!")
        return model

    if arch == 'segvggt_dpt_joint':
        model = SegVGGTDPTJointInfer(cfg)
        model.prepare(device)
        print("SegVGGTDPTJoint loaded successfully!")
        return model

    if arch == 'roma':
        from muviseg.evaluation.roma_infer import RoMaSegInfer
        mc = cfg['MODEL']
        model = RoMaSegInfer(
            variant=mc.get('ROMA_VARIANT', 'indoor'),
            num_samples=mc.get('ROMA_NUM_SAMPLES', 5000),
            certainty_threshold=mc.get('ROMA_CERTAINTY', 0.0),
            upsample=mc.get('ROMA_UPSAMPLE', False),
            scoring=mc.get('ROMA_SCORING', 'certainty'),
        )
        model.prepare(device)
        print(f"RoMa loaded successfully (scoring={model.scoring}, "
              f"upsample={model.upsample})")
        return model

    if arch == 'segvggt':
        model = SegVGGTInfer(cfg)
        model.prepare(device)
        print("SegVGGT loaded successfully!")
        return model

    if arch == 'lightglue_v2':
        model = SegMASt3RLGv2Infer(cfg)
        model.prepare(device)
        print("SegMASt3RLGv2 loaded successfully!")
        return model

    # Default: MASt3R + Sinkhorn
    model = MASt3RSegFeatInfer(cfg)

    checkpoint_path = cfg['MODEL']['CHECKPOINT']
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.prepare(device)
    print("Model loaded successfully!")
    return model


def evaluate_batch(
    model,
    batch: dict,
    device: torch.device,
    vis_dir: Path = None,
    vis_counter: list = None,
    num_vis: int = 20,
    num_examples_per_pair: int = 5,
) -> list:
    """
    Evaluate a single batch and compute metrics.

    Args:
        model: Initialized model (MASt3RSegFeatInfer or SegVGGTDPTInfer)
        batch: Batch dictionary from dataloader
        device: Device for computation
        vis_dir: If set, save match visualizations here
        vis_counter: Mutable [int] tracking total visualizations saved
        num_vis: Max total visualization pairs to save
        num_examples_per_pair: GT match rows to show per pair image

    Returns:
        List of dicts with 'pose_bin' and 'metrics' keys
    """
    img0 = batch['img0'].to(device)
    img1 = batch['img1'].to(device)
    masks0_list = batch['masks0']
    masks1_list = batch['masks1']
    instance_ids0_list = batch['instance_ids0']
    instance_ids1_list = batch['instance_ids1']
    pose_bins = batch['pose_bin']

    is_tuple_mode = hasattr(model, 'infer_tuple')
    if is_tuple_mode:
        images_bnchw = batch['images'].to(device)
        masks_nested = batch['masks']
        N = batch['n_frames']
        B = images_bnchw.shape[0]
        masks_per_view = [
            pad_masks_to_batch([masks_nested[b][v] for b in range(B)], device)
            for v in range(N)
        ]

    results = []

    for i in range(img0.shape[0]):
        img0_single = img0[i:i+1]
        img1_single = img1[i:i+1]
        masks0_single = masks0_list[i].unsqueeze(0).to(device)
        masks1_single = masks1_list[i].unsqueeze(0).to(device)
        instance_ids0 = instance_ids0_list[i]
        instance_ids1 = instance_ids1_list[i]
        pose_bin = pose_bins[i]

        if len(instance_ids0) == 0 or len(instance_ids1) == 0:
            results.append({'pose_bin': pose_bin,
                            'metrics': {'AUPRC': 0.0, 'R@1': 0.0, 'R@5': 0.0, 'num_queries': 0}})
            continue

        with torch.no_grad():
            if is_tuple_mode:
                images_s = images_bnchw[i:i+1]
                masks_s = [mv[i:i+1] for mv in masks_per_view]
                _, scores = model.infer_tuple(images_s, masks_s)
                M0 = masks0_single.shape[1]
                M1 = masks1_single.shape[1]
                scores = scores[:, :M0, :M1]
            else:
                match_result, scores = model.infer_pair(
                    img0_single, img1_single, masks0_single, masks1_single
                )

        scores_np = scores[0].cpu().numpy()       # (M, N)
        gt_matrix = generate_instance_correspondences(instance_ids0, instance_ids1)
        metrics = compute_metrics(scores_np, gt_matrix)
        results.append({'pose_bin': pose_bin, 'metrics': metrics})

        # Visualizations
        if vis_dir is not None and vis_counter is not None and vis_counter[0] < num_vis:
            scene = batch['scene'][i] if 'scene' in batch else ''
            idx0 = batch['idx0'][i] if 'idx0' in batch else vis_counter[0]
            idx1 = batch['idx1'][i] if 'idx1' in batch else ''
            pair_title = f"{scene}  {idx0}→{idx1}  bin={pose_bin}"

            if is_tuple_mode:
                frame_indices = batch['frame_indices'][i]
                tuple_title = pair_title + f"  tuple={frame_indices}"
                visualize_match_tuple(
                    batch['images'][i],
                    batch['masks'][i],
                    batch['instance_ids'][i],
                    scores_np, gt_matrix,
                    save_path=vis_dir / f"tuple_{vis_counter[0]:04d}_matches.png",
                    num_examples=num_examples_per_pair,
                    title=tuple_title,
                )
            else:
                vis_path = vis_dir / f"pair_{vis_counter[0]:04d}_matches.png"
                visualize_match_pair(
                    img0_single[0], img1_single[0],
                    masks0_list[i], masks1_list[i],
                    instance_ids0, instance_ids1,
                    scores_np, gt_matrix,
                    save_path=vis_path,
                    num_examples=num_examples_per_pair,
                    title=pair_title,
                )

            heat_path = vis_dir / f"pair_{vis_counter[0]:04d}_heatmap.png"
            visualize_score_heatmap(
                img0_single[0], img1_single[0],
                scores_np, gt_matrix,
                save_path=heat_path,
                title=pair_title,
            )

            vis_counter[0] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SegMASt3R on Replica dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_eval_replica.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=None,
        help="Limit evaluation to first N pairs (for smoke testing)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save match visualizations to output_dir/visualizations/"
    )
    parser.add_argument(
        "--num_vis",
        type=int,
        default=30,
        help="Max number of pair visualizations to save (default 30)"
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Override output dir if specified
    if args.output_dir is not None:
        cfg['EVAL']['OUTPUT_DIR'] = args.output_dir

    output_dir = Path(cfg['EVAL']['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pairs
    pairs_file = cfg['EVAL']['PAIRS_FILE']
    if not Path(pairs_file).exists():
        raise FileNotFoundError(
            f"Pairs file not found: {pairs_file}\n"
            "Please run sample_pairs.py first to generate the pairs file."
        )

    pairs = load_pairs_from_json(pairs_file)
    if args.num_pairs is not None:
        pairs = pairs[:args.num_pairs]
        print(f"Loaded {len(pairs)} pairs (limited to {args.num_pairs}) from {pairs_file}")
    else:
        print(f"Loaded {len(pairs)} pairs from {pairs_file}")

    # Print distribution by pose bin
    bin_names = [f"{b[0]}-{b[1]}" for b in cfg['EVAL']['POSE_BINS']]
    bin_counts = [0] * len(bin_names)
    for pair in pairs:
        bin_counts[pair['pose_bin']] += 1

    print("\nPair distribution by pose bin:")
    for bin_name, count in zip(bin_names, bin_counts):
        print(f"  {bin_name}°: {count} pairs")

    # Create dataset
    n_frames = int(cfg['EVAL'].get('N_FRAMES', 2))
    context_step = int(cfg['EVAL'].get('CONTEXT_STEP', 5))
    dataset = ReplicaSegmentMatchDataset(
        data_root=cfg['DATASET']['DATA_ROOT'],
        instance_mask_root=cfg['DATASET']['INSTANCE_MASK_ROOT'],
        pairs=pairs,
        target_h=cfg['DATASET']['RESIZE_H'],
        target_w=cfg['DATASET']['RESIZE_W'],
        m_prime=cfg['DATASET'].get('M_PRIME', None),
        n_frames=n_frames,
        context_step=context_step,
    )

    print(f"\nDataset size: {len(dataset)}  (n_frames={n_frames}, context_step={context_step})")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg['EVAL']['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['EVAL']['NUM_WORKERS'],
        collate_fn=collate_fn_tuple if n_frames >= 2 else collate_fn,
    )

    print(f"Batch size: {cfg['EVAL']['BATCH_SIZE']}")
    print(f"Number of batches: {len(dataloader)}")

    # Setup model
    model = setup_model(cfg, device)

    # Visualization setup
    vis_dir = None
    vis_counter = [0]
    if args.visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        print(f"Visualizations → {vis_dir}  (max {args.num_vis} pairs)")

    # Run evaluation
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80 + "\n")

    results_by_bin = defaultdict(list)

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch_results = evaluate_batch(
            model, batch, device,
            vis_dir=vis_dir,
            vis_counter=vis_counter,
            num_vis=args.num_vis,
        )

        # Accumulate results by pose bin
        for result in batch_results:
            pose_bin = result['pose_bin']
            bin_name = bin_names[pose_bin]
            results_by_bin[bin_name].append(result['metrics'])

    # Aggregate metrics by pose bin
    print("\n" + "="*80)
    print("Aggregating results...")
    print("="*80 + "\n")

    aggregated_metrics = aggregate_metrics_by_bin(results_by_bin)

    # Print results in Table 2 format
    table_str = print_table2_format(aggregated_metrics)
    print(table_str)

    # Save results to JSON
    results_json_path = output_dir / "metrics_by_bin.json"
    with open(results_json_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nSaved detailed metrics to: {results_json_path}")

    # Save formatted table to text file
    table_txt_path = output_dir / "table2_results.txt"
    with open(table_txt_path, 'w') as f:
        f.write(table_str)
    print(f"Saved formatted table to: {table_txt_path}")

    # Save all raw results
    raw_results_path = output_dir / "raw_results.json"
    raw_results = {
        'config': cfg,
        'results_by_bin': {k: v for k, v in results_by_bin.items()},
        'aggregated_metrics': aggregated_metrics
    }
    with open(raw_results_path, 'w') as f:
        json.dump(raw_results, f, indent=2)
    print(f"Saved raw results to: {raw_results_path}")

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
