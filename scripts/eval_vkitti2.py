"""
Evaluate SegVGGTDPT (or any supported arch) on Virtual KITTI 2 dataset.

Uses trackID-based ground truth correspondences.

Run from evaluation/ directory:
    python3 scripts/eval_segvggt_vkitti2.py \
        --config configs/config_eval_vkitti2_segvggt.yaml \
        --output_dir results/segvggt_dpt_vkitti2/v3-001

Smoke test (5 pairs):
    python3 scripts/eval_segvggt_vkitti2.py \
        --config configs/config_eval_vkitti2_segvggt.yaml \
        --num_pairs 5
"""

import sys
from pathlib import Path

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

from muviseg.datasets.vkitti2_dataset import (
    VKitti2SegmentMatchDataset,
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
from muviseg.config.paths import resolve_config_paths
from muviseg.evaluation.ground_truth_generator import generate_instance_correspondences
from muviseg.evaluation.eval_metrics import compute_metrics, aggregate_metrics_by_bin, print_table2_format
from muviseg.evaluation.vis_utils import visualize_match_pair, visualize_match_tuple, visualize_score_heatmap


def load_config(config_path: str) -> dict:
    """Load a YAML config and resolve its paths against the repository root."""
    with open(config_path, 'r') as f:
        return resolve_config_paths(yaml.safe_load(f))


def setup_model(cfg: dict, device: torch.device):
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
            variant=mc.get('ROMA_VARIANT', 'outdoor'),
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

    # MASt3R + Sinkhorn
    model = MASt3RSegFeatInfer(cfg)
    ckpt_path = cfg['MODEL']['CHECKPOINT']
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.prepare(device)
    print("Model loaded successfully!")
    return model


def evaluate_batch(
    model,
    batch: dict,
    device: torch.device,
    vis_dir: Path = None,
    vis_counter: list = None,
    num_vis: int = 30,
    num_examples_per_pair: int = 5,
) -> list:
    img0 = batch['img0'].to(device)
    img1 = batch['img1'].to(device)
    masks0_list = batch['masks0']
    masks1_list = batch['masks1']
    instance_ids0_list = batch['instance_ids0']
    instance_ids1_list = batch['instance_ids1']
    pose_bins = batch['pose_bin']

    # Multi-frame joint mode: batch contains stacked (B, N, 3, H, W) in 'images'
    is_tuple_mode = hasattr(model, 'infer_tuple')
    if is_tuple_mode:
        images_bnchw = batch['images'].to(device)  # (B, N, 3, H, W)
        masks_nested = batch['masks']              # list[B] of list[N] of (M_v, H, W)
        N = batch['n_frames']
        B = images_bnchw.shape[0]
        # Pad per-view masks across the batch: list[N] of (B, M_max_v, H, W)
        masks_per_view = [
            pad_masks_to_batch([masks_nested[b][v] for b in range(B)], device)
            for v in range(N)
        ]

    results = []
    for i in range(img0.shape[0]):
        img0_s = img0[i:i+1]
        img1_s = img1[i:i+1]
        masks0_s = masks0_list[i].unsqueeze(0).to(device)
        masks1_s = masks1_list[i].unsqueeze(0).to(device)
        instance_ids0 = instance_ids0_list[i]
        instance_ids1 = instance_ids1_list[i]
        pose_bin = pose_bins[i]

        if len(instance_ids0) == 0 or len(instance_ids1) == 0:
            results.append({'pose_bin': pose_bin,
                            'metrics': {'AUPRC': 0.0, 'R@1': 0.0, 'R@5': 0.0, 'num_queries': 0}})
            continue

        with torch.no_grad():
            if is_tuple_mode:
                images_s = images_bnchw[i:i+1]                          # (1, N, 3, H, W)
                masks_s = [mv[i:i+1] for mv in masks_per_view]          # list[N] of (1, M_max_v, H, W)
                _, scores = model.infer_tuple(images_s, masks_s)
                # scores shape: (1, M_max_0, M_max_1) — trim to the real M_0/M_1 for metrics
                M0 = masks0_s.shape[1]
                M1 = masks1_s.shape[1]
                scores = scores[:, :M0, :M1]
            else:
                _, scores = model.infer_pair(img0_s, img1_s, masks0_s, masks1_s)

        scores_np = scores[0].cpu().numpy()
        gt_matrix = generate_instance_correspondences(instance_ids0, instance_ids1)
        metrics = compute_metrics(scores_np, gt_matrix)
        results.append({'pose_bin': pose_bin, 'metrics': metrics})

        if vis_dir is not None and vis_counter is not None and vis_counter[0] < num_vis:
            scene = batch['scene'][i]
            variant = batch.get('variant', [''])[i]
            idx0 = batch['idx0'][i]
            idx1 = batch['idx1'][i]
            pair_title = f"{scene}/{variant}  {idx0}→{idx1}  bin={pose_bin}"

            if is_tuple_mode:
                frame_indices = batch['frame_indices'][i]
                tuple_title = pair_title + f"  tuple={frame_indices}"
                visualize_match_tuple(
                    batch['images'][i],               # (N, 3, H, W) CPU tensor
                    batch['masks'][i],                # list[N] of (M_v, H, W)
                    batch['instance_ids'][i],
                    scores_np, gt_matrix,
                    save_path=vis_dir / f"tuple_{vis_counter[0]:04d}_matches.png",
                    num_examples=num_examples_per_pair,
                    title=tuple_title,
                )
            else:
                visualize_match_pair(
                    img0_s[0], img1_s[0],
                    masks0_list[i], masks1_list[i],
                    instance_ids0, instance_ids1,
                    scores_np, gt_matrix,
                    save_path=vis_dir / f"pair_{vis_counter[0]:04d}_matches.png",
                    num_examples=num_examples_per_pair,
                    title=pair_title,
                )
            visualize_score_heatmap(
                img0_s[0], img1_s[0],
                scores_np, gt_matrix,
                save_path=vis_dir / f"pair_{vis_counter[0]:04d}_heatmap.png",
                title=pair_title,
            )
            vis_counter[0] += 1

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on Virtual KITTI 2")
    parser.add_argument("--config", type=str, default="configs/config_eval_vkitti2_segvggt.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_pairs", type=int, default=None,
                        help="Limit to first N pairs (smoke test)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save match visualizations to output_dir/visualizations/")
    parser.add_argument("--num_vis", type=int, default=30,
                        help="Max number of pair visualizations to save (default 30)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    if args.output_dir is not None:
        cfg['EVAL']['OUTPUT_DIR'] = args.output_dir

    output_dir = Path(cfg['EVAL']['OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs_file = cfg['EVAL']['PAIRS_FILE']
    if not Path(pairs_file).exists():
        raise FileNotFoundError(
            f"Pairs file not found: {pairs_file}\n"
            "Run: python3 sampling/sample_pairs_vkitti2.py --output pairs_vkitti2_4000.json "
            "--num_pairs_per_variant 160"
        )

    pairs = load_pairs_from_json(pairs_file)
    if args.num_pairs is not None:
        pairs = pairs[:args.num_pairs]
        print(f"Loaded {len(pairs)} pairs (limited to {args.num_pairs})")
    else:
        print(f"Loaded {len(pairs)} pairs from {pairs_file}")

    bin_names = [f"{b[0]}-{b[1]}" for b in cfg['EVAL']['POSE_BINS']]

    n_frames = int(cfg['EVAL'].get('N_FRAMES', 2))
    context_step = int(cfg['EVAL'].get('CONTEXT_STEP', 5))
    dataset = VKitti2SegmentMatchDataset(
        data_root=cfg['DATASET']['DATA_ROOT'],
        pairs=pairs,
        target_h=cfg['DATASET']['RESIZE_H'],
        target_w=cfg['DATASET']['RESIZE_W'],
        m_prime=cfg['DATASET'].get('M_PRIME', None),
        n_frames=n_frames,
        context_step=context_step,
    )
    print(f"Dataset size: {len(dataset)}  (n_frames={n_frames}, context_step={context_step})")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg['EVAL']['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['EVAL']['NUM_WORKERS'],
        collate_fn=collate_fn_tuple if n_frames >= 2 else collate_fn,
    )

    model = setup_model(cfg, device)

    vis_dir = None
    vis_counter = [0]
    if args.visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        print(f"Visualizations → {vis_dir}  (max {args.num_vis} pairs)")

    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80 + "\n")

    results_by_bin = defaultdict(list)
    for batch in tqdm(dataloader, desc="Evaluating"):
        for result in evaluate_batch(
            model, batch, device,
            vis_dir=vis_dir,
            vis_counter=vis_counter,
            num_vis=args.num_vis,
        ):
            bin_name = bin_names[result['pose_bin']]
            results_by_bin[bin_name].append(result['metrics'])

    print("\n" + "="*80)
    print("Aggregating results...")
    print("="*80 + "\n")

    aggregated_metrics = aggregate_metrics_by_bin(results_by_bin)
    table_str = print_table2_format(aggregated_metrics)
    print(table_str)

    results_json_path = output_dir / "metrics_by_bin.json"
    with open(results_json_path, 'w') as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nSaved metrics to: {results_json_path}")

    table_txt_path = output_dir / "table_results.txt"
    with open(table_txt_path, 'w') as f:
        f.write(table_str)
    print(f"Saved table to: {table_txt_path}")

    raw_results_path = output_dir / "raw_results.json"
    with open(raw_results_path, 'w') as f:
        json.dump({'config': cfg, 'results_by_bin': {k: v for k, v in results_by_bin.items()},
                   'aggregated_metrics': aggregated_metrics}, f, indent=2)
    print(f"Saved raw results to: {raw_results_path}")

    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
