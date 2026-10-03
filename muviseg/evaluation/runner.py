"""One evaluation pipeline for both benchmarks.

Replaces two drivers that were ~85% identical, including the whole six-way
architecture dispatch and the whole per-sample metric loop. They genuinely
differed in five places, all captured by `DatasetSpec` below:

* the dataset class, and whether it takes a separate instance-mask root
* the name of the formatted-table output file
* RoMa's default variant when a config does not set one
* whether the visualisation title carries a `variant` field
* whether a pair-count-per-bin histogram is printed before loading

Everything else was the same code written twice.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from muviseg.config.paths import resolve_config_paths
from muviseg.evaluation.eval_metrics import (
    aggregate_metrics_by_bin,
    compute_metrics,
    print_table2_format,
)
from muviseg.evaluation.ground_truth_generator import generate_instance_correspondences
from muviseg.evaluation.model_infer import (
    MASt3RSegFeatInfer,
    SegMASt3RLGv2Infer,
    SegVGGTDPTInfer,
    SegVGGTDPTJointInfer,
    SegVGGTInfer,
    pad_masks_to_batch,
)
from muviseg.evaluation.vis_utils import (
    visualize_match_pair,
    visualize_match_tuple,
    visualize_score_heatmap,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    build: Callable[..., Any]          # dataset constructor
    collate: Callable[..., Any]        # pairwise collate
    collate_tuple: Callable[..., Any]  # N-frame collate
    load_pairs: Callable[[str], list]
    table_filename: str                # kept per dataset: existing tooling reads these
    roma_default_variant: str
    needs_instance_mask_root: bool
    print_bin_histogram: bool


def _replica_spec() -> DatasetSpec:
    from muviseg.datasets.replica_dataset import (
        ReplicaSegmentMatchDataset,
        collate_fn,
        collate_fn_tuple,
        load_pairs_from_json,
    )
    return DatasetSpec(
        name="replica",
        build=ReplicaSegmentMatchDataset,
        collate=collate_fn,
        collate_tuple=collate_fn_tuple,
        load_pairs=load_pairs_from_json,
        table_filename="table2_results.txt",
        roma_default_variant="indoor",
        needs_instance_mask_root=True,
        print_bin_histogram=True,
    )


def _vkitti2_spec() -> DatasetSpec:
    from muviseg.datasets.vkitti2_dataset import (
        VKitti2SegmentMatchDataset,
        collate_fn,
        collate_fn_tuple,
        load_pairs_from_json,
    )
    return DatasetSpec(
        name="vkitti2",
        build=VKitti2SegmentMatchDataset,
        collate=collate_fn,
        collate_tuple=collate_fn_tuple,
        load_pairs=load_pairs_from_json,
        table_filename="table_results.txt",
        roma_default_variant="outdoor",
        needs_instance_mask_root=False,
        print_bin_histogram=False,
    )


DATASETS: dict[str, Callable[[], DatasetSpec]] = {
    "replica": _replica_spec,
    "vkitti2": _vkitti2_spec,
}


def load_config(config_path: str) -> dict:
    """Load a YAML config and resolve its paths against the repository root."""
    with open(config_path, "r") as f:
        return resolve_config_paths(yaml.safe_load(f))


def setup_model(cfg: dict, device: torch.device, spec: DatasetSpec):
    """Build the model named by MODEL.ARCH and move it to `device`."""
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
            variant=mc.get('ROMA_VARIANT', spec.roma_default_variant),
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
    spec: DatasetSpec,
    vis_dir: Path = None,
    vis_counter: list = None,
    num_vis: int = 30,
    num_examples_per_pair: int = 5,
) -> list:
    """Score every pair in a batch. Returns one {'pose_bin', 'metrics'} per pair.

    Note that the acceptance rule the wrappers compute (mutual nearest neighbour
    plus matchability) is deliberately *not* used here: the reported metrics are
    computed from the dense score matrix. The accept/reject rule only matters to
    the downstream navigation experiment.
    """
    img0 = batch['img0'].to(device)
    img1 = batch['img1'].to(device)
    masks0_list = batch['masks0']
    masks1_list = batch['masks1']
    instance_ids0_list = batch['instance_ids0']
    instance_ids1_list = batch['instance_ids1']
    pose_bins = batch['pose_bin']

    # Multi-frame joint mode: the batch carries stacked (B, N, 3, H, W) in 'images'
    is_tuple_mode = hasattr(model, 'infer_tuple')
    if is_tuple_mode:
        images_bnchw = batch['images'].to(device)
        masks_nested = batch['masks']
        N = batch['n_frames']
        B = images_bnchw.shape[0]
        # Pad per-view masks across the batch: list[N] of (B, M_max_v, H, W)
        masks_per_view = [
            pad_masks_to_batch([masks_nested[b][v] for b in range(B)], device)
            for v in range(N)
        ]

    results = []
    for i in range(img0.shape[0]):
        img0_s = img0[i:i + 1]
        img1_s = img1[i:i + 1]
        masks0_s = masks0_list[i].unsqueeze(0).to(device)
        masks1_s = masks1_list[i].unsqueeze(0).to(device)
        instance_ids0 = instance_ids0_list[i]
        instance_ids1 = instance_ids1_list[i]
        pose_bin = pose_bins[i]

        if len(instance_ids0) == 0 or len(instance_ids1) == 0:
            results.append({
                'pose_bin': pose_bin,
                'metrics': {'AUPRC': 0.0, 'R@1': 0.0, 'R@5': 0.0, 'num_queries': 0},
            })
            continue

        with torch.no_grad():
            if is_tuple_mode:
                images_s = images_bnchw[i:i + 1]
                masks_s = [mv[i:i + 1] for mv in masks_per_view]
                _, scores = model.infer_tuple(images_s, masks_s)
                # scores is (1, M_max_0, M_max_1); trim to the real M_0/M_1
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
            scene = batch['scene'][i] if 'scene' in batch else ''
            idx0 = batch['idx0'][i] if 'idx0' in batch else vis_counter[0]
            idx1 = batch['idx1'][i] if 'idx1' in batch else ''
            if 'variant' in batch:
                variant = batch['variant'][i]
                pair_title = f"{scene}/{variant}  {idx0}→{idx1}  bin={pose_bin}"
            else:
                pair_title = f"{scene}  {idx0}→{idx1}  bin={pose_bin}"

            if is_tuple_mode:
                frame_indices = batch['frame_indices'][i]
                visualize_match_tuple(
                    batch['images'][i],
                    batch['masks'][i],
                    batch['instance_ids'][i],
                    scores_np, gt_matrix,
                    save_path=vis_dir / f"tuple_{vis_counter[0]:04d}_matches.png",
                    num_examples=num_examples_per_pair,
                    title=pair_title + f"  tuple={frame_indices}",
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


def run(cfg: dict, spec: DatasetSpec, *, device: str = "cuda", num_pairs: int = None,
        output_dir: str = None, visualize: bool = False, num_vis: int = 30) -> dict:
    """Run the full evaluation and write the three output files."""
    if output_dir is not None:
        cfg['EVAL']['OUTPUT_DIR'] = output_dir

    out = Path(cfg['EVAL']['OUTPUT_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out}")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {dev}")

    pairs_file = cfg['EVAL']['PAIRS_FILE']
    if not Path(pairs_file).exists():
        raise FileNotFoundError(
            f"Pairs file not found: {pairs_file}\n"
            f"The benchmark pair lists ship in assets/pairs/. Regenerate with "
            f"muviseg.sampling only if you do not need to match the reported numbers "
            f"-- see docs/reproduction.md."
        )

    pairs = spec.load_pairs(pairs_file)
    if num_pairs is not None:
        pairs = pairs[:num_pairs]
        print(f"Loaded {len(pairs)} pairs (limited to {num_pairs}) from {pairs_file}")
    else:
        print(f"Loaded {len(pairs)} pairs from {pairs_file}")

    bin_names = [f"{b[0]}-{b[1]}" for b in cfg['EVAL']['POSE_BINS']]

    if spec.print_bin_histogram:
        counts = [0] * len(bin_names)
        for pair in pairs:
            counts[pair['pose_bin']] += 1
        print("\nPair distribution by pose bin:")
        for bin_name, count in zip(bin_names, counts):
            print(f"  {bin_name}°: {count} pairs")

    n_frames = int(cfg['EVAL'].get('N_FRAMES', 2))
    context_step = int(cfg['EVAL'].get('CONTEXT_STEP', 5))
    kwargs = dict(
        data_root=cfg['DATASET']['DATA_ROOT'],
        pairs=pairs,
        target_h=cfg['DATASET']['RESIZE_H'],
        target_w=cfg['DATASET']['RESIZE_W'],
        m_prime=cfg['DATASET']['M_PRIME'],
        n_frames=n_frames,
        context_step=context_step,
    )
    if spec.needs_instance_mask_root:
        kwargs['instance_mask_root'] = cfg['DATASET']['INSTANCE_MASK_ROOT']
    dataset = spec.build(**kwargs)
    print(f"\nDataset size: {len(dataset)}  (n_frames={n_frames}, context_step={context_step})")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg['EVAL']['BATCH_SIZE'],
        shuffle=False,
        num_workers=cfg['EVAL']['NUM_WORKERS'],
        collate_fn=spec.collate_tuple if n_frames >= 2 else spec.collate,
    )
    print(f"Batch size: {cfg['EVAL']['BATCH_SIZE']}")
    print(f"Number of batches: {len(dataloader)}")

    model = setup_model(cfg, dev, spec)

    vis_dir = None
    vis_counter = [0]
    if visualize:
        vis_dir = out / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        print(f"Visualizations → {vis_dir}  (max {num_vis} pairs)")

    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80 + "\n")

    results_by_bin = defaultdict(list)
    for batch in tqdm(dataloader, desc="Evaluating"):
        for result in evaluate_batch(
            model, batch, dev, spec,
            vis_dir=vis_dir, vis_counter=vis_counter, num_vis=num_vis,
        ):
            results_by_bin[bin_names[result['pose_bin']]].append(result['metrics'])

    print("\n" + "=" * 80)
    print("Aggregating results...")
    print("=" * 80 + "\n")

    aggregated = aggregate_metrics_by_bin(results_by_bin)
    table_str = print_table2_format(aggregated)
    print(table_str)

    metrics_path = out / "metrics_by_bin.json"
    with open(metrics_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nSaved detailed metrics to: {metrics_path}")

    table_path = out / spec.table_filename
    with open(table_path, 'w') as f:
        f.write(table_str)
    print(f"Saved formatted table to: {table_path}")

    raw_path = out / "raw_results.json"
    with open(raw_path, 'w') as f:
        json.dump({
            'config': cfg,
            'results_by_bin': {k: v for k, v in results_by_bin.items()},
            'aggregated_metrics': aggregated,
        }, f, indent=2)
    print(f"Saved raw results to: {raw_path}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    return aggregated


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a MuViSeg model on a segment-matching benchmark")
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS),
                   help="which benchmark to run")
    p.add_argument("--config", required=True, help="path to an evaluation config")
    p.add_argument("--output_dir", default=None, help="override EVAL.OUTPUT_DIR")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--num_pairs", type=int, default=None,
                   help="limit to the first N pairs (smoke test)")
    p.add_argument("--visualize", action="store_true",
                   help="save match visualizations to <output_dir>/visualizations/")
    p.add_argument("--num_vis", type=int, default=30,
                   help="max number of pair visualizations to save (default 30)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = DATASETS[args.dataset]()
    cfg = load_config(args.config)
    print(f"Loaded config from: {args.config}")
    run(cfg, spec,
        device=args.device, num_pairs=args.num_pairs, output_dir=args.output_dir,
        visualize=args.visualize, num_vis=args.num_vis)
    return 0
