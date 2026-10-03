"""
Sanity check for ScanNet++ preprocessing pipeline.

Checks:
  1. scannetpp_processed  — depth + images present for all scenes
  2. masks_resize_mast3r  — per-image SAM2 pkl files complete & valid
  3. masks_resize_mast3r_res — pair corr pkl files: progress + spot-check

Usage:
    python -m training.scripts.sanity_check \
        --root /mnt/vol1/datasets/ScanNet++ \
        [--sample 50]          # number of pkls to spot-check per step
        [--check_rle]          # decode a few RLEs to verify they're valid bitmaps
"""

import argparse
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def fail(msg):  print(f"  {RED}✗{RESET} {msg}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 – scannetpp_processed
# ─────────────────────────────────────────────────────────────────────────────

def check_processed(processed_root: Path, meta_scenes, meta_images_per_scene):
    section("Step 1 — scannetpp_processed")

    dirs = sorted(d for d in processed_root.iterdir() if d.is_dir())
    print(f"  Scenes on disk  : {len(dirs)}")
    print(f"  Scenes in meta  : {len(meta_scenes)}")

    meta_set  = set(meta_scenes)
    disk_set  = {d.name for d in dirs}

    in_meta_not_disk = meta_set - disk_set
    in_disk_not_meta = disk_set - meta_set

    if in_meta_not_disk:
        fail(f"In metadata but missing on disk: {in_meta_not_disk}")
    else:
        ok("All metadata scenes present on disk")

    if in_disk_not_meta:
        warn(f"Extra dirs not in metadata: {in_disk_not_meta}")

    # Per-scene: check depth/ and images/ exist, image count matches metadata
    depth_missing = []
    images_missing = []
    count_mismatch = []

    for d in dirs:
        if d.name not in meta_set:
            continue
        if not (d / "depth").exists():
            depth_missing.append(d.name)
        if not (d / "images").exists():
            images_missing.append(d.name)
        else:
            n_imgs = len(list((d / "images").iterdir()))
            expected = meta_images_per_scene.get(d.name, 0)
            if n_imgs != expected:
                count_mismatch.append((d.name, n_imgs, expected))

    if depth_missing:
        fail(f"Missing depth/: {depth_missing[:5]}")
    else:
        ok("All scenes have depth/")

    if images_missing:
        fail(f"Missing images/: {images_missing[:5]}")
    else:
        ok("All scenes have images/")

    if count_mismatch:
        fail(f"Image count mismatch (first 5): {count_mismatch[:5]}")
    else:
        ok("Image counts match metadata for all scenes")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 – masks_resize_mast3r (per-image)
# ─────────────────────────────────────────────────────────────────────────────

def check_masks(masks_root: Path, meta_scenes, meta_images, meta_sceneids,
                sample: int, check_rle: bool):
    section("Step 2 — masks_resize_mast3r (per-image SAM2 masks)")

    if not masks_root.exists():
        fail(f"Directory does not exist: {masks_root}")
        return

    mask_scenes = {d.name for d in masks_root.iterdir() if d.is_dir()}
    meta_set    = set(meta_scenes)

    in_meta_not_mask = meta_set - mask_scenes
    if in_meta_not_mask:
        fail(f"Scenes in metadata but missing from masks ({len(in_meta_not_mask)}): "
             f"{sorted(in_meta_not_mask)[:5]}")
    else:
        ok(f"All {len(meta_set)} metadata scenes have a masks directory")

    # Count pkls
    all_pkls = list(masks_root.glob("*/*.pkl"))
    print(f"  Total pkl files : {len(all_pkls)}")
    print(f"  Expected (meta) : {len(meta_images)}")

    if len(all_pkls) == len(meta_images):
        ok("pkl count exactly matches metadata image count")
    elif len(all_pkls) < len(meta_images):
        missing = len(meta_images) - len(all_pkls)
        warn(f"Missing {missing} pkl files ({missing/len(meta_images)*100:.1f}%)")
    else:
        warn(f"More pkl files than expected: {len(all_pkls)} > {len(meta_images)}")

    # Spot-check: load N random pkls
    sample_pkls = random.sample(all_pkls, min(sample, len(all_pkls)))
    corrupt = []
    empty   = []
    wrong_keys = []

    for p in sample_pkls:
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            if "mask_coco_rles_resized" not in data:
                wrong_keys.append(p.name)
                continue
            rles = data["mask_coco_rles_resized"]
            if len(rles) == 0:
                empty.append(p.name)
            elif check_rle:
                # Decode one RLE to verify it's a valid bitmap
                from pycocotools import mask as mask_utils
                rle = rles[0]
                if isinstance(rle["counts"], str):
                    rle = dict(rle, counts=rle["counts"].encode())
                bm = mask_utils.decode(rle)
                assert bm.ndim == 2 and bm.max() <= 1
        except Exception as e:
            corrupt.append((p.name, str(e)))

    if corrupt:
        fail(f"Corrupt pkls in sample: {corrupt[:3]}")
    else:
        ok(f"Spot-check ({len(sample_pkls)} files): no corrupt pkls")

    if wrong_keys:
        fail(f"Wrong keys in sample: {wrong_keys[:3]}")
    else:
        ok("Keys look correct (mask_coco_rles_resized present)")

    empty_frac = len(empty) / len(sample_pkls) if sample_pkls else 0
    if empty_frac > 0.1:
        warn(f"{len(empty)}/{len(sample_pkls)} sampled files have 0 masks ({empty_frac*100:.1f}%)")
    else:
        ok(f"Empty-masks rate in sample: {len(empty)}/{len(sample_pkls)} ({empty_frac*100:.1f}%)")

    # Mask count distribution
    counts = []
    for p in sample_pkls:
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            counts.append(len(data.get("mask_coco_rles_resized", [])))
        except Exception:
            pass
    if counts:
        arr = np.array(counts)
        print(f"  Mask count (sample): min={arr.min()} median={np.median(arr):.0f} "
              f"max={arr.max()} mean={arr.mean():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 – masks_resize_mast3r_res (pair correspondences)
# ─────────────────────────────────────────────────────────────────────────────

def check_corr_pairs(res_root: Path, meta_scenes, meta_sceneids, meta_images, meta_pairs,
                     sample: int):
    section("Step 3 — masks_resize_mast3r_res (pair correspondences)")

    if not res_root.exists():
        fail(f"Directory does not exist: {res_root}")
        print(f"  corr_pairs.py has NOT been run yet.")
        return

    # Pairs per scene from metadata
    pairs_per_scene = Counter()
    for i, j in meta_pairs:
        scene = meta_scenes[meta_sceneids[i]]
        pairs_per_scene[scene] += 1

    # Actual processed pairs per scene
    res_scenes = sorted(d for d in res_root.iterdir() if d.is_dir())
    print(f"  Scenes with output : {len(res_scenes)} / {len(meta_scenes)}")

    total_done  = 0
    total_expected = len(meta_pairs)
    per_scene_progress = {}

    for s in res_scenes:
        done = len(list(s.glob("*.pkl")))
        exp  = pairs_per_scene.get(s.name, 0)
        total_done += done
        per_scene_progress[s.name] = (done, exp)

    pct = total_done / total_expected * 100 if total_expected else 0
    print(f"  Pairs done  : {total_done:,} / {total_expected:,} ({pct:.2f}%)")
    print(f"  Pairs left  : {total_expected - total_done:,}")

    if total_done == 0:
        fail("No pair pkl files found — corr_pairs.py needs to be run")
        return
    elif pct < 100:
        warn(f"corr_pairs.py only {pct:.2f}% complete — needs to be resumed")
    else:
        ok("All pairs processed")

    # Per-scene breakdown
    print(f"\n  Per-scene progress (scenes with output):")
    for scene, (done, exp) in sorted(per_scene_progress.items()):
        bar_done = int(done / exp * 20) if exp else 0
        bar = "[" + "#" * bar_done + "." * (20 - bar_done) + "]"
        print(f"    {scene}: {bar} {done}/{exp} ({done/exp*100:.1f}%)" if exp
              else f"    {scene}: {done} pairs (not in metadata?)")

    # Spot-check content
    all_pair_pkls = list(res_root.glob("*/*.pkl"))
    sample_pkls = random.sample(all_pair_pkls, min(sample, len(all_pair_pkls)))
    corrupt = []
    zero_corr = 0

    for p in sample_pkls:
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            if "seg_corr_list" not in data:
                corrupt.append((p.name, f"missing key: seg_corr_list (keys={list(data.keys())})"))
            elif data["seg_corr_list"] is not None and len(data["seg_corr_list"]) == 0:
                zero_corr += 1
        except Exception as e:
            corrupt.append((p.name, str(e)))

    if corrupt:
        fail(f"Corrupt pair pkls in sample: {corrupt[:3]}")
    else:
        ok(f"Spot-check ({len(sample_pkls)} pair pkls): seg_corr_list key present")

    zero_frac = zero_corr / len(sample_pkls) if sample_pkls else 0
    if zero_frac > 0.5:
        warn(f"High fraction of pairs with 0 correspondences: "
             f"{zero_corr}/{len(sample_pkls)} ({zero_frac*100:.0f}%)")
    else:
        ok(f"Pairs with 0 correspondences in sample: "
           f"{zero_corr}/{len(sample_pkls)} ({zero_frac*100:.0f}%)")

    # Corr count distribution
    corr_counts = []
    for p in sample_pkls:
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            cl = data.get("seg_corr_list") or []
            corr_counts.append(len(cl))
        except Exception:
            pass
    if corr_counts:
        arr = np.array(corr_counts)
        print(f"\n  Corr count (sample): min={arr.min()} median={np.median(arr):.0f} "
              f"max={arr.max()} mean={arr.mean():.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/mnt/vol1/datasets/ScanNet++",
                   help="ScanNet++ dataset root")
    p.add_argument("--sample", type=int, default=100,
                   help="Number of pkl files to spot-check per step")
    p.add_argument("--check_rle", action="store_true",
                   help="Decode one RLE per sampled file to verify bitmap validity")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    root           = Path(args.root)
    processed_root = root / "scannetpp_processed"
    masks_root     = root / "masks_resize_mast3r"
    res_root       = root / "masks_resize_mast3r_res"
    meta_path      = processed_root / "all_metadata.npz"

    print(f"\nScanNet++ root : {root}")
    print(f"Metadata       : {meta_path}")

    if not meta_path.exists():
        print(f"{RED}ERROR{RESET}: metadata not found at {meta_path}")
        sys.exit(1)

    # Load metadata once
    print("\nLoading all_metadata.npz ...")
    meta = np.load(str(meta_path), allow_pickle=True)
    meta_scenes   = meta["scenes"]        # (S,)
    meta_sceneids = meta["sceneids"]      # (N,)
    meta_images   = meta["images"]        # (N,)
    meta_pairs    = meta["pairs"][:, :2].astype(int)  # (P, 2)

    # images per scene
    imgs_per_scene = Counter()
    for sc_idx in meta_sceneids:
        imgs_per_scene[meta_scenes[sc_idx]] += 1

    print(f"  Scenes : {len(meta_scenes)}")
    print(f"  Images : {len(meta_images)}")
    print(f"  Pairs  : {len(meta_pairs):,}")

    # Run checks
    check_processed(processed_root, meta_scenes, imgs_per_scene)
    check_masks(masks_root, meta_scenes, meta_images, meta_sceneids,
                args.sample, args.check_rle)
    check_corr_pairs(res_root, meta_scenes, meta_sceneids, meta_images, meta_pairs,
                     args.sample)

    section("Summary")
    print("  Re-run with --check_rle for deeper RLE validation.")
    print("  corr_pairs.py is idempotent (skips existing files) — safe to resume.\n")


if __name__ == "__main__":
    main()
