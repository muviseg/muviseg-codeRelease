"""
Generate seg_corr_list for each image pair in scannetpp_processed.

Reads:  all_metadata.npz  (pairs, intrinsics, trajectories, images, scenes, sceneids)
        scannetpp_processed/<scene>/depth/<name>.png
        masks_resize_mast3r/<scene>/<name>.pkl   (mask_coco_rles_resized)

Writes: masks_resize_mast3r_res/<scene>/<name_i>__<name_j>.pkl
            {
              'seg_corr_list': [(i0,j0), (i1,j1), ...]
            }
        NOTE: RLEs are intentionally NOT stored here — load them from
        masks_resize_mast3r/<scene>/<name>.pkl when needed for training.

Usage:
    python -m training.scripts.corr_pairs \
        --metadata   /mnt/vol1/datasets/ScanNet++/scannetpp_processed/all_metadata.npz \
        --processed  /mnt/vol1/datasets/ScanNet++/scannetpp_processed \
        --masks      /mnt/vol1/datasets/ScanNet++/masks_resize_mast3r \
        --output     /mnt/vol1/datasets/ScanNet++/masks_resize_mast3r_res \
        --num_workers 32 \
        --iou_threshold 0.25
"""

import os
# Prevent each worker process from spawning extra numpy/BLAS threads.
# With many workers, N_workers × N_threads threads would over-subscribe
# the CPU and thrash cache — single-threaded numpy per worker is faster.
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS",     "1")

import argparse
import pickle
import traceback
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Process-level mask cache
#
# Within a ProcessPoolExecutor worker, process_image_batch is called many
# times.  Because DUSt3R pairs are spatially local, the img_j appearing in
# group (scene, img_i=t) heavily overlaps with those in group (scene, img_i=t+1).
# Caching decoded masks_j across calls avoids redundant pycocotools decodes.
#
# Capacity: _MASK_CACHE_MAX entries.  Each entry is ~(M × H*W) uint8.
# With M≈50 and H*W≈200 K the memory cost is ~10 MB/entry.
# ---------------------------------------------------------------------------

_MASK_CACHE     = OrderedDict()   # str(path) -> (masks_flat, areas, (H, W))
_MASK_CACHE_MAX = 16              # increase if the machine has plenty of RAM


def _load_masks_cached(pkl_path: Path):
    """Like _load_and_decode_masks but with an LRU cache per worker process."""
    key = str(pkl_path)
    if key in _MASK_CACHE:
        _MASK_CACHE.move_to_end(key)
        return _MASK_CACHE[key]
    result = _load_and_decode_masks(pkl_path)
    if len(_MASK_CACHE) >= _MASK_CACHE_MAX:
        _MASK_CACHE.popitem(last=False)   # evict least-recently-used
    _MASK_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def load_depth(depth_path: Path) -> np.ndarray:
    """Returns float32 depth in metres. 0 = invalid."""
    d = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    d /= 1000.0          # mm → m
    d[d <= 0] = 0.0
    return d             # (H, W)


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    depth: (H, W) float32 metres
    K:     (4,) [fx, fy, cx, cy]  (colmap convention)
    returns xyz: (H*W, 3), valid: (H*W,) bool
    """
    H, W = depth.shape
    fx, fy, cx, cy = K
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)               # (H, W)
    valid = depth > 0
    z = depth                                # (H, W)
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z
    xyz = np.stack([x, y, z], axis=-1)       # (H, W, 3)
    return xyz.reshape(-1, 3), valid.reshape(-1)


def project(xyz: np.ndarray, pose_i: np.ndarray, pose_j: np.ndarray,
            K_j: np.ndarray, H: int, W: int):
    """
    xyz:    (N, 3) in camera_i frame
    pose_*: (4, 4) cam-to-world matrices
    K_j:    (4,) [fx, fy, cx, cy]
    Returns uv: (N, 2) float32, in_bounds: (N,) bool
    """
    # cam_i → world
    R_i = pose_i[:3, :3]
    t_i = pose_i[:3, 3]
    xyz_world = xyz @ R_i.T + t_i            # (N, 3)

    # world → cam_j
    pose_j_inv = np.linalg.inv(pose_j)
    R_j = pose_j_inv[:3, :3]
    t_j = pose_j_inv[:3, 3]
    xyz_j = xyz_world @ R_j.T + t_j         # (N, 3)

    # project
    fx, fy, cx, cy = K_j
    z = xyz_j[:, 2]
    valid_z = z > 0.01
    u = np.where(valid_z, fx * xyz_j[:, 0] / np.maximum(z, 1e-8) + cx, -1)
    v = np.where(valid_z, fy * xyz_j[:, 1] / np.maximum(z, 1e-8) + cy, -1)

    in_bounds = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uv = np.stack([u, v], axis=-1)
    return uv, in_bounds


# ---------------------------------------------------------------------------
# RLE helpers
# ---------------------------------------------------------------------------

def decode_rle(rle: dict) -> np.ndarray:
    """COCO RLE → bool (H, W)"""
    rle_copy = dict(rle)
    if isinstance(rle_copy["counts"], str):
        rle_copy["counts"] = rle_copy["counts"].encode("utf-8")
    return mask_utils.decode(rle_copy).astype(bool)


# ---------------------------------------------------------------------------
# Worker function — one task = all pairs sharing the same img_i
#
# Savings vs per-pair worker:
#   • img_i depth + masks loaded & decoded exactly ONCE per ~55 pairs
#   • img_i backprojection computed once per batch
#   • masks pre-flattened (M, H*W) + linearized pixel indexing for speed
#   • output pkl stores only seg_corr_list (~few hundred bytes, not 40 KB)
# ---------------------------------------------------------------------------

def _load_and_decode_masks(pkl_path: Path):
    """
    Load pkl, decode all RLEs.
    Returns:
        masks_flat : (M, H*W) uint8   — pre-flattened for fast indexing
        areas      : (M,)    int64
        shape      : (H, W)            — original image shape
    """
    with open(pkl_path, "rb") as f:
        rles = pickle.load(f)["mask_coco_rles_resized"]
    if not rles:
        # Return a sentinel shape; caller must handle M=0
        return np.zeros((0, 1), np.uint8), np.zeros(0, np.int64), (1, 1)
    masks = np.stack([decode_rle(r).astype(np.uint8) for r in rles])  # (M, H, W)
    H, W = masks.shape[1], masks.shape[2]
    masks_flat = masks.reshape(len(rles), -1)          # (M, H*W)
    areas = masks_flat.sum(axis=1).astype(np.int64)
    return masks_flat, areas, (H, W)


def process_image_batch(args_tuple):
    """
    Process all pairs that share the same img_i.

    Optimisations vs a per-pair worker:
      • img_i depth + masks loaded & decoded ONCE per ~55 pairs
      • mi_at_valid = masks_i_flat[:, valid_depth_pixels] computed ONCE
        per batch → per-pair only needs a cheap in_bounds subset
      • img_j masks retrieved from a process-level LRU cache (hit rate is
        high because DUSt3R pairs are spatially local)
      • mj_at_proj = masks_j_flat[:, proj_pixels] computed ONCE per pair
        so the inner mask-i loop works on a small (M_j, V_proj) array

    args_tuple:
        scene_name   : str
        img_name_i   : str
        K_i          : (4,) float32
        pose_i       : (4,4) float32
        j_list       : list of (img_name_j, K_j, pose_j)
        processed_root, masks_root, output_root : str
        iou_thresh   : float
    """
    import time

    (scene_name, img_name_i, K_i, pose_i, j_list,
     processed_root, masks_root, output_root, iou_thresh) = args_tuple

    processed_root = Path(processed_root)
    masks_root     = Path(masks_root)
    output_root    = Path(output_root)
    out_dir        = output_root / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)

    errors    = []
    n_done    = 0
    n_skipped = 0

    try:
        # ── Load img_i once for the entire batch ──────────────────────────
        depth_i = load_depth(
            processed_root / scene_name / "depth" / f"{img_name_i}.png"
        )
        masks_i_flat, _, _ = _load_and_decode_masks(
            masks_root / scene_name / f"{img_name_i}.pkl"
        )
    except Exception:
        tb = traceback.format_exc()
        for img_name_j, *_ in j_list:
            errors.append((img_name_i, img_name_j, tb))
        return scene_name, img_name_i, 0, len(j_list), errors

    # ── Pre-compute depth-valid subset of img_i ONCE for the whole batch ──
    # valid_i marks pixels with finite depth.  mi_at_valid restricts
    # masks_i_flat to only those pixels — every pair then just needs a
    # cheap in_bounds slice of this smaller array (avoids M_i full-image
    # boolean gathers per pair).
    xyz_i, valid_i = backproject(depth_i, K_i)
    valid_i_idx    = np.where(valid_i)[0]                # (V_depth,) int indices
    xyz_i_valid    = xyz_i[valid_i]                      # (V_depth, 3)
    mi_at_valid    = masks_i_flat[:, valid_i_idx]        # (M_i, V_depth)  ONCE
    n_masks_i      = len(mi_at_valid)
    del masks_i_flat                                     # free full-image copy

    # Timing: print wall-clock breakdown for the first 2 non-skipped pairs
    _timing_budget = 2
    _timing_done   = 0

    # ── Process each (img_i, img_j) pair ─────────────────────────────────
    for img_name_j, K_j, pose_j in j_list:
        out_pkl = out_dir / f"{img_name_i}__{img_name_j}.pkl"
        if out_pkl.exists():
            n_skipped += 1
            continue

        _t0 = time.perf_counter() if _timing_done < _timing_budget else None
        try:
            depth_j = load_depth(
                processed_root / scene_name / "depth" / f"{img_name_j}.png"
            )
            _t1 = time.perf_counter() if _t0 else None

            # Use process-level LRU cache to avoid repeated RLE decodes
            _pkl_j = masks_root / scene_name / f"{img_name_j}.pkl"
            _cache_hit = _t0 is not None and str(_pkl_j) in _MASK_CACHE
            masks_j_flat, area_j, (H_j, W_j) = _load_masks_cached(_pkl_j)
            _t2 = time.perf_counter() if _t0 else None

            if n_masks_i == 0 or len(masks_j_flat) == 0:
                seg_corr = []
            else:
                seg_corr = _compute_corr(
                    mi_at_valid, xyz_i_valid,
                    masks_j_flat, area_j, W_j,
                    pose_i, pose_j, K_j, depth_j,
                    iou_thresh,
                )
            _t3 = time.perf_counter() if _t0 else None

            with open(out_pkl, "wb") as f:
                pickle.dump({"seg_corr_list": seg_corr},
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            n_done += 1

            if _t0 is not None:
                _t4 = time.perf_counter()
                print(
                    f"[TIMING] depth_j={(_t1-_t0)*1e3:.1f}ms  "
                    f"masks_j={(_t2-_t1)*1e3:.1f}ms({'hit' if _cache_hit else 'miss'})  "
                    f"corr={(_t3-_t2)*1e3:.1f}ms  "
                    f"write={(_t4-_t3)*1e3:.1f}ms  "
                    f"total={(_t4-_t0)*1e3:.1f}ms  "
                    f"M_i={n_masks_i}  M_j={len(masks_j_flat)}"
                )
                _timing_done += 1

        except Exception:
            errors.append((img_name_i, img_name_j, traceback.format_exc()))

    return scene_name, img_name_i, n_done, n_skipped, errors


def _compute_corr(
    mi_at_valid,    # (M_i, V_depth) uint8 — pre-computed valid-depth subset
    xyz_i_valid,    # (V_depth, 3)         — 3-D coords of valid-depth pixels
    masks_j_flat,   # (M_j, H_j*W_j) uint8 — pre-flattened
    area_j,         # (M_j,) int
    W_j,            # int — width of img_j (for linearised indexing)
    pose_i, pose_j,
    K_j,
    depth_j,
    iou_thresh: float,
):
    """
    For each mask in img_i, find the best-matching mask in img_j using
    depth backprojection + forward projection.

    Speedups vs naive per-pair loop:
      • Receives mi_at_valid (M_i, V_depth) pre-computed by the caller once
        per batch.  Only a cheap in_bounds slice is needed per pair.
      • mj_at_proj = masks_j_flat[:, proj_pixels] computed ONCE per pair;
        the inner mask-i loop indexes into this compact (M_j, V_proj) array
        rather than the full (M_j, H*W) one.
    """
    # Project valid img_i pixels into img_j
    uv_j, in_bounds = project(xyz_i_valid, pose_i, pose_j, K_j, *depth_j.shape)
    # in_bounds: (V_depth,) bool — which depth-valid pixels land in img_j

    if not in_bounds.any():
        return []

    # Cheap slice of the pre-computed mi_at_valid
    mi_at_proj = mi_at_valid[:, in_bounds]          # (M_i, V_proj)
    uv_proj    = uv_j[in_bounds].astype(np.int32)   # (V_proj, 2)  [u=col, v=row]
    pix_j_lin  = uv_proj[:, 1] * W_j + uv_proj[:, 0]  # (V_proj,) linearised

    # Pre-compute masks_j at ALL proj positions ONCE per pair.
    # Inner loop then reads from (M_j, V_proj) instead of the full image.
    mj_at_proj = masks_j_flat[:, pix_j_lin]         # (M_j, V_proj)

    corr_list = []
    for idx_i in range(len(mi_at_proj)):
        sel = np.where(mi_at_proj[idx_i])[0]        # (S,) contiguous row → fast
        if not sel.size:
            continue

        hits  = mj_at_proj[:, sel].sum(axis=1)      # (M_j,) from compact array
        union = sel.size + area_j - hits
        iou   = hits / (union + 1e-8)

        best_j = int(iou.argmax())
        if iou[best_j] >= iou_thresh:
            corr_list.append((idx_i, best_j))

    return corr_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata",  required=True,
                   help="Path to all_metadata.npz")
    p.add_argument("--processed", required=True,
                   help="Root of scannetpp_processed (for depth maps)")
    p.add_argument("--masks",     required=True,
                   help="Root of per-image mask pickles (masks_resize_mast3r)")
    p.add_argument("--output",    required=True,
                   help="Where to write pair pickles (can equal --masks)")
    p.add_argument("--iou_threshold", type=float, default=0.25,
                   help="Min IoU to accept a segment correspondence")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Limit total pairs (for debugging)")
    return p.parse_args()


def main():
    args = parse_args()

    from collections import Counter, defaultdict
    from concurrent.futures import wait, FIRST_COMPLETED
    import itertools
    import time

    # ── Load metadata ────────────────────────────────────────────────────────
    print("Loading all_metadata.npz ...")
    meta         = np.load(args.metadata, allow_pickle=True)
    scenes       = meta["scenes"]
    sceneids     = meta["sceneids"]
    images       = meta["images"]
    intrinsics   = meta["intrinsics"]
    trajectories = meta["trajectories"]
    pairs        = meta["pairs"][:, :2].astype(int)

    print(f"  Scenes : {len(scenes)}")
    print(f"  Images : {len(images)}")
    print(f"  Pairs  : {len(pairs):,}")

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    def get_K(intr):
        intr = np.array(intr)
        if intr.shape == (3, 3):
            return np.array([intr[0,0], intr[1,1], intr[0,2], intr[1,2]], dtype=np.float32)
        return intr.astype(np.float32)

    # ── Group pairs by (scene, img_i) ───────────────────────────────────────
    # Key insight: the same img_i appears in ~55 pairs on average.
    # Grouping lets the worker load & decode img_i exactly ONCE per group.
    print("Grouping pairs by img_i ...")
    # groups[(scene, img_i_name)] = {K_i, pose_i, j_list: [(img_j, K_j, pose_j)]}
    group_meta  = {}   # (scene, img_i) -> (K_i, pose_i)
    group_jlist = defaultdict(list)  # (scene, img_i) -> [(img_j, K_j, pose_j)]

    for idx_i, idx_j in pairs:
        scene  = scenes[sceneids[idx_i]]
        img_i  = images[idx_i]
        key    = (scene, img_i)
        if key not in group_meta:
            group_meta[key] = (
                get_K(intrinsics[idx_i]),
                trajectories[idx_i].copy(),
            )
        group_jlist[key].append((
            images[idx_j],
            get_K(intrinsics[idx_j]),
            trajectories[idx_j].copy(),
        ))

    n_groups = len(group_meta)
    pairs_per_scene = Counter(scene for scene, _ in group_meta)
    print(f"  img_i groups : {n_groups:,}  (avg {len(pairs)/n_groups:.1f} pairs each)")

    # ── Count already-done pairs ─────────────────────────────────────────────
    output_root = Path(args.output)
    print("Counting already-done pairs ...")
    scene_names_set = set(scene for scene, _ in group_meta)
    already_done = sum(
        len(list((output_root / sc).glob("*.pkl")))
        for sc in scene_names_set
        if (output_root / sc).exists()
    )
    print(f"  Already done : {already_done:,} / {len(pairs):,}")
    print(f"  Remaining    : {len(pairs) - already_done:,}")
    print(f"  Scenes       : {len(scene_names_set)}")
    print(f"  Workers      : {args.num_workers}")
    print()

    # ── Lazy task generator (group-level, not pair-level) ────────────────────
    def task_gen():
        for (scene, img_i), (K_i, pose_i) in group_meta.items():
            j_list = group_jlist[(scene, img_i)]
            yield (scene, img_i, K_i, pose_i, j_list,
                   args.processed, args.masks, args.output,
                   args.iou_threshold)

    # Bounded executor: num_workers * 4 groups in flight
    # (each group takes ~55x longer than a single pair, so smaller buffer is fine)
    BUFFER = max(args.num_workers * 4, 16)

    # ── Tracking ─────────────────────────────────────────────────────────────
    scene_pairs_done  = Counter()   # scene -> pairs completed
    scene_pairs_total = Counter(    # scene -> total pairs
        scene for scene, _ in group_meta
        for _ in group_jlist[(scene, _)]
    )
    # Rebuild correctly
    scene_pairs_total = Counter()
    for (scene, img_i), j_list in group_jlist.items():
        scene_pairs_total[scene] += len(j_list)

    scene_started = set()
    all_errors    = []
    pairs_done    = 0
    groups_done   = 0
    t_start       = time.time()
    LOG_INTERVAL  = 20   # log every N completed groups

    gen = task_gen()

    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        in_flight = set()
        for task in itertools.islice(gen, BUFFER):
            in_flight.add(ex.submit(process_image_batch, task))

        print(f"Executor seeded with {len(in_flight)} group-tasks, starting ...\n")

        with tqdm(total=len(pairs), unit="pair",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                  ) as pbar:

            while in_flight:
                done_futs, remaining_futs = wait(in_flight, return_when=FIRST_COMPLETED)
                in_flight = set(remaining_futs)

                for fut in done_futs:
                    scene, img_i, n_done, n_skip, errs = fut.result()
                    groups_done += 1
                    pairs_done  += n_done + n_skip
                    scene_pairs_done[scene] += n_done + n_skip

                    if errs:
                        for img_i_e, img_j_e, tb in errs:
                            all_errors.append((scene, img_i_e, img_j_e))
                            tqdm.write(
                                f"[ERROR] {scene}  {img_i_e}__{img_j_e}:\n"
                                f"        {tb.splitlines()[-1]}"
                            )

                    if scene not in scene_started:
                        scene_started.add(scene)
                        tqdm.write(
                            f"[START] scene {scene}  "
                            f"({scene_pairs_total[scene]:,} pairs)"
                        )

                    if scene_pairs_done[scene] >= scene_pairs_total[scene]:
                        elapsed = time.time() - t_start
                        scenes_left = sum(
                            1 for s in scene_names_set
                            if scene_pairs_done[s] < scene_pairs_total[s]
                        )
                        tqdm.write(
                            f"[DONE ] scene {scene}  "
                            f"{scene_pairs_total[scene]:,} pairs  |  "
                            f"scenes left: {scenes_left}  |  "
                            f"errors: {len(all_errors)}  |  "
                            f"elapsed: {elapsed/60:.1f} min"
                        )
                    elif groups_done % LOG_INTERVAL == 0:
                        elapsed = time.time() - t_start
                        rate    = pairs_done / elapsed if elapsed > 0 else 0
                        eta     = (len(pairs) - pairs_done) / rate / 60 if rate > 0 else float("inf")
                        tqdm.write(
                            f"[INFO ] {pairs_done:,}/{len(pairs):,} pairs  |  "
                            f"{rate:.0f} p/s  |  ETA {eta:.1f} min  |  "
                            f"errors: {len(all_errors)}"
                        )

                    pbar.update(n_done + n_skip)

                    task = next(gen, None)
                    if task is not None:
                        in_flight.add(ex.submit(process_image_batch, task))

    elapsed_total = time.time() - t_start
    print(f"\nDone in {elapsed_total/60:.1f} min.  Errors: {len(all_errors)}")
    if all_errors:
        print("First errors:", all_errors[:5])


if __name__ == "__main__":
    main()
