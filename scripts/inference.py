#!/usr/bin/env python3
"""Run MuViSeg over a sequence of frames and write segments, matches and tracks.

Segment proposals come from FastSAM (needs `uv sync --extra eval`), unless you
pass --masks with pre-computed ones.

    python scripts/inference.py --frames path/to/frames --out out/run1
    python scripts/inference.py --frames frames --n 6 --stride 2 --batch 4
    python scripts/inference.py --frames frames --max-frames 60 --save-overlays

Outputs, under --out:
    summary.json     run settings, timings, track statistics
    tracks.json      {frame: {segment_index: track_id}}
    masks.pt         the segment proposals, so a re-render needs no re-segmenting
    overlays/        per-frame overlays, with --save-overlays
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from muviseg.config.paths import resolve_config_paths, resolve_path  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Images reach the model in [-1, 1]; it rescales internally per backbone.
TO_TENSOR = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])


def find_frames(frames_dir: Path, max_frames: int | None, every: int) -> list[Path]:
    paths = sorted(p for p in frames_dir.iterdir()
                   if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no images in {frames_dir}")
    paths = paths[::every]
    if max_frames is not None:
        paths = paths[:max_frames]
    return paths


def load_images(paths: list[Path], h: int, w: int) -> list[torch.Tensor]:
    return [TO_TENSOR(Image.open(p).convert("RGB").resize((w, h), Image.BILINEAR))
            for p in paths]


def draw_overlay(path: Path, masks: torch.Tensor, gids: list[int],
                 lens: dict, min_len: int, out_path: Path, alpha: float = 0.34):
    """Colour each tracked segment by a hue derived from its track id."""
    import colorsys

    img = np.array(Image.open(path).convert("RGB").resize(
        (masks.shape[-1], masks.shape[-2]), Image.BILINEAR)).astype(np.float32)
    for i in range(masks.shape[0]):
        g = gids[i]
        if lens.get(g, 0) < min_len:
            continue
        r, gg, b = colorsys.hsv_to_rgb((g * 0.618034) % 1.0, 0.75, 1.0)
        colour = np.array([r * 255, gg * 255, b * 255], dtype=np.float32)
        m = masks[i].numpy().astype(bool)
        img[m] = (1 - alpha) * img[m] + alpha * colour
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True, help="directory of frames")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--config", default="configs/eval/replica_segvggt_joint.yaml",
                    help="config supplying MODEL.* (checkpoint, backbone, head)")
    ap.add_argument("--n", type=int, default=4, help="joint window size N (default 4)")
    ap.add_argument("--stride", type=int, default=2, help="window stride (default 2)")
    ap.add_argument("--batch", type=int, default=4, help="windows per batch (default 4)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--masks", type=Path, default=None,
                    help="reuse masks.pt from an earlier run instead of segmenting")
    ap.add_argument("--fastsam", default="FastSAM-x.pt",
                    help="FastSAM weights (repo-root relative by default)")
    ap.add_argument("--m-prime", type=int, default=50,
                    help="max segment proposals per frame (default 50)")
    ap.add_argument("--save-overlays", action="store_true")
    ap.add_argument("--min-track-len", type=int, default=3,
                    help="only colour tracks at least this long (default 3)")
    args = ap.parse_args()

    cfg = resolve_config_paths(yaml.safe_load(open(args.config)))
    h = int(cfg["DATASET"]["RESIZE_H"])
    w = int(cfg["DATASET"]["RESIZE_W"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    paths = find_frames(args.frames, args.max_frames, args.every)
    print(f"{len(paths)} frames from {args.frames}, model input {w}x{h}")
    if len(paths) < args.n:
        raise SystemExit(f"need at least --n {args.n} frames, found {len(paths)}")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.masks is not None:
        masks_per_frame = torch.load(args.masks, map_location="cpu")
        print(f"reused {len(masks_per_frame)} mask sets from {args.masks}")
        if len(masks_per_frame) != len(paths):
            raise SystemExit(
                f"{args.masks} has {len(masks_per_frame)} mask sets but "
                f"{len(paths)} frames were selected")
        seg_timing = {"reused_from": str(args.masks)}
    else:
        from muviseg.inference.segmentation import segment_frames
        fastsam = resolve_path(str(args.fastsam))
        if not Path(fastsam).is_file():
            raise SystemExit(
                f"FastSAM weights not found: {fastsam}\n"
                f"Download FastSAM-x.pt (see docs/checkpoints.md), or pass --masks.")
        masks_per_frame, seg_timing = segment_frames(
            paths, fastsam, target_h=h, target_w=w, m_prime=args.m_prime,
            device=str(device))
        torch.save(masks_per_frame, args.out / "masks.pt")
        counts = [int(m.shape[0]) for m in masks_per_frame]
        print(f"segments per frame: min {min(counts)} / mean "
              f"{sum(counts)/len(counts):.1f} / max {max(counts)}")

    from muviseg.evaluation.model_infer import SegVGGTDPTJointInfer
    from muviseg.inference.sliding_window import run_sequence

    wrapper = SegVGGTDPTJointInfer(cfg)
    wrapper.prepare(device)
    model = wrapper.inner

    images = load_images(paths, h, w)
    gid_of, lens, stats = run_sequence(
        model, device, images, masks_per_frame,
        n=args.n, stride=args.stride, batch=args.batch)

    tracks = {}
    for (f, i), g in sorted(gid_of.items()):
        tracks.setdefault(str(f), {})[str(i)] = int(g)
    (args.out / "tracks.json").write_text(json.dumps(tracks, indent=2))

    summary = {
        "frames": [str(p) for p in paths],
        "n_frames": len(paths),
        "model_input": {"h": h, "w": w},
        "config": args.config,
        "checkpoint": cfg["MODEL"].get("CHECKPOINT"),
        "segmentation": seg_timing,
        "window": stats,
        "track_lengths": {str(k): v for k, v in sorted(lens.items())},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\ntracks: {stats['n_tracks']}  "
          f"(length >= 3: {stats['n_tracks_len_ge_3']}, "
          f"longest {stats['longest_track']})")
    print(f"latency p50 {stats['latency_ms_per_batch']['p50']} ms/batch "
          f"(backbone {stats['backbone_ms_per_batch']} ms, "
          f"head {stats['matching_head_ms_per_batch']} ms)")

    if args.save_overlays:
        print("writing overlays...")
        for f, p in enumerate(paths):
            gids = [gid_of[(f, i)] for i in range(masks_per_frame[f].shape[0])]
            draw_overlay(p, masks_per_frame[f], gids, lens, args.min_track_len,
                         args.out / "overlays" / f"{f:06d}.png")

    print(f"\nwrote {args.out}/summary.json, tracks.json"
          + (", overlays/" if args.save_overlays else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
