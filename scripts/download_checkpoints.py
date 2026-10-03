#!/usr/bin/env python3
"""Download the trained MuViSeg heads from the Hugging Face Hub.

Places each file where the shipped configs expect it, under `results/`.
Frozen backbones are NOT downloaded here -- see setup_third_party.sh.

Usage:
    python scripts/download_checkpoints.py                 # the three main heads
    python scripts/download_checkpoints.py --all           # + the ablation heads
    python scripts/download_checkpoints.py --only lgv2
    python scripts/download_checkpoints.py --repo-id other/repo --verify
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ID = "MuViSeg/muviseg"
REPO_ROOT = Path(__file__).resolve().parents[1]

# name -> (path in the HF repo, local path, md5, role)
CHECKPOINTS: dict[str, tuple[str, str, str, str]] = {
    "lgv2": (
        "segmast3r_lg_v2/best.pth",
        "results/segmast3r_lg_v2/best.pth",
        "fa0dcac9c83877669782c38ef9e93ebf",
        "main",
    ),
    "segvggt_dpt_v3": (
        "segvggt_dpt/v3-001/best.pth",
        "results/segvggt_dpt/v3-001/best.pth",
        "e0753fb7286e5ae389632e02c165ef4c",
        "main",
    ),
    "segvggt_dpt_joint": (
        "segvggt_dpt/joint-001/best.pth",
        "results/segvggt_dpt/joint-001/best.pth",
        "4f375a6cd97e82543c375e345f7c3264",
        "main",
    ),
    "segvggt_single_layer": (
        "segvggt/best.pth",
        "results/segvggt/best.pth",
        "31a917a2cda462a0644d7b02b77cf794",
        "ablation",
    ),
    # The reported single-layer row uses this file on Virtual KITTI 2 and
    # segvggt/best.pth on Replica. Both are published for that reason.
    "segvggt_single_layer_step140k": (
        "segvggt/step_0140000.pth",
        "results/segvggt/step_0140000.pth",
        "34e8779da0dfa3a415676ce97caffa8f",
        "ablation",
    ),
}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=REPO_ID, help=f"HF repo (default {REPO_ID})")
    ap.add_argument("--revision", default=None, help="tag, branch or commit to pin")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT,
                    help="root to place files under (default: repository root)")
    ap.add_argument("--all", action="store_true", help="also fetch the ablation heads")
    ap.add_argument("--only", action="append", choices=sorted(CHECKPOINTS),
                    help="fetch just these (repeatable)")
    ap.add_argument("--verify", action="store_true", help="check md5 after download")
    ap.add_argument("--list", action="store_true", help="list what would be fetched")
    args = ap.parse_args()

    if args.only:
        wanted = list(args.only)
    elif args.all:
        wanted = list(CHECKPOINTS)
    else:
        wanted = [k for k, v in CHECKPOINTS.items() if v[3] == "main"]

    if args.list:
        for name in wanted:
            remote, local, digest, role = CHECKPOINTS[name]
            print(f"{name:30s} {role:8s} {args.repo_id}/{remote} -> {local}")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface-hub is missing; it is a core dependency, so run `uv sync`",
              file=sys.stderr)
        return 1

    failures = []
    for name in wanted:
        remote, local, digest, _role = CHECKPOINTS[name]
        target = args.dest / local
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"==> {name}: {args.repo_id}/{remote}")
        try:
            cached = hf_hub_download(repo_id=args.repo_id, filename=remote,
                                     revision=args.revision)
        except Exception as exc:  # network, auth, missing file
            print(f"    failed: {exc}", file=sys.stderr)
            failures.append(name)
            continue
        # copy rather than symlink, so the tree stays usable if the HF cache is cleared
        target.write_bytes(Path(cached).read_bytes())
        if args.verify:
            got = md5(target)
            if got != digest:
                print(f"    md5 MISMATCH: expected {digest}, got {got}", file=sys.stderr)
                failures.append(name)
                continue
            print(f"    ok, md5 verified -> {local}")
        else:
            print(f"    ok -> {local}")

    if failures:
        print(f"\nfailed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll requested checkpoints are in place. Frozen backbones are separate:"
          "\n  bash setup_third_party.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
