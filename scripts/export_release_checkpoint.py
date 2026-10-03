"""
Export a slim, release-ready checkpoint.

Training checkpoints store the full model state (including the frozen
MASt3R/VGGT backbone, ~3 GB) plus optimizer/scheduler state. For release only
the trained head weights and a little metadata are needed: the backbone is
re-loaded from third_party/ at inference time, and every checkpoint consumer
in this repo (muviseg/evaluation/model_infer.py, the downstream matcher adapters)
loads with strict=False.

Note: slim checkpoints cannot be used with `scripts/train.py --resume` (no optimizer
state).

Usage:
    python scripts/export_release_checkpoint.py \
        --input  results/segvggt_dpt/v3-001/best.pth \
        --output release_checkpoints/segvggt_dpt/v3-001/best.pth
"""

import argparse
from pathlib import Path

import torch

# Parameters under these prefixes are frozen backbones, re-loaded from
# third_party/ at inference time (MASt3R -> "backbone.", VGGT -> "aggregator.").
FROZEN_PREFIXES = ("backbone.", "aggregator.")
KEEP_META = ("epoch", "global_step", "best_val_ma", "metrics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Full training checkpoint (.pth)")
    parser.add_argument("--output", required=True, help="Slim checkpoint to write")
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    head = {k: v for k, v in state.items() if not k.startswith(FROZEN_PREFIXES)}
    if not head:
        raise SystemExit("Nothing left after stripping frozen prefixes — wrong checkpoint?")
    if len(head) == len(state):
        print("  note: no frozen-backbone tensors found; checkpoint is already slim")

    slim = {k: ckpt[k] for k in KEEP_META if k in ckpt}
    slim["model_state"] = head

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out)

    kept_mb = sum(v.numel() * v.element_size() for v in head.values()) / 1e6
    print(f"{args.input} -> {out}")
    print(f"  kept {len(head)}/{len(state)} tensors ({kept_mb:.1f} MB of head weights)")


if __name__ == "__main__":
    main()
