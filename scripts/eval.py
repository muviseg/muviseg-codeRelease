#!/usr/bin/env python3
"""Evaluate a MuViSeg model on Replica or Virtual KITTI 2.

    python scripts/eval.py --dataset replica --config configs/eval/replica_segvggt.yaml
    python scripts/eval.py --dataset vkitti2 --config configs/eval/vkitti2_lgv2.yaml

Flags: --num_pairs N (smoke test), --visualize, --num_vis, --device, --output_dir.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from muviseg.evaluation.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
