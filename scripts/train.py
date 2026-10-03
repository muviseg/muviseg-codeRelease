"""
Usage (single GPU):
    python -m training.train --config training/config/segmast3r_train.yaml
    python -m training.train --mock

Usage (multi-GPU / mixed precision with Accelerate):
    accelerate launch -m training.train --config training/config/segmast3r_train_lg.yaml

Tensorboard:
    tensorboard --logdir results/segmast3r_repro/tb
"""

import sys
from pathlib import Path

# Fix FD exhaustion: torch multiprocessing default 'file_descriptor' strategy
# uses one FD per shared tensor. With batch_size=64, num_workers=4,
# persistent_workers=True, prefetch_factor=2 → easily 3000+ FDs.
# 'file_system' uses /dev/shm instead → no FD pressure.
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Bootstrap: ensure project root is on sys.path when script is run directly
# (accelerate launch adds the script's dir, not the project root)
_root = Path(__file__).resolve().parents[1]

# if str(_root) not in sys.path:
sys.path.insert(0, str(_root))

import argparse

from muviseg.training.utils import load_cfg
from muviseg.training.trainer import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--mock",   action="store_true")
    args, overrides = parser.parse_known_args()

    cfg = load_cfg(args.config, overrides=overrides)
    train(cfg, mock=args.mock, resume=args.resume)
