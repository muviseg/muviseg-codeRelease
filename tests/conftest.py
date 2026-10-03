import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cfg_dir(kind: str) -> Path:
    return REPO_ROOT / "configs" / kind


def all_configs(kind: str):
    return sorted(_cfg_dir(kind).glob("*.yaml"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def pairs_dir(repo_root: Path) -> Path:
    return repo_root / "assets" / "pairs"


def requires_dataset(root) -> None:
    """Skip a test when the configured dataset root is not present."""
    if root is None or not Path(root).exists():
        pytest.skip(f"dataset root not available: {root}")


def requires_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def requires_third_party(name: str) -> None:
    if not (REPO_ROOT / "third_party" / name).exists():
        pytest.skip(f"third_party/{name} not set up; run setup_third_party.sh")
