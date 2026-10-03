"""Training is reproducible from a seed, and the legacy behaviour is preserved.

The original trainer seeded only the train/val split, so two identical runs
diverged. These tests pin the new contract rather than the old numbers, because
there were no stable old numbers to pin.
"""
import random

import pytest

torch = pytest.importorskip("torch")

from muviseg.config.default import cfg as base_cfg  # noqa: E402
from muviseg.training.trainer import (  # noqa: E402
    _run_provenance,
    _seed_everything,
    _worker_init_fn,
)


def test_config_exposes_seed_and_legacy_switch():
    assert base_cfg.TRAINING.SEED == 42
    assert base_cfg.TRAINING.LEGACY_RNG is False, "seeded by default"


def test_seed_everything_is_reproducible():
    def draw():
        _seed_everything(42)
        return (
            random.random(),
            torch.rand(3).tolist(),
            torch.nn.Linear(4, 4).weight.flatten().tolist(),
        )

    assert draw() == draw()


def test_different_seeds_differ():
    _seed_everything(42)
    a = torch.rand(8).tolist()
    _seed_everything(7)
    b = torch.rand(8).tolist()
    assert a != b


def test_seed_everything_returns_a_seeded_loader_generator():
    g1 = _seed_everything(42)
    g2 = _seed_everything(42)
    assert torch.equal(
        torch.randperm(16, generator=g1), torch.randperm(16, generator=g2)
    )


def test_worker_init_fn_gives_workers_distinct_but_reproducible_streams():
    def stream(worker_id):
        torch.manual_seed(1234)  # emulate the base seed the loader sets
        _worker_init_fn(worker_id)
        return [random.random() for _ in range(4)]

    a0, b0 = stream(0), stream(0)
    a1 = stream(1)
    assert a0 == b0, "the same worker must be reproducible"
    assert a0 != a1, "different workers must not draw identical samples"


def test_provenance_records_what_produced_a_checkpoint():
    c = base_cfg.clone()
    info = _run_provenance(c)
    assert info["seed"] == 42
    assert info["legacy_rng"] is False
    assert info["torch"] == torch.__version__
    assert "argv" in info
    assert "git_sha" in info  # None outside a checkout, but the key is always there
