"""Every module in the package must import without side effects."""
import importlib
import pkgutil

import pytest

import muviseg

# Modules that import a third-party source checkout at import time. They are
# exercised by test_model_components.py, which skips when the checkout is absent.
NEEDS_THIRD_PARTY = {
    "muviseg.models.sinkhorn",
    "muviseg.models.lightglue",
    "muviseg.models.lightglue_v2",
    "muviseg.models.vggt_lightglue",
    "muviseg.models.vggt_dpt_lg",
}

# Modules whose dependency lives in an optional extra. Importing them without
# that extra must fail cleanly, so the test skips rather than pretending.
OPTIONAL_EXTRA = {
    "muviseg.evaluation.segmentor": ("ultralytics", "eval"),
    "muviseg.evaluation.roma_infer": ("romatch", "roma"),
}

ALL = [m.name for m in pkgutil.walk_packages(muviseg.__path__, "muviseg.")]


def test_package_discovers_modules():
    assert len(ALL) > 10, ALL


@pytest.mark.parametrize("name", [m for m in ALL if m not in NEEDS_THIRD_PARTY])
def test_import(name):
    if name in OPTIONAL_EXTRA:
        dist, extra = OPTIONAL_EXTRA[name]
        pytest.importorskip(dist, reason=f"{dist} comes from `uv sync --extra {extra}`")
    importlib.import_module(name)


def test_every_optional_module_is_declared():
    """Guards against a new module quietly importing an optional dependency."""
    for name in OPTIONAL_EXTRA:
        assert name in ALL, f"{name} no longer exists; update OPTIONAL_EXTRA"


def test_paths_resolve_from_file_not_cwd(tmp_path, monkeypatch):
    from muviseg import paths

    monkeypatch.chdir(tmp_path)
    importlib.reload(paths)
    assert paths.REPO_ROOT.is_dir()
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert paths.SEGMAST3R_ROOT == paths.REPO_ROOT / "third_party" / "segmast3r"
    assert paths.VGGT_ROOT == paths.REPO_ROOT / "third_party" / "vggt"
