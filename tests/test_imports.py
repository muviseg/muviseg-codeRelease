"""Every module in the package must import without side effects."""
import importlib
import pkgutil

import pytest

import muviseg

# Modules that pull in a third-party checkout at import time are covered by
# test_models.py, which skips when that checkout is absent.
NEEDS_THIRD_PARTY = {
    "muviseg.models.sinkhorn",
    "muviseg.models.lightglue",
    "muviseg.models.lightglue_v2",
    "muviseg.models.vggt_lightglue",
    "muviseg.models.vggt_dpt_lg",
    "muviseg.evaluation.roma_infer",
}

ALL = [m.name for m in pkgutil.walk_packages(muviseg.__path__, "muviseg.")]


def test_package_discovers_modules():
    assert len(ALL) > 10, ALL


@pytest.mark.parametrize("name", [m for m in ALL if m not in NEEDS_THIRD_PARTY])
def test_import(name):
    importlib.import_module(name)


def test_paths_resolve_from_file_not_cwd(tmp_path, monkeypatch):
    from muviseg import paths

    monkeypatch.chdir(tmp_path)
    importlib.reload(paths)
    assert paths.REPO_ROOT.is_dir()
    assert (paths.REPO_ROOT / "pyproject.toml").is_file()
    assert paths.SEGMAST3R_ROOT == paths.REPO_ROOT / "third_party" / "segmast3r"
    assert paths.VGGT_ROOT == paths.REPO_ROOT / "third_party" / "vggt"
