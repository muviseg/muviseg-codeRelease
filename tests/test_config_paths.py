"""Config paths resolve against the repository root, never the working directory."""
import os

import pytest
import yaml

from conftest import all_configs
from muviseg.config.paths import UnsetVariable, resolve_config_paths, resolve_path
from muviseg.paths import REPO_ROOT


def test_absolute_is_untouched():
    assert resolve_path("/abs/path/x.pth") == "/abs/path/x.pth"


def test_relative_is_repo_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_path("results/x.pth") == str(REPO_ROOT / "results" / "x.pth")


def test_variable_is_expanded(monkeypatch):
    monkeypatch.setenv("MUVISEG_TEST_ROOT", "/data/sets")
    assert resolve_path("${MUVISEG_TEST_ROOT}/Replica") == "/data/sets/Replica"


def test_unset_variable_fails_loudly(monkeypatch):
    monkeypatch.delenv("MUVISEG_DEFINITELY_UNSET", raising=False)
    with pytest.raises(UnsetVariable, match="MUVISEG_DEFINITELY_UNSET"):
        resolve_path("${MUVISEG_DEFINITELY_UNSET}/x")


def test_non_path_values_are_left_alone():
    cfg = {"MODEL": {"CHECKPOINT": "results/a.pth", "N_LAYERS": 3, "ARCH": "segvggt_dpt"}}
    resolve_config_paths(cfg)
    assert cfg["MODEL"]["N_LAYERS"] == 3
    assert cfg["MODEL"]["ARCH"] == "segvggt_dpt"
    assert cfg["MODEL"]["CHECKPOINT"].startswith(str(REPO_ROOT))


@pytest.mark.parametrize("path", all_configs("eval"), ids=lambda p: p.stem)
def test_shipped_eval_configs_have_no_machine_paths(path):
    """A released config must not hardcode someone's home or mount point."""
    raw = path.read_text()
    for bad in ("/mnt/", "/home/", "/scratch/", "nabo-", "vol2_raid"):
        assert bad not in raw, f"{path.name} still contains {bad!r}"


@pytest.mark.parametrize("path", all_configs("train"), ids=lambda p: p.stem)
def test_shipped_train_configs_have_no_machine_paths(path):
    raw = path.read_text()
    for bad in ("/mnt/", "/home/", "/scratch/", "nabo-", "vol2_raid"):
        assert bad not in raw, f"{path.name} still contains {bad!r}"


@pytest.mark.parametrize("path", all_configs("eval"), ids=lambda p: p.stem)
def test_eval_configs_resolve_with_the_documented_variable(path, monkeypatch):
    monkeypatch.setenv("MUVISEG_DATA_ROOT", "/datasets")
    cfg = resolve_config_paths(yaml.safe_load(path.read_text()))
    for key in ("DATA_ROOT", "INSTANCE_MASK_ROOT"):
        if key in cfg["DATASET"]:
            assert os.path.isabs(cfg["DATASET"][key])
            assert "${" not in cfg["DATASET"][key]
    assert os.path.isabs(cfg["EVAL"]["PAIRS_FILE"])
    for key in ("CHECKPOINT", "MAST3R_CKPT", "VGGT_CKPT"):
        if key in cfg["MODEL"]:
            assert os.path.isabs(cfg["MODEL"][key])


def test_pairs_files_are_shipped_and_resolve(monkeypatch):
    monkeypatch.setenv("MUVISEG_DATA_ROOT", "/datasets")
    seen = set()
    for path in all_configs("eval"):
        cfg = resolve_config_paths(yaml.safe_load(path.read_text()))
        seen.add(cfg["EVAL"]["PAIRS_FILE"])
    assert seen, "no eval configs found"
    for p in seen:
        assert os.path.isfile(p), f"pair list missing from the repo: {p}"
