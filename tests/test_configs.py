"""Every shipped config must parse, and training configs must satisfy the schema."""
import yaml
import pytest

from conftest import all_configs


@pytest.mark.parametrize("path", all_configs("eval"), ids=lambda p: p.stem)
def test_eval_config_parses(path):
    cfg = yaml.safe_load(path.read_text())
    assert "DATASET" in cfg and "EVAL" in cfg and "MODEL" in cfg, sorted(cfg)
    assert cfg["EVAL"]["PAIRS_FILE"]
    assert cfg["DATASET"]["DATA_ROOT"]


@pytest.mark.parametrize("path", all_configs("train"), ids=lambda p: p.stem)
def test_train_config_merges_into_schema(path):
    from muviseg.config.default import cfg as base

    c = base.clone()
    c.defrost()
    # merge_from_file is strict: an unknown key raises, which is the check.
    c.merge_from_file(str(path))
    c.freeze()
    assert c.MODEL.ARCH
    assert c.TRAINING.BATCH_SIZE > 0
