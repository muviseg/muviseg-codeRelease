"""The unified evaluation runner keeps the per-dataset differences it must keep."""
import pytest

from muviseg.evaluation.runner import DATASETS, build_parser


def test_both_benchmarks_are_registered():
    assert sorted(DATASETS) == ["replica", "vkitti2"]


@pytest.mark.parametrize("name", sorted(DATASETS))
def test_spec_builds(name):
    spec = DATASETS[name]()
    assert spec.name == name
    assert callable(spec.build) and callable(spec.load_pairs)
    assert callable(spec.collate) and callable(spec.collate_tuple)


def test_table_filenames_stay_distinct():
    """Existing tooling reads these exact names; do not 'tidy' them."""
    assert DATASETS["replica"]().table_filename == "table2_results.txt"
    assert DATASETS["vkitti2"]().table_filename == "table_results.txt"


def test_roma_default_variant_is_per_dataset():
    assert DATASETS["replica"]().roma_default_variant == "indoor"
    assert DATASETS["vkitti2"]().roma_default_variant == "outdoor"


def test_only_replica_takes_a_separate_instance_mask_root():
    assert DATASETS["replica"]().needs_instance_mask_root is True
    assert DATASETS["vkitti2"]().needs_instance_mask_root is False


def test_cli_requires_a_dataset_and_a_config():
    p = build_parser()
    args = p.parse_args(["--dataset", "replica", "--config", "x.yaml"])
    assert args.dataset == "replica" and args.num_vis == 30
    with pytest.raises(SystemExit):
        p.parse_args(["--config", "x.yaml"])
    with pytest.raises(SystemExit):
        p.parse_args(["--dataset", "nope", "--config", "x.yaml"])
