"""Metric arithmetic, checked against values worked out by hand."""
import numpy as np
import pytest

from muviseg.evaluation.eval_metrics import aggregate_metrics_by_bin, compute_metrics
from muviseg.evaluation.ground_truth_generator import generate_instance_correspondences


def test_ground_truth_matches_equal_instance_ids():
    gt = generate_instance_correspondences([3, 7, 9], [7, 3])
    assert gt.shape == (3, 2)
    np.testing.assert_array_equal(gt, np.array([[0, 1], [1, 0], [0, 0]], dtype=gt.dtype))


def test_perfect_scores():
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    gt = np.array([[1, 0], [0, 1]])
    m = compute_metrics(scores, gt)
    assert m["num_queries"] == 2
    assert m["AUPRC"] == pytest.approx(1.0)
    assert m["R@1"] == pytest.approx(1.0)
    assert m["R@5"] == pytest.approx(1.0)


def test_rows_without_a_positive_are_skipped():
    scores = np.array([[0.9, 0.1], [0.2, 0.8]])
    gt = np.array([[1, 0], [0, 0]])  # second query has no correspondence
    m = compute_metrics(scores, gt)
    assert m["num_queries"] == 1
    assert m["R@1"] == pytest.approx(1.0)


def test_r_at_1_counts_the_argmax_only():
    # the positive is ranked 2nd of 3 -> R@1 = 0, R@5 = 1, AP = 1/2
    scores = np.array([[0.1, 0.9, 0.5]])
    gt = np.array([[0, 0, 1]])
    m = compute_metrics(scores, gt)
    assert m["R@1"] == pytest.approx(0.0)
    assert m["R@5"] == pytest.approx(1.0)
    assert m["AUPRC"] == pytest.approx(0.5)


def test_bins_are_weighted_by_query_count_not_pair_count():
    # bin A: 1 pair, 1 query, AUPRC 1.0;  bin B: 1 pair, 9 queries, AUPRC 0.0
    # query-weighted overall must be 0.1, not the unweighted 0.5
    by_bin = {
        "0-45": [{"AUPRC": 1.0, "R@1": 1.0, "R@5": 1.0, "num_queries": 1}],
        "45-90": [{"AUPRC": 0.0, "R@1": 0.0, "R@5": 0.0, "num_queries": 9}],
    }
    agg = aggregate_metrics_by_bin(by_bin)
    assert agg["0-45"]["num_queries"] == 1
    assert agg["45-90"]["num_queries"] == 9
    total = sum(v["num_queries"] for v in agg.values())
    overall = sum(v["AUPRC"] * v["num_queries"] for v in agg.values()) / total
    assert overall == pytest.approx(0.1)
