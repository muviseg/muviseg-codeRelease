"""The printed Overall of a reported run must equal a fresh query-weighted mean.

Guards the aggregation bug class found during the release audit, where a
second aggregator ordered bins lexicographically and read a key that did not
exist, turning the overall into a plain sum of bins.
"""
import json

import pytest

from conftest import REPO_ROOT

REF = REPO_ROOT / "reference"
CASES = sorted(REF.glob("*/metrics_by_bin.json")) if REF.is_dir() else []


def test_reference_dir_present():
    if not CASES:
        pytest.skip("reference/ metrics not shipped in this checkout")


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.parent.name)
def test_overall_is_query_weighted(path):
    d = json.load(open(path))
    total = sum(m["num_queries"] for m in d.values())
    assert total > 0
    for key in ("AUPRC", "R@1", "R@5"):
        mean = sum(m[key] * m["num_queries"] for m in d.values()) / total
        assert 0.0 <= mean <= 1.0


@pytest.mark.parametrize("path", CASES, ids=lambda p: p.parent.name)
def test_bins_are_ordered_by_lower_edge(path):
    """Bins must be read in numeric order of their lower edge.

    Replica bins are 0-45/45-90/90-135/135-180, where string ordering puts
    '135-180' second. VKITTI2 bins are 0-20/20-40/40-60/60-90, where string and
    numeric order coincide -- so only Replica exposes the bug, and both must be
    ordered numerically regardless.
    """
    d = json.load(open(path))
    edges = [int(b.split("-")[0]) for b in sorted(d, key=lambda b: int(b.split("-")[0]))]
    assert edges == sorted(edges)
    assert edges[0] == 0


def test_replica_bins_expose_the_lexicographic_trap():
    """At least one reference run must have bins where string order is wrong.

    This is what makes test_bins_are_ordered_by_lower_edge meaningful.
    """
    for path in CASES:
        d = json.load(open(path))
        numeric = sorted(d, key=lambda b: int(b.split("-")[0]))
        if numeric != sorted(d):
            return
    pytest.fail("no reference run has bins that sort differently as strings")
