"""Sliding-window geometry and track construction."""
import pytest

torch = pytest.importorskip("torch")

from muviseg.inference.sliding_window import window_starts
from muviseg.inference.tracks import UnionFind, build_tracks


def test_window_starts_covers_the_tail():
    starts = window_starts(10, 4, 3)
    assert starts[0] == 0
    assert starts[-1] + 4 == 10, "the last window must reach the final frame"


def test_window_counts_match_the_reported_demo():
    """540 frames, stride 2: 269 windows at N=4 and 268 at N=6."""
    assert len(window_starts(540, 4, 2)) == 269
    assert len(window_starts(540, 6, 2)) == 268


def test_window_needs_enough_frames():
    assert window_starts(4, 4, 2) == [0]


def test_union_find_path_compression():
    uf = UnionFind()
    for n in "abcd":
        uf.add(n)
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("d") != uf.find("a")


def _masks(n_frames, per_frame=2):
    return [torch.zeros(per_frame, 4, 4) for _ in range(n_frames)]


def test_chained_matches_form_one_track():
    best = {(0, 1): {0: (0, 0.9)}, (1, 2): {0: (1, 0.8)}}
    gid, lens, n = build_tracks(3, _masks(3), best)
    assert gid[(0, 0)] == gid[(1, 0)] == gid[(2, 1)]
    assert max(lens.values()) == 3


def test_a_track_never_holds_two_segments_of_one_frame():
    best = {(0, 0): {0: (1, 0.99)}}
    gid, _lens, _n = build_tracks(3, _masks(3), best)
    assert gid[(0, 0)] != gid[(0, 1)]


def test_higher_scoring_edge_wins_under_the_constraint():
    # frame1 seg0 and seg1 both want frame0 seg0; only the better one may link
    best = {(0, 1): {0: (0, 0.4)}, (1, 0): {1: (0, 0.9)}}
    gid, lens, _n = build_tracks(2, _masks(2), best)
    assert gid[(0, 0)] == gid[(1, 1)], "the 0.9 edge should have won"
    assert gid[(0, 0)] != gid[(1, 0)]


def test_unmatched_segments_each_get_their_own_track():
    gid, lens, n = build_tracks(3, _masks(3), {})
    assert n == 6
    assert set(lens.values()) == {1}
