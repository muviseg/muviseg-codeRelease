"""Build multi-view tracks from accepted pairwise matches.

Union-find over accepted matches, with one constraint: a track may never contain
two segments from the same frame. Edges are consumed in descending score order,
so a higher-scoring link wins when the constraint would otherwise be violated.
"""
from __future__ import annotations


class UnionFind:
    """Disjoint sets with path compression, keyed by any hashable node."""

    def __init__(self) -> None:
        self.parent: dict = {}

    def add(self, n) -> None:
        self.parent.setdefault(n, n)

    def find(self, n):
        r = n
        while self.parent[r] != r:
            r = self.parent[r]
        while self.parent[n] != r:
            self.parent[n], n = r, self.parent[n]
        return r

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_tracks(n_frames: int, masks_per_frame: list, best: dict):
    """Group segments across frames into tracks.

    Args:
        n_frames: number of frames (kept for signature stability; the per-frame
            segment counts come from `masks_per_frame`).
        masks_per_frame: list of (M_f, H, W) mask tensors, one entry per frame.
        best: {(frame_a, frame_b): {i: (j, score)}} accepted matches.

    Returns:
        gid_of: {(frame, local_segment_index): track_id}
        lens:   {track_id: number of segments in it}
        n_tracks
    """
    # Keep only the best incoming edge per target segment, then sort by score.
    edges = []
    for (fa, fb), inner in best.items():
        jb: dict[int, tuple[int, float]] = {}
        for i, (j, s) in inner.items():
            if jb.get(j) is None or s > jb[j][1]:
                jb[j] = (i, s)
        for j, (i, s) in jb.items():
            edges.append((s, fa, i, fb, j))
    edges.sort(key=lambda e: -e[0])

    uf = UnionFind()
    comp: dict = {}
    for f, m in enumerate(masks_per_frame):
        for i in range(m.shape[0]):
            uf.add((f, i))
            comp[(f, i)] = {f}

    for _s, fa, i, fb, j in edges:
        a, b = (fa, i), (fb, j)
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        if comp[ra] & comp[rb]:
            # merging would put two segments of one frame in the same track
            continue
        uf.union(a, b)
        comp[uf.find(a)] = comp[ra] | comp[rb]

    root2gid: dict = {}
    gid_of: dict = {}
    for f, m in enumerate(masks_per_frame):
        for i in range(m.shape[0]):
            r = uf.find((f, i))
            if r not in root2gid:
                root2gid[r] = len(root2gid)
            gid_of[(f, i)] = root2gid[r]

    lens: dict[int, int] = {}
    for g in gid_of.values():
        lens[g] = lens.get(g, 0) + 1
    return gid_of, lens, len(root2gid)
