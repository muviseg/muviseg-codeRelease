# Reference metrics

Per-bin metrics for each row reported in the paper, exactly as produced by the
evaluation runs. `tests/test_reference_metrics.py` checks them, and
`docs/reproduction.md` explains how to compare a fresh run against them.

Each `metrics_by_bin.json` maps a pose bin to `AUPRC`, `R@1`, `R@5` and
`num_queries`. The overall figure for a run is the **query-count-weighted** mean
across bins, and bins must be ordered numerically — `135-180` sorts before
`45-90` as a string.

Two rows reported in the paper have no reference file here:

* **SegMASt3R + Sinkhorn** — needs `segmast3r_spp.ckpt`, which we do not
  redistribute, so no artifact was produced from this tree.
* **SegMASt3R + LG v2 on Replica** — the run directory was not preserved. Its
  VKITTI2 counterpart is included.

See `docs/reproduction.md` before comparing against the paper's tables.
