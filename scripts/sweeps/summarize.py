#!/usr/bin/env python3
"""Summarise a sweep of evaluation runs into one table.

Replaces three copies of a shell helper that each re-aggregated
`metrics_by_bin.json` themselves, and did it wrongly in two ways:

* bins were ordered with `sorted(keys)`, which is lexicographic, so the Replica
  bins came out as 0-45, 135-180, 45-90, 90-135 and the columns were mislabelled;
* the per-bin weight was read from a key named `num_pairs`, which does not exist
  in the file -- the key is `num_queries` -- so every weight was 0 and the
  "overall" column became the plain sum of the four bin values.

Usage:
    python scripts/sweeps/summarize.py RUN_DIR [RUN_DIR ...]
    python scripts/sweeps/summarize.py --glob 'results/segvggt_dpt_sweep/*'
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
from pathlib import Path

METRICS = ("AUPRC", "R@1", "R@5")


def bin_order(keys) -> list[str]:
    """Numeric order by the bin's lower edge. '135-180' must not sort second."""
    return sorted(keys, key=lambda b: int(str(b).split("-")[0]))


def overall(by_bin: dict) -> tuple[dict[str, float], int]:
    """Query-count-weighted mean across bins, as the evaluator defines it."""
    total = sum(m["num_queries"] for m in by_bin.values())
    if total == 0:
        return {k: float("nan") for k in METRICS}, 0
    return (
        {k: sum(m[k] * m["num_queries"] for m in by_bin.values()) / total for k in METRICS},
        total,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="*", type=Path)
    ap.add_argument("--glob", dest="pattern", help="shell glob of run directories")
    ap.add_argument("--percent", action="store_true", help="print as percentages")
    args = ap.parse_args()

    dirs = list(args.run_dirs)
    if args.pattern:
        dirs += [Path(p) for p in sorted(globmod.glob(args.pattern))]
    if not dirs:
        ap.error("give at least one run directory, or --glob")

    rows, bin_names = [], []
    for d in dirs:
        f = d / "metrics_by_bin.json"
        if not f.is_file():
            rows.append((d.name, None, None, None))
            continue
        by_bin = json.load(open(f))
        names = bin_order(by_bin)
        if not bin_names:
            bin_names = names
        ov, total = overall(by_bin)
        rows.append((d.name, names, by_bin, (ov, total)))

    scale = 100.0 if args.percent else 1.0
    fmt = "%8.2f" if args.percent else "%8.4f"
    width = max((len(r[0]) for r in rows), default=4) + 2

    header = f"{'run':<{width}}" + "".join(f"{b:>10}" for b in bin_names)
    header += f" |{'AUPRC':>9}{'R@1':>9}{'R@5':>9}{'queries':>10}"
    print(header)
    print("-" * len(header))
    for name, names, by_bin, agg in rows:
        if names is None:
            print(f"{name:<{width}}FAILED (no metrics_by_bin.json)")
            continue
        line = f"{name:<{width}}"
        for b in bin_names:
            line += (fmt % (by_bin[b]["AUPRC"] * scale)).rjust(10) if b in by_bin else " " * 10
        ov, total = agg
        line += " |" + "".join((fmt % (ov[k] * scale)).rjust(9) for k in METRICS)
        line += f"{total:>10}"
        print(line)
    print("-" * len(header))
    print("Bins are AUPRC, ordered by lower edge. Overall columns are weighted by "
          "query count, not a mean of bins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
