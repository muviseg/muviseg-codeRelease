# Reproducing the reported results

This page states what reproduces exactly, what reproduces approximately, and
what cannot be reproduced from this release. Read it before comparing numbers.

## The benchmark is a fixed pair list

Both benchmarks are frozen pair lists shipped in `assets/pairs/`, so every row is
evaluated on identical inputs:

| | pairs | scenes | cells | per cell | bins (relative rotation) |
|---|---|---|---|---|---|
| Replica | 3200 | 8 | 32 | 100 | 0-45 / 45-90 / 90-135 / 135-180 |
| Virtual KITTI 2 | 2400 | 5 × 4 variants | 48 of 80 populated | 50 | 0-20 / 20-40 / 40-60 / 60-90 |

Two things to note about the VKITTI2 list, both verifiable from the file:

* It contains **2400** pairs. The filename used during development said 4000 and
  the submitted text said 3200; 2400 is the number that matches the reported
  10308 query segments.
* Only 48 of the 80 (scene, variant, bin) cells are populated. `Scene02` and
  `Scene18` contribute a single bin each, so the stratification is not uniform
  across scenes, and the largest observed rotation is 79.3°.

`muviseg/sampling/` regenerates the lists, but regeneration is sensitive to
the number of trajectory lines per scene: the per-cell choice is a seeded
`RandomState(42).choice` over the enumerated candidates, so a different sequence
length silently yields a different pair set. **Use the shipped lists** to compare
against the reported numbers.

## How the metrics are defined

`muviseg/evaluation/eval_metrics.py` is the only metric implementation.

Per query segment (a row of the M×N score matrix):

* rows with no ground-truth correspondence are **skipped**, so `num_queries`
  counts only query segments that have a match to find;
* `AUPRC` is `average_precision_score` over that row's raw scores;
* `R@1` is whether the argmax is correct; `R@5` whether a correct match is in the
  top `min(5, N)`.

A pair's value is the unweighted mean over its surviving rows. A bin's value is
the **query-count-weighted** mean over its pairs, and the overall value is the
query-count-weighted mean over bins.

Two consequences worth stating, because both have produced wrong numbers:

1. **The overall is not the mean of the four bin values.** Weight by
   `num_queries`.
2. **Bins must be ordered by their numeric lower edge.** As strings, `135-180`
   sorts before `45-90`. `tests/test_reference_metrics.py` guards both.

Reference per-bin metrics for the reported runs are in `reference/`.

## What the model is given as input

On both benchmarks the segment proposals are the **datasets' own instance
annotations**, not the output of a segmenter: Replica's `semantic_instance_*.png`
and VKITTI2's `instancegt_*.png` decoded as `trackID = pixel - 1`. Masks are
resized with nearest-neighbour and truncated to the first `M' = 50` instance IDs
in ascending-ID order.

FastSAM appears only in the separate FastSAM-mask ablations and in the demo. So
on Replica the instance annotations are simultaneously the model's input and the
ground truth; a colour-coded overlay keyed on instance id is not a model output.

## Determinism

Evaluation is deterministic given a fixed pair list: the loader does not shuffle,
context-frame selection walks outward from the pair midpoint with no randomness,
and mask truncation is by ID order. Re-running a config reproduces its
`metrics_by_bin.json` byte for byte.

Two exceptions:

* **RoMa** samples 5000 correspondences per pair with no seed, so the two RoMa
  rows are not bitwise reproducible. Expect variation in the last digit.
* Changing the PyTorch or CUDA version changes GEMM kernels. The reported numbers
  come from torch 2.10.0+cu128, which `uv.lock` pins.

**Training is not reproducible run to run** in the released code path. The only
seeding is a fixed `RandomState(42)` for the train/val split; weight
initialisation, batch order and — for the multi-frame and joint models — the
composition of each frame tuple are unseeded. See **Training determinism** below.

### Training determinism in detail

If you retrain, be aware that the released trainer seeds only the train/val split
(`torch.randperm` with a fixed generator). It sets no global torch, numpy or
`random` seed, no `worker_init_fn`, and no cuDNN determinism flags; TF32 is
enabled globally. The dataset's `__getitem__` uses the global `random` module, so
with several dataloader workers the frame tuples themselves vary between runs.
`--resume` restores model, optimizer and scheduler state but no RNG state.

One more trainer behaviour worth knowing: `global_step` advances by the number of
processes, so with multi-GPU training a `VAL_INTERVAL` or `SAVE_INTERVAL` that is
not divisible by the world size can be stepped over entirely.

## Caveats on specific reported rows

**SegVGGT single-layer** is reported as one row but comes from two different
checkpoint files of the same run: `segvggt/best.pth` on Replica and
`segvggt/step_0140000.pth` on Virtual KITTI 2. Both are shipped.

**SegMASt3R + Sinkhorn cannot be reproduced from this release.** It needs
`segmast3r_spp.ckpt` (3.7 GB) from the SegMASt3R release, which we do not
redistribute. Separately, note that the navigation experiment's Sinkhorn adapter
imports a copy of the inference wrapper that lives inside the segmast3r checkout,
not the one in this package, so the two code paths were never identical.

**SegMASt3R + LG v2 on Replica** has no reference metrics file; that run
directory was not preserved. The checkpoint is shipped, so the row can be
regenerated.

**Multi-frame N is not a clean ablation.** In tuple mode the per-view mask count
is padded to a common maximum, and the padded all-zero masks produce descriptors
that still take part in the joint attention. So N=2 against N>2 differs by more
than the attention span alone.

## Comparing a fresh run

```bash
uv run python scripts/eval_replica.py \
    --config configs/eval/replica_segvggt.yaml \
    --output_dir /tmp/myrun
python - <<'PY'
import json
a = json.load(open("reference/segvggt_dpt_v3_replica/metrics_by_bin.json"))
b = json.load(open("/tmp/myrun/metrics_by_bin.json"))
print("identical" if a == b else "differs")
PY
```

For a fast check, restrict to a subset with `--num_pairs`. Note that the pair
lists are grouped scene-major then bin-major, so a small prefix lands entirely
inside the first scene's lowest bin and exercises one bin only.
