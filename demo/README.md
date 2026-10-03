# Demo: the joint model on a recorded robot trajectory

A real run of SegVGGT-DPT Joint on a 540-frame walk through a university library,
in two joint-window modes (N = 4 and N = 6), with a per-batch latency benchmark.

**There is no ground truth for this recording.** Every segment and every link
below is a model output, so this shows what the method does on unconstrained
real footage — not how accurate it is.

```
FastSAM-x                        segment proposals, ~46 per frame
SegVGGT-DPT Joint (joint-001)    joint attention over all segments of a window
mutual + matchability filter     accepted matches (the stock rule, no threshold)
constrained union-find           tracks (never two segments of one frame)
```

## Input

540 frames at 3840x2160, resized to 512x336 for the model. The odometry path is
395.5 m long, so consecutive frames are **0.73 m apart on average** — this is a
walk through a building, not a slow orbit around an object. FastSAM returns
7 / 46.2 / 50 masks per frame (min/mean/max); the mean sits against the
`m_prime = 50` cap, so the model runs near its maximum segment load.

The recording is not redistributed with the code. Any directory of frames works.

## Reproducing it

```bash
uv sync --extra eval        # ultralytics, for FastSAM
uv run python scripts/download_checkpoints.py

uv run python scripts/inference.py \
    --frames path/to/frames --out demo/output/N4 \
    --n 4 --stride 2 --batch 4 --save-overlays

# a second mode over the same cached masks, for a like-for-like comparison
uv run python scripts/inference.py \
    --frames path/to/frames --out demo/output/N6 \
    --n 6 --stride 2 --batch 4 --masks demo/output/N4/masks.pt
```

Reusing `masks.pt` matters: it makes the two modes differ only in the joint
window, not in their segment proposals.

## Measured results

Sliding window, stride 2, batch 4, all C(N,2) pairs per window. Model on an
RTX PRO 6000 Blackwell, segmentation on an RTX 5090, nothing else on either GPU.

| | **N = 4** | **N = 6** |
|---|---|---|
| windows | 269 | 268 |
| pairs per window | 6 | 15 |
| wall clock, whole trajectory | 52.0 s | 85.8 s |
| **latency per batch (B=4), p50** | **698.8 ms** | **1161.3 ms** |
| — of which VGGT backbone | 682.1 ms (98.3 %) | 1141.8 ms (98.4 %) |
| — of which matching head | 11.5 ms (1.7 %) | 18.2 ms (1.6 %) |
| peak VRAM | 7.73 GiB | 9.86 GiB |
| segments total | 24 945 | 24 945 |
| tracks of length >= 3 | 1 825 | 1 652 |

FastSAM itself cost 242.7 ms/frame, most of it 4K JPEG decode rather than
inference (12.6 s decode and 8.8 s of network time over 540 frames).

**The backbone is 98% of the cost.** The trainable matching head — the part this
paper contributes — is 1.5-2% of inference time, and going from 6 pairs per
window to 15 adds about 7 ms to it. So the joint formulation is close to free
relative to the frozen backbone it sits on.

## What these numbers do and do not show

* **N = 6 is not better here.** It produces *fewer* long tracks than N = 4
  (1 652 against 1 825) for 1.65x the wall clock. On a trajectory where
  consecutive frames are 0.73 m apart, a wider window spans a larger baseline
  than the matcher handles well. Do not read "larger N is better" into this.
* **Latency depends on having the GPU to yourself.** A verification run of the
  released code on 60 of these frames, sharing a GPU with an evaluation job,
  measured 1121 ms/batch at N = 4 instead of 698.8 — a 1.6x penalty purely from
  contention. Benchmark on an idle device.
* **Track counts are not accuracy.** With no ground truth, a track being long
  only means the model kept linking something. The matchability head is
  conservative: on annotated data it accepts few matches but is usually right
  about the ones it accepts, so expect many short tracks and some genuine objects
  missed entirely.

## Verification of the released code path

The code here is the demo script refactored into `muviseg/inference/` and
`scripts/inference.py`. Checked against the original measurement:

* window counts identical — 269 at N = 4, 268 at N = 6, pinned in
  `tests/test_inference.py`
* peak VRAM 7.75 GiB against 7.73 GiB at N = 4
* segment counts per frame consistent (38 / 49.1 / 50 over the first 60 frames,
  against 7 / 46.2 / 50 over all 540 — the opening stretch is busier)

Latency was not re-verified cleanly, for the contention reason above.
