# MuViSeg: Multi-View Segment Correspondences from Dense Geometry Priors

[![Project page](https://img.shields.io/badge/project-muviseg.github.io-blue)](https://muviseg.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2607.17938-b31b1b)](https://arxiv.org/abs/2607.17938)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97-checkpoints-yellow)](https://huggingface.co/MuViSeg/muviseg)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official code release for the ACCV 2026 paper.

**Task.** Given two or more images of the same scene plus class-agnostic
segmentation masks for each, predict which segments correspond to the same
physical object. A frozen 3D foundation-model backbone (MASt3R or VGGT) provides
dense descriptors; masked average pooling produces one descriptor per segment; a
lightweight trainable head (LightGlue-style attention with a DoubleSoftmax
matcher) predicts the assignment, including a dustbin for unmatched segments.

## Install

```bash
git clone https://github.com/MuViSeg/muviseg-codeRelease.git
cd muviseg-codeRelease
uv sync                                  # pinned env: Python 3.11, torch 2.10.0+cu128
bash setup_third_party.sh                # pinned MASt3R/SegMASt3R and VGGT checkouts
uv run python scripts/download_checkpoints.py
```

Point the configs at your datasets with one environment variable:

```bash
export MUVISEG_DATA_ROOT=/path/to/datasets   # expects Replica/, VirtualKITTI2/, ScanNet++/
```

Paths in configs are absolute as given, `${VAR}`-expanded from the environment, or
resolved against the repository root — so every command below works from any
directory. Details in [`docs/installation.md`](docs/installation.md) and
[`docs/dataset_preparation.md`](docs/dataset_preparation.md).

Check the install (no GPU or dataset needed):

```bash
uv run pytest -q
```

## Model zoo

| Model | Backbone (frozen) | Trainable head | Params | Training config |
|---|---|---|---|---|
| SegMASt3R + Sinkhorn | MASt3R ViT-L | none (OT matcher only) | 0 | `configs/train/segmast3r_train.yaml` |
| SegMASt3R + LG v1 | MASt3R ViT-L | linear proj + 3× attention | ~0.45 M | `configs/train/segmast3r_train_lg.yaml` |
| **SegMASt3R + LG v2** | MASt3R ViT-L | MLP proj + 3× attention (4× FFN) | ~0.8 M | `configs/train/segmast3r_train_lg_v2.yaml` |
| SegVGGT (single-layer) | VGGT Aggregator (layer 23) | MLP proj + 3× attention | ~0.8 M | `configs/train/segvggt_train.yaml` |
| **SegVGGT-DPT** | VGGT Aggregator (5/11/17/23) | DPT fusion + attention | ~10.7 M | `configs/train/segvggt_dpt_train_v3.yaml` |
| SegVGGT-DPT Multiframe | VGGT Aggregator | DPT fusion + pairwise attention over N frames | ~10.7 M | `configs/train/segvggt_dpt_train_multiframe.yaml` |
| **SegVGGT-DPT Joint** | VGGT Aggregator | DPT fusion + joint attention over all N frames | ~11.0 M | `configs/train/segvggt_dpt_train_joint.yaml` |

## Evaluation

```bash
# Replica (indoor), one config per model
uv run python scripts/eval_replica.py --config configs/eval/replica_lgv2.yaml
uv run python scripts/eval_replica.py --config configs/eval/replica_segvggt.yaml
uv run python scripts/eval_replica.py --config configs/eval/replica_segvggt_joint.yaml

# Virtual KITTI 2 (outdoor)
uv run python scripts/eval_vkitti2.py --config configs/eval/vkitti2_lgv2.yaml

# useful flags
#   --num_pairs 40      smoke test on a prefix of the pair list
#   --visualize         write match overlays to <output_dir>/visualizations/
#   --device cpu
#   --output_dir DIR    override EVAL.OUTPUT_DIR
```

Each run writes `metrics_by_bin.json`, a formatted table, and `raw_results.json`.
Reference per-bin metrics for the reported runs are in `reference/`.

Sweeps used in the paper:

```bash
bash scripts/sweeps/run_eval_segvggt_dpt_all.sh    # checkpoint epochs
bash scripts/sweeps/run_eval_segvggt_joint_all.sh  # joint N in {2,4,6,8}
```

### Expected results (overall AUPRC / R@1 / R@5)

| Model | Replica | Virtual KITTI 2 |
|---|---|---|
| SegMASt3R + LG v2 | **84.5** / 77.5 / 93.6 [^lgv2] | **81.2** / 72.7 / 92.8 |
| SegVGGT-DPT (pairwise v3) | 79.7 / 74.1 / 90.1 | 51.9 / 42.9 / 67.6 |
| SegVGGT-DPT Joint (N=4) | 81.6 / 76.5 / 90.6 | 52.8 / 43.2 / 66.8 |
| SegVGGT-DPT Joint (N=2) | 81.5 / 76.2 / 90.7 | 52.2 / 42.3 / 66.7 |
| SegVGGT single-layer | 79.2 / 73.3 / 89.6 | 51.5 / 42.1 / 66.3 |

[^lgv2]: Every other number in this table is reproduced from the `reference/`
    artifacts shipped here. This one is taken from the paper: the Replica run
    directory for LG v2 was not preserved, so there is no artifact to check it
    against. The checkpoint is published, so the row can be regenerated.

MASt3R + LG v2 dominates at small viewpoint change; joint multi-frame attention
narrows the gap at large rotations. **Read
[`docs/reproduction.md`](docs/reproduction.md) before comparing against the
paper** — it lists which rows are exactly reproducible, which are not, and why.

## Training

Training uses ScanNet++ with SAM 2 masks and depth-backprojected ground-truth
correspondences. Prepare the data first
([`docs/dataset_preparation.md`](docs/dataset_preparation.md)), then:

```bash
# smoke test: synthetic data, no dataset and no backbone weights needed
uv run python scripts/train.py --mock

# single GPU
uv run python scripts/train.py --config configs/train/segvggt_dpt_train_v3.yaml

# multi-GPU with bf16 via Accelerate; trailing KEY VALUE pairs override the config
uv run accelerate launch scripts/train.py \
    --config configs/train/segvggt_dpt_train_joint.yaml \
    TRAINING.BATCH_SIZE 16

# resume (needs a full checkpoint, not a released slim one)
uv run python scripts/train.py --config configs/train/segvggt_dpt_train_v3.yaml \
    --resume results/segvggt_dpt/v3-001/last.pth

# monitor
uv run tensorboard --logdir results/
```

Configuration is YACS: `muviseg/config/default.py` holds the schema and every
default, a `--config` YAML overrides it, and trailing `KEY VALUE` pairs on the
command line override that. The merge is strict, so a misspelled key is an error
rather than a silent no-op. Checkpoints and TensorBoard logs go to
`results/<run_name>/`.

Two things to know before you retrain:

* **Runs are not reproducible from a seed.** Only the train/val split is seeded.
  Weight init, batch order and — for the multi-frame models — the composition of
  each frame tuple are not. See `docs/reproduction.md`.
* **MASt3R descriptors are only valid per pair**, because its decoder
  cross-attends between the two views. Per-image precompute is therefore
  unsupported; SegVGGT-DPT always runs its backbone online.

## Using the model from Python

The inference wrappers take a loaded config, a device, and batched tensors.
Images must arrive in `[-1, 1]`; each model rescales internally for its backbone.

```python
import numpy as np, torch, yaml
import torchvision.transforms as T
from PIL import Image

from muviseg.config.paths import resolve_config_paths
from muviseg.evaluation.model_infer import SegVGGTDPTInfer

to_tensor = T.Compose([T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])

def load_image(path, size=(512, 336)):
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return to_tensor(img).unsqueeze(0)                      # (1, 3, H, W)

def masks_from_label_png(path, size=(512, 336), m_prime=50):
    lab = np.array(Image.open(path).resize(size, Image.NEAREST))
    ids = [i for i in np.unique(lab) if i != 0][:m_prime]
    stack = np.stack([lab == i for i in ids]).astype(np.float32)
    return torch.from_numpy(stack).unsqueeze(0), ids        # (1, M, H, W)

cfg = resolve_config_paths(yaml.safe_load(open("configs/eval/replica_segvggt.yaml")))
model = SegVGGTDPTInfer(cfg)
model.prepare(torch.device("cuda"))

img0, img1 = load_image("view0.png"), load_image("view1.png")
masks0, ids0 = masks_from_label_png("view0_instances.png")
masks1, ids1 = masks_from_label_png("view1_instances.png")

match_result, scores = model.infer_pair(img0, img1, masks0, masks1)
# scores:       (1, M, N) mutual-agreement probabilities
# match_result: (1, M) int64, -1 where a segment was left unmatched

for i, j in enumerate(match_result[0].tolist()):
    if j >= 0:
        print(f"segment {i} -> {j}   score {scores[0, i, j]:.3f}")
```

`match_result` applies the stock acceptance rule: mutual nearest neighbour plus a
positive matchability logit. It is deliberately conservative — precision is high
and many true matches are left as `-1` — so use `scores` directly if you want to
apply your own threshold.

Swap `SegVGGTDPTInfer` for `SegMASt3RLGv2Infer`, `SegVGGTInfer` or
`MASt3RSegFeatInfer` with the matching config; all four share this interface.
`SegVGGTDPTJointInfer` additionally offers `infer_tuple` for N frames at once.

Other useful pieces of the package:

```python
from muviseg.evaluation.eval_metrics import compute_metrics, aggregate_metrics_by_bin
from muviseg.evaluation.ground_truth_generator import generate_instance_correspondences
from muviseg.datasets.replica_dataset import ReplicaSegmentMatchDataset
from muviseg.models.vggt_dpt_lg import SegVGGTDPT, SegVGGTDPTJoint
```

## Downstream: topological object-goal navigation

Our matchers replace the LightGlue localizer of RoboHop / ObjectReact for
InstanceImageNav on HM3D, reported as Success Rate / SPL / Soft-SPL.

This runs in a **separate conda environment**, because habitat-sim needs conda and
Python 3.10 and cannot be installed with uv:

```bash
bash downstream/topological_navigation/setup.sh     # clone upstream @6da34c3 + apply our overlay

conda create -n nav python=3.10 && conda activate nav
conda install habitat-sim=0.3.3 headless -c conda-forge -c aihabitat
pip install -r downstream/topological_navigation/object-rel-nav/requirements.txt
pip install -e .                                    # the muviseg package

cd downstream/topological_navigation/object-rel-nav
python main.py -c configs/segvggt_dpt_minival_Ss16.yaml
python scripts/summarize_matchers.py                # SR / SPL / Soft-SPL table
python scripts/analyze_table3.py                    # paired McNemar + bootstrap
```

**This pipeline is not deterministic** — repeated identical runs diverge and can
flip an episode's outcome, with roughly ±6 SPL of run-to-run spread. At minival
sizes the comparisons are also underpowered. Read
[`downstream/topological_navigation/README.md`](downstream/topological_navigation/README.md)
before quoting any number from it; it explains the cold-start setting, which
dominates the result, and what our own re-measurement did and did not reproduce.

## Repository layout

```
muviseg/            the installable package
  models/           sinkhorn, lightglue, lightglue_v2, vggt_lightglue, vggt_dpt_lg
  data/             ScanNet++ pair and tuple datasets (+ synthetic data for --mock)
  training/         trainer, validator, shared utils
  datasets/         Replica and Virtual KITTI 2 loaders
  evaluation/       inference wrappers, metrics, ground truth, visualisation
  sampling/         stratified pair samplers
  config/           YACS schema and path resolution
  paths.py          locates third_party checkouts, independent of cwd
configs/train|eval/ one YAML per training run and per (model, benchmark)
scripts/            train, eval, inference, data preparation, sweeps
assets/pairs/       the frozen benchmark pair lists
reference/          per-bin metrics for the reported runs
downstream/         object-goal navigation overlay
docs/               installation, data, checkpoints, reproduction, publishing
tests/              smoke tests
third_party/        pinned external checkouts (populated by setup_third_party.sh)
```

## Documentation

| | |
|---|---|
| [installation.md](docs/installation.md) | supported versions, extras, Docker |
| [dataset_preparation.md](docs/dataset_preparation.md) | where to get each dataset and how to prepare ScanNet++ |
| [checkpoints.md](docs/checkpoints.md) | what to download and from where |
| [reproduction.md](docs/reproduction.md) | **what reproduces, what does not, and why** |
| [publishing_checkpoints.md](docs/publishing_checkpoints.md) | for maintainers |

## License

MIT, see [LICENSE](LICENSE). Third-party components (MASt3R, VGGT, SegMASt3R,
LightGlue, RoMa, object-rel-nav / RoboHop / ObjectReact, SAM 2, FastSAM) remain
under their respective upstream licenses; see
[`third_party/README.md`](third_party/README.md).

## Citation

See [CITATION.cff](CITATION.cff), or cite the paper:

```bibtex
@inproceedings{muviseg2026,
  title     = {MuViSeg: Multi-View Segment Correspondences from Dense Geometry Priors},
  author    = {Fatykhoph, Denis and Akhtyamov, Timur and Pakulev, Konstantin
               and Devchich, German and Ferrer, Gonzalo},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## Acknowledgements

This code builds on [MASt3R](https://github.com/naver/mast3r),
[VGGT](https://github.com/facebookresearch/vggt),
[SegMASt3R](https://github.com/SegMASt3R/segmast3r),
[LightGlue](https://github.com/cvg/LightGlue),
[RoMa](https://github.com/Parskatt/RoMa),
[object-rel-nav / RoboHop / ObjectReact](https://github.com/oravus/object-rel-nav),
[SAM 2](https://github.com/facebookresearch/sam2) and
[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM).
