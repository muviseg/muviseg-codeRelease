# MuViSeg: Multi-View Segment Correspondences from Dense Geometry Priors

[![Project page](https://img.shields.io/badge/project-muviseg.github.io-blue)](https://muviseg.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-2607.17938-b31b1b)](https://arxiv.org/abs/2607.17938)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official code release for the ACCV 2026 paper.

> **Status: release in progress.** This repository is being assembled from the
> research tree one reviewed step at a time, so that every refactor is checked
> against the numbers the paper reports. Sections below are filled in as the
> corresponding step lands. See [`docs/`](docs/) for what is already usable.

**Task.** Given two or more images of the same scene plus class-agnostic
segmentation masks for each, predict which segments correspond to the same
physical object. A frozen 3D foundation-model backbone (MASt3R or VGGT)
provides dense descriptors; masked average pooling produces one descriptor per
segment; a lightweight trainable head (LightGlue-style attention with a
DoubleSoftmax matcher) predicts the assignment, including a dustbin for
unmatched segments.

## Contents

| | |
|---|---|
| `muviseg/` | the installable package: models, data, training, evaluation, inference |
| `configs/` | one YAML per training run and per (model, benchmark) evaluation |
| `scripts/` | entry points: train, eval, inference, data preparation |
| `downstream/` | object-goal navigation on HM3D inside RoboHop / ObjectReact |
| `demo/` | the joint model on a recorded real-robot trajectory |
| `docs/` | installation, data preparation, reproduction |
| `tests/` | smoke tests for the critical pipeline |

## Quickstart

```bash
git clone https://github.com/muviseg/muviseg-codeRelease.git
cd muviseg-codeRelease
uv sync
bash setup_third_party.sh
```

<!-- TODO(step 13): training / evaluation / inference / reproduction sections -->

## License

MIT, see [LICENSE](LICENSE). Third-party components remain under their
upstream licenses; see `third_party/README.md`.

## Citation

See [CITATION.cff](CITATION.cff).
