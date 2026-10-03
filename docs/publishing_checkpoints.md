# Publishing the checkpoints to the Hugging Face Hub

What to upload, where, and how. One-time setup for maintainers.

## What gets published

Only the **slim, head-only** checkpoints — five files, 113 MB in total. Not the
full training checkpoints (3-4 GB each, mostly frozen backbone and optimizer
state), and not the backbones themselves, which belong to their upstream authors.

If you are starting from full training checkpoints, strip them first:

```bash
for src in segmast3r_lg_v2/best.pth segvggt/best.pth segvggt/step_0140000.pth \
           segvggt_dpt/v3-001/best.pth segvggt_dpt/joint-001/best.pth; do
  uv run python scripts/export_release_checkpoint.py \
      --input  "results/$src" \
      --output "upload/$src"
done
du -sh upload   # expect ~113 MB
```

The layout inside `upload/` is exactly the layout in the Hub repo, and exactly
what `scripts/download_checkpoints.py` expects:

```
segmast3r_lg_v2/best.pth              3.3 MB
segvggt/best.pth                      5.5 MB
segvggt/step_0140000.pth              5.5 MB
segvggt_dpt/v3-001/best.pth            50 MB
segvggt_dpt/joint-001/best.pth         49 MB
```

## Create the repo and upload

`huggingface-hub` is a core dependency, so the `hf` CLI is already in the synced
environment.

```bash
uv run hf auth login                                   # paste a WRITE token
uv run hf repos create muviseg/muviseg --type model     # omit if it exists
uv run hf upload muviseg/muviseg upload/ . --type model
uv run hf repos tag create muviseg/muviseg v1.0.0 --type model
```

`hf upload` takes `REPO_ID [LOCAL_PATH] [PATH_IN_REPO]`, splits large folders over
several commits, and resumes if you re-run the same command after an interruption.
It also creates the repo when it does not exist, so `repos create` is optional.

Verified against `huggingface_hub` 2.1.1. Note the subcommand is `repos`, not
`repo`, and older releases shipped the CLI as `huggingface-cli` instead of `hf`.

## Add the model card

Hub repos show `README.md` as the model card. Create `upload/README.md` before
uploading, or add it afterwards:

```markdown
---
license: mit
tags:
  - segment-matching
  - multi-view
  - correspondence
  - 3d-vision
library_name: pytorch
---

# MuViSeg checkpoints

Trained heads for **MuViSeg: Multi-View Segment Correspondences from Dense
Geometry Priors** (ACCV 2026).

- Code: https://github.com/muviseg/muviseg-codeRelease
- Project page: https://muviseg.github.io
- Paper: https://arxiv.org/abs/2607.17938

These are **head-only** checkpoints: the frozen MASt3R / VGGT backbone is stripped
and reloaded separately at inference time, so each file is a few MB rather than a
few GB. They cannot resume training.

| file | model | params | size |
|---|---|---|---|
| `segmast3r_lg_v2/best.pth` | SegMASt3R + LG v2 | ~0.8 M | 3.3 MB |
| `segvggt_dpt/v3-001/best.pth` | SegVGGT-DPT, pairwise | ~10.7 M | 50 MB |
| `segvggt_dpt/joint-001/best.pth` | SegVGGT-DPT, joint attention | ~11.0 M | 49 MB |
| `segvggt/best.pth` | SegVGGT single-layer (ablation) | ~0.8 M | 5.5 MB |
| `segvggt/step_0140000.pth` | SegVGGT single-layer, step 140k | ~0.8 M | 5.5 MB |

The single-layer row in the paper uses `best.pth` on Replica and
`step_0140000.pth` on Virtual KITTI 2.

## Usage

```bash
git clone https://github.com/muviseg/muviseg-codeRelease.git
cd muviseg-codeRelease && uv sync && bash setup_third_party.sh
uv run python scripts/download_checkpoints.py --all --verify
```

Backbones are not included here: VGGT-1B comes from `facebook/VGGT-1B` and MASt3R
from the MASt3R release. See the repository's `docs/checkpoints.md`.

## Citation

See `CITATION.cff` in the code repository.
```

## After publishing

1. Check the md5s in `docs/checkpoints.md` still match what you uploaded:

   ```bash
   uv run python scripts/download_checkpoints.py --all --verify --dest /tmp/ckpt-check
   ```

2. If the repo id or layout changed, update `REPO_ID` and the `CHECKPOINTS` table
   in `scripts/download_checkpoints.py` and the table in `docs/checkpoints.md`.

## A note on access

Keep the Hub repo **public**. A gated or private repo makes
`scripts/download_checkpoints.py` fail for everyone without a token, which defeats
the point of the release. If you must gate it, say so in the README and document
`huggingface-cli login` as a required step.
