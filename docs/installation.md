# Installation

## Supported configuration

The reported results were produced with:

| | |
|---|---|
| Python | 3.11 (3.12 and 3.13 also resolve) |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| GPUs | 1× RTX PRO 6000 Blackwell (97 GB) for training; a single 24 GB card is enough for evaluation and inference |

Training uses bf16 mixed precision, which needs compute capability 8.0 or newer
(Ampere and later). Evaluation runs on CPU with `--device cpu`, slowly.

## Install

```bash
git clone https://github.com/MuViSeg/muviseg-codeRelease.git
cd muviseg-codeRelease
uv sync
```

`uv sync` creates `.venv` from the committed `uv.lock`, so everyone gets the same
resolved versions. Run commands with `uv run`, which needs no activation:

```bash
uv run python scripts/eval_replica.py --help
```

### Extras

```bash
uv sync --extra eval       # FastSAM ablations, figure scripts, the demo
uv sync --extra roma       # the RoMa dense-matcher baseline
uv sync --extra dataprep   # SAM 2, for generating ScanNet++ masks
uv sync --extra dev        # pytest and ruff
```

`roma` is separate from `eval` on purpose: `romatch` pulls in `wandb` and
`polars`, which nothing else here needs.

### CUDA other than 12.8

`pyproject.toml` pins torch and torchvision to the cu128 wheel index, because
that is what the reported numbers were produced with. For a different build,
override the index:

```bash
uv sync --no-sources                               # whatever PyPI defaults to
uv pip install torch torchvision --index https://download.pytorch.org/whl/cu121
```

Changing the CUDA or PyTorch version can change results in the last digit; see
[`reproduction.md`](reproduction.md).

## Third-party dependencies

```bash
bash setup_third_party.sh
```

This clones the two pinned source dependencies, deploys our patches into the
segmast3r checkout, and tells you which large backbone weights are still
missing and where to get them. See [`../third_party/README.md`](../third_party/README.md).

## Dataset roots

No dataset is vendored. Point the configs at your copies. Paths in the shipped
configs are currently absolute and must be edited to match your layout; the
intended resolution order is:

* absolute, used as given
* relative, resolved against the repository root, not the current directory
* `${VAR}`, expanded from the environment

Evaluation currently has to be run from the repository root. Making every path
resolve against the repository root regardless of working directory is in
progress.

## Docker

If the CUDA stack is awkward to install natively, or you want the downstream
navigation environment, use the image:

```bash
docker build -t muviseg .
docker run --gpus all -it -v /path/to/datasets:/datasets muviseg
```

## Check the install

```bash
uv run pytest -q
```

These are smoke tests: imports, config parsing, metric arithmetic and the head
components. They need neither a GPU nor a dataset, and skip what is unavailable.
