# Third-party dependencies

Everything here is fetched by `bash setup_third_party.sh` from the repository
root. Nothing in this directory except `segmast3r_patches/` is committed.

## Pinned source checkouts

| directory | upstream | pin | needed for |
|---|---|---|---|
| `segmast3r/` | [SegMASt3R/segmast3r](https://github.com/SegMASt3R/segmast3r) | `0977d06` | MASt3R-backbone models: `ARCH = sinkhorn`, `lightglue`, `lightglue_v2` |
| `vggt/` | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | `44b3afb` | VGGT-backbone models: `ARCH = segvggt`, `segvggt_dpt`, `segvggt_dpt_joint` |

They are put on `sys.path` at import time by `muviseg/paths.py`
(`setup_segmast3r_path()`, `setup_vggt_path()`), not installed as packages.

Note that `segmast3r`'s own top-level package is called `src`, and
`setup_segmast3r_path()` inserts it at the front of `sys.path`. After that call,
`import src.*` resolves inside the segmast3r checkout.

## Our patches

`segmast3r_patches/` holds two first-party modules that have to sit *inside* the
segmast3r checkout to be importable, where they would otherwise be untracked and
lost on any re-checkout. `setup_third_party.sh` copies them in. See
[`segmast3r_patches/README.md`](segmast3r_patches/README.md).

## Weights you have to obtain yourself

`setup_third_party.sh` reports which of these are missing; it does not download
them, because they are large and come from third parties under their own terms.

| file | size | source |
|---|---|---|
| `vggt_weights.pt` | 5.0 GB | [facebook/VGGT-1B](https://huggingface.co/facebook/VGGT-1B); the code can also auto-download it |
| `segmast3r/mast3r_src/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` | 2.8 GB | [naver/mast3r](https://github.com/naver/mast3r#checkpoints) |
| `../FastSAM-x.pt` (repo root) | 145 MB | [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM#model-checkpoints) — only for the FastSAM ablations and the demo |

Trained MuViSeg heads are separate; see `docs/checkpoints.md`.

## Downstream navigation

The HM3D navigation experiment uses a different upstream
([oravus/object-rel-nav](https://github.com/oravus/object-rel-nav)) in its own
conda environment, and is set up by `downstream/topological_navigation/setup.sh`
rather than by this script.

## Licenses

Each dependency stays under its upstream license. MuViSeg's own code is MIT; see
[`../LICENSE`](../LICENSE).
