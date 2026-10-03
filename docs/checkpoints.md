# Checkpoints

## Trained MuViSeg heads

Hosted on the Hugging Face Hub: **[muviseg/muviseg](https://huggingface.co/muviseg/muviseg)**.

```bash
# the three heads used for the main results
uv run python scripts/download_checkpoints.py

# everything, including the two single-layer ablation heads
uv run python scripts/download_checkpoints.py --all --verify
```

Files land where the shipped configs expect them, under `results/`:

| model | local path | size | md5 |
|---|---|---|---|
| SegMASt3R + LG v2 | `results/segmast3r_lg_v2/best.pth` | 3.3 MB | `fa0dcac9c83877669782c38ef9e93ebf` |
| SegVGGT-DPT (pairwise, v3) | `results/segvggt_dpt/v3-001/best.pth` | 50 MB | `e0753fb7286e5ae389632e02c165ef4c` |
| SegVGGT-DPT Joint | `results/segvggt_dpt/joint-001/best.pth` | 49 MB | `4f375a6cd97e82543c375e345f7c3264` |
| SegVGGT single-layer (ablation) | `results/segvggt/best.pth` | 5.5 MB | `31a917a2cda462a0644d7b02b77cf794` |
| SegVGGT single-layer, step 140k | `results/segvggt/step_0140000.pth` | 5.5 MB | `34e8779da0dfa3a415676ce97caffa8f` |

`--verify` checks these md5s after downloading. Pin a specific upload with
`--revision <tag-or-sha>`, and point elsewhere with `--repo-id`.

To load one directly instead:

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("muviseg/muviseg", "segvggt_dpt/v3-001/best.pth")
```

### These are head-only checkpoints

The frozen backbone is stripped by `scripts/export_release_checkpoint.py`, so each
file holds only the trainable head plus `epoch`, `global_step`, `best_val_ma` and
the validation `metrics`. The backbone is reloaded from `third_party/` at inference
time, and every checkpoint consumer here accepts both slim and full checkpoints.

The md5s above are for the published files, for verifying your download.
Re-running the export does not reproduce them byte for byte -- torch writes archive
metadata into the file -- but it does reproduce the contents exactly (verified: all
61 tensors and all metadata equal for the LG v2 head).

A slim checkpoint **cannot resume training**: `scripts/train.py --resume` needs the
optimizer and scheduler state, which only a full checkpoint carries.

### Which checkpoint produced which reported row

The SegVGGT single-layer row comes from two different files of the same run:
`segvggt/best.pth` on Replica and `segvggt/step_0140000.pth` on Virtual KITTI 2.
Both are published for that reason. See [`reproduction.md`](reproduction.md).

## Backbones (third party, not redistributed here)

Fetched or reported by `bash setup_third_party.sh`:

| backbone | expected path | size |
|---|---|---|
| VGGT-1B | `third_party/vggt_weights.pt` | 5.0 GB |
| MASt3R ViT-L | `third_party/segmast3r/mast3r_src/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` | 2.8 GB |
| FastSAM-x | `FastSAM-x.pt` | 145 MB |

FastSAM is needed only for the FastSAM-mask ablations and the demo; the main
Replica and VKITTI2 benchmarks feed the datasets' own instance annotations to the
model as segment proposals.

## Not available

The SegMASt3R + Sinkhorn baseline needs `segmast3r_spp.ckpt` (3.7 GB) from the
SegMASt3R release, which we do not redistribute. Place it at
`checkpoints/segmast3r_spp.ckpt` if you have it. See
[`reproduction.md`](reproduction.md) for what its absence means.

## Publishing a new set (maintainers)

See [`publishing_checkpoints.md`](publishing_checkpoints.md).
