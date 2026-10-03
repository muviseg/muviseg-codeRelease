# Checkpoints

## Trained MuViSeg heads

Hosted on OSF: <https://osf.io/z46uj/> (`muviseg-checkpoints.tar.gz`, 104 MB).

```bash
curl -L -o muviseg-checkpoints.tar.gz "https://osf.io/z46uj/download"
tar -xzf muviseg-checkpoints.tar.gz -C .
```

Extracting at the repository root puts every file where the shipped evaluation
configs expect it.

| model | path | size | md5 |
|---|---|---|---|
| SegMASt3R + LG v2 | `checkpoints/segmast3r_lg_v2/best.pth` | 3.3 MB | `fa0dcac9c83877669782c38ef9e93ebf` |
| SegVGGT-DPT (pairwise, v3) | `checkpoints/segvggt_dpt/v3-001/best.pth` | 50 MB | `e0753fb7286e5ae389632e02c165ef4c` |
| SegVGGT-DPT Joint | `checkpoints/segvggt_dpt/joint-001/best.pth` | 49 MB | `4f375a6cd97e82543c375e345f7c3264` |
| SegVGGT single-layer (ablation) | `checkpoints/segvggt/best.pth` | 5.5 MB | `31a917a2cda462a0644d7b02b77cf794` |
| SegVGGT single-layer, step 140k | `checkpoints/segvggt/step_0140000.pth` | 5.5 MB | `34e8779da0dfa3a415676ce97caffa8f` |

### These are head-only checkpoints

The frozen backbone is stripped with `scripts/export_release_checkpoint.py`, so
each file holds only the trainable head plus `epoch`, `global_step`,
`best_val_ma` and the validation `metrics`. The backbone is reloaded from
`third_party/` at inference time. Every checkpoint consumer in this repository
accepts both the slim and the original full checkpoints.

The md5s above are for the published files, for verifying your download.
Re-running the export does not reproduce them byte for byte — torch writes
archive metadata into the file — but it does reproduce the contents exactly
(verified: all 61 tensors and all metadata equal for the LG v2 head).

A slim checkpoint **cannot resume training** — `scripts/train.py --resume`
needs the optimizer and scheduler state, which only a full checkpoint carries.

### Which checkpoint produced which reported row

Two of the reported ablation numbers come from **different** files of the same
run: the SegVGGT single-layer row uses `segvggt/best.pth` on Replica and
`segvggt/step_0140000.pth` on Virtual KITTI 2. Both are shipped for that reason.
See `docs/reproduction.md`.

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
SegMASt3R release, which we do not redistribute. See `docs/reproduction.md` for
what that means for reproducing that row.
