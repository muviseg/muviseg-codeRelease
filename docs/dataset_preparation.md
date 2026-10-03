# Dataset preparation

Nothing is vendored. Obtain each dataset from its own source, then point the
configs at your copy.

| dataset | used for | obtain from |
|---|---|---|
| ScanNet++ | training | <https://kaldir.vc.in.tum.de/scannetpp/> (registration required) |
| Replica (semantic) | indoor evaluation | <https://github.com/facebookresearch/Replica-Dataset> plus the instance-segmentation release |
| Virtual KITTI 2 | outdoor evaluation | <https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/> |
| HM3D + InstanceImageNav | downstream navigation | see `downstream/topological_navigation/README.md` |

## Evaluation data

### Replica

Two roots, because the RGB frames and the instance masks ship separately:

```
<replica_root>/<scene>/Sequence_1/rgb/rgb_<i>.png
<instance_root>/<scene>/Sequence_1/semantic_instance/semantic_instance_<i>.png
<replica_root>/<scene>/Sequence_1/traj_w_c.txt        # poses, for pair sampling
```

Eight scenes are used: `office_0..4`, `room_0..2`. Set `DATASET.DATA_ROOT` and
`DATASET.INSTANCE_MASK_ROOT` in `configs/eval/replica*.yaml`.

### Virtual KITTI 2

One root; RGB and instance masks sit in sibling trees:

```
<vkitti_root>/vkitti_rgb/<scene>/<variant>/frames/rgb/Camera_0/rgb_<i:05d>.jpg
<vkitti_root>/vkitti_instanceSegmentation/<scene>/<variant>/frames/instanceSegmentation/Camera_0/instancegt_<i:05d>.png
<vkitti_root>/vkitti_textgt/<scene>/<variant>/extrinsic.txt
```

Five scenes (`Scene01`, `Scene02`, `Scene06`, `Scene18`, `Scene20`) × four
variants (`clone`, `15-deg-left`, `30-deg-right`, `sunset`). Set
`DATASET.DATA_ROOT` in `configs/eval/vkitti2*.yaml`.

Instance ids are decoded as `trackID = pixel_value - 1`, and `trackID <= 0` is
background.

### Pair lists

Already generated and shipped in `assets/pairs/`. Do not regenerate them if you
intend to compare against the reported numbers — see
[`reproduction.md`](reproduction.md).

## Training data (ScanNet++)

Three preparation stages, in order. All are driven by `scripts/data/`.

```bash
export SPP=/path/to/ScanNet++
```

**1. Automatic masks for every frame** (needs `--extra dataprep` for SAM 2):

```bash
uv run python scripts/data/generate_masks.py \
    --processed_root   $SPP/scannetpp_processed \
    --masks_root       $SPP/masks_resize_mast3r \
    --sam2_checkpoint  /path/to/sam2_hiera_large.pt \
    --sam2_config      /path/to/sam2_hiera_l.yaml \
    --gpu_ids 0,1
```

**2. Ground-truth segment correspondences** by depth backprojection:

```bash
uv run python scripts/data/corr_pairs.py \
    --metadata  $SPP/scannetpp_processed/all_metadata.npz \
    --processed $SPP/scannetpp_processed \
    --masks     $SPP/masks_resize_mast3r \
    --output    $SPP/masks_resize_mast3r_res \
    --num_workers 32 --iou_threshold 0.25
```

**3. Check all three stages:**

```bash
uv run python scripts/data/sanity_check.py --root $SPP --sample 200 --check_rle
```

**4. Optional descriptor precompute** — only for the MASt3R configs that set
`DATASET.PAIR_DSC_ROOT`:

```bash
uv run accelerate launch --num_processes=2 scripts/data/precompute_features.py \
    --config configs/train/segmast3r_train_precompute.yaml \
    --output $SPP/segmast3r_pair_dsc
```

### Why precompute is per pair, not per image

MASt3R's decoder cross-attends between the two views, so the descriptors for one
image depend on which image it is paired with. Caching descriptors per image would
be wrong, and is deliberately unsupported. SegVGGT-DPT fuses spatial features
before pooling and always runs its backbone online, so it has no precompute path
at all.

Precomputed VGGT descriptors are roughly 85× larger than MASt3R's; store them as
fp16 and expect about 6 GB per 1000 pairs.

Then set the `DATASET.*` paths in `configs/train/*.yaml`.
