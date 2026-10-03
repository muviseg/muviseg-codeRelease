"""
Generate SAM2 AMG masks for all images in scannetpp_processed.

Usage:
    python generate_sam2_masks.py \
        --processed_root /mnt/vol1/datasets/ScanNet++/data/scannetpp_processed \
        --masks_root     /mnt/vol1/datasets/ScanNet++/data/masks_resize_mast3r \
        --sam2_checkpoint /mnt/vol0/home/dennis/Projects/segmast3r/sam/sam2_hiera_large.pt \
        --sam2_config    /mnt/vol0/home/dennis/Projects/segmast3r/sam/sam2_hiera_l.yaml \
        --num_workers 2 --gpu_ids 0,1
"""

import argparse
import pickle
import queue
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


SAM2_AMG_KWARGS = dict(
    points_per_side=32,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.92,
    stability_score_offset=1.0,
    box_nms_thresh=0.7,
    crop_n_layers=1,
    crop_nms_thresh=0.7,
    crop_overlap_ratio=512 / 1500,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

# Serialize model initialization — fixes torch.jit.script conflict
# when multiple threads initialize SAM2 simultaneously
_init_lock = threading.Lock()


def get_image_paths(scene_dir: Path):
    exts = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}
    return sorted(p for p in (scene_dir / "images").iterdir() if p.suffix in exts)


def load_rgb_numpy(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def masks_to_coco_rle(mask_tensor: torch.Tensor):
    from sam2.utils.amg import mask_to_rle_pytorch
    from pycocotools import mask as mask_utils

    rles_uncompressed = mask_to_rle_pytorch(mask_tensor.bool())
    result = []
    for rle in rles_uncompressed:
        h, w = rle["size"]
        compressed = mask_utils.frPyObjects(rle, h, w)
        compressed["counts"] = compressed["counts"].decode("utf-8")
        result.append(compressed)
    return result


def build_model(sam2_config: str, sam2_checkpoint: str, device):
    from sam2.build_sam import build_sam2
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_path = Path(sam2_config)
    if config_path.exists():
        config_dir = str(config_path.parent.resolve())
        config_name = config_path.stem  # без .yaml

        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
            model = build_sam2(
                config_file=config_name,
                ckpt_path=sam2_checkpoint,
                device=device,
                mode="eval",
            )
    else:
        GlobalHydra.instance().clear()
        model = build_sam2(sam2_config, sam2_checkpoint, device=device, mode="eval")
    return model


def worker(gpu_id: int, scene_queue: queue.Queue, masks_root: Path,
           sam2_checkpoint: str, sam2_config: str, pbar_lock, pbar):

    # --- Initialize model (serialized to avoid JIT conflict) ---
    with _init_lock:
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            device = torch.device(f"cuda:{gpu_id}")
            model = build_model(sam2_config, sam2_checkpoint, device)
            generator = SAM2AutomaticMaskGenerator(model, **SAM2_AMG_KWARGS)
            generator.predictor.model.eval()
            print(f"[GPU {gpu_id}] SAM2 loaded OK")
        except Exception:
            print(f"[GPU {gpu_id}] Failed to load SAM2:")
            traceback.print_exc()
            return

    # --- Process scenes ---
    while True:
        try:
            scene_dir = scene_queue.get_nowait()
        except queue.Empty:
            break

        scene_id = scene_dir.name
        out_dir = masks_root / scene_id
        out_dir.mkdir(parents=True, exist_ok=True)

        img_paths = get_image_paths(scene_dir)

        for img_path in img_paths:
            out_pkl = out_dir / (img_path.stem + ".pkl")
            if out_pkl.exists():
                continue

            try:
                image_np = load_rgb_numpy(img_path)
                with torch.inference_mode():
                    sam_masks = generator.generate(image_np)

                if len(sam_masks) == 0:
                    coco_rles = []
                else:
                    mask_tensors = torch.stack(
                        [torch.from_numpy(m["segmentation"]) for m in sam_masks]
                    )
                    coco_rles = masks_to_coco_rle(mask_tensors)

                with open(out_pkl, "wb") as f:
                    pickle.dump(
                        {"mask_coco_rles_resized": coco_rles, "seg_corr_list": None},
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            except Exception:
                print(f"\n[GPU {gpu_id}] ERROR on {img_path}")
                traceback.print_exc()

        with pbar_lock:
            pbar.update(1)
        scene_queue.task_done()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed_root", required=True)
    p.add_argument("--masks_root", required=True)
    p.add_argument("--sam2_checkpoint", required=True)
    p.add_argument("--sam2_config", required=True)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--gpu_ids", default="0",
                   help="Comma-separated GPU IDs, e.g. '0,1'")
    p.add_argument("--skip_existing_scenes", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    processed_root = Path(args.processed_root)
    masks_root = Path(args.masks_root)
    masks_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    num_workers = min(args.num_workers, len(gpu_ids))

    scene_dirs = sorted(
        d for d in processed_root.iterdir()
        if d.is_dir() and (d / "images").exists()
    )

    if args.skip_existing_scenes:
        scene_dirs = [
            d for d in scene_dirs
            if not (masks_root / d.name).exists()
            or not any((masks_root / d.name).iterdir())
        ]

    print(f"Scenes to process : {len(scene_dirs)}")
    print(f"Workers / GPUs    : {num_workers} ({gpu_ids[:num_workers]})")
    print(f"Output root       : {masks_root}")

    q = queue.Queue()
    for d in scene_dirs:
        q.put(d)

    pbar_lock = threading.Lock()
    pbar = tqdm(total=len(scene_dirs), desc="scenes", unit="scene")

    threads = []
    for i in range(num_workers):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        t = threading.Thread(
            target=worker,
            args=(gpu_id, q, masks_root, args.sam2_checkpoint,
                  args.sam2_config, pbar_lock, pbar),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    pbar.close()
    print("Done.")


if __name__ == "__main__":
    main()
