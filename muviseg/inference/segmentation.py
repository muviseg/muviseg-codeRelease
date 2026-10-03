"""Class-agnostic segment proposals with FastSAM.

Only needed when you have no instance annotations. The Replica and Virtual
KITTI 2 benchmarks feed the datasets' own annotations to the model instead.

Requires the `eval` extra (`uv sync --extra eval`) for ultralytics.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Defaults are the settings the reported demo used.
SEG_IMGSZ = 1024
SEG_CONF = 0.4
SEG_IOU = 0.9
DECODE_SIZE = (1920, 1080)


def _decode(path: Path, size) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.array(img)


def segment_frames(
    paths: list[Path],
    checkpoint: str | Path,
    *,
    target_h: int,
    target_w: int,
    m_prime: int = 50,
    device: str = "cuda",
    batch: int = 8,
    workers: int = 12,
    decode_size=DECODE_SIZE,
    imgsz: int = SEG_IMGSZ,
    conf: float = SEG_CONF,
    iou: float = SEG_IOU,
    progress: bool = True,
) -> tuple[list[torch.Tensor], dict]:
    """Run FastSAM over every frame.

    Returns a list of (M, target_h, target_w) uint8 CPU tensors, sorted by
    descending area and truncated to `m_prime`, plus a timing dict.

    Masks are area-sorted and then cut at `m_prime`, which is what makes the
    proposal set stable: FastSAM's own output order is not.
    """
    from ultralytics import FastSAM

    seg = FastSAM(str(checkpoint))
    out: list[torch.Tensor] = []
    t_dec = t_inf = 0.0
    pool = ThreadPoolExecutor(max_workers=workers)
    t_all = time.perf_counter()

    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        t0 = time.perf_counter()
        imgs = list(pool.map(lambda p: _decode(p, decode_size), chunk))
        t_dec += time.perf_counter() - t0

        t0 = time.perf_counter()
        res = seg(imgs, device=device, retina_masks=True, imgsz=imgsz,
                  conf=conf, iou=iou, verbose=False)
        if str(device).startswith("cuda"):
            # ultralytics rewrites CUDA_VISIBLE_DEVICES when handed an explicit
            # index, so only the current device can safely be synced here.
            torch.cuda.synchronize()
        t_inf += time.perf_counter() - t0

        for r in res:
            if r.masks is None or r.masks.data.shape[0] == 0:
                out.append(torch.zeros((0, target_h, target_w), dtype=torch.uint8))
                continue
            m = r.masks.data.detach().to("cpu", dtype=torch.float32)
            m = F.interpolate(m.unsqueeze(1), size=(target_h, target_w),
                              mode="nearest").squeeze(1)
            m = (m > 0.5).to(torch.uint8)
            areas = m.float().sum(dim=(1, 2))
            m = m[torch.argsort(areas, descending=True)][:m_prime]
            m = m[m.float().sum(dim=(1, 2)) > 0]
            out.append(m.contiguous())

        if progress:
            n_batches = (len(paths) + batch - 1) // batch
            # Report every batch on short sequences; every tenth on long ones,
            # so a 60-frame run does not look stuck after one line.
            every = 1 if n_batches <= 20 else 10
            if (i // max(1, batch)) % every == 0 or i + batch >= len(paths):
                print(f"  [seg] {min(i + batch, len(paths))}/{len(paths)}", flush=True)

    pool.shutdown()
    elapsed = time.perf_counter() - t_all
    timing = dict(total_s=elapsed, decode_s=t_dec, fastsam_s=t_inf,
                  frames=len(paths), batch=batch,
                  ms_per_frame=elapsed / max(1, len(paths)) * 1000)
    return out, timing
