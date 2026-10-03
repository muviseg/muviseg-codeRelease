import torch
from tqdm import tqdm

from muviseg.training.utils import pad_masks_to_batch, pad_descriptors_to_batch


def _unnorm(t: torch.Tensor) -> "np.ndarray":
    """(3,H,W) [-1,1] → (H,W,3) uint8"""
    import numpy as np
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _random_colors(n: int, seed: int = 0):
    import numpy as np
    rng = np.random.RandomState(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    from matplotlib.colors import hsv_to_rgb
    return [hsv_to_rgb([h, 0.85, 0.95]) for h in hues]


def make_match_vis(img0: torch.Tensor,    # (3,H,W)
                   img1: torch.Tensor,    # (3,H,W)
                   masks0: torch.Tensor,  # (M,H,W)
                   masks1: torch.Tensor,  # (N,H,W)
                   log_P: torch.Tensor,   # (M+1,N+1)
                   seg_corr: torch.Tensor,# (K,2)
                   max_show: int = 8) -> "np.ndarray":
    """
    Returns an HWC uint8 image suitable for writer.add_image().

    Layout:  [ img0 + GT overlays | img1 + predicted overlays ]
    Colors:  same color = same GT correspondence
    Green outline on predicted match = correct
    Red outline on predicted match   = wrong
    """
    import numpy as np

    img0_np = _unnorm(img0).astype(np.float32) / 255.0
    img1_np = _unnorm(img1).astype(np.float32) / 255.0
    H0, W0  = img0_np.shape[:2]
    H1, W1  = img1_np.shape[:2]

    M  = masks0.shape[0]
    N  = masks1.shape[0]
    K  = min(seg_corr.shape[0], max_show)

    if K == 0:
        # No correspondences — just return side-by-side originals
        H  = max(H0, H1)
        c0 = np.pad(img0_np, ((0, H - H0), (0, 0), (0, 0)))
        c1 = np.pad(img1_np, ((0, H - H1), (0, 0), (0, 0)))
        canvas = np.concatenate([c0, c1], axis=1)
        return (canvas * 255).clip(0, 255).astype(np.uint8)

    colors = _random_colors(K)
    ov0 = img0_np.copy()
    ov1 = img1_np.copy()
    alpha = 0.5

    # Predicted matches: row-wise argmax on log_P (excluding dustbin)
    lP_crop  = log_P[:M, :N]          # (M, N)
    pred_j   = lP_crop.argmax(dim=1)  # (M,)

    for k in range(K):
        i0, j0 = seg_corr[k, 0].item(), seg_corr[k, 1].item()
        if i0 >= M or j0 >= N:
            continue
        c = np.array(colors[k])

        # GT overlay on img0 (always show GT mask)
        m0 = masks0[i0].bool().cpu().numpy()
        if H0 == ov0.shape[0]:
            ov0[m0] = ov0[m0] * (1 - alpha) + c * alpha

        # Predicted overlay on img1
        pred  = pred_j[i0].item()
        m1_pr = masks1[pred].bool().cpu().numpy() if pred < N else None

        # Show predicted mask; color = same as GT pair
        if m1_pr is not None and H1 == ov1.shape[0]:
            correct = (pred == j0)
            ov1[m1_pr] = ov1[m1_pr] * (1 - alpha) + c * alpha
            # Add thin border: green if correct, red if wrong
            border_c = np.array([0.0, 0.9, 0.0]) if correct else np.array([0.9, 0.0, 0.0])
            import scipy.ndimage as ndi
            border = m1_pr & ~ndi.binary_erosion(m1_pr, iterations=3)
            ov1[border] = border_c

    H  = max(H0, H1)
    c0 = np.pad(ov0, ((0, H - H0), (0, 0), (0, 0)))
    c1 = np.pad(ov1, ((0, H - H1), (0, 0), (0, 0)))

    # White divider
    divider = np.ones((H, 3, 3), dtype=np.float32)
    canvas  = np.concatenate([c0, divider, c1], axis=1)
    return (canvas * 255).clip(0, 255).astype(np.uint8)


@torch.no_grad()
def run_validation(model, loader_val, device, accelerator,
                   loss_fn, metrics_fn, score_mat_fn,
                   writer=None, global_step=0,
                   n_vis_batches: int = 4,
                   vis_img_size: int = 512,
                   vis_resize_mode: str = "square",
                   max_batches: int = 0) -> dict:
    """
    Full validation pass → aggregated metrics dict (gathered across all processes).
    Saves n_vis_batches visualizations to TensorBoard (main process only).

    IMPORTANT: uses accelerator.unwrap_model() to bypass DDP wrapper.
    DDP forward hooks expect a backward pass; calling DDP-wrapped model in
    eval/no_grad corrupts its internal state and causes NCCL desync on the
    next training step.

    loss_fn(output, seg_corr, masks0, masks1)  → scalar tensor
    metrics_fn(output, seg_corr, masks0, masks1) → dict
    score_mat_fn(output, b, M, N) → (M, N) cpu tensor for vis
    """
    # Unwrap DDP — forward passes during validation must NOT go through
    # the DDP wrapper to avoid registering gradient hooks that never fire.
    unwrapped = accelerator.unwrap_model(model)

    agg = {
        "loss": [], "matching_accuracy": [], "mean_gt_logprob": [], "dustbin_rate": [],
        "recall_at_1": [], "recall_at_5": [], "auprc": [],
        "false_dustbin_rate": [], "wrong_match_rate": [], "false_match_rate": [],
    }
    vis_count = 0

    for batch_idx, batch in enumerate(tqdm(
        loader_val, desc="  Val", leave=False, dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
        total=max_batches if max_batches > 0 else None,
    )):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        if "dsc0" in batch:
            dsc0   = pad_descriptors_to_batch(batch["dsc0"], device)
            dsc1   = pad_descriptors_to_batch(batch["dsc1"], device)
            output = unwrapped(None, None, dsc0_pre=dsc0, dsc1_pre=dsc1)
        else:
            img0   = batch["img0"].to(device)
            img1   = batch["img1"].to(device)
            masks0 = pad_masks_to_batch(batch["masks0"], device)
            masks1 = pad_masks_to_batch(batch["masks1"], device)
            output = unwrapped(img0, img1, masks0, masks1)

        loss = loss_fn(output, batch["seg_corr"], batch["masks0"], batch["masks1"])
        m    = metrics_fn(output, batch["seg_corr"], batch["masks0"], batch["masks1"])
        agg["loss"].append(loss.item())
        for k in m:
            agg[k].append(m[k])

        # ── Visualize first sample from first n_vis_batches (main process only) ──
        if accelerator.is_main_process and writer is not None and vis_count < n_vis_batches:
            b = 0
            if "img0" in batch:
                img0_vis = batch["img0"][b]
                img1_vis = batch["img1"][b]
            elif "img_path_i" in batch and batch["img_path_i"][b]:
                # Precomputed mode: lazy-load images only for visualization.
                from muviseg.data.dataset import _load_image
                from pathlib import Path as _Path
                img0_vis = _load_image(_Path(batch["img_path_i"][b]),
                                       vis_img_size, vis_resize_mode)
                img1_vis = _load_image(_Path(batch["img_path_j"][b]),
                                       vis_img_size, vis_resize_mode)
            else:
                img0_vis = img1_vis = None

            if img0_vis is not None:
                M_b = batch["masks0"][b].shape[0]
                N_b = batch["masks1"][b].shape[0]

                # In precomputed mode masks are dummy (M,1,1) stubs.
                # Lazy-load full masks only for the few vis batches.
                if "mask_path_i" in batch and batch["mask_path_i"][b]:
                    from muviseg.data.dataset import (
                        _decode_rles_batched, _resize_masks, MAX_MASKS,
                    )
                    import pickle as _pkl
                    with open(batch["mask_path_i"][b], "rb") as _f:
                        _rles0 = _pkl.load(_f)["mask_coco_rles_resized"]
                    with open(batch["mask_path_j"][b], "rb") as _f:
                        _rles1 = _pkl.load(_f)["mask_coco_rles_resized"]
                    _H, _W = img0_vis.shape[-2:]
                    masks0_vis = _decode_rles_batched(_rles0[:MAX_MASKS])
                    if masks0_vis.shape[-2:] != (_H, _W):
                        masks0_vis = _resize_masks(masks0_vis, (_H, _W))
                    masks1_vis = _decode_rles_batched(_rles1[:MAX_MASKS])
                    if masks1_vis.shape[-2:] != (_H, _W):
                        masks1_vis = _resize_masks(masks1_vis, (_H, _W))
                else:
                    masks0_vis = batch["masks0"][b]
                    masks1_vis = batch["masks1"][b]

                vis = make_match_vis(
                    img0=img0_vis,
                    img1=img1_vis,
                    masks0=masks0_vis,
                    masks1=masks1_vis,
                    log_P=score_mat_fn(output, b, M_b, N_b),
                    seg_corr=batch["seg_corr"][b],
                    max_show=8,
                )
                writer.add_image(
                    f"val/match_vis_{vis_count}",
                    torch.from_numpy(vis).permute(2, 0, 1),  # → (3,H,W)
                    global_step=global_step,
                )
            vis_count += 1

    # ── Gather metrics across all processes ─────────────────────────────────
    # accelerator.gather is a collective op — all processes must call it.
    # With 1 GPU it's a no-op. With N GPUs we average the per-process means.
    means = {}
    for k, v in agg.items():
        local_mean = torch.tensor(
            sum(v) / max(len(v), 1), device=accelerator.device
        )
        gathered = accelerator.gather(local_mean.unsqueeze(0))   # (num_processes,)
        means[k] = gathered.mean().item()

    return means
