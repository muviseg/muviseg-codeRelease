"""
Visualization utilities for segment matching evaluation.

Shared between eval_replica_table2.py and eval_segvggt_vkitti2.py.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from pathlib import Path


def tensor_to_numpy_image(img_tensor) -> np.ndarray:
    """
    Convert (3, H, W) tensor in [-1, 1] to (H, W, 3) float32 in [0, 1].
    Also accepts (H, W, 3) numpy arrays (returned as-is after clipping).
    """
    if hasattr(img_tensor, "cpu"):
        img = img_tensor.cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = (img + 1.0) / 2.0
    else:
        img = np.asarray(img_tensor, dtype=np.float32)
        if img.max() > 1.0:
            img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def visualize_match_pair(
    img0,
    img1,
    masks0,
    masks1,
    instance_ids0,
    instance_ids1,
    scores: np.ndarray,
    gt_matrix: np.ndarray,
    save_path,
    num_examples: int = 5,
    title: str = "",
):
    """
    Visualise GT vs predicted segment matches for one image pair.

    Layout per row (5 columns):
      [Image 0] | [Query mask (yellow)] | [GT match (green)] | [Predicted (green/red)] | [Info]

    Args:
        img0, img1  : (3,H,W) tensor in [-1,1]  OR  (H,W,3) uint8 numpy
        masks0      : (M,H,W) tensor or numpy — binary masks for img0
        masks1      : (N,H,W) tensor or numpy — binary masks for img1
        instance_ids0, instance_ids1 : list[int]
        scores      : (M,N) numpy float — model output (higher = better match)
        gt_matrix   : (M,N) numpy uint8/float — 1 where GT match exists
        save_path   : path to write PNG
        num_examples: max number of GT pairs to show
        title       : optional suptitle string
    """
    img0_np = tensor_to_numpy_image(img0)
    img1_np = tensor_to_numpy_image(img1)

    # Convert masks to numpy (H,W) bool arrays
    def to_np(m):
        if hasattr(m, "cpu"):
            return m.cpu().numpy().astype(bool)
        return np.asarray(m, dtype=bool)

    masks0_np = [to_np(masks0[i]) for i in range(len(masks0))]
    masks1_np = [to_np(masks1[j]) for j in range(len(masks1))]

    M, N = len(masks0_np), len(masks1_np)

    # Collect GT pairs with predicted match info
    gt_pairs = []
    for i in range(M):
        for j in range(N):
            if gt_matrix[i, j] == 1:
                pred_j = int(np.argmax(scores[i]))
                gt_pairs.append({
                    "query_idx": i,
                    "gt_idx": j,
                    "pred_idx": pred_j,
                    "instance_id": instance_ids0[i] if i < len(instance_ids0) else -1,
                    "gt_score": float(scores[i, j]),
                    "pred_score": float(scores[i, pred_j]),
                    "is_correct": pred_j == j,
                })

    if not gt_pairs:
        return  # nothing to visualise

    # Sort: correct matches first, then by GT score descending
    gt_pairs.sort(key=lambda x: (not x["is_correct"], -x["gt_score"]))
    rows = gt_pairs[:num_examples]
    n_rows = len(rows)

    fig, axes = plt.subplots(n_rows, 5, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    _Y = np.array([1.0, 1.0, 0.0])   # yellow
    _G = np.array([0.0, 1.0, 0.0])   # green
    _R = np.array([1.0, 0.0, 0.0])   # red

    for row, p in enumerate(rows):
        q = p["query_idx"]
        gj = p["gt_idx"]
        pj = p["pred_idx"]
        correct = p["is_correct"]

        # Col 0: raw image 0
        axes[row, 0].imshow(img0_np)
        axes[row, 0].set_title(f"Image 0\n(query {q})", fontsize=9)
        axes[row, 0].axis("off")

        # Col 1: query mask (yellow)
        hi = img0_np.copy()
        hi[masks0_np[q]] = hi[masks0_np[q]] * 0.4 + _Y * 0.6
        axes[row, 1].imshow(hi)
        axes[row, 1].set_title(f"Query mask\nID={p['instance_id']}", fontsize=9)
        axes[row, 1].axis("off")

        # Col 2: GT match (green)
        hi = img1_np.copy()
        hi[masks1_np[gj]] = hi[masks1_np[gj]] * 0.4 + _G * 0.6
        axes[row, 2].imshow(hi)
        axes[row, 2].set_title(f"GT match\ntgt={gj}  s={p['gt_score']:.3f}",
                               fontsize=9, color="green")
        axes[row, 2].axis("off")

        # Col 3: predicted match (green if correct, red if wrong)
        hi = img1_np.copy()
        col = _G if correct else _R
        hi[masks1_np[pj]] = hi[masks1_np[pj]] * 0.4 + col * 0.6
        status = "CORRECT" if correct else "WRONG"
        axes[row, 3].imshow(hi)
        axes[row, 3].set_title(f"Predicted ({status})\ntgt={pj}  s={p['pred_score']:.3f}",
                               fontsize=9, color="green" if correct else "red")
        axes[row, 3].axis("off")

        # Col 4: text info
        txt = (
            f"query:  {q}\n"
            f"ID:     {p['instance_id']}\n\n"
            f"GT tgt: {gj}\n"
            f"GT s:   {p['gt_score']:.3f}\n\n"
            f"Pred:   {pj}\n"
            f"Pred s: {p['pred_score']:.3f}\n\n"
            + ("✓ correct" if correct else "✗ wrong")
        )
        bg = "lightgreen" if correct else "lightcoral"
        axes[row, 4].text(0.05, 0.5, txt, fontsize=8, va="center",
                          fontfamily="monospace",
                          bbox=dict(boxstyle="round", facecolor=bg, alpha=0.5))
        axes[row, 4].set_xlim(0, 1)
        axes[row, 4].set_ylim(0, 1)
        axes[row, 4].axis("off")

    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _mask_overlay(
    img_np: np.ndarray,
    masks_np: list,
    alpha: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Overlay all masks onto img with distinct random colors."""
    if not masks_np:
        return img_np
    out = img_np.copy()
    rng = np.random.RandomState(seed)
    for m in masks_np:
        color = rng.rand(3)
        out[m] = out[m] * (1 - alpha) + color * alpha
    return np.clip(out, 0.0, 1.0)


def visualize_match_tuple(
    images,
    masks,
    instance_ids,
    scores: np.ndarray,
    gt_matrix: np.ndarray,
    save_path,
    num_examples: int = 5,
    query_pair: tuple = (0, 1),
    title: str = "",
):
    """
    Multi-frame visualisation for joint model eval.

    Layout:
      Row 0: N frames in a horizontal strip, each with all masks overlaid
             in distinct random colors. Query frames (q0, q1) labelled in
             the axis title; context frames labelled ctx_k.
      Rows 1..1+num_examples: query-pair match examples (same 5-column
             layout as visualize_match_pair).

    Args:
        images       : (N, 3, H, W) tensor  OR  list[N] of (3, H, W) tensor/ndarray
        masks        : list[N] of (M_v, H, W)
        instance_ids : list[N] of list[int]
        scores       : (M_0, M_1) numpy for query pair (0, 1)
        gt_matrix    : (M_0, M_1) numpy
        save_path    : output PNG path
        num_examples : max GT-pair rows to show
        query_pair   : indices of the query frames in the tuple (default (0, 1))
        title        : optional suptitle
    """
    # Normalise images input → list[N] of (H, W, 3) numpy in [0, 1]
    if hasattr(images, "ndim") and getattr(images, "ndim", 0) == 4:
        images_list = [images[v] for v in range(images.shape[0])]
    else:
        images_list = list(images)
    images_np = [tensor_to_numpy_image(im) for im in images_list]
    N = len(images_np)

    # Convert per-frame masks to numpy bool
    def to_np(m):
        if hasattr(m, "cpu"):
            return m.cpu().numpy().astype(bool)
        return np.asarray(m, dtype=bool)
    masks_np_per_view = [[to_np(masks[v][i]) for i in range(len(masks[v]))] for v in range(N)]

    q0, q1 = query_pair
    masks0_np = masks_np_per_view[q0]
    masks1_np = masks_np_per_view[q1]
    M, N_mask = len(masks0_np), len(masks1_np)
    ids0 = instance_ids[q0] if q0 < len(instance_ids) else []

    gt_pairs = []
    for i in range(M):
        for j in range(N_mask):
            if gt_matrix[i, j] == 1:
                pred_j = int(np.argmax(scores[i]))
                gt_pairs.append({
                    "query_idx": i,
                    "gt_idx": j,
                    "pred_idx": pred_j,
                    "instance_id": ids0[i] if i < len(ids0) else -1,
                    "gt_score": float(scores[i, j]),
                    "pred_score": float(scores[i, pred_j]),
                    "is_correct": pred_j == j,
                })
    gt_pairs.sort(key=lambda x: (not x["is_correct"], -x["gt_score"]))
    rows = gt_pairs[:num_examples]
    n_rows = len(rows)

    # Grid: top row = N overview panels, below = n_rows × 5 match panels.
    n_cols = max(N, 5)
    fig = plt.figure(figsize=(4 * n_cols, 4 * (1 + n_rows)))
    gs = fig.add_gridspec(1 + n_rows, n_cols)

    # Top row: N overview panels
    for v in range(N):
        ax = fig.add_subplot(gs[0, v])
        overlay = _mask_overlay(images_np[v], masks_np_per_view[v], alpha=0.5, seed=v)
        ax.imshow(overlay)
        if v == q0:
            tag = f"q0 (frame {v})"
        elif v == q1:
            tag = f"q1 (frame {v})"
        else:
            tag = f"ctx_{v} (frame {v})"
        ax.set_title(f"{tag}\n{len(masks_np_per_view[v])} masks", fontsize=9)
        ax.axis("off")

    if n_rows == 0:
        if title:
            fig.suptitle(title, fontsize=11)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return

    _Y = np.array([1.0, 1.0, 0.0])
    _G = np.array([0.0, 1.0, 0.0])
    _R = np.array([1.0, 0.0, 0.0])

    img0_np = images_np[q0]
    img1_np = images_np[q1]

    for row, p in enumerate(rows):
        q, gj, pj = p["query_idx"], p["gt_idx"], p["pred_idx"]
        correct = p["is_correct"]

        ax = fig.add_subplot(gs[1 + row, 0])
        ax.imshow(img0_np)
        ax.set_title(f"Image q0\n(query {q})", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(gs[1 + row, 1])
        hi = img0_np.copy()
        hi[masks0_np[q]] = hi[masks0_np[q]] * 0.4 + _Y * 0.6
        ax.imshow(hi)
        ax.set_title(f"Query mask\nID={p['instance_id']}", fontsize=9)
        ax.axis("off")

        ax = fig.add_subplot(gs[1 + row, 2])
        hi = img1_np.copy()
        hi[masks1_np[gj]] = hi[masks1_np[gj]] * 0.4 + _G * 0.6
        ax.imshow(hi)
        ax.set_title(f"GT match\ntgt={gj}  s={p['gt_score']:.3f}",
                    fontsize=9, color="green")
        ax.axis("off")

        ax = fig.add_subplot(gs[1 + row, 3])
        hi = img1_np.copy()
        col = _G if correct else _R
        hi[masks1_np[pj]] = hi[masks1_np[pj]] * 0.4 + col * 0.6
        status = "CORRECT" if correct else "WRONG"
        ax.imshow(hi)
        ax.set_title(f"Predicted ({status})\ntgt={pj}  s={p['pred_score']:.3f}",
                    fontsize=9, color="green" if correct else "red")
        ax.axis("off")

        ax = fig.add_subplot(gs[1 + row, 4])
        txt = (
            f"query:  {q}\n"
            f"ID:     {p['instance_id']}\n\n"
            f"GT tgt: {gj}\n"
            f"GT s:   {p['gt_score']:.3f}\n\n"
            f"Pred:   {pj}\n"
            f"Pred s: {p['pred_score']:.3f}\n\n"
            + ("correct" if correct else "wrong")
        )
        bg = "lightgreen" if correct else "lightcoral"
        ax.text(0.05, 0.5, txt, fontsize=8, va="center", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor=bg, alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def visualize_score_heatmap(
    img0,
    img1,
    scores: np.ndarray,
    gt_matrix: np.ndarray,
    save_path,
    title: str = "",
):
    """
    Side-by-side: images + score matrix heatmap + GT matrix.

    Args:
        img0, img1  : (3,H,W) tensor in [-1,1]
        scores      : (M,N) numpy
        gt_matrix   : (M,N) numpy
        save_path   : output PNG path
    """
    img0_np = tensor_to_numpy_image(img0)
    img1_np = tensor_to_numpy_image(img1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(img0_np)
    axes[0].set_title("Image 0")
    axes[0].axis("off")

    axes[1].imshow(img1_np)
    axes[1].set_title("Image 1")
    axes[1].axis("off")

    im = axes[2].imshow(scores, cmap="hot", aspect="auto", vmin=0, vmax=scores.max())
    plt.colorbar(im, ax=axes[2], label="Score")
    axes[2].set_title(f"Score matrix ({scores.shape[0]}×{scores.shape[1]})")
    axes[2].set_xlabel("Mask idx img1")
    axes[2].set_ylabel("Mask idx img0")

    axes[3].imshow(gt_matrix, cmap="Greys", aspect="auto", vmin=0, vmax=1)
    axes[3].set_title("GT matrix")
    axes[3].set_xlabel("Mask idx img1")
    axes[3].set_ylabel("Mask idx img0")

    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
