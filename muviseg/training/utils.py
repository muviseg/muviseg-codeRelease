import importlib.util
import math
from pathlib import Path

import torch


def pad_masks_to_batch(masks_list: list, device) -> torch.Tensor:
    """Pad variable-M masks → (B, M_max, H, W) float."""
    M_max = max(m.shape[0] for m in masks_list)
    H, W  = masks_list[0].shape[-2:]
    out = []
    for m in masks_list:
        m = m.to(device)
        pad = M_max - m.shape[0]
        if pad > 0:
            m = torch.cat([m, torch.zeros(pad, H, W, dtype=m.dtype, device=m.device)], 0)
        out.append(m)
    return torch.stack(out).float()


def pad_descriptors_to_batch(dsc_list: list, device) -> torch.Tensor:
    """Pad variable-M precomputed descriptors → (B, M_max, D) float."""
    M_max = max(d.shape[0] for d in dsc_list)
    D     = dsc_list[0].shape[1]
    out   = []
    for d in dsc_list:
        d   = d.to(device).float()
        pad = M_max - d.shape[0]
        if pad > 0:
            d = torch.cat([d, torch.zeros(pad, D, dtype=d.dtype, device=d.device)], 0)
        out.append(d)
    return torch.stack(out)   # (B, M_max, D)


def load_cfg(config_path=None, overrides=None):
    """
    Load config from default.py, optionally merge YAML, then apply CLI overrides.

    CLI overrides use YACS merge_from_list format:
        KEY VALUE [KEY VALUE ...]
    Examples:
        DEBUG true
        TRAINING.BATCH_SIZE 32
        TRAINING.LR 0.001
    """
    # Load our own default — not the original project's configs/default.py
    _here = Path(__file__).resolve().parent.parent / "config"
    _candidates = [
        _here / "default.py",
    ]
    spec = None
    for p in _candidates:
        if p.exists():
            spec = importlib.util.spec_from_file_location("segmast3r_default", p)
            break
    if spec is None:
        raise FileNotFoundError(
            f"default.py not found. Looked in: {_candidates}"
        )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.cfg

    cfg.defrost()
    if config_path:
        cfg.merge_from_file(config_path)
    else:
        # --mock: small/fast overrides, no mixed precision needed
        cfg.SAVE_DIR = "results/mock_run"
        cfg.TRAINING.BATCH_SIZE    = 4
        cfg.TRAINING.NUM_WORKERS   = 0
        cfg.TRAINING.EPOCHS        = 2
        cfg.TRAINING.WARMUP_STEPS  = 50
        cfg.TRAINING.LOG_INTERVAL  = 5
        cfg.TRAINING.VAL_INTERVAL  = 20
        cfg.TRAINING.SAVE_INTERVAL = 50
        cfg.ACCELERATE.MIXED_PRECISION = "no"
    if overrides:
        # YACS can't coerce "true"→bool or "32"→int from CLI strings.
        # Pre-cast values to match the types already in the config.
        typed = []
        for i in range(0, len(overrides), 2):
            key, val = overrides[i], overrides[i + 1]
            cur = cfg
            for part in key.split("."):
                cur = getattr(cur, part)
            if isinstance(cur, bool):
                val = val.lower() in ("true", "1", "yes")
            elif isinstance(cur, int):
                val = int(val)
            elif isinstance(cur, float):
                val = float(val)
            typed.extend([key, val])
        cfg.merge_from_list(typed)
    cfg.freeze()
    return cfg


def compute_matching_metrics(log_P: torch.Tensor,
                              seg_corr_list: list,
                              masks0_list: list,
                              masks1_list: list) -> dict:
    """
    Matching quality metrics from Sinkhorn log-prob matrix.

    matching_accuracy — fraction of GT pairs where row-wise argmax
                        (excluding dustbin col) picks the correct target segment.

    mean_gt_logprob   — mean log_P[i,j] for GT pairs.

    dustbin_rate      — fraction of img0 segments whose argmax (including dustbin)
                        lands on the dustbin column.

    recall_at_1       — same as matching_accuracy (top-1 recall among GT pairs).

    recall_at_5       — fraction of GT pairs where correct target is in top-5
                        predictions (excluding dustbin).

    auprc             — Area Under Precision-Recall Curve. For each img0 segment,
                        score = P(matched to correct j | not dustbin). Treats GT pairs
                        as positives, all other (i, j') as negatives.
    """
    total_correct = 0
    total_correct_top5 = 0
    total_gt      = 0
    total_logprob = 0.0
    total_segs    = 0
    total_dustbin = 0

    # For AUPRC: collect (score_for_gt_pair, score_for_all_non_gt) per GT pair
    all_scores = []   # list of (gt_score, is_positive)
    all_labels = []

    for b in range(log_P.shape[0]):
        M    = masks0_list[b].shape[0]
        N    = masks1_list[b].shape[0]
        corr = seg_corr_list[b]           # (K, 2)
        lP   = log_P[b, :M, :N+1]        # (M, N+1) — last col = dustbin

        if corr.shape[0] > 0:
            # Scores excluding dustbin column
            scores_no_dust = lP[:, :N]                  # (M, N)
            pred_j = scores_no_dust.argmax(dim=1)       # (M,)
            _, topk_j = scores_no_dust.topk(min(5, N), dim=1)  # (M, min(5,N))

            gt_i_set = set()
            for k in range(corr.shape[0]):
                i0, j0 = corr[k, 0].item(), corr[k, 1].item()
                if i0 < M and j0 < N:
                    total_correct += int(pred_j[i0].item() == j0)
                    total_correct_top5 += int(j0 in topk_j[i0].tolist())
                    total_gt      += 1
                    total_logprob += log_P[b, i0, j0].item()
                    gt_i_set.add(i0)

            # AUPRC: for each row i that has a GT match, collect scores
            # Build GT lookup: i → j
            gt_map = {}
            for k in range(corr.shape[0]):
                i0, j0 = corr[k, 0].item(), corr[k, 1].item()
                if i0 < M and j0 < N:
                    gt_map[i0] = j0

            for i0, j0 in gt_map.items():
                row_scores = scores_no_dust[i0]  # (N,)
                for j in range(N):
                    all_scores.append(row_scores[j].item())
                    all_labels.append(1.0 if j == j0 else 0.0)

        pred_full      = lP.argmax(dim=1)   # argmax over all N+1 cols
        total_segs    += M
        total_dustbin += (pred_full == N).sum().item()

    # Compute AUPRC
    auprc = 0.0
    if all_scores:
        scores_t = torch.tensor(all_scores)
        labels_t = torch.tensor(all_labels)
        auprc = _compute_auprc(scores_t, labels_t)

    return {
        "matching_accuracy": total_correct / max(total_gt,   1),
        "recall_at_1":       total_correct / max(total_gt,   1),
        "recall_at_5":       total_correct_top5 / max(total_gt, 1),
        "auprc":             auprc,
        "mean_gt_logprob":   total_logprob / max(total_gt,   1),
        "dustbin_rate":      total_dustbin / max(total_segs, 1),
    }


def _compute_auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute Area Under Precision-Recall Curve via trapezoidal rule."""
    sorted_idx = scores.argsort(descending=True)
    labels_sorted = labels[sorted_idx]

    tp = labels_sorted.cumsum(0)
    total_pos = labels_sorted.sum().item()
    if total_pos == 0:
        return 0.0

    precision = tp / torch.arange(1, len(labels_sorted) + 1, dtype=torch.float32)
    recall = tp / total_pos

    # Prepend (recall=0, precision=1) for proper AUC
    recall = torch.cat([torch.zeros(1), recall])
    precision = torch.cat([torch.ones(1), precision])

    # Trapezoidal AUC
    dr = recall[1:] - recall[:-1]
    return (dr * precision[1:]).sum().item()
