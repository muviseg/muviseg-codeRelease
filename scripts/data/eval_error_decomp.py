"""
One-shot error decomposition on validation set using a saved checkpoint.

Usage (multi-GPU):
  accelerate launch -m training.scripts.eval_error_decomp \
      --config training/config/segvggt_dpt_train.yaml \
      --checkpoint results/segvggt_dpt/best.pth \
      --max-batches 200

Prints per-segment error breakdown:
  false_dustbin — model → dustbin, but GT match exists
  wrong_match   — model → match, but picked wrong segment
  false_match   — model → match, but no GT match exists (should be dustbin)
"""

import sys
from pathlib import Path

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from tqdm import tqdm
from accelerate import Accelerator

from torch.utils.data import DataLoader, Subset
from muviseg.training.utils import load_cfg, pad_masks_to_batch, pad_descriptors_to_batch
from muviseg.data.dataset import ScanNetPPSegDataset, get_collate_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int, default=0, help="0 = full val set")
    args, overrides = parser.parse_known_args()

    cfg = load_cfg(args.config, overrides=overrides)

    accelerator = Accelerator(
        mixed_precision=getattr(getattr(cfg, "ACCELERATE", None), "MIXED_PRECISION", "no"),
    )
    device = accelerator.device

    # Build model
    arch = cfg.MODEL.ARCH
    if arch == "vggt_dpt":
        from muviseg.models.vggt_dpt_lg import (
            SegVGGTDPT, compute_matching_metrics_lg,
        )
        lg = cfg.MODEL.LG
        lg_v2 = cfg.MODEL.LG_V2
        layer_indices = tuple(getattr(cfg.MODEL, "VGGT_LAYER_INDICES", (5, 11, 17, 23)))
        fusion_dim = getattr(cfg.MODEL, "VGGT_FUSION_DIM", 256)
        model = SegVGGTDPT(
            vggt_ckpt=cfg.MODEL.VGGT_CKPT,
            layer_indices=layer_indices,
            fusion_dim=fusion_dim,
            proj_dim=lg.PROJ_DIM,
            n_layers=lg.N_LAYERS,
            n_heads=lg.N_HEADS,
            ffn_expansion=lg_v2.FFN_EXPANSION,
            use_grad_checkpoint=lg.GRAD_CHECKPOINT,
            deep_supervision=lg.DEEP_SUPERVISION,
            device="cpu",
        )

        def metrics_fn(output, seg_corr, masks0, masks1):
            return compute_matching_metrics_lg(output[0], output[1],
                                               seg_corr, masks0, masks1)
    else:
        raise ValueError(f"Unsupported arch for error decomp: {arch}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    if accelerator.is_main_process:
        print(f"Loaded checkpoint: step={ckpt.get('global_step')}, "
              f"epoch={ckpt.get('epoch')}, best_val_ma={ckpt.get('best_val_ma', '?')}")

    # Build val loader (same split logic as trainer.py)
    pair_dsc_root = getattr(cfg.DATASET, "PAIR_DSC_ROOT", "") or None
    ds_full = ScanNetPPSegDataset(
        metadata_path=cfg.DATASET.METADATA_PATH,
        processed_root=cfg.DATASET.DATA_ROOT,
        masks_root=cfg.DATASET.SEGDATA_ROOT,
        pairs_root=cfg.DATASET.PAIRS_ROOT,
        target_size=max(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH),
        resize_mode=cfg.DATASET.RESIZE_MODE,
        pair_dsc_root=pair_dsc_root or "",
    )
    n_total = len(ds_full)
    n_val = max(1, int(n_total * cfg.DATASET.VAL_FRACTION))
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    ds_val = Subset(ds_full, indices[:n_val].tolist())
    collate_fn = get_collate_fn(cfg.DATASET.RESIZE_MODE)
    loader_val = DataLoader(
        ds_val, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False,
        num_workers=min(4, cfg.TRAINING.NUM_WORKERS), collate_fn=collate_fn,
        pin_memory=(accelerator.device.type == "cuda"),
    )
    if accelerator.is_main_process:
        print(f"Val set: {len(ds_val):,} pairs")

    model, loader_val = accelerator.prepare(model, loader_val)
    model.eval()

    unwrapped = accelerator.unwrap_model(model)

    # Per-rank counters (integers, not lists — for precise gather)
    keys = [
        "total_correct", "total_correct_top5", "total_gt", "total_segs",
        "total_unmatch", "total_false_dustbin", "total_wrong_match", "total_false_match",
    ]
    counters = {k: 0 for k in keys}
    total_logprob = 0.0
    all_scores = []
    all_labels = []
    n_batches = 0

    max_b = args.max_batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(
            loader_val, desc="Eval", leave=False,
            disable=not accelerator.is_local_main_process,
            total=max_b if max_b > 0 else None,
        )):
            if max_b > 0 and batch_idx >= max_b:
                break

            if "dsc0" in batch:
                dsc0 = pad_descriptors_to_batch(batch["dsc0"], device)
                dsc1 = pad_descriptors_to_batch(batch["dsc1"], device)
                output = unwrapped(None, None, dsc0_pre=dsc0, dsc1_pre=dsc1)
            else:
                img0 = batch["img0"].to(device)
                img1 = batch["img1"].to(device)
                masks0 = pad_masks_to_batch(batch["masks0"], device)
                masks1 = pad_masks_to_batch(batch["masks1"], device)
                output = unwrapped(img0, img1, masks0, masks1)

            # Raw per-batch decomposition
            log_mutual = output[0]
            match0 = output[1]
            seg_corr_list = batch["seg_corr"]
            masks0_list = batch["masks0"]
            masks1_list = batch["masks1"]

            for b in range(log_mutual.shape[0]):
                M = masks0_list[b].shape[0]
                N = masks1_list[b].shape[0]
                corr = seg_corr_list[b]
                lm = log_mutual[b, :M, :N]
                m0_b = match0[b, :M]

                gt_map = {}
                if corr.shape[0] > 0:
                    pred_j = lm.argmax(dim=1)
                    _, topk_j = lm.topk(min(5, N), dim=1)

                    for k in range(corr.shape[0]):
                        i0, j0 = corr[k, 0].item(), corr[k, 1].item()
                        if i0 < M and j0 < N:
                            counters["total_correct"] += int(pred_j[i0].item() == j0)
                            counters["total_correct_top5"] += int(
                                j0 in topk_j[i0].tolist()
                            )
                            counters["total_gt"] += 1
                            total_logprob += lm[i0, j0].item()
                            gt_map[i0] = j0

                    for i0, j0 in gt_map.items():
                        row = lm[i0]
                        for j in range(N):
                            all_scores.append(row[j].item())
                            all_labels.append(1.0 if j == j0 else 0.0)

                for i in range(M):
                    predicted_dustbin = m0_b[i].item() < 0
                    has_gt = i in gt_map
                    if has_gt and predicted_dustbin:
                        counters["total_false_dustbin"] += 1
                    elif has_gt and not predicted_dustbin:
                        pred = lm.argmax(dim=1)[i].item() if corr.shape[0] > 0 else -1
                        if pred != gt_map[i]:
                            counters["total_wrong_match"] += 1
                    elif not has_gt and not predicted_dustbin:
                        counters["total_false_match"] += 1

                counters["total_segs"] += M
                counters["total_unmatch"] += int((m0_b < 0).sum().item())
            n_batches += 1

            # Intermediate print every 50 batches (on main process only)
            if accelerator.is_main_process and n_batches % 10 == 0:
                _tgt = counters["total_gt"]
                _tsegs = counters["total_segs"]
                _ma = counters["total_correct"] / max(_tgt, 1)
                _r5 = counters["total_correct_top5"] / max(_tgt, 1)
                _db = counters["total_unmatch"] / max(_tsegs, 1)
                _fd = counters["total_false_dustbin"] / max(_tsegs, 1)
                _wm = counters["total_wrong_match"] / max(_tsegs, 1)
                _fm = counters["total_false_match"] / max(_tsegs, 1)
                tqdm.write(
                    f"  [{n_batches:4d} batches | {_tsegs} segs] "
                    f"MA={_ma:.3f}  R@5={_r5:.3f}  dust={_db:.3f}  "
                    f"f_dust={_fd:.4f}  w_match={_wm:.4f}  f_match={_fm:.4f}"
                )

    # Gather across GPUs
    accelerator.wait_for_everyone()

    counter_tensor = torch.tensor(
        [counters[k] for k in keys] + [total_logprob, n_batches],
        dtype=torch.float64, device=device,
    )
    gathered = accelerator.gather(counter_tensor)  # (num_procs * 10,)
    counter_tensor = gathered.reshape(accelerator.num_processes, -1).sum(dim=0)

    if accelerator.is_main_process:
        vals = counter_tensor.cpu().tolist()
        tc, tc5, tgt, tsegs, tunm, tfd, twm, tfm, tlp, nb = vals

        ma = tc / max(tgt, 1)
        r5 = tc5 / max(tgt, 1)
        dustbin = tunm / max(tsegs, 1)
        fd_rate = tfd / max(tsegs, 1)
        wm_rate = twm / max(tsegs, 1)
        fm_rate = tfm / max(tsegs, 1)
        mlp = tlp / max(tgt, 1)

        print("\n" + "=" * 60)
        print(f"ERROR DECOMPOSITION  ({int(nb)} batches, {int(tsegs)} segments)")
        print("=" * 60)
        print(f"  matching_accuracy       = {ma:.4f}")
        print(f"  recall_at_5             = {r5:.4f}")
        print(f"  dustbin_rate            = {dustbin:.4f}")
        print(f"  mean_gt_logprob         = {mlp:.4f}")
        print(f"  ---")
        print(f"  false_dustbin_rate      = {fd_rate:.4f}  ({int(tfd)} segs)")
        print(f"  wrong_match_rate        = {wm_rate:.4f}  ({int(twm)} segs)")
        print(f"  false_match_rate        = {fm_rate:.4f}  ({int(tfm)} segs)")
        print("=" * 60)

        total_err = tfd + twm + tfm
        if total_err > 0:
            print(f"\nError budget (% of total errors = {int(total_err)}):")
            print(f"  false_dustbin: {tfd/total_err*100:5.1f}%  "
                  f"← model too conservative (has GT, predicted dustbin)")
            print(f"  wrong_match:   {twm/total_err*100:5.1f}%  "
                  f"← weak discriminability (has GT, matched wrong)")
            print(f"  false_match:   {fm/total_err*100:5.1f}%  "
                  f"← model too aggressive (no GT, predicted match)")

        # GT match fraction for context
        gt_frac = tgt / max(tsegs, 1)
        print(f"\nContext: {gt_frac*100:.1f}% of segments have a GT match "
              f"→ ideal dustbin ≈ {(1-gt_frac)*100:.1f}%")


if __name__ == "__main__":
    main()
