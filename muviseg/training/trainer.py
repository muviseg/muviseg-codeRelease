import math
import time
from pathlib import Path

import torch
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+: ~1.8x faster matmul
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from muviseg.paths import setup_segmast3r_path
setup_segmast3r_path()

from muviseg.training.utils import (
    pad_masks_to_batch,
    pad_descriptors_to_batch,
    compute_matching_metrics,
)
from muviseg.training.validator import run_validation


def _seed_everything(seed: int) -> "torch.Generator":
    """Seed every RNG the training loop draws from, and return a loader generator.

    Covers what the original code left unseeded: weight initialisation, batch
    order, and -- via the worker_init_fn below -- the `random` module that
    ScanNetPPSegDataset/ScanNetPPTupleDataset use inside __getitem__.

    TF32 stays enabled and cuDNN autotuning is untouched, so results are
    repeatable on the same hardware and software stack but not necessarily
    across different ones.
    """
    import random as _random

    import numpy as _np

    _random.seed(seed)
    _np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def _worker_init_fn(worker_id: int) -> None:
    """Give each dataloader worker a distinct, reproducible RNG state."""
    import random as _random

    import numpy as _np

    base = torch.initial_seed() % (2 ** 31)
    _random.seed(base + worker_id)
    _np.random.seed((base + worker_id) % (2 ** 32))


def build_lr_scheduler(optimizer, cfg, total_steps: int):
    warmup = cfg.TRAINING.WARMUP_STEPS
    sched  = cfg.TRAINING.LR_SCHEDULER

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        if sched == "cosine":
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_step(model, batch, device, loss_fn):
    if "dsc0" in batch:
        # Precomputed mode: skip backbone + pooling
        dsc0   = pad_descriptors_to_batch(batch["dsc0"], device)  # (B, M_max, 24)
        dsc1   = pad_descriptors_to_batch(batch["dsc1"], device)
        output = model(None, None, dsc0_pre=dsc0, dsc1_pre=dsc1)
    else:
        img0   = batch["img0"].to(device)
        img1   = batch["img1"].to(device)
        masks0 = pad_masks_to_batch(batch["masks0"], device)
        masks1 = pad_masks_to_batch(batch["masks1"], device)
        output = model(img0, img1, masks0, masks1)
    loss = loss_fn(output, batch["seg_corr"], batch["masks0"], batch["masks1"])
    # output[0]  = score matrix (log_P for sinkhorn, log_mutual for LG)
    # output[-2] = dsc0,  output[-1] = dsc1
    return loss, output


def train_step_multiframe(model, batch, device, loss_fn):
    """Train step for multi-frame SegVGGT (N-frame tuples)."""
    images = batch["images"].to(device)  # (B, N, 3, H, W)
    pair_indices = batch["pair_indices"]  # list of (a, b)

    # Pad masks per view: list[N] of (B, M_max_v, H, W)
    B, N = images.shape[:2]
    masks_padded = []
    for v in range(N):
        v_masks = [batch["masks_list"][b][v] for b in range(B)]
        masks_padded.append(pad_masks_to_batch(v_masks, device))

    output = model.forward_multiframe(images, masks_padded, pair_indices)
    loss = loss_fn(output, batch["pair_corrs"], batch["masks_list"])
    return loss, output


def _run_provenance(cfg=None) -> dict:
    """What produced this checkpoint: seed, command line, code version.

    None of this was recorded before, so a released checkpoint could not be
    traced back to the run that made it.
    """
    import subprocess
    import sys

    info = {"argv": sys.argv, "torch": torch.__version__}
    if cfg is not None:
        info["seed"] = int(getattr(cfg.TRAINING, "SEED", 42))
        info["legacy_rng"] = bool(getattr(cfg.TRAINING, "LEGACY_RNG", False))
    try:
        info["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[2]),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        info["git_sha"] = None
    return info


def _save_ckpt(path, model, optimizer, scheduler, epoch, global_step,
               metrics: dict, best_val_ma=None, accelerator=None, cfg=None):
    m = accelerator.unwrap_model(model) if accelerator is not None else model
    payload = {
        "epoch":           epoch,
        "global_step":     global_step,
        "model_state":     m.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics":         metrics,
        "provenance":      _run_provenance(cfg),
    }
    if best_val_ma is not None:
        payload["best_val_ma"] = best_val_ma
    torch.save(payload, path)


def train(cfg, mock=False, resume=None):
    legacy_rng = bool(getattr(cfg.TRAINING, "LEGACY_RNG", False))
    seed = int(getattr(cfg.TRAINING, "SEED", 42))
    loader_generator = None if legacy_rng else _seed_everything(seed)

    # ── Accelerator ───────────────────────────────────────────────
    mixed_prec = getattr(getattr(cfg, "ACCELERATE", None), "MIXED_PRECISION", "no")
    grad_accum = getattr(getattr(cfg, "ACCELERATE", None), "GRADIENT_ACCUMULATION_STEPS", 1)
    accelerator = Accelerator(
        mixed_precision=mixed_prec,
        gradient_accumulation_steps=grad_accum,
        log_with=None,
    )
    device = accelerator.device

    # ── Save dir / TensorBoard ────────────────────────────────────
    save_dir = Path(cfg.SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if accelerator.is_main_process:
        tb_dir = save_dir / "tb"
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"Device: {device}  |  num_processes: {accelerator.num_processes}  "
              f"|  mixed_precision: {mixed_prec}")
        print(f"Tensorboard: tensorboard --logdir {tb_dir}")

    # ── Dataset ───────────────────────────────────────────────────
    pair_dsc_root = None
    if mock:
        if accelerator.is_main_process:
            print("MOCK run — synthetic data, lightweight backbone")
        from muviseg.data.mock import MockDataset, mock_collate
        ds_train   = MockDataset(n=128)
        ds_val     = MockDataset(n=32)
        collate_fn = mock_collate
    else:
        from muviseg.data.dataset import ScanNetPPSegDataset, get_collate_fn
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
        max_pairs = getattr(cfg.DATASET, "MAX_PAIRS", 0)
        if max_pairs > 0 and max_pairs < n_total:
            if accelerator.is_main_process:
                print(f"MAX_PAIRS={max_pairs:,} — subsampling from {n_total:,}")
            n_total = max_pairs
        n_val   = max(1, int(n_total * cfg.DATASET.VAL_FRACTION))
        indices = torch.randperm(len(ds_full), generator=torch.Generator().manual_seed(42))
        indices = indices[:n_total]  # cap to max_pairs (or full dataset)
        ds_val   = Subset(ds_full, indices[:n_val].tolist())
        ds_train = Subset(ds_full, indices[n_val:].tolist())
        collate_fn = get_collate_fn(cfg.DATASET.RESIZE_MODE)
        if accelerator.is_main_process:
            print(f"Train: {len(ds_train):,}  |  Val: {len(ds_val):,}")

    loader_train = DataLoader(
        ds_train,
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAINING.NUM_WORKERS if not mock else 0,
        collate_fn=collate_fn,
        pin_memory=(accelerator.device.type == "cuda"),
        prefetch_factor=cfg.TRAINING.PREFETCH_FACTOR if not mock else None,
        persistent_workers=(cfg.TRAINING.NUM_WORKERS > 0 and not mock),
        generator=loader_generator,
        worker_init_fn=None if legacy_rng else _worker_init_fn,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=cfg.TRAINING.BATCH_SIZE,
        shuffle=False,
        num_workers=min(4, cfg.TRAINING.NUM_WORKERS) if not mock else 0,
        collate_fn=collate_fn,
        pin_memory=(accelerator.device.type == "cuda"),
        worker_init_fn=None if legacy_rng else _worker_init_fn,
    )

    # ── Model / loss / metrics ────────────────────────────────────
    arch = getattr(cfg.MODEL, "ARCH", "sinkhorn")

    if not mock and pair_dsc_root and arch not in (
        "sinkhorn", "lightglue", "lightglue_v2", "vggt",
    ):
        raise ValueError(
            f"PAIR_DSC_ROOT is set but ARCH='{arch}' does not support "
            f"precomputed descriptors."
        )
    if not mock and arch == "vggt_dpt" and pair_dsc_root:
        raise ValueError(
            "ARCH='vggt_dpt' is online-only (DPT fusion needs spatial features "
            "before pooling). Set PAIR_DSC_ROOT='' or use ARCH='vggt'."
        )

    matcher_cfg = {
        "TYPE": cfg.FEATURE_MATCHER.TYPE,
        "SINKHORN": {
            "NUM_IT":             cfg.FEATURE_MATCHER.SINKHORN.NUM_IT,
            "DUSTBIN_SCORE_INIT": cfg.FEATURE_MATCHER.SINKHORN.DUSTBIN_SCORE_INIT,
        }
    }

    if mock:
        import torch.nn.functional as F

        def _mock_masked_avg_pool(feat, masks):
            """(B, D, H, W), (B, M, H, W) → (B, M, D)"""
            B, D, H, W = feat.shape
            M = masks.shape[1]
            feat_flat  = feat.view(B, D, H * W)         # (B, D, HW)
            masks_flat = masks.view(B, M, H * W).float() # (B, M, HW)
            area = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, M, 1)
            return torch.einsum("bdp,bmp->bmd", feat_flat, masks_flat) / area

        class _MockSinkhornMatcher(torch.nn.Module):
            def __init__(self, num_it=5, dustbin_init=1.0):
                super().__init__()
                self.dustbin_score = torch.nn.Parameter(torch.tensor(dustbin_init))
                self.num_it = num_it

            def forward(self, dsc0, dsc1):
                """(B, M, D), (B, N, D) → (B, M+1, N+1) log-assignment"""
                sim = torch.einsum("bmd,bnd->bmn", dsc0, dsc1)  # (B, M, N)
                B, M, N = sim.shape
                dust = self.dustbin_score.expand(B, 1, 1)
                # augment with dustbin row/col
                sim = torch.cat([sim, dust.expand(B, M, 1)], dim=2)       # (B,M,N+1)
                sim = torch.cat([sim, dust.expand(B, 1, N + 1)], dim=1)   # (B,M+1,N+1)
                for _ in range(self.num_it):
                    sim = sim - torch.logsumexp(sim, dim=2, keepdim=True)
                    sim = sim - torch.logsumexp(sim, dim=1, keepdim=True)
                return sim

        class _MockSegMASt3R(torch.nn.Module):
            def __init__(self, matcher_cfg, desc_dim=24):
                super().__init__()
                self.conv    = torch.nn.Conv2d(3, desc_dim, 3, padding=1)
                self.matcher = _MockSinkhornMatcher(
                    num_it=matcher_cfg["SINKHORN"]["NUM_IT"],
                    dustbin_init=matcher_cfg["SINKHORN"]["DUSTBIN_SCORE_INIT"],
                )
                self.desc_dim = desc_dim

            def extract_desc(self, imgs):
                return self.conv(imgs)  # (B, D, H, W)

            def forward(self, img0, img1, masks0, masks1):
                d0 = self.extract_desc(img0)
                d1 = self.extract_desc(img1)
                dH, dW = d0.shape[-2:]
                if masks0.shape[-2:] != (dH, dW):
                    masks0 = F.interpolate(masks0.float(), (dH, dW), mode="nearest")
                    masks1 = F.interpolate(masks1.float(), (dH, dW), mode="nearest")
                dsc0 = _mock_masked_avg_pool(d0, masks0.float())
                dsc1 = _mock_masked_avg_pool(d1, masks1.float())
                return self.matcher(dsc0, dsc1), dsc0, dsc1

        if arch in ("vggt", "vggt_dpt"):
            # Mock VGGT/VGGT-DPT: LightGlue-style output (log_mutual, match0, match1, dsc0, dsc1)
            class _MockSegVGGT(torch.nn.Module):
                def __init__(self, desc_dim=2048, proj_dim=128):
                    super().__init__()
                    self.conv = torch.nn.Conv2d(3, desc_dim, 3, padding=1)
                    self.proj = torch.nn.Linear(desc_dim, proj_dim)
                    self.match_head = torch.nn.Linear(proj_dim, 1)
                    self.desc_dim = desc_dim

                def forward(self, img0, img1, masks0, masks1,
                            dsc0_pre=None, dsc1_pre=None):
                    if dsc0_pre is not None:
                        dsc0 = dsc0_pre.transpose(1, 2)  # (B, D, M)
                        dsc1 = dsc1_pre.transpose(1, 2)
                    else:
                        d0 = self.conv(img0)
                        d1 = self.conv(img1)
                        dH, dW = d0.shape[-2:]
                        if masks0.shape[-2:] != (dH, dW):
                            masks0 = F.interpolate(
                                masks0.float(), (dH, dW), mode="nearest"
                            )
                            masks1 = F.interpolate(
                                masks1.float(), (dH, dW), mode="nearest"
                            )
                        dsc0 = _mock_masked_avg_pool(d0, masks0.float()).transpose(1, 2)
                        dsc1 = _mock_masked_avg_pool(d1, masks1.float()).transpose(1, 2)
                    # Project + mutual score
                    x0 = self.proj(dsc0.transpose(1, 2))  # (B, M, proj)
                    x1 = self.proj(dsc1.transpose(1, 2))
                    sim = torch.bmm(x0, x1.transpose(1, 2))  # (B, M, N)
                    log_mutual = sim.log_softmax(dim=-1) + sim.log_softmax(dim=-2)
                    match0 = self.match_head(x0).squeeze(-1)
                    match1 = self.match_head(x1).squeeze(-1)
                    return log_mutual, match0, match1, dsc0, dsc1

            model = _MockSegVGGT()

            # LightGlue-style mock loss
            def _mock_lg_loss(log_m, seg_corr, masks0, masks1):
                loss = torch.tensor(0.0, device=log_m.device)
                B = log_m.shape[0]
                for b in range(B):
                    corr = seg_corr[b]
                    if corr.shape[0] == 0:
                        continue
                    loss = loss - log_m[b, corr[:, 0], corr[:, 1]].mean()
                return loss / max(B, 1)

            def loss_fn(output, seg_corr, masks0, masks1):
                return _mock_lg_loss(output[0], seg_corr, masks0, masks1)
            def metrics_fn(output, seg_corr, masks0, masks1):
                return compute_matching_metrics(output[0].cpu(), seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][b, :M, :N].cpu()

        else:
            model = _MockSegMASt3R(matcher_cfg)

            # Mock uses sinkhorn-compatible output: (B,M+1,N+1) log-assignment, dsc0, dsc1
            # Inline NLL loss avoids importing from the (possibly empty) submodule.
            def _mock_nll(log_P, seg_corr, masks0, masks1):
                loss = torch.tensor(0.0, device=log_P.device)
                B = log_P.shape[0]
                for b in range(B):
                    corr = seg_corr[b]
                    if corr.shape[0] == 0:
                        continue
                    loss = loss - log_P[b, corr[:, 0], corr[:, 1]].mean()
                return loss / max(B, 1)

            def loss_fn(output, seg_corr, masks0, masks1):
                return _mock_nll(output[0], seg_corr, masks0, masks1)
            def metrics_fn(output, seg_corr, masks0, masks1):
                return compute_matching_metrics(output[0].cpu(), seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][b, :M, :N].cpu()

    elif arch == "lightglue":
        from muviseg.models.lightglue import (
            SegMASt3RLG, lightglue_loss, lightglue_loss_deep,
            compute_matching_metrics_lg,
        )
        lg = cfg.MODEL.LG
        model = SegMASt3RLG(
            mast3r_ckpt=cfg.MODEL.MAST3R_CKPT,
            proj_dim=lg.PROJ_DIM,
            n_layers=lg.N_LAYERS,
            n_heads=lg.N_HEADS,
            use_grad_checkpoint=lg.GRAD_CHECKPOINT,
            deep_supervision=lg.DEEP_SUPERVISION,
            device="cpu",
        )
        _deep   = lg.DEEP_SUPERVISION
        _lambda = lg.LAMBDA_MATCH
        if _deep:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss_deep(output[0], seg_corr, masks0, masks1,
                                           lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                lm, m0, _ = output[0][-1]   # last layer
                return compute_matching_metrics_lg(lm, m0, seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][-1][0][b, :M, :N].cpu()
        else:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss(output[0], output[1], output[2],
                                      seg_corr, masks0, masks1, lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                return compute_matching_metrics_lg(output[0], output[1],
                                                   seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][b, :M, :N].cpu()

    elif arch == "lightglue_v2":
        from muviseg.models.lightglue_v2 import (
            SegMASt3RLGv2, lightglue_loss, lightglue_loss_deep,
            compute_matching_metrics_lg,
        )
        lg    = cfg.MODEL.LG
        lg_v2 = cfg.MODEL.LG_V2
        model = SegMASt3RLGv2(
            mast3r_ckpt=cfg.MODEL.MAST3R_CKPT,
            proj_dim=lg.PROJ_DIM,
            proj_mid_dim=lg_v2.PROJ_MID_DIM,
            n_layers=lg.N_LAYERS,
            n_heads=lg.N_HEADS,
            ffn_expansion=lg_v2.FFN_EXPANSION,
            use_grad_checkpoint=lg.GRAD_CHECKPOINT,
            deep_supervision=lg.DEEP_SUPERVISION,
            device="cpu",
        )
        _deep   = lg.DEEP_SUPERVISION
        _lambda = lg.LAMBDA_MATCH
        if _deep:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss_deep(output[0], seg_corr, masks0, masks1,
                                           lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                lm, m0, _ = output[0][-1]
                return compute_matching_metrics_lg(lm, m0, seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][-1][0][b, :M, :N].cpu()
        else:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss(output[0], output[1], output[2],
                                      seg_corr, masks0, masks1, lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                return compute_matching_metrics_lg(output[0], output[1],
                                                   seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][b, :M, :N].cpu()

    elif arch == "vggt":
        from muviseg.models.vggt_lightglue import (
            SegVGGT,
            lightglue_loss, lightglue_loss_deep,
            compute_matching_metrics_lg,
        )
        if not pair_dsc_root and accelerator.is_main_process:
            print(
                "WARNING: VGGT online mode is very slow. "
                "Consider running precompute_vggt_features.py first."
            )
        lg = cfg.MODEL.LG
        lg_v2 = cfg.MODEL.LG_V2
        model = SegVGGT(
            vggt_ckpt=cfg.MODEL.VGGT_CKPT,
            layer_idx=cfg.MODEL.VGGT_LAYER_IDX,
            proj_dim=lg.PROJ_DIM,
            proj_mid_dim=lg_v2.PROJ_MID_DIM,
            n_layers=lg.N_LAYERS,
            n_heads=lg.N_HEADS,
            ffn_expansion=lg_v2.FFN_EXPANSION,
            use_grad_checkpoint=lg.GRAD_CHECKPOINT,
            deep_supervision=lg.DEEP_SUPERVISION,
            device="cpu",
        )
        _deep = lg.DEEP_SUPERVISION
        _lambda = lg.LAMBDA_MATCH
        if _deep:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss_deep(output[0], seg_corr, masks0, masks1,
                                           lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                lm, m0, _ = output[0][-1]
                return compute_matching_metrics_lg(lm, m0, seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][-1][0][b, :M, :N].cpu()
        else:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss(output[0], output[1], output[2],
                                      seg_corr, masks0, masks1, lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                return compute_matching_metrics_lg(output[0], output[1],
                                                   seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][b, :M, :N].cpu()

    elif arch == "vggt_dpt":
        from muviseg.models.vggt_dpt_lg import (
            SegVGGTDPT,
            lightglue_loss, lightglue_loss_deep,
            compute_matching_metrics_lg,
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
            matchability_bias=getattr(cfg.MODEL, "MATCHABILITY_BIAS", 0.0),
            temperature_init=getattr(cfg.MODEL, "TEMPERATURE_INIT", 1.0),
            device="cpu",
        )
        _deep = lg.DEEP_SUPERVISION
        _lambda = lg.LAMBDA_MATCH
        if accelerator.is_main_process:
            print(f"SegVGGT-DPT: layers={list(layer_indices)}, "
                  f"fusion_dim={fusion_dim}, proj_dim={lg.PROJ_DIM}")
        if _deep:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss_deep(output[0], seg_corr, masks0, masks1,
                                           lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                lm, m0, _ = output[0][-1]
                return compute_matching_metrics_lg(lm, m0, seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][-1][0][b, :M, :N].cpu()
        else:
            def loss_fn(output, seg_corr, masks0, masks1):
                return lightglue_loss(output[0], output[1], output[2],
                                      seg_corr, masks0, masks1, lambda_match=_lambda)
            def metrics_fn(output, seg_corr, masks0, masks1):
                return compute_matching_metrics_lg(output[0], output[1],
                                                   seg_corr, masks0, masks1)
            def score_mat(output, b, M, N):
                return output[0][b, :M, :N].cpu()

    elif arch == "vggt_dpt_multiframe":
        from muviseg.models.vggt_dpt_lg import (
            SegVGGTDPT,
            lightglue_loss as _lg_loss,
            multiframe_lightglue_loss,
            compute_matching_metrics_lg,
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
            deep_supervision=False,
            matchability_bias=getattr(cfg.MODEL, "MATCHABILITY_BIAS", 0.0),
            temperature_init=getattr(cfg.MODEL, "TEMPERATURE_INIT", 1.0),
            device="cpu",
        )
        _lambda = lg.LAMBDA_MATCH
        _n_frames = getattr(cfg.DATASET, "N_FRAMES", 4)
        if accelerator.is_main_process:
            print(f"SegVGGT-DPT-Multiframe: N={_n_frames}, "
                  f"layers={list(layer_indices)}, "
                  f"fusion_dim={fusion_dim}, proj_dim={lg.PROJ_DIM}")

        # Training loss: multiframe (3-arg signature for train_step_multiframe)
        def loss_fn_train(output, pair_corrs_batch, masks_batch):
            return multiframe_lightglue_loss(
                output, pair_corrs_batch, masks_batch,
                lambda_match=_lambda,
            )
        # Validation uses pairwise forward() → standard 4-arg signature
        def loss_fn(output, seg_corr, masks0, masks1):
            return _lg_loss(output[0], output[1], output[2],
                            seg_corr, masks0, masks1, lambda_match=_lambda)
        def metrics_fn(output, seg_corr, masks0, masks1):
            return compute_matching_metrics_lg(output[0], output[1],
                                               seg_corr, masks0, masks1)
        def score_mat(output, b, M, N):
            return output[0][b, :M, :N].cpu()

        # Override only the training dataset to use tuples;
        # validation stays pairwise (run_validation expects pairwise batches).
        # Reuse ds_full (pairwise, already created above) to avoid re-filtering
        # 3.5M pairs a second time. Build tuple dataset from same filtered pairs.
        if not mock:
            from muviseg.data.dataset import (
                ScanNetPPTupleDataset, collate_fn_tuple,
            )
            # Wrap existing pairwise dataset into a tuple dataset
            # (shares metadata, pairs, img_cache — no re-filtering)
            ds_tuple = ScanNetPPTupleDataset.from_pairwise(
                ds_full,
                n_frames=_n_frames,
                min_pairs_per_tuple=getattr(cfg.DATASET, "MIN_PAIRS_PER_TUPLE", 2),
                max_pairs_per_tuple=getattr(cfg.DATASET, "MAX_PAIRS_PER_TUPLE", 6),
                random_neighbor_prob=getattr(cfg.DATASET, "RANDOM_NEIGHBOR_PROB", 0.2),
            )
            # ds_full and ds_tuple share the same pairs list / length
            # → reuse the indices already computed for ds_full
            n_total = len(ds_full)
            max_pairs = getattr(cfg.DATASET, "MAX_PAIRS", 0)
            if max_pairs > 0 and max_pairs < n_total:
                if accelerator.is_main_process:
                    print(f"MAX_PAIRS={max_pairs:,} — subsampling from {n_total:,}")
                n_total = max_pairs
            n_val = max(1, int(n_total * cfg.DATASET.VAL_FRACTION))
            indices = torch.randperm(len(ds_full), generator=torch.Generator().manual_seed(42))
            indices = indices[:n_total]

            ds_train = Subset(ds_tuple, indices[n_val:].tolist())
            ds_val   = Subset(ds_full,  indices[:n_val].tolist())

            if accelerator.is_main_process:
                print(f"Train (tuples): {len(ds_train):,}  |  "
                      f"Val (pairwise): {len(ds_val):,}")

            # Rebuild training loader with tuple collate
            loader_train = DataLoader(
                ds_train,
                batch_size=cfg.TRAINING.BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.TRAINING.NUM_WORKERS,
                collate_fn=collate_fn_tuple,
                pin_memory=(accelerator.device.type == "cuda"),
                prefetch_factor=cfg.TRAINING.PREFETCH_FACTOR,
                persistent_workers=(cfg.TRAINING.NUM_WORKERS > 0),
            )
            # Val loader stays pairwise (reuses collate_fn from original setup)
            loader_val = DataLoader(
                ds_val,
                batch_size=cfg.TRAINING.BATCH_SIZE,
                shuffle=False,
                num_workers=min(4, cfg.TRAINING.NUM_WORKERS),
                collate_fn=collate_fn,
                pin_memory=(accelerator.device.type == "cuda"),
            )

    elif arch == "vggt_dpt_joint":
        from muviseg.models.vggt_dpt_lg import (
            SegVGGTDPTJoint,
            lightglue_loss as _lg_loss,
            multiframe_lightglue_loss,
            compute_matching_metrics_lg,
        )
        lg = cfg.MODEL.LG
        lg_v2 = cfg.MODEL.LG_V2
        layer_indices = tuple(getattr(cfg.MODEL, "VGGT_LAYER_INDICES", (5, 11, 17, 23)))
        fusion_dim = getattr(cfg.MODEL, "VGGT_FUSION_DIM", 256)
        _n_joint_layers = getattr(cfg.MODEL, "N_JOINT_LAYERS", 3)
        _max_frames = getattr(cfg.MODEL, "MAX_FRAMES", 16)
        model = SegVGGTDPTJoint(
            vggt_ckpt=cfg.MODEL.VGGT_CKPT,
            layer_indices=layer_indices,
            fusion_dim=fusion_dim,
            proj_dim=lg.PROJ_DIM,
            n_joint_layers=_n_joint_layers,
            n_heads=lg.N_HEADS,
            ffn_expansion=lg_v2.FFN_EXPANSION,
            max_frames=_max_frames,
            use_grad_checkpoint=lg.GRAD_CHECKPOINT,
            matchability_bias=getattr(cfg.MODEL, "MATCHABILITY_BIAS", 0.0),
            temperature_init=getattr(cfg.MODEL, "TEMPERATURE_INIT", 1.0),
            device="cpu",
        )
        _lambda = lg.LAMBDA_MATCH
        _n_frames = getattr(cfg.DATASET, "N_FRAMES", 4)
        if accelerator.is_main_process:
            print(f"SegVGGT-DPT-Joint: N={_n_frames}, "
                  f"joint_layers={_n_joint_layers}, max_frames={_max_frames}, "
                  f"layers={list(layer_indices)}, "
                  f"fusion_dim={fusion_dim}, proj_dim={lg.PROJ_DIM}")

        # Training loss: multiframe (3-arg signature for train_step_multiframe)
        def loss_fn_train(output, pair_corrs_batch, masks_batch):
            return multiframe_lightglue_loss(
                output, pair_corrs_batch, masks_batch,
                lambda_match=_lambda,
            )
        # Validation uses pairwise forward() → standard 4-arg signature
        def loss_fn(output, seg_corr, masks0, masks1):
            return _lg_loss(output[0], output[1], output[2],
                            seg_corr, masks0, masks1, lambda_match=_lambda)
        def metrics_fn(output, seg_corr, masks0, masks1):
            return compute_matching_metrics_lg(output[0], output[1],
                                               seg_corr, masks0, masks1)
        def score_mat(output, b, M, N):
            return output[0][b, :M, :N].cpu()

        # Dataset: same tuple train / pairwise val as multiframe
        if not mock:
            from muviseg.data.dataset import (
                ScanNetPPTupleDataset, collate_fn_tuple,
            )
            ds_tuple = ScanNetPPTupleDataset.from_pairwise(
                ds_full,
                n_frames=_n_frames,
                min_pairs_per_tuple=getattr(cfg.DATASET, "MIN_PAIRS_PER_TUPLE", 2),
                max_pairs_per_tuple=getattr(cfg.DATASET, "MAX_PAIRS_PER_TUPLE", 6),
                random_neighbor_prob=getattr(cfg.DATASET, "RANDOM_NEIGHBOR_PROB", 0.2),
            )
            n_total = len(ds_full)
            max_pairs = getattr(cfg.DATASET, "MAX_PAIRS", 0)
            if max_pairs > 0 and max_pairs < n_total:
                if accelerator.is_main_process:
                    print(f"MAX_PAIRS={max_pairs:,} — subsampling from {n_total:,}")
                n_total = max_pairs
            n_val = max(1, int(n_total * cfg.DATASET.VAL_FRACTION))
            indices = torch.randperm(len(ds_full), generator=torch.Generator().manual_seed(42))
            indices = indices[:n_total]

            ds_train = Subset(ds_tuple, indices[n_val:].tolist())
            ds_val   = Subset(ds_full,  indices[:n_val].tolist())

            if accelerator.is_main_process:
                print(f"Train (tuples): {len(ds_train):,}  |  "
                      f"Val (pairwise): {len(ds_val):,}")

            loader_train = DataLoader(
                ds_train,
                batch_size=cfg.TRAINING.BATCH_SIZE,
                shuffle=True,
                num_workers=cfg.TRAINING.NUM_WORKERS,
                collate_fn=collate_fn_tuple,
                pin_memory=(accelerator.device.type == "cuda"),
                prefetch_factor=cfg.TRAINING.PREFETCH_FACTOR,
                persistent_workers=(cfg.TRAINING.NUM_WORKERS > 0),
            )
            loader_val = DataLoader(
                ds_val,
                batch_size=cfg.TRAINING.BATCH_SIZE,
                shuffle=False,
                num_workers=min(4, cfg.TRAINING.NUM_WORKERS),
                collate_fn=collate_fn,
                pin_memory=(accelerator.device.type == "cuda"),
            )

    else:  # sinkhorn (default)
        from muviseg.models.sinkhorn import SegMASt3R, superglue_nll_loss as _nll
        model = SegMASt3R(
            mast3r_ckpt=cfg.MODEL.MAST3R_CKPT,
            matcher_cfg=matcher_cfg,
            device="cpu",
            precompute_mode=bool(pair_dsc_root),
        )
        def loss_fn(output, seg_corr, masks0, masks1):
            return _nll(output[0], seg_corr, masks0, masks1)
        def metrics_fn(output, seg_corr, masks0, masks1):
            return compute_matching_metrics(output[0].cpu(), seg_corr, masks0, masks1)
        def score_mat(output, b, M, N):
            return output[0][b, :M, :N].cpu()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.TRAINING.LR, weight_decay=cfg.TRAINING.WEIGHT_DECAY)

    # ── Prepare: wraps model in DDP, handles mixed precision ──────
    model, optimizer, loader_train, loader_val = accelerator.prepare(
        model, optimizer, loader_train, loader_val
    )

    # ── Scheduler ─────────────────────────────────────────────────
    # Built after prepare so len(loader_train) reflects the per-process count.
    # With gradient accumulation, optimizer steps = batches / grad_accum.
    iters_per_epoch = len(loader_train) // grad_accum
    total_steps     = iters_per_epoch * cfg.TRAINING.EPOCHS
    scheduler       = build_lr_scheduler(optimizer, cfg, total_steps)

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in trainable)
        print(f"Trainable params: {n_params:,}")
        print(f"Iters/epoch: {iters_per_epoch:,}  |  Total steps: {total_steps:,}")
        if writer:
            writer.add_text("config/trainable_params", str(n_params), 0)
            writer.add_text("config/yaml", str(dict(cfg)), 0)
            writer.add_text("config/iters_per_epoch", str(iters_per_epoch), 0)

    # ── Resume ────────────────────────────────────────────────────
    start_epoch  = 0
    global_step  = 0
    best_val_ma  = 0.0

    ckpt_path = resume or getattr(cfg, "RESUME", None) or ""
    if not ckpt_path:
        ckpt_path = None
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_ma = ckpt.get("best_val_ma", 0.0)
        if accelerator.is_main_process:
            print(f"Resumed from {ckpt_path}  (epoch {start_epoch}, step {global_step})")

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.TRAINING.EPOCHS):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        pbar = tqdm(loader_train,
                    desc=f"Epoch {epoch+1}/{cfg.TRAINING.EPOCHS}",
                    dynamic_ncols=True,
                    disable=not accelerator.is_local_main_process)

        _debug = cfg.DEBUG
        _sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
        if _debug and accelerator.is_main_process:
            _dbg_times = {"data": [], "forward": [], "loss_bwd": [], "optim": []}

        _t_iter_start = time.time()
        for batch in pbar:
          with accelerator.accumulate(model):
            if _debug:
                _sync()
                _t_after_data = time.time()

            optimizer.zero_grad()
            if arch in ("vggt_dpt_multiframe", "vggt_dpt_joint"):
                loss, output = train_step_multiframe(model, batch, device, loss_fn_train)
            else:
                loss, output = train_step(model, batch, device, loss_fn)

            if _debug:
                _sync()
                _t_after_fwd = time.time()

            accelerator.backward(loss)

            if cfg.TRAINING.GRAD_CLIP > 0:
                accelerator.clip_grad_norm_(trainable, cfg.TRAINING.GRAD_CLIP)

            if _debug:
                _sync()
                _t_after_bwd = time.time()

            optimizer.step()

            if _debug:
                _sync()
                _t_after_opt = time.time()

            epoch_loss  += loss.item()

            # Only count optimizer steps (not accumulation sub-steps)
            if accelerator.sync_gradients:
                scheduler.step()
                global_step += accelerator.num_processes
            lr_now       = scheduler.get_last_lr()[0]

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_now:.2e}")

            # ── Debug timing ─────────────────────────────────────
            if _debug and accelerator.is_main_process:
                _dbg_times["data"].append(_t_after_data - _t_iter_start)
                _dbg_times["forward"].append(_t_after_fwd - _t_after_data)
                _dbg_times["loss_bwd"].append(_t_after_bwd - _t_after_fwd)
                _dbg_times["optim"].append(_t_after_opt - _t_after_bwd)
                if len(_dbg_times["data"]) % 5 == 0:
                    n = 5
                    def _avg(lst): return sum(lst[-n:]) / len(lst[-n:]) * 1e3
                    _mem = ""
                    if device.type == "cuda":
                        _alloc = torch.cuda.memory_allocated() / 1e6
                        _resrv = torch.cuda.memory_reserved() / 1e6
                        _mem = f"  mem={_alloc:.0f}/{_resrv:.0f}MB"
                    tqdm.write(
                        f"  [DEBUG step {global_step}] "
                        f"data={_avg(_dbg_times['data']):.0f}ms  "
                        f"fwd={_avg(_dbg_times['forward']):.0f}ms  "
                        f"bwd={_avg(_dbg_times['loss_bwd']):.0f}ms  "
                        f"optim={_avg(_dbg_times['optim']):.0f}ms  "
                        f"total={sum(_avg(_dbg_times[k]) for k in _dbg_times):.0f}ms"
                        f"{_mem}"
                    )
            _t_iter_start = time.time()

            # Skip logging/val/save on accumulation sub-steps
            if not accelerator.sync_gradients:
                continue

            # ── TB: train scalars (main process only) ─────────────
            if accelerator.is_main_process and global_step % cfg.TRAINING.LOG_INTERVAL == 0:
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar("train/lr",   lr_now,      global_step)
                    unwrapped = accelerator.unwrap_model(model)
                    if hasattr(unwrapped, "matcher") and hasattr(unwrapped.matcher, "dustbin_score"):
                        ds_score = unwrapped.matcher.dustbin_score.item()
                        writer.add_scalar("train/dustbin_score", ds_score, global_step)
                    if hasattr(unwrapped, "matcher") and hasattr(unwrapped.matcher, "log_tau"):
                        tau = unwrapped.matcher.log_tau.exp().item()
                        writer.add_scalar("train/temperature", tau, global_step)
                tqdm.write(f"  step {global_step:>7d} | "
                           f"loss {loss.item():.4f} | lr {lr_now:.2e}")

            # ── Validation ────────────────────────────────────────
            if global_step % cfg.TRAINING.VAL_INTERVAL == 0:
                # Barrier: ensure all ranks finish the last training step
                # (DDP gradient allreduce) before entering validation.
                accelerator.wait_for_everyone()
                model.eval()
                vm = run_validation(model, loader_val, device, accelerator,
                                    loss_fn=loss_fn, metrics_fn=metrics_fn,
                                    score_mat_fn=score_mat,
                                    writer=writer, global_step=global_step,
                                    n_vis_batches=4,
                                    vis_img_size=max(cfg.DATASET.HEIGHT, cfg.DATASET.WIDTH),
                                    vis_resize_mode=cfg.DATASET.RESIZE_MODE,
                                    max_batches=cfg.TRAINING.VAL_MAX_BATCHES)
                model.train()

                if accelerator.is_main_process:
                    if writer:
                        writer.add_scalar("val/loss",              vm["loss"],              global_step)
                        writer.add_scalar("val/matching_accuracy", vm["matching_accuracy"], global_step)
                        writer.add_scalar("val/mean_gt_logprob",   vm["mean_gt_logprob"],   global_step)
                        writer.add_scalar("val/dustbin_rate",      vm["dustbin_rate"],      global_step)
                        writer.add_scalar("val/recall_at_1",       vm["recall_at_1"],       global_step)
                        writer.add_scalar("val/recall_at_5",       vm["recall_at_5"],       global_step)
                        writer.add_scalar("val/auprc",             vm["auprc"],             global_step)
                        # Error decomposition
                        for ek in ("false_dustbin_rate", "wrong_match_rate", "false_match_rate"):
                            if ek in vm:
                                writer.add_scalar(f"val/{ek}", vm[ek], global_step)
                    tqdm.write(
                        f"  *** VAL {global_step} | "
                        f"loss={vm['loss']:.4f} | "
                        f"MA={vm['matching_accuracy']:.3f} | "
                        f"R@5={vm['recall_at_5']:.3f} | "
                        f"AUPRC={vm['auprc']:.3f} | "
                        f"dustbin={vm['dustbin_rate']:.3f} | "
                        f"f_dust={vm.get('false_dustbin_rate', 0):.3f} | "
                        f"w_match={vm.get('wrong_match_rate', 0):.3f} | "
                        f"f_match={vm.get('false_match_rate', 0):.3f} ***"
                    )

                # wait_for_everyone is a collective op — must be called by ALL ranks.
                # Moving it outside is_main_process prevents the NCCL deadlock where
                # rank 0 blocks on the barrier while rank 1 has already moved forward.
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if vm["matching_accuracy"] > best_val_ma:
                        best_val_ma = vm["matching_accuracy"]
                        _save_ckpt(save_dir / "best.pth", model, optimizer, scheduler,
                                   epoch, global_step, vm, best_val_ma=best_val_ma,
                                   accelerator=accelerator, cfg=cfg)
                        tqdm.write(f"  → New best  MA={best_val_ma:.3f}")

            # ── Periodic checkpoint ───────────────────────────────
            if global_step % cfg.TRAINING.SAVE_INTERVAL == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    _save_ckpt(save_dir / f"step_{global_step:07d}.pth",
                               model, optimizer, scheduler,
                               epoch, global_step, {"loss": loss.item()},
                               best_val_ma=best_val_ma,
                               accelerator=accelerator, cfg=cfg)

        # ── End of epoch ──────────────────────────────────────────
        avg_loss = epoch_loss / iters_per_epoch
        elapsed  = time.time() - epoch_start

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if writer:
                writer.add_scalar("train/epoch_loss", avg_loss, epoch + 1)
            print(f"Epoch {epoch+1} | avg_loss={avg_loss:.4f} | {elapsed/60:.1f} min")
            _save_ckpt(save_dir / f"epoch_{epoch+1:03d}.pth",
                       model, optimizer, scheduler,
                       epoch + 1, global_step, {"loss": avg_loss},
                       best_val_ma=best_val_ma,
                       accelerator=accelerator, cfg=cfg)

    if writer:
        writer.close()
    if accelerator.is_main_process:
        print(f"\nDone. Best val MA: {best_val_ma:.3f}")
        print(f"Tensorboard: tensorboard --logdir {save_dir / 'tb'}")
