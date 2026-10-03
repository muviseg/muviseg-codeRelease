"""Run the joint model over a sequence with a sliding window of N frames.

Every window is scored on all C(N,2) pairs in a single joint-attention pass,
which is the point of the joint model: the cost is one backbone pass per window,
not one per pair.

This reaches into the model (`extract_multi_layer_nframes`, `fusion`, `proj`,
`joint_attn_layers`, `matcher`) rather than calling `infer_tuple`, for two
reasons: it needs every pair of a window from one pass, and it measures the
backbone/head split that the reported latency figures quote.
"""
from __future__ import annotations

import itertools
import statistics
import time

import torch
import torch.nn.functional as F

from muviseg.evaluation.model_infer import _mutual_match_filter, pad_masks_to_batch
from muviseg.inference.tracks import build_tracks


def window_starts(n_frames: int, n: int, stride: int) -> list[int]:
    """Window start indices, always including the last full window."""
    starts = list(range(0, max(1, n_frames - n + 1), stride))
    if starts and starts[-1] != n_frames - n and n_frames >= n:
        starts.append(n_frames - n)
    return starts


def run_sequence(
    model,
    device: torch.device,
    img_tensors: list[torch.Tensor],
    masks_per_frame: list[torch.Tensor],
    *,
    n: int = 4,
    stride: int = 2,
    batch: int = 4,
    progress: bool = True,
):
    """Score every sliding window, accept matches, and build tracks.

    Args:
        model: a `SegVGGTDPTJoint` (the inner model, not the Infer wrapper).
        img_tensors: list of (3, H, W) tensors in [-1, 1].
        masks_per_frame: list of (M_f, H, W) mask tensors.

    Returns:
        gid_of, lens, stats  -- see `muviseg.inference.tracks.build_tracks`.
    """
    n_frames = len(img_tensors)
    if n_frames < n:
        raise ValueError(f"need at least n={n} frames, got {n_frames}")
    starts = window_starts(n_frames, n, stride)
    pairs = list(itertools.combinations(range(n), 2))

    best: dict[tuple[int, int], dict[int, tuple[int, float]]] = {}
    lat_batch, lat_backbone, lat_head = [], [], []

    is_cuda = device.type == "cuda"
    sync = (lambda: torch.cuda.synchronize(device)) if is_cuda else (lambda: None)

    t_all = time.perf_counter()
    for bi in range(0, len(starts), batch):
        grp = starts[bi:bi + batch]
        B = len(grp)
        images = torch.stack([
            torch.stack([img_tensors[s + v] for v in range(n)]) for s in grp
        ]).to(device)                                            # (B, N, 3, H, W)

        masks_in, real_M = [], []
        for v in range(n):
            per = [masks_per_frame[s + v] for s in grp]
            real_M.append([p.shape[0] for p in per])
            masks_in.append(pad_masks_to_batch(per, device))      # (B, M_max, H, W)

        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            feats, H_p, W_p = model.extract_multi_layer_nframes(images)
            sync()
            t_bb = time.perf_counter() - t0

            t1 = time.perf_counter()
            fused = [model.fusion(feats[v]) for v in range(n)]
            pdt = next(model.proj.parameters()).dtype
            projs = []
            for v in range(n):
                mk = masks_in[v]
                if mk.shape[-2:] != (H_p, W_p):
                    mk = F.interpolate(mk.float(), (H_p, W_p), mode="nearest")
                Bv, D, Hh, Ww = fused[v].shape
                Mv = mk.shape[1]
                ff = fused[v].view(Bv, D, Hh * Ww)
                mf = mk.view(Bv, Mv, Hh * Ww).float()
                area = mf.sum(-1, keepdim=True).clamp(min=1)
                dsc = torch.bmm(ff, mf.transpose(1, 2)) / area.transpose(1, 2)
                projs.append(model.proj(dsc.transpose(1, 2).to(dtype=pdt)))
            for li, layer in enumerate(model.joint_attn_layers):
                projs = layer(projs, frame_embeds_added=(li > 0))
            out = {
                (a, b): model.matcher(projs[a].transpose(1, 2), projs[b].transpose(1, 2))
                for (a, b) in pairs
            }
            sync()
            t_hd = time.perf_counter() - t1
        dt = time.perf_counter() - t0
        lat_batch.append(dt * 1000)
        lat_backbone.append(t_bb * 1000)
        lat_head.append(t_hd * 1000)

        for (a, b) in pairs:
            log_mutual, m0, _ = out[(a, b)]
            sc_all = torch.exp(log_mutual)
            for k, s in enumerate(grp):
                Ma, Mb = real_M[a][k], real_M[b][k]
                if Ma == 0 or Mb == 0:
                    continue
                sc = sc_all[k:k + 1, :Ma, :Mb]
                mr = _mutual_match_filter(sc, m0[k:k + 1, :Ma], device)[0]
                fa, fb = s + a, s + b
                d = best.setdefault((fa, fb), {})
                row = sc[0]
                for i in range(Ma):
                    j = int(mr[i])
                    if j < 0:
                        continue
                    v = float(row[i, j])
                    if d.get(i) is None or v > d[i][1]:
                        d[i] = (j, v)

        if progress:
            n_batches = (len(starts) + batch - 1) // batch
            every = 1 if n_batches <= 20 else 10
            if (bi // max(1, batch)) % every == 0 or bi + B >= len(starts):
                print(f"  [N={n}] window batch {bi + B}/{len(starts)}", flush=True)

    wall = time.perf_counter() - t_all
    gid_of, lens, n_gid = build_tracks(n_frames, masks_per_frame, best)

    def pct(xs, q):
        return round(sorted(xs)[int(q * (len(xs) - 1))], 1)

    stats = dict(
        N=n, stride=stride, batch_size=batch, n_windows=len(starts),
        n_pairs_per_window=len(pairs), n_batches=len(lat_batch),
        wall_s=round(wall, 2),
        latency_ms_per_batch=dict(
            mean=round(statistics.mean(lat_batch), 1),
            p50=round(statistics.median(lat_batch), 1),
            p95=pct(lat_batch, 0.95),
            min=round(min(lat_batch), 1), max=round(max(lat_batch), 1)),
        latency_ms_per_window=round(statistics.mean(lat_batch) / batch, 1),
        backbone_ms_per_batch=round(statistics.mean(lat_backbone), 1),
        matching_head_ms_per_batch=round(statistics.mean(lat_head), 1),
        windows_per_s=round(len(starts) / wall, 2),
        peak_vram_GiB=(round(torch.cuda.max_memory_allocated(device) / 2 ** 30, 2)
                       if is_cuda else None),
        n_tracks=n_gid,
        n_tracks_len_ge_3=sum(1 for v in lens.values() if v >= 3),
        n_tracks_len_ge_5=sum(1 for v in lens.values() if v >= 5),
        longest_track=max(lens.values()) if lens else 0,
        n_segments_total=int(sum(m.shape[0] for m in masks_per_frame)),
        n_segments_in_tracks_len_ge_3=int(
            sum(1 for k, g in gid_of.items() if lens[g] >= 3)),
    )
    return gid_of, lens, stats
