"""
Synthetic timing benchmark: SegVGGTDPT (pairwise) vs SegVGGTDPTJoint.

Measures backbone (VGGT Aggregator) and matching-head wall-clock time
on fake images + masks for several N (frames per window). Both models
share backbone, so backbone time is identical; only the matching head
differs (C(N,2) self+cross attn passes vs single joint self-attn pass).

Output: per-N table with backbone_ms, matching_ms, total_ms, plus
matching-head speedup (pairwise / joint) and total wall-clock speedup.

Run:
  source ~/nabo-projects-mounted/claude-venv/bin/activate
  python3 evaluation/scripts/bench_joint_vs_pairwise.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from muviseg.models.vggt_dpt_lg import SegVGGTDPT, SegVGGTDPTJoint  # noqa: E402

VGGT_CKPT = str(_PROJECT_ROOT / "third_party" / "vggt_weights.pt")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

H, W = 336, 512
SEGMENTS_PER_FRAME = 50
N_VALUES = (2, 4, 6, 8)
WARMUP = 5
ITERS = 30


def fake_inputs(n_frames: int, batch: int = 1):
    """Random images in [-1, 1] and disjoint random masks per frame."""
    images = torch.rand(batch, n_frames, 3, H, W, device=DEVICE) * 2.0 - 1.0
    masks_list = []
    for _ in range(n_frames):
        # Random binary masks, M segments each. We don't enforce disjointness;
        # masked-average pooling tolerates overlaps.
        m = (torch.rand(batch, SEGMENTS_PER_FRAME, H, W, device=DEVICE) > 0.7).float()
        masks_list.append(m)
    return images, masks_list


def all_pairs(n: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(n) for b in range(a + 1, n)]


def time_forward(model, images, masks_list, pair_indices) -> tuple[float, float]:
    """Returns (backbone_ms, matching_ms) split via cuda.synchronize()."""
    is_cuda = DEVICE.type == "cuda"
    inner = model

    with torch.no_grad():
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        all_feats, H_p, W_p = inner.extract_multi_layer_nframes(images)
        if is_cuda:
            torch.cuda.synchronize()
        backbone_s = time.perf_counter() - t0

        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        N = images.shape[1]
        fused = [inner.fusion(all_feats[v]) for v in range(N)]

        proj_dtype = next(inner.proj.parameters()).dtype
        projs = []
        for v in range(N):
            m = masks_list[v]
            if m.shape[-2:] != (H_p, W_p):
                m = torch.nn.functional.interpolate(
                    m.float(), (H_p, W_p), mode="nearest"
                )
            B, D, Hp_, Wp_ = fused[v].shape
            M = m.shape[1]
            feat_flat = fused[v].view(B, D, Hp_ * Wp_)
            masks_flat = m.view(B, M, Hp_ * Wp_).float()
            area = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1)
            dsc = torch.bmm(feat_flat, masks_flat.transpose(1, 2)) / area.transpose(1, 2)
            proj = inner.proj(dsc.transpose(1, 2).to(dtype=proj_dtype))
            projs.append(proj)

        if isinstance(inner, SegVGGTDPTJoint):
            for i, layer in enumerate(inner.joint_attn_layers):
                projs = layer(projs, frame_embeds_added=(i > 0))
            for (a, b) in pair_indices:
                _ = inner.matcher(projs[a].transpose(1, 2), projs[b].transpose(1, 2))
        else:
            for (a, b) in pair_indices:
                x0, x1 = projs[a].clone(), projs[b].clone()
                for layer in inner.attn_layers:
                    x0, x1 = layer(x0, x1)
                _ = inner.matcher(x0.transpose(1, 2), x1.transpose(1, 2))

        if is_cuda:
            torch.cuda.synchronize()
        matching_s = time.perf_counter() - t0

    return backbone_s * 1000, matching_s * 1000


def bench(model, n_frames: int, label: str) -> dict:
    pairs = all_pairs(n_frames)
    images, masks_list = fake_inputs(n_frames)

    for _ in range(WARMUP):
        time_forward(model, images, masks_list, pairs)

    backbone_times = []
    matching_times = []
    for _ in range(ITERS):
        b, m = time_forward(model, images, masks_list, pairs)
        backbone_times.append(b)
        matching_times.append(m)

    bm = sum(backbone_times) / len(backbone_times)
    mm = sum(matching_times) / len(matching_times)
    bs = (sum((t - bm) ** 2 for t in backbone_times) / len(backbone_times)) ** 0.5
    ms = (sum((t - mm) ** 2 for t in matching_times) / len(matching_times)) ** 0.5

    print(f"  [{label}] N={n_frames} pairs={len(pairs)}  "
          f"backbone={bm:6.1f}±{bs:.1f} ms  matching={mm:5.1f}±{ms:.1f} ms  "
          f"total={bm + mm:6.1f} ms")
    return {
        "label": label, "n_frames": n_frames, "n_pairs": len(pairs),
        "backbone_ms": round(bm, 2), "backbone_std": round(bs, 2),
        "matching_ms": round(mm, 2), "matching_std": round(ms, 2),
        "total_ms": round(bm + mm, 2),
    }


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"Shape:  H={H} W={W}  segments/frame={SEGMENTS_PER_FRAME}")
    print(f"Iters:  warmup={WARMUP}  measured={ITERS}")
    print()

    print("Loading SegVGGTDPT (pairwise)...")
    pairwise = SegVGGTDPT(
        vggt_ckpt=VGGT_CKPT,
        layer_indices=(5, 11, 17, 23),
        fusion_dim=256, proj_dim=128,
        n_layers=3, n_heads=4, ffn_expansion=4,
    ).to(DEVICE).eval()

    print("Loading SegVGGTDPTJoint (joint)...")
    joint = SegVGGTDPTJoint(
        vggt_ckpt=VGGT_CKPT,
        layer_indices=(5, 11, 17, 23),
        fusion_dim=256, proj_dim=128,
        n_joint_layers=3, n_heads=4, ffn_expansion=4, max_frames=16,
    ).to(DEVICE).eval()

    results = []
    for n in N_VALUES:
        print(f"\n--- N={n} (C({n},2)={n * (n - 1) // 2} pairs) ---")
        rp = bench(pairwise, n, "pairwise")
        rj = bench(joint, n, "joint   ")
        results.append({"n_frames": n, "pairwise": rp, "joint": rj})

    print("\n=== Summary ===")
    print(f"{'N':>3} {'pairs':>6}  "
          f"{'pw_back':>8} {'pw_match':>9} {'pw_total':>9}  "
          f"{'jt_back':>8} {'jt_match':>9} {'jt_total':>9}  "
          f"{'match_x':>8} {'total_x':>8}")
    for row in results:
        n = row["n_frames"]
        p = row["pairwise"]
        j = row["joint"]
        match_x = p["matching_ms"] / max(j["matching_ms"], 1e-6)
        total_x = p["total_ms"] / max(j["total_ms"], 1e-6)
        print(f"{n:>3} {p['n_pairs']:>6}  "
              f"{p['backbone_ms']:>8.1f} {p['matching_ms']:>9.1f} {p['total_ms']:>9.1f}  "
              f"{j['backbone_ms']:>8.1f} {j['matching_ms']:>9.1f} {j['total_ms']:>9.1f}  "
              f"{match_x:>7.2f}x {total_x:>7.2f}x")

    out_dir = _PROJECT_ROOT / "results" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "joint_vs_pairwise.json"
    with open(out_file, "w") as f:
        json.dump({
            "device": str(DEVICE),
            "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
            "image_hw": [H, W],
            "segments_per_frame": SEGMENTS_PER_FRAME,
            "warmup": WARMUP, "iters": ITERS,
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
