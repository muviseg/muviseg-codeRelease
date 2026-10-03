"""RoMa dense matcher aggregated to segment level, for the Tables 1-2 evaluation.

A modern dense-matching baseline. RoMa produces a dense warp between two images;
we sample pixel correspondences and aggregate them into a segment-level score
matrix with the same vote used by the SuperPoint+LightGlue navigation adapter
(`third_party/object-rel-nav/libs/matcher/lightglue.py:49-57, 91-103`). Keeping
the aggregation identical means a RoMa row differs from a keypoint row only in
where the correspondences come from.

Scoring
-------
AUPRC and R@k are computed per query row, so any per-row rescaling leaves the
ranking untouched. Only a scheme that reweights *within* a row changes the
result, which is why two variants exist:

  votes      number of sampled correspondences landing in (segment i, segment j).
             Integer and sparse: most entries are exactly zero, so the ranking
             below the top few candidates is arbitrary and R@5 / AUPRC are
             depressed for reasons unrelated to matching quality.

  certainty  sum of RoMa's own certainty over the correspondences landing in
             (i, j). Continuous, far fewer ties, and it lets a segment pair
             backed by a few confident matches outrank one backed by many
             uncertain ones.

`certainty` is the default because scoring a baseline in a way that guarantees
ties is not a fair test of it. Both are reported, and the variant more
favourable to RoMa is the one that belongs in the table.
"""

import numpy as np
import torch
from PIL import Image


class RoMaSegInfer:
    """Matches the (match_result, scores) contract of muviseg.evaluation.model_infer classes."""

    def __init__(self, variant="indoor", num_samples=5000, certainty_threshold=0.0,
                 upsample=False, scoring="certainty"):
        if scoring not in ("certainty", "votes"):
            raise ValueError(f"scoring must be 'certainty' or 'votes', got {scoring}")
        self.variant = variant
        self.num_samples = int(num_samples)
        self.certainty_threshold = float(certainty_threshold)
        self.upsample = bool(upsample)
        self.scoring = scoring
        self.model = None

    def prepare(self, device):
        from romatch import roma_indoor, roma_outdoor

        self.device = device
        builder = roma_indoor if self.variant == "indoor" else roma_outdoor
        self.model = builder(device=device)
        # RoMa's refinement stage targets ~(864, 1152). Replica and VKITTI2 frames
        # are smaller than that, so it costs time without adding usable precision.
        self.model.upsample_preds = self.upsample
        return self

    # ------------------------------------------------------------------
    @staticmethod
    def _to_pil(img):
        """(3, H, W) in [-1, 1] -> PIL RGB, the form RoMa's match() consumes."""
        arr = ((img.detach().float().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

    def _correspondences(self, pil0, pil1, H, W):
        """(K,2) pixel coords in each image, plus per-correspondence certainty."""
        warp, certainty = self.model.match(pil0, pil1, device=self.device)
        matches, conf = self.model.sample(warp, certainty, num=self.num_samples)
        if self.certainty_threshold > 0.0:
            keep = conf >= self.certainty_threshold
            matches, conf = matches[keep], conf[keep]
        if matches.shape[0] == 0:
            return np.zeros((0, 2), int), np.zeros((0, 2), int), np.zeros((0,))
        k0, k1 = self.model.to_pixel_coordinates(matches, H, W, H, W)
        k0 = k0.detach().cpu().numpy()
        k1 = k1.detach().cpu().numpy()
        # to_pixel_coordinates can land on the upper bound; clip so the mask
        # lookup below stays in range.
        k0[:, 0] = np.clip(k0[:, 0], 0, W - 1); k0[:, 1] = np.clip(k0[:, 1], 0, H - 1)
        k1[:, 0] = np.clip(k1[:, 0], 0, W - 1); k1[:, 1] = np.clip(k1[:, 1], 0, H - 1)
        return k0.astype(int), k1.astype(int), conf.detach().cpu().numpy()

    def _score_matrix(self, k0, k1, conf, m0, m1):
        """(M, N) score matrix from pixel correspondences and binary masks."""
        # membership: which segment does each correspondence endpoint fall in
        occ0 = m0[:, k0[:, 1], k0[:, 0]]      # (M, K) bool
        occ1 = m1[:, k1[:, 1], k1[:, 0]]      # (N, K) bool
        if self.scoring == "votes":
            return (occ0.astype(np.float32) @ occ1.astype(np.float32).T)
        w = conf.astype(np.float32)
        return (occ0.astype(np.float32) * w) @ occ1.astype(np.float32).T

    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer_pair(self, img0, img1, masks0, masks1):
        assert self.model is not None, "Call prepare(device) before infer_pair"
        B, _, H, W = img0.shape
        M, N = masks0.shape[1], masks1.shape[1]
        scores = torch.zeros((B, M, N), dtype=torch.float32, device=self.device)

        for b in range(B):
            k0, k1, conf = self._correspondences(
                self._to_pil(img0[b]), self._to_pil(img1[b]), H, W)
            if len(k0) == 0:
                continue
            m0 = masks0[b].detach().cpu().numpy() > 0.5
            m1 = masks1[b].detach().cpu().numpy() > 0.5
            scores[b] = torch.from_numpy(
                self._score_matrix(k0, k1, conf, m0, m1)).to(self.device)

        # Mutual nearest neighbour, matching how the other rows decide a match.
        # RoMa has no learned reject head, so an all-zero row means "no evidence"
        # and is reported as unmatched rather than forced onto an arbitrary peer.
        ref_to_target = scores.argmax(dim=-1)
        target_to_ref = scores.argmax(dim=-2)
        reciprocal = torch.gather(target_to_ref, 1, ref_to_target.clamp(max=N - 1))
        rows = torch.arange(M, device=self.device).unsqueeze(0).expand(B, M)
        has_evidence = scores.amax(dim=-1) > 0
        valid = (reciprocal == rows) & has_evidence

        match_result = -torch.ones((B, M), dtype=torch.int64, device=self.device)
        match_result[valid] = ref_to_target[valid]
        return match_result, scores
