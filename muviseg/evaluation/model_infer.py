# PyTorch Imports
import sys
import torch
from pathlib import Path


def _mutual_match_filter(scores: torch.Tensor, match0: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
    """
    Mutual + matchability filter for DoubleSoftmax-style outputs.
    scores: (B, M, N) probabilities, match0: (B, M) matchability logit.
    Returns match_result (B, M) int64 — -1 for unmatched.
    """
    B, M, N = scores.shape
    ref_to_target = scores.argmax(dim=-1)
    target_to_ref = scores.argmax(dim=-2)
    is_matchable = match0[:, :M] > 0
    safe_indices = ref_to_target.clamp(max=N - 1)
    reciprocal = torch.gather(target_to_ref, 1, safe_indices)
    row_indices = torch.arange(M, device=device).unsqueeze(0).expand(B, M)
    is_mutual = reciprocal == row_indices
    valid_mask = is_matchable & is_mutual
    match_result = -torch.ones((B, M), dtype=torch.int64, device=device)
    match_result[valid_mask] = ref_to_target[valid_mask]
    return match_result


def pad_masks_to_batch(masks_list: list, device) -> torch.Tensor:
    """Pad variable-M masks → (B, M_max, H, W) float. Empty views → M=1 (dummy zero)."""
    M_max = max(m.shape[0] for m in masks_list)
    if M_max == 0:
        M_max = 1  # avoid empty tensor that breaks downstream F.interpolate
    H, W = masks_list[0].shape[-2:]
    out = []
    for m in masks_list:
        m = m.to(device)
        pad = M_max - m.shape[0]
        if pad > 0:
            m = torch.cat([m, torch.zeros(pad, H, W, dtype=m.dtype, device=m.device)], 0)
        out.append(m)
    return torch.stack(out).float()


class MASt3RSegFeatInfer(torch.nn.Module):
    def __init__(self, cfg):
        # The segmast3r checkout has to be on sys.path before these imports.
        # The other wrappers in this file get that as a side effect of importing
        # muviseg.models.*, which calls setup_segmast3r_path() for them; this one
        # imports the segmast3r tree directly, so it has to ask explicitly.
        from muviseg.paths import setup_segmast3r_path  # noqa: PLC0415

        setup_segmast3r_path()

        # Lazy imports — the segmast3r checkout may not be set up
        import mast3r_src.mast3r.model as mast3r_model  # noqa: PLC0415
        from src.models.mast3r_segfeat.diff_feature_matcher import featureMatcher  # noqa: PLC0415
        from src.models.mast3r_segfeat.diff_masked_pooling import masked_average_pooling  # noqa: PLC0415
        self._mast3r_model = mast3r_model
        self._masked_average_pooling = masked_average_pooling
        self._featureMatcher = featureMatcher
        super().__init__()
        self.cfg = cfg

        model_params = self._get_model_params(cfg)
        self.encoder = mast3r_model.AsymmetricMASt3R(**model_params)

        cfg["FEATURE_MATCHER"]["SINKHORN"]["DUSTBIN_SCORE_INIT"] = 5.3937
        self.feature_matcher = featureMatcher(cfg["FEATURE_MATCHER"])

        self._configure_grad()

    def _configure_grad(self):
        """
        Centralized method to configure gradient settings for model.
        """
        # Freeze the complete model
        self.encoder.requires_grad_(False)
        # Freeze the feature matcher
        self.feature_matcher.requires_grad_(False)

    def _get_model_params(self, cfg):
        """
        Generate model parameters based on dataset type.
        """
        m = self._mast3r_model
        base_params = {
            "pos_embed": "RoPE100",
            "patch_embed_cls": "ManyAR_PatchEmbed",
            "img_size": (336, 512),
            "head_type": "catmlp+dpt",
            "output_mode": "pts3d+desc24",
            "depth_mode": ("exp", -m.inf, m.inf),
            "conf_mode": ("exp", 1, m.inf),
            "enc_embed_dim": 1024,
            "enc_depth": 24,
            "enc_num_heads": 16,
            "dec_embed_dim": 768,
            "dec_depth": 12,
            "dec_num_heads": 12,
            "two_confs": True,
        }

        dataset_type = cfg["DATASET"]["DATA_SOURCE"].lower()

        # NOTE: this was `if dataset_type == "mapfree" or "hm3d":`, which is
        # always True ("hm3d" is a truthy string), so this block has been applied
        # to EVERY dataset -- Replica and VKITTI2 included. All published Table 1
        # and Table 2 numbers were produced that way, so the branch is made
        # unconditional to preserve behaviour rather than "fixed" to
        # `in ("mapfree", "hm3d")`, which would silently invalidate them.
        if True:
            base_params.update(
                {
                    "patch_embed_cls": "PatchEmbedDust3R",
                    "img_size": (512, 512),
                    "desc_conf_mode": ("exp", 0, m.inf),
                    "landscape_only": False,
                }
            )

        return base_params

    def forward(self, view1, view2):
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = (
                self.encoder._encode_symmetrized(view1, view2)
            )

            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

            pred1 = self.encoder._downstream_head(
                1, [tok.float() for tok in dec1], shape1
            )
            pred2 = self.encoder._downstream_head(
                2, [tok.float() for tok in dec2], shape2
            )

        return pred1["desc"], pred2["desc"]

    def prepare(self, device):
        """Call once before inference loop"""
        self.device = device
        self.eval()
        self.to(device)

    def infer_pair(self, img0, img1, masks0, masks1):
        assert hasattr(self, "device"), "Call model.prepare(device) before infer_pair"

        with torch.no_grad():
            B, _, H_resize, W_resize = img0.shape
            img0 = img0.to(self.device)  # (B, 3, H_resize, W_resize)
            img1 = img1.to(self.device)  # (B, 3, H_resize, W_resize)

            masks0 = masks0.to(self.device)  # (B, M, H_resize, W_resize)
            masks1 = masks1.to(self.device)  # (B, N, H_resize, W_resize)

            # Construct view 0 (reference id est img0) and view 1 (query id est img1)
            view0 = {
                "img": img0,
                "true_shape": torch.tensor(
                    [[H_resize, W_resize]], dtype=torch.int64, device=self.device
                ),
                "instance": [
                    f"view1_{i}" for i in range(B)
                ],  # NOTE: view1 is MASt3R's internal name for the reference view (our view0)
            }

            view1 = {
                "img": img1,
                "true_shape": torch.tensor(
                    [[H_resize, W_resize]], dtype=torch.int64, device=self.device
                ),
                "instance": [
                    f"view2_{i}" for i in range(B)
                ],  # NOTE: view2 is MASt3R's internal name for the query view (our view1)
            }

            # Model forward pass and get feature maps
            pred0, pred1 = self.forward(view0, view1)
            desc0 = pred0.permute(
                0, 3, 1, 2
            )  # (B, H_resize, W_resize, 24) -> (B, 24, H_resize, W_resize)
            desc1 = pred1.permute(
                0, 3, 1, 2
            )  # (B, H_resize, W_resize, 24) -> (B, 24, H_resize, W_resize)

            #  Aggregation to obtain segment features (Masked Average Pooling)
            agg_desc0 = self._masked_average_pooling(desc0, masks0)  # (B, 24, M)
            agg_desc1 = self._masked_average_pooling(desc1, masks1)  # (B, 24, N)

            log_P_dustb = self.feature_matcher(agg_desc0, agg_desc1)  # (B, M+1, N+1)
            scores = torch.exp(log_P_dustb)  # (B, M+1, N+1)

            # Build pred masks
            B, M_p1, N_p1 = scores.shape
            M = M_p1 - 1  # exclude dustbin row
            N = N_p1 - 1  # exclude dustbin column

            # Regular Row-Wise ArgMax ############################################
            # # Step 1: Get best matches for each ref mask (excluding dustbin row)
            # matching_indices = scores[:, :M, :].argmax(dim=-1)  # (B, M)

            # # Step 2: Create a mask indicating where the match is not to the dustbin
            # valid_mask = matching_indices < N  # (B, M)

            # # Initialize with -1s for unmatched
            # match_result = -1 * torch.ones(
            #     (B, M), dtype=torch.int64, device=self.device
            # )  # (B, M)
            # match_result[valid_mask] = matching_indices[valid_mask]  # (B, M)
            # ######################################################################

            # Mutual Nearest Neighbor Matching ##############################
            # 1. Forward Match (Ref -> Target)
            # Find best column j for each row i (excluding ref dustbin row)
            # shape: (B, M), values in [0, N] (where N is dustbin)
            ref_to_target = scores[:, :M, :].argmax(dim=-1)

            # 2. Backward Match (Target -> Ref)
            # Find best row i for each column j (excluding target dustbin col)
            # shape: (B, N), values in [0, M] (where M is dustbin)
            target_to_ref = scores[:, :, :N].argmax(dim=-2)

            # 3. Dustbin Check
            # Ensure the forward match is not pointing to the dustbin column
            is_not_dustbin = ref_to_target < N

            # 4. Mutual Check Implementation
            # We need to verify: target_to_ref[b, ref_to_target[b, i]] == i

            # Safe Gather: ref_to_target contains indices up to N (dustbin).
            # target_to_ref only has size N. Accessing index N would cause OOB.
            # We clamp indices to N-1 just for the gather operation;
            # the 'is_not_dustbin' mask will filter out the invalid ones anyway.
            safe_indices = ref_to_target.clamp(max=N - 1)

            # Gather the reverse choice for every forward choice
            # shape: (B, M)
            reciprocal_choice = torch.gather(target_to_ref, 1, safe_indices)

            # Create grid of current row indices for comparison
            # shape: (B, M)
            row_indices = torch.arange(M, device=self.device).unsqueeze(0).expand(B, M)

            # Check if the target's choice points back to the current row
            is_mutual = reciprocal_choice == row_indices

            # 5. Final Masking
            valid_mask = is_not_dustbin & is_mutual

            # Initialize result with -1
            match_result = -1 * torch.ones(
                (B, M), dtype=torch.int64, device=self.device
            )

            # Fill valid matches
            match_result[valid_mask] = ref_to_target[valid_mask]
            ######################################################################

        # Return both match indices and raw scores (excluding dustbin)
        # scores_no_dustbin: (B, M, N) - matching scores for evaluation
        scores_no_dustbin = scores[:, :M, :N]

        return match_result, scores_no_dustbin  # match_result: (B, M), scores: (B, M, N)


class SegMASt3RLGv2Infer(torch.nn.Module):
    """
    Inference wrapper for SegMASt3RLGv2 (frozen MASt3R + LightGlue-style head v2).

    Checkpoint format (trainer.py):
        {"model_state": ..., "epoch": ..., "global_step": ..., ...}
    """

    def __init__(self, cfg):
        super().__init__()

        _project_root = Path(__file__).parent.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

        from muviseg.models.lightglue_v2 import SegMASt3RLGv2  # noqa: PLC0415

        model_cfg = cfg["MODEL"]
        self.inner = SegMASt3RLGv2(
            mast3r_ckpt=model_cfg["MAST3R_CKPT"],
            proj_dim=model_cfg.get("PROJ_DIM", 128),
            proj_mid_dim=model_cfg.get("PROJ_MID_DIM", 64),
            n_layers=model_cfg.get("N_LAYERS", 3),
            n_heads=model_cfg.get("N_HEADS", 4),
            ffn_expansion=model_cfg.get("FFN_EXPANSION", 4),
        )

        ckpt_path = model_cfg["CHECKPOINT"]
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading SegMASt3RLGv2 checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
        # Load only head weights (backbone is loaded from MAST3R_CKPT)
        missing, unexpected = self.inner.load_state_dict(state, strict=False)
        # Backbone keys are expected to be in the checkpoint
        if missing:
            non_backbone = [k for k in missing if not k.startswith("backbone.")]
            if non_backbone:
                print(f"  WARNING: missing non-backbone keys: {non_backbone}")
        if 'best_val_ma' in ckpt:
            print(f"  epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}, "
                  f"best_val_ma={ckpt['best_val_ma']:.4f}")
        else:
            print(f"  epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}")

    def prepare(self, device):
        self.device = device
        self.inner.eval()
        self.inner.to(device)

    def infer_pair(self, img0, img1, masks0, masks1):
        assert hasattr(self, "device"), "Call prepare(device) before infer_pair"

        with torch.no_grad():
            B = img0.shape[0]
            img0 = img0.to(self.device)
            img1 = img1.to(self.device)
            masks0 = masks0.to(self.device)
            masks1 = masks1.to(self.device)

            log_mutual, match0, match1, _, _ = self.inner(
                img0, img1, masks0, masks1,
            )

            scores = torch.exp(log_mutual)

            M = masks0.shape[1]
            N = masks1.shape[1]

            ref_to_target = scores.argmax(dim=-1)
            target_to_ref = scores.argmax(dim=-2)

            is_matchable = match0[:, :M] > 0

            safe_indices = ref_to_target.clamp(max=N - 1)
            reciprocal = torch.gather(target_to_ref, 1, safe_indices)
            row_indices = torch.arange(M, device=self.device).unsqueeze(0).expand(B, M)
            is_mutual = reciprocal == row_indices

            valid_mask = is_matchable & is_mutual

            match_result = -torch.ones((B, M), dtype=torch.int64, device=self.device)
            match_result[valid_mask] = ref_to_target[valid_mask]

        return match_result, scores


class SegVGGTInfer(torch.nn.Module):
    """
    Inference wrapper for SegVGGT (VGGT Aggregator single-layer + LightGlue head).

    Checkpoint format (trainer.py):
        {"model_state": ..., "epoch": ..., "global_step": ..., ...}
    """

    def __init__(self, cfg):
        super().__init__()

        _project_root = Path(__file__).parent.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

        from muviseg.models.vggt_lightglue import SegVGGT  # noqa: PLC0415

        model_cfg = cfg["MODEL"]
        self.inner = SegVGGT(
            vggt_ckpt=model_cfg["VGGT_CKPT"],
            layer_idx=model_cfg.get("LAYER_IDX", 23),
            proj_dim=model_cfg.get("PROJ_DIM", 128),
            proj_mid_dim=model_cfg.get("PROJ_MID_DIM", 256),
            n_layers=model_cfg.get("N_LAYERS", 3),
            n_heads=model_cfg.get("N_HEADS", 4),
            ffn_expansion=model_cfg.get("FFN_EXPANSION", 4),
        )

        ckpt_path = model_cfg["CHECKPOINT"]
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading SegVGGT checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.inner.load_state_dict(ckpt["model_state"])
        if 'best_val_ma' in ckpt:
            print(f"  epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}, "
                  f"best_val_ma={ckpt.get('best_val_ma', 'n/a'):.4f}")
        else:
            print(f"  epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}")

    def prepare(self, device):
        self.device = device
        self.inner.eval()
        self.inner.to(device)

    def infer_pair(self, img0, img1, masks0, masks1):
        assert hasattr(self, "device"), "Call prepare(device) before infer_pair"

        with torch.no_grad():
            B = img0.shape[0]
            img0 = img0.to(self.device)
            img1 = img1.to(self.device)
            masks0 = masks0.to(self.device)
            masks1 = masks1.to(self.device)

            log_mutual, match0, match1, _, _ = self.inner(
                img0, img1, masks0, masks1,
            )

            scores = torch.exp(log_mutual)

            M = masks0.shape[1]
            N = masks1.shape[1]

            ref_to_target = scores.argmax(dim=-1)
            target_to_ref = scores.argmax(dim=-2)

            is_matchable = match0[:, :M] > 0

            safe_indices = ref_to_target.clamp(max=N - 1)
            reciprocal = torch.gather(target_to_ref, 1, safe_indices)
            row_indices = torch.arange(M, device=self.device).unsqueeze(0).expand(B, M)
            is_mutual = reciprocal == row_indices

            valid_mask = is_matchable & is_mutual

            match_result = -torch.ones((B, M), dtype=torch.int64, device=self.device)
            match_result[valid_mask] = ref_to_target[valid_mask]

        return match_result, scores


class SegVGGTDPTInfer(torch.nn.Module):
    """
    Inference wrapper for SegVGGTDPT (VGGT Aggregator + DPT fusion + LightGlue head).

    Mirrors the MASt3RSegFeatInfer interface so eval_replica_table2.py works
    without modification (only setup_model needs to branch on arch).

    Checkpoint format (trainer.py):
        {"model_state": ..., "epoch": ..., "global_step": ..., ...}
    """

    def __init__(self, cfg):
        super().__init__()

        # Add project root to sys.path so `training.*` is importable
        _project_root = Path(__file__).parent.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

        from muviseg.models.vggt_dpt_lg import SegVGGTDPT  # noqa: PLC0415

        model_cfg = cfg["MODEL"]
        self.inner = SegVGGTDPT(
            vggt_ckpt=model_cfg["VGGT_CKPT"],
            layer_indices=tuple(model_cfg.get("LAYER_INDICES", [5, 11, 17, 23])),
            fusion_dim=model_cfg.get("FUSION_DIM", 256),
            proj_dim=model_cfg.get("PROJ_DIM", 128),
            n_layers=model_cfg.get("N_LAYERS", 3),
            n_heads=model_cfg.get("N_HEADS", 4),
            ffn_expansion=model_cfg.get("FFN_EXPANSION", 4),
            matchability_bias=model_cfg.get("MATCHABILITY_BIAS", 0.0),
            temperature_init=model_cfg.get("TEMPERATURE_INIT", 1.0),
        )

        ckpt_path = model_cfg["CHECKPOINT"]
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading SegVGGTDPT checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.inner.load_state_dict(ckpt["model_state"])
        print(f"  epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}, "
              f"best_val_ma={ckpt.get('best_val_ma', 'n/a'):.4f}" if 'best_val_ma' in ckpt
              else f"  epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}")

    def prepare(self, device):
        self.device = device
        self.inner.eval()
        self.inner.to(device)

    def infer_pair(self, img0, img1, masks0, masks1):
        assert hasattr(self, "device"), "Call prepare(device) before infer_pair"

        with torch.no_grad():
            B = img0.shape[0]
            img0 = img0.to(self.device)
            img1 = img1.to(self.device)
            masks0 = masks0.to(self.device)  # (B, M, H, W)
            masks1 = masks1.to(self.device)  # (B, N, H, W)

            # Forward: log_mutual (B,M,N), match0 (B,M), match1 (B,N)
            log_mutual, match0, match1, _, _ = self.inner(img0, img1, masks0, masks1)

            scores = torch.exp(log_mutual)  # (B, M, N)
            match_result = _mutual_match_filter(scores, match0, self.device)

        return match_result, scores  # (B, M), (B, M, N)


class SegVGGTDPTJointInfer(torch.nn.Module):
    """
    Inference wrapper for SegVGGTDPTJoint (joint multi-frame model).

    Exposes infer_tuple(images, masks_list) that runs forward_multiframe on
    N frames and returns scores for the query pair (0, 1) only — so eval
    metrics and GT machinery stay pairwise and comparable to pairwise
    baselines.
    """

    def __init__(self, cfg):
        super().__init__()
        _project_root = Path(__file__).parent.parent.parent
        if str(_project_root) not in sys.path:
            sys.path.insert(0, str(_project_root))

        from muviseg.models.vggt_dpt_lg import SegVGGTDPTJoint  # noqa: PLC0415

        model_cfg = cfg["MODEL"]
        self.inner = SegVGGTDPTJoint(
            vggt_ckpt=model_cfg["VGGT_CKPT"],
            layer_indices=tuple(model_cfg.get("LAYER_INDICES", [5, 11, 17, 23])),
            fusion_dim=model_cfg.get("FUSION_DIM", 256),
            proj_dim=model_cfg.get("PROJ_DIM", 128),
            n_joint_layers=model_cfg.get("N_JOINT_LAYERS", 3),
            n_heads=model_cfg.get("N_HEADS", 4),
            ffn_expansion=model_cfg.get("FFN_EXPANSION", 4),
            max_frames=model_cfg.get("MAX_FRAMES", 16),
            matchability_bias=model_cfg.get("MATCHABILITY_BIAS", 0.0),
            temperature_init=model_cfg.get("TEMPERATURE_INIT", 1.0),
        )

        ckpt_path = model_cfg["CHECKPOINT"]
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"Loading SegVGGTDPTJoint checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.inner.load_state_dict(ckpt["model_state"])
        print(f"  epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}, "
              f"best_val_ma={ckpt.get('best_val_ma', 'n/a'):.4f}" if 'best_val_ma' in ckpt
              else f"  epoch={ckpt.get('epoch')}, step={ckpt.get('global_step')}")

    def prepare(self, device):
        self.device = device
        self.inner.eval()
        self.inner.to(device)

    def infer_tuple(self, images, masks_list):
        """
        images: (B, N, 3, H, W) in [-1, 1].
        masks_list: list[N] of (B, M_v, H, W) float.
        Returns match_result (B, M_0), scores (B, M_0, M_1) for pair (0, 1).
        """
        assert hasattr(self, "device"), "Call prepare(device) before infer_tuple"
        with torch.no_grad():
            images = images.to(self.device)
            masks_list = [m.to(self.device) for m in masks_list]
            out = self.inner.forward_multiframe(
                images, masks_list, pair_indices=[(0, 1)]
            )
            log_mutual, match0, _ = out[(0, 1)]
            scores = torch.exp(log_mutual)  # (B, M_0, M_1)
            match_result = _mutual_match_filter(scores, match0, self.device)
        return match_result, scores
