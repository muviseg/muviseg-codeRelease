"""
Default config for SegMASt3R training.
Keeps our keys separate from the original project's default.py.
"""
from yacs.config import CfgNode as CN

_CN = CN()

# ── Model ─────────────────────────────────────────────────────────
_CN.MODEL = CN()
_CN.MODEL.MAST3R_CKPT = ""
_CN.MODEL.ARCH        = "sinkhorn"   # sinkhorn | lightglue | lightglue_v2 | vggt | vggt_dpt
_CN.MODEL.VGGT_CKPT      = "facebook/VGGT-1B"  # local .pt path or HF model id
_CN.MODEL.VGGT_LAYER_IDX = 23                   # aggregator layer index (0–23), single-layer mode
_CN.MODEL.VGGT_LAYER_INDICES = (5, 11, 17, 23)  # multi-layer extraction for DPT fusion
_CN.MODEL.VGGT_FUSION_DIM    = 256              # DPT fusion output dim (D_hat)
_CN.MODEL.MATCHABILITY_BIAS  = 0.0             # init bias for matchability logit (>0 → less dustbin)
_CN.MODEL.TEMPERATURE_INIT   = 1.0             # learnable temperature init (1.0 = neutral)
_CN.MODEL.N_JOINT_LAYERS     = 3              # joint multi-frame attention layers (multiframe arch)
_CN.MODEL.MAX_FRAMES         = 16             # max frames for learnable frame embeddings

# LightGlue-style architecture params (MODEL.ARCH = "lightglue")
_CN.MODEL.LG = CN()
_CN.MODEL.LG.PROJ_DIM            = 128   # project MASt3R 24-dim → this
_CN.MODEL.LG.N_LAYERS            = 3     # self+cross attention blocks
_CN.MODEL.LG.N_HEADS             = 4     # attention heads
_CN.MODEL.LG.GRAD_CHECKPOINT     = False # gradient checkpointing (saves VRAM)
_CN.MODEL.LG.DEEP_SUPERVISION    = False # compute loss at every layer
_CN.MODEL.LG.LAMBDA_MATCH        = 1.0   # weight for matchability BCE loss

# LightGlue v2 extra params (MODEL.ARCH = "lightglue_v2")
# Shared params (PROJ_DIM, N_LAYERS, etc.) are read from MODEL.LG.
_CN.MODEL.LG_V2 = CN()
_CN.MODEL.LG_V2.PROJ_MID_DIM     = 64    # MLP projector hidden dim (24→mid→proj_dim)
_CN.MODEL.LG_V2.FFN_EXPANSION    = 4     # FFN multiplier (v1 uses 2×, v2 uses 4×)

# ── Misc ──────────────────────────────────────────────────────────
_CN.DEBUG    = False
_CN.SAVE_DIR = "results/segmast3r_repro"
_CN.RESUME   = ""          # path to checkpoint, or "" for fresh start

# ── Feature matcher ───────────────────────────────────────────────
_CN.FEATURE_MATCHER = CN()
_CN.FEATURE_MATCHER.TYPE = "Sinkhorn"
_CN.FEATURE_MATCHER.SINKHORN = CN()
_CN.FEATURE_MATCHER.SINKHORN.NUM_IT             = 50
_CN.FEATURE_MATCHER.SINKHORN.DUSTBIN_SCORE_INIT = 1.0

# ── Dataset ───────────────────────────────────────────────────────
_CN.DATASET = CN()
_CN.DATASET.METADATA_PATH = ""
_CN.DATASET.DATA_ROOT     = ""
_CN.DATASET.SEGDATA_ROOT  = ""
_CN.DATASET.PAIRS_ROOT    = ""
_CN.DATASET.HEIGHT        = 336
_CN.DATASET.WIDTH         = 512
_CN.DATASET.RESIZE_MODE   = "square"    # square | longest_side
_CN.DATASET.VAL_FRACTION          = 0.02
_CN.DATASET.MAX_PAIRS     = 0   # 0 = use all pairs; >0 = cap dataset size (before train/val split)
_CN.DATASET.PAIR_DSC_ROOT = ""  # per-pair precomputed descriptors; "" = online backbone
# Multi-frame tuple settings (used by vggt_dpt_multiframe arch)
_CN.DATASET.N_FRAMES              = 2    # number of frames per tuple (>=2)
_CN.DATASET.MIN_PAIRS_PER_TUPLE   = 2    # minimum GT pairs required in a tuple
_CN.DATASET.MAX_PAIRS_PER_TUPLE   = 6    # subsample if more GT pairs available
_CN.DATASET.RANDOM_NEIGHBOR_PROB  = 0.2  # probability of random (vs greedy) expansion

# ── Training ──────────────────────────────────────────────────────
_CN.TRAINING = CN()
_CN.TRAINING.BATCH_SIZE      = 36
# Seeding. SEED makes a run repeatable: weight init, batch order and the
# dataloader workers' RNG all derive from it. LEGACY_RNG restores the
# original behaviour, in which nothing but the train/val split was seeded;
# it exists to document that behaviour, not because it reproduces anything.
_CN.TRAINING.SEED = 42
_CN.TRAINING.LEGACY_RNG = False
_CN.TRAINING.NUM_WORKERS     = 8
_CN.TRAINING.PREFETCH_FACTOR = 2
_CN.TRAINING.LR              = 1e-4
_CN.TRAINING.WEIGHT_DECAY    = 1e-4
_CN.TRAINING.EPOCHS          = 5
_CN.TRAINING.GRAD_CLIP       = 0.0
_CN.TRAINING.LR_SCHEDULER    = "cosine"  # cosine | none
_CN.TRAINING.WARMUP_STEPS    = 500
_CN.TRAINING.LOG_INTERVAL    = 100
_CN.TRAINING.VAL_INTERVAL    = 5000
_CN.TRAINING.VAL_MAX_BATCHES = 0      # 0 = full val set; >0 = subsample (prevents NCCL timeout)
_CN.TRAINING.SAVE_INTERVAL   = 5000

# ── Accelerate ────────────────────────────────────────────────────
_CN.ACCELERATE = CN()
_CN.ACCELERATE.MIXED_PRECISION = "no"   # no | bf16 | fp16
_CN.ACCELERATE.GRADIENT_ACCUMULATION_STEPS = 1

cfg = _CN
