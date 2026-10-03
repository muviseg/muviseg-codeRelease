# Release checklist

What was done to turn the research tree into this repository, and what is still
open. Kept in the repo so the state is not folklore.

## Done

- [x] Single installable package, importable and runnable from any directory
- [x] `uv sync` from the committed lock reproduces the paper environment
      (Python 3.11, torch 2.10.0+cu128)
- [x] No machine-specific paths in any config; one `MUVISEG_DATA_ROOT`
- [x] Pinned third-party setup over HTTPS, with our patches deployed
- [x] Docker image that actually builds and installs the project
- [x] One evaluation entry point for both benchmarks
- [x] Reproducible training from a seed (`TRAINING.SEED`), run provenance
      recorded in every checkpoint
- [x] Inference entry point for arbitrary frame sequences
- [x] Downstream navigation overlay, as a patch against a pinned upstream
- [x] Checkpoints published on the Hugging Face Hub, fetched by script
- [x] Smoke tests that need neither GPU nor dataset
- [x] README covering install, evaluation, training, inference, library use and
      the downstream task
- [x] `docs/reproduction.md` stating what reproduces and what does not

## Verification standard used

Every refactor step was gated on a numeric comparison, not on review alone.

The golden set is the 14 shipped evaluation configs over a stratified pair
subset: 2 pairs from every (scene, variant, bin) cell, so all four pose bins are
exercised (Replica 64 pairs, VKITTI2 96). A plain `--num_pairs N` prefix will not
do, because both pair lists are grouped scene-major then bin-major, so a small
prefix sits entirely inside one bin of one scene.

A step was accepted only when all runnable configs produced byte-identical
`metrics_by_bin.json` against the pre-refactor tree, and the configs that cannot
run failed in the same way. The package move, the path refactor and the
evaluation unification each passed at 8 identical / 0 differing.

## Open

- [ ] **Confirm `SegMASt3R/segmast3r` is publicly cloneable.** If it is not, no
      external user can build the MASt3R-backbone models, and
      `setup_third_party.sh` fails at its first step. Check while logged out.
- [ ] **Reconcile the Table 2 Overall column with the camera-ready.** The per-bin
      values in the lab notes match the artifacts exactly, but the Overall
      columns of five of seven Replica rows do not: the artifacts give
      79.70 / 74.10 / 90.11 for SegVGGT-DPT v3 where the notes say
      79.10 / 72.73 / 89.65. The artifact values are the query-weighted means of
      the per-bin numbers, so they are the correct ones. This also narrows the
      reported Joint-vs-pairwise gap from +2.27 to +1.87 AUPRC.
- [ ] Regenerate the VKITTI2 figures if the published ones used the old axis
      labels: the data is binned 0-20/20-40/40-60/60-90, not 0-45/.../135-180.
- [ ] Decide how to present the two rows that cannot be reproduced from this
      release (Sinkhorn, and LG v2 on Replica) -- see `reproduction.md`.
- [ ] Optional: a pairwise-vs-joint ablation at matched N. The current
      comparison varies N within the joint model only.

## Not planned

- Unifying the four diverged copies of `SegmentAttentionLayerV2` /
  `DoubleSoftmaxMatcher`. They are not identical any more -- the `vggt_dpt_lg`
  copy has `matchability_bias` and a learnable temperature that the others lack
  -- so collapsing them would change behaviour for some checkpoints.
- Reproducing the published checkpoints by retraining. Training was unseeded when
  they were made, so no seed recovers them. Seeding only makes *new* runs
  repeatable.
