# Downstream: topological object-goal navigation

Our segment matchers replace the LightGlue localizer of RoboHop / ObjectReact for
InstanceImageNav on HM3D (Habitat), reported as Success Rate / SPL / Soft-SPL.

This experiment runs **outside** the main `muviseg` environment, because
habitat-sim needs conda and Python 3.10 and cannot be installed with uv.

## Setup

```bash
bash downstream/topological_navigation/setup.sh
```

That clones [oravus/object-rel-nav](https://github.com/oravus/object-rel-nav) at
`6da34c3` and applies `overlay.patch` — 48 new files and 7 patched upstream
files. The overlay is kept as a patch rather than a file tree so that it fails
loudly if upstream moves, instead of silently overwriting a different version.

Then, by hand:

```bash
conda create -n nav python=3.10 && conda activate nav
conda install habitat-sim=0.3.3 headless -c conda-forge -c aihabitat
pip install -r object-rel-nav/requirements.txt
pip install -e ../..          # the muviseg package, for the matcher adapters
python ../../scripts/download_checkpoints.py
```

Datasets (HM3D scenes plus the InstanceImageNav episodes) go in
`object-rel-nav/data/`; see that repo's `Readme.md` for the download steps.

## Running

```bash
cd downstream/topological_navigation/object-rel-nav
python main.py -c configs/<name>.yaml
python scripts/summarize_matchers.py        # SR / SPL / Soft-SPL table
python scripts/analyze_table3.py            # paired McNemar + paired bootstrap
```

A matcher is selected by `goal_gen.matcher_name` in the config, dispatched in
`libs/localizer/loc_topo.py` to an adapter in `libs/matcher/`:

| `matcher_name` | adapter | what it is |
|---|---|---|
| `lightglue` | upstream | the RoboHop / ObjectReact baseline |
| `segmast3r_sinkhorn` | `segmast3r_sinkhorn.py` | MASt3R backbone, Sinkhorn OT matcher |
| `segmast3r_lgv2` | `segmast3r_lgv2.py` | MASt3R backbone, LG v2 head |
| `segvggt_dpt` | `segvggt_dpt.py` | VGGT backbone, DPT fusion, pairwise |
| `segvggt_dpt_joint` | `segvggt_dpt_joint.py` | VGGT backbone, joint attention over N frames |
| `roma` | `roma_seg.py` | RoMa dense matcher, votes aggregated per segment |

Representative configs: `baseline_robohop_minival.yaml`, `lgv2_minival_Ss16.yaml`,
`sinkhorn_minival_Ss16.yaml`, `segvggt_dpt_minival_Ss16.yaml`,
`vggt_joint_N4_Ss16.yaml`, `roma_minival_Ss16.yaml`. The `*_val.yaml` variants run
the full validation split rather than minival; `sweep_*` vary the localizer radius
and reference subsampling.

## Read this before quoting any number

**The pipeline is not deterministic.** Repeated identical runs diverge and can flip
an episode's outcome. In our measurements 11 of 16 repeats of the same
matcher/episode/config diverged, and run-to-run spread is roughly ±6 SPL. Report
variance over repeats; never claim zero. Of the matchers we tried, only Sinkhorn
reproduced bitwise.

**Cold start matters more than the matcher.** `goal_gen.init_localization` controls
how the agent first localizes:

* `global` (default here) — match frame 0 against the whole map, using only the
  method's own matcher
* `pose` — seed from the ground-truth start pose
* `zero` — the legacy behaviour: the localizer window starts at index 0 while
  `max_start_distance: hard` drops the agent mid-trajectory, so the true start can
  fall outside the search window entirely

With `zero`, 6 of 12 minival episodes start outside the search window at
`subsample_ref: 16`. `scripts/start_window_analysis.py` measures this.

**At minival sizes nothing here is statistically significant.** With n=10 episodes,
a paired comparison has power around 0.03; our power analysis puts n=36 at 0.39 and
n=108 at 0.91. Prefer more episodes over more table rows, and use
`scripts/analyze_table3.py`, which does a paired McNemar test and a paired
bootstrap over `per_episode_metrics.csv` rather than comparing two aggregate
numbers.

**Two further caveats on the adapters.** The joint adapter discards the trained
matchability head by default; enabling it (`use_matchability`) removes roughly 95%
of accepted matches on HM3D, so it is a large behavioural change rather than a free
improvement. And the Sinkhorn adapter imports the inference wrapper that ships
inside the segmast3r checkout, not `muviseg/evaluation/model_infer.py`, so the
navigation and pairwise Sinkhorn paths are not the same code.

**One upstream footgun:** do not run `scripts/create_maps_hm3d.py` against
`data/hm3d_iin_val/`. It writes graphs under a different filename than the runtime
reads, and would recompute and overwrite all 108 prebuilt maps.

## Reproducing the paper's table

The published table was produced before the cold-start fix. Our own re-measurement
on the fixed code, 3 repeats of 12 minival episodes, did **not** reproduce the
reported advantage of the LG v2 matcher over the LightGlue baseline: both averaged
46.67 SR, paired McNemar p = 1.000. The most coherent reading is that the published
gap was substantially an artifact of the cold-start bug. At n=10, p = 1.000 means
"cannot resolve", not "no effect" — so treat the published ordering of these two
rows as unconfirmed and re-run at a larger episode count if you depend on it.
