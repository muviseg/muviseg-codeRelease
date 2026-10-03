#!/usr/bin/env bash
# SegVGGT-DPT Joint across joint-window sizes N, on both benchmarks.
#   bash scripts/sweeps/run_eval_segvggt_joint_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-uv run python}"
cd "$ROOT"

OUT="results/sweeps/segvggt_dpt_joint"
TMP="$OUT/tmp_configs"
mkdir -p "$TMP"

for N in 2 4 6 8; do
  for ds in replica vkitti2; do
    src="configs/eval/${ds}_segvggt_joint.yaml"
    cfg="$TMP/${ds}_N${N}.yaml"
    sed "s|^\( *N_FRAMES: *\).*|\1$N|" "$src" > "$cfg"
    echo "==> $ds / N=$N"
    $PY "scripts/eval_${ds}.py" --config "$cfg" --output_dir "$OUT/N${N}_${ds}" \
      2>&1 | tee "$OUT/N${N}_${ds}.log" | tail -2
  done
done

echo
$PY scripts/sweeps/summarize.py --percent --glob "$OUT/N*_replica" | tee "$OUT/summary_replica.txt"
echo
$PY scripts/sweeps/summarize.py --percent --glob "$OUT/N*_vkitti2" | tee "$OUT/summary_vkitti2.txt"
