#!/usr/bin/env bash
# SegVGGT-DPT across training checkpoints, on both benchmarks.
#   bash scripts/sweeps/run_eval_segvggt_dpt_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-uv run python}"
cd "$ROOT"

CKPT_DIR="results/segvggt_dpt/v3-001"
OUT="results/sweeps/segvggt_dpt"
TMP="$OUT/tmp_configs"
mkdir -p "$TMP"

CKPTS=("best" "epoch_001" "epoch_002" "epoch_003")

for name in "${CKPTS[@]}"; do
  for ds in replica vkitti2; do
    src="configs/eval/${ds}_segvggt.yaml"
    cfg="$TMP/${ds}_${name}.yaml"
    sed "s|^\( *CHECKPOINT: *\).*|\1$CKPT_DIR/$name.pth|" "$src" > "$cfg"
    echo "==> $ds / $name"
    $PY "scripts/eval_${ds}.py" --config "$cfg" --output_dir "$OUT/${name}_${ds}" \
      2>&1 | tee "$OUT/${name}_${ds}.log" | tail -2
  done
done

echo
$PY scripts/sweeps/summarize.py --percent --glob "$OUT/*_replica" | tee "$OUT/summary_replica.txt"
echo
$PY scripts/sweeps/summarize.py --percent --glob "$OUT/*_vkitti2" | tee "$OUT/summary_vkitti2.txt"
