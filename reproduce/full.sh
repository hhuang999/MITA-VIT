#!/usr/bin/env bash
# Rerun the paper experiments with external data and checkpoints. A CUDA GPU is required.
# Rerun the frozen-study experiments. Validated completed runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=${1:-outputs/rerun}
DEST=${2:-build/gpu-rerun}
for config in diagnose_probe diagnose_mean_maps k_sweep attention_off high_norm clickme_layers \
              table1 register table1_register table1_mita value_norm sanity c2_analysis \
              supplementary_heldout supplementary_eval; do
  "$PY" scripts/run.py --config "configs/$config.yaml" --output "$OUT"
done
bash reproduce/from_outputs.sh "$OUT" "$DEST"
