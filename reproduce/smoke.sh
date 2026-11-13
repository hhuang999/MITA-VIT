#!/usr/bin/env bash
# Run each paper configuration on 16 images, using external data and a CUDA GPU.
# Existing completed smoke runs are kept and skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=outputs/smoke
for config in diagnose_probe diagnose_mean_maps k_sweep attention_off high_norm clickme_layers \
              table1 register table1_register value_norm sanity c2_analysis supplementary_heldout supplementary_eval; do
  "$PY" scripts/run.py --config "configs/$config.yaml" --output "$OUT" --smoke 16 --model deit
done
"$PY" scripts/run.py --config configs/table1_mita.yaml --output "$OUT" --smoke 16
"$PY" scripts/make_tables.py --outputs "$OUT" --results build/smoke/results
echo SMOKE_DONE
