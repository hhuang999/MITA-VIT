#!/usr/bin/env bash
# Smoke test: every experiment on its first 16 images with DeiT-S (MiTA-DeiT-S for its Table 1 row), then the
# tables, into outputs/smoke and results/smoke. Covers every reading, metric and execution check; under 10 minutes
# on one GPU. Needs the datasets and models, like full.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=outputs/smoke
rm -rf "$OUT" results/smoke
run() { echo "== $1"; $PY scripts/run.py --config "configs/$1.yaml" --output "$OUT" --smoke 16 "${@:2}"; }
for config in diagnose_probe diagnose_mean_maps fig2a_k_sweep attention_off single_layer high_norm clickme_layers \
              table1 table1_register order sanity part_reliance map_similarity value_norm register c2_analysis \
              supplementary_heldout supplementary_eval d2; do
  run "$config" --model deit
done
run table1_mita
$PY scripts/make_tables.py --outputs "$OUT" --results results/smoke
echo SMOKE_DONE
