#!/usr/bin/env bash
# Rerun every experiment of the paper from scratch, then rebuild every table and figure.
# Needs a CUDA GPU (about 12 GPU-hours on one NVIDIA L4), the datasets (scripts/prepare_data.py) and the models
# (scripts/download_models.py). Completed runs are skipped, so the script can be restarted after an interruption
# (an interrupted run continues with: python scripts/run.py --config <config> --resume).
# Order: the lens (diagnose) first, then the main results, then the supporting and exploratory checks.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
run() { echo "== $1 ($(date -u +%FT%TZ))"; $PY scripts/run.py --config "configs/$1.yaml" "${@:2}"; }

# Lens: position probes, blur widths, mean maps (premises of C2 and C1; needed by every reading below)
run diagnose_probe
run diagnose_mean_maps
# Main results: execution test (k sweep, attention removed), high-norm premise, per-layer ClickMe alignment,
# main results table with its model-side variants
run k_sweep
run attention_off
run high_norm
run clickme_layers
run table1
run table1_register
run table1_mita
# Supporting checks: single-layer and part-reliance checks, top-k map similarity, value norms, register search,
# other order of the corrections, sanity check; then the exploratory analyses (C2, combinations, discount)
run single_layer
run part_reliance
run map_similarity
run value_norm
run register
run order
run sanity
run c2_analysis
run supplementary_heldout
run supplementary_eval
run d2

$PY scripts/make_tables.py
$PY scripts/make_figures.py
$PY scripts/make_latex_tables.py
