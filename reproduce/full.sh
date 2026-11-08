#!/usr/bin/env bash
# Rerun every experiment of the paper from scratch, then rebuild every table and figure.
# Needs a CUDA GPU (about 12 GPU-hours on one NVIDIA L4), the datasets (scripts/prepare_data.py) and the models
# (scripts/download_models.py). Completed runs are skipped, so the script can be restarted after an interruption
# (an interrupted run continues with: python scripts/run.py --config <config> --resume).
# Order: the lens (diagnose) first, then the main-text elements, then the appendix.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
run() { echo "== $1 ($(date -u +%FT%TZ))"; $PY scripts/run.py --config "configs/$1.yaml" "${@:2}"; }

# Lens: position probes, blur widths, mean maps (figure 3 a-b; needed by every reading below)
run diagnose_probe
run diagnose_mean_maps
# Main text: figure 2(a), figure 3(c-d), Table 1 and sections 4.2-4.4
run fig2a_k_sweep
run attention_off
run high_norm
run clickme_layers
run table1
run table1_register
run table1_mita
# Appendix: B (execution details), C (sanity), D (order, combinations), E (exploratory)
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
