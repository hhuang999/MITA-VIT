#!/usr/bin/env bash
# Rebuild every summary table (results/), figure (figures/) and LaTeX table (paper/tables/) from saved run
# outputs (outputs/, written by full.sh or unpacked from an archive of a full run). CPU only, a few minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY scripts/make_tables.py --outputs "${1:-outputs}" --results "${2:-results}"
$PY scripts/make_figures.py --outputs "${1:-outputs}" --results "${2:-results}" --figures "${3:-figures}"
$PY scripts/make_latex_tables.py --results "${2:-results}"
