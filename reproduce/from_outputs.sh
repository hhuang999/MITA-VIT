#!/usr/bin/env bash
# Rebuild only the final paper's results, figures and tables from saved outputs. CPU only.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=${1:-outputs}
RESULTS=${2:-results}
FIGURES=${3:-figures}
"$PY" scripts/make_tables.py --outputs "$OUT" --results "$RESULTS"
"$PY" scripts/make_figures.py --outputs "$OUT" --results "$RESULTS" --figures "$FIGURES"
"$PY" scripts/make_latex_tables.py --results "$RESULTS"
"$PY" -I paper/make/make_main_table.py "$RESULTS" paper/tables_own/main_results.tex
"$PY" -I paper/make/diag_table.py "$RESULTS/diagnose.csv" paper/tables_own/diagnostics_per_layer.tex
"$PY" -I paper/make/make_paired_table.py paper/tables paper/tables_own/paired_differences.tex
"$PY" -I paper/make/make_diagnosis_figure.py "$RESULTS" paper/figures_own/diagnosis.pdf
