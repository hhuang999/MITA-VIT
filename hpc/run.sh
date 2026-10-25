#!/usr/bin/env bash
# Run inside a user-approved GPU development environment; this is not a Slurm job.
set -euo pipefail
cd "$(dirname "$0")/.."
rq="${1:?Choose rq1}"
case "$rq" in rq1) ;; *) echo "Only RQ1 is implemented" >&2; exit 2 ;; esac
entry="scripts/run_${rq}.py"
if [[ ! -f "$entry" ]]; then
    echo "$entry has not been implemented; see docs/progress.md" >&2
    exit 1
fi
output="${2:?Provide a new experiment output directory}"
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2
export MPLBACKEND=Agg
.venv/bin/python scripts/hpc_preflight.py --stage ready --output "${output}_preflight.json"
.venv/bin/python -m pytest -q --junitxml="${output}_tests.xml"
.venv/bin/python "$entry" --config "configs/${rq}.yaml" --output "$output"
.venv/bin/python scripts/make_figures.py "$output" --title "RQ1 | ImageNet validation, 5,000 images"
