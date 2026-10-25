#!/usr/bin/env bash
# User-reviewed first GPU job: preflight, tests, then 50 ImageNet images only.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:?Usage: bash hpc/first_job.sh experiments/YYYY-MM-DD_hpc_first}"
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 2; }
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2
export MPLBACKEND=Agg
.venv/bin/python scripts/hpc_preflight.py --stage ready --output "$output/preflight.json"
.venv/bin/python -m pytest -q --junitxml="$output/tests.xml"
.venv/bin/python scripts/run_rq1.py --config configs/rq1_hpc_smoke.yaml --output "$output/smoke"
.venv/bin/python scripts/make_figures.py "$output/smoke" --title "HPC smoke | ImageNet, 50 images"
echo "First job complete. Results: $output"
echo "Formal RQ1 is a separate command; it has not been started by this script."
