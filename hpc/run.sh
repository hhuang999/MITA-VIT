#!/usr/bin/env bash
# Run inside a user-approved GPU development environment; this is not a Slurm job.
set -euo pipefail
cd "$(dirname "$0")/.."
rq="${1:?Choose rq1, rq2 or rq3}"
case "$rq" in rq1|rq2|rq3) ;; *) echo "Unknown RQ: $rq" >&2; exit 2 ;; esac
entry="scripts/run_${rq}.py"
if [[ ! -f "$entry" ]]; then
    echo "$entry has not been implemented; see docs/progress.md" >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$PWD/checkpoints/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/checkpoints/torch}"
.venv/bin/python -m pytest
.venv/bin/python "$entry" --config "configs/${rq}.yaml"
