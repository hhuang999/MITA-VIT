#!/usr/bin/env bash
# Authorized HPC-only completion of the existing A800 RQ1 sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
output="${1:?Usage: bash hpc/rq1_k128_a800.sh experiments/YYYY-MM-DD_rq1_k128_a800}"
checks="${output}_checks"
[[ ! -e "$output" && ! -e "$checks" ]] || {
  echo "Output or checks directory already exists" >&2
  exit 2
}
mkdir -p "$checks"
exec > >(tee "$checks/job.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$checks/exit_code.txt"' EXIT
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2
export MPLBACKEND=Agg
date -Iseconds
git rev-parse HEAD
.venv/bin/python scripts/hpc_preflight.py --stage ready --output "$checks/preflight.json"
.venv/bin/python - "$checks/preflight.json" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
expected = "NVIDIA A800-SXM4-80GB"
actual = report.get("gpu", {}).get("name")
if actual != expected:
    raise SystemExit(f"Need {expected} to complete the original hardware sweep; got {actual}")
PY
.venv/bin/python -m pytest -q --basetemp "$checks/pytest-tmp" --junitxml="$checks/tests.xml"
# run_rq1 checks k=197 on all 5,000 images before running the seven settings.
.venv/bin/python scripts/run_rq1.py --config configs/rq1_k128.yaml --output "$output"
.venv/bin/python scripts/audit_overnight.py rq1 "$output" \
  --reference experiments/2026-10-26_rq1_imagenet
date -Iseconds
echo "A800 k=128 complete; retrieve outputs, then stop the platform environment."
