#!/usr/bin/env bash
# CPU-only rebuild. Usage: bash reproduce/from_outputs.sh [outputs-root] [fresh-build-directory]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
OUT=${1:-outputs}
DEST=${2:-build/rebuilt}
"$PY" scripts/rebuild_results.py --outputs "$OUT" --destination "$DEST"
