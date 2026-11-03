#!/usr/bin/env bash
# Compress the large per-image tables (> 5 MB) of attention-budget runs to .csv.gz before committing them;
# scripts/analyze_budget.py reads either form. Usage: scripts/compress_tables.sh experiments/<run> [...]
set -euo pipefail
for run in "$@"; do
  find "$run" -maxdepth 1 -name '*.csv' -size +5M -exec gzip -9 {} \;
done
