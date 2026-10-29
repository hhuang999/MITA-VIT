#!/usr/bin/env bash
# Runs ON the Colab VM. Run the jobs in /content/jobs.txt in order ("name<TAB>command" per line), logging each
# to /content/run_env/<name>.log and its exit code to run_status.txt. A failing job stops the queue.
# The GPU test suite runs first.
set -uo pipefail
cd /content/MITA-VIT
ENV=/content/run_env
mkdir -p $ENV
STATUS=$ENV/run_status.txt
note() { echo "$(date -u +%FT%TZ) $*" | tee -a $STATUS; }
.venv/bin/python -m pytest -q -p no:cacheprovider > $ENV/gpu_tests.log 2>&1; note "gpu_tests exit=$?"
grep -q "failed" $ENV/gpu_tests.log && { note "ABORT gpu tests failed"; exit 1; }
while IFS=$'\t' read -r name command; do
  [ -z "$name" ] && continue
  bash -c "$command" > "$ENV/$name.log" 2>&1
  code=$?; note "$name exit=$code"
  [ $code -eq 0 ] || { note "ABORT $name failed"; exit 1; }
done < /content/jobs.txt
note "ALL_DONE"
