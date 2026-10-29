#!/usr/bin/env bash
# Runs ON the Colab VM. Download and verify datasets. HF_TOKEN must be in the environment for imagenet.
# Usage: bash /content/data.sh imagenet pet
set -euo pipefail
cd /content/MITA-VIT
PY=.venv/bin/python
for dataset in "$@"; do
  if [ "$dataset" = imagenet ]; then
    test -n "${HF_TOKEN:-}" || { echo "HF_TOKEN missing"; exit 1; }
    $PY scripts/download_data.py imagenet
    $PY scripts/verify_imagenet.py --output /content/run_env/data_verification.json
  else
    $PY scripts/download_data.py "$dataset"
  fi
done
echo DATA_DONE
