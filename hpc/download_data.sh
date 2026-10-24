#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/checkpoints/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/checkpoints/torch}"
.venv/bin/python scripts/download_models.py
.venv/bin/python scripts/download_data.py "${1:?Choose imagenette, pet or imagenet}"
