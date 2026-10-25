#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/download_models.py
.venv/bin/python scripts/download_data.py "${1:?Choose imagenette, pet or imagenet}"
