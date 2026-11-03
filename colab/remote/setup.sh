#!/usr/bin/env bash
# Runs ON the Colab VM. Clone the uploaded bundle, build the pinned environment, fetch and verify weights.
# Usage: bash /content/setup.sh <branch>
set -euo pipefail
BRANCH=${1:?branch}
cd /content
rm -rf MITA-VIT && git clone -q -b "$BRANCH" /content/mita.bundle MITA-VIT
cd MITA-VIT && git log --oneline -1
uv venv -q --python 3.12 .venv
PY=/content/MITA-VIT/.venv/bin/python
uv pip install -q --python $PY torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python $PY -r requirements.txt
uv pip install -q --python $PY -e . --no-deps
uv pip install -q --python $PY gdown
mkdir -p checkpoints/mita
.venv/bin/gdown -q --folder https://drive.google.com/drive/folders/175gOzODWbS_6sap-tXz-9ufh8-y2Mq7k -O /content/mita_ckpt
mv /content/mita_ckpt/best_checkpoint.pth /content/mita_ckpt/log.txt checkpoints/mita/
mkdir -p checkpoints/mita_tiny   # MiTA-DeiT-T (attention budget); folder from the MiTA model zoo
.venv/bin/gdown -q --folder https://drive.google.com/drive/folders/1KnFqHpJLc_NddIjKnRJEb9McQjN4DXgk -O /content/mita_tiny_ckpt
mv /content/mita_tiny_ckpt/best_checkpoint.pth /content/mita_tiny_ckpt/log.txt checkpoints/mita_tiny/
$PY scripts/download_models.py   # DeiT-S, DINO + head; verifies every SHA-256
$PY scripts/check_environment.py
echo SETUP_DONE
