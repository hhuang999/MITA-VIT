#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to an isolated Python 3.11/3.12 interpreter; the platform image has Python 3.10}"
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Use Python 3.11 or 3.12"'
if [[ ! -e .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert sys.prefix != sys.base_prefix; assert (3, 11) <= sys.version_info[:2] < (3, 13)'
# Fetch only platform wheels here; resolve their runtime dependencies together
# with the project pins on PyPI rather than the slower PyTorch mirror.
.venv/bin/python -m pip install --no-deps -r requirements-torch.txt --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
.venv/bin/python -m pip install -r requirements-torch.txt -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda); print("CUDA visible:", torch.cuda.is_available())'
echo "Environment installed. This script does not launch tests or GPU inference."
