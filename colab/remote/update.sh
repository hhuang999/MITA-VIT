#!/usr/bin/env bash
# Runs ON the Colab VM. Move the existing clone to the newly uploaded bundle without touching data/ or
# checkpoints/ (setup.sh re-clones and would delete them). Usage: bash /content/update.sh <branch>
set -euo pipefail
BRANCH=${1:?branch}
cd /content/MITA-VIT
git pull -q --ff-only /content/mita.bundle "$BRANCH"
git log --oneline -1
echo UPDATE_DONE
