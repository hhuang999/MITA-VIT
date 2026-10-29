#!/usr/bin/env bash
# Runs ON the Colab VM. Move the existing clone to the newly uploaded bundle without touching data/ or
# checkpoints/ (setup.sh re-clones and would delete them). Usage: bash /content/update.sh <branch>
# The forced checkout replaces untracked files in the way, i.e. result files that were downloaded, tidied and
# committed locally (same content); other untracked files, such as the large raw tables, stay.
set -euo pipefail
BRANCH=${1:?branch}
cd /content/MITA-VIT
git fetch -q /content/mita.bundle "$BRANCH"
git merge-base --is-ancestor HEAD FETCH_HEAD  # fast-forward only
git checkout -q -f -B "$BRANCH" FETCH_HEAD
git log --oneline -1
echo UPDATE_DONE
