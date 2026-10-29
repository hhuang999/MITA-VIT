#!/usr/bin/env bash
# Runs LOCALLY. Bundle the branch, upload it to a Colab session in 8 MB parts (large single uploads are cut
# off by the network proxy), join and verify the SHA-256 on the VM, and upload the remote/ scripts.
# Usage: colab/local/send_code.sh <session> <branch> [base commit]
# With a base commit the bundle holds only the commits after it (for update.sh on a VM that has the base).
set -euo pipefail
SESSION=${1:?session}; BRANCH=${2:?branch}; BASE=${3:-}
COLAB=${COLAB:-colab}
ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
git -C "$ROOT" bundle create "$WORK/mita.bundle" ${BASE:+"^$BASE"} "$BRANCH"
SHA=$(sha256sum "$WORK/mita.bundle" | cut -d' ' -f1)
split -b 8M -d "$WORK/mita.bundle" "$WORK/part."
printf "import os\nos.makedirs('/content/parts', exist_ok=True)\nprint('ok')\n" > "$WORK/mkdir.py"
"$COLAB" exec -s "$SESSION" -f "$WORK/mkdir.py" --timeout 60
upload() {  # retry with backoff: the tunnel occasionally drops mid-transfer
  for wait in 0 2 4 8 16; do
    sleep $wait
    "$COLAB" upload -s "$SESSION" "$1" "$2" 2>&1 | grep -q Uploaded && return 0
  done
  echo "upload failed: $1" >&2; return 1
}
for part in "$WORK"/part.*; do upload "$part" "/content/parts/$(basename "$part")"; done
cat > "$WORK/join.py" <<PY
import hashlib, subprocess
subprocess.run(['bash', '-lc', 'cat /content/parts/part.* > /content/mita.bundle && rm -r /content/parts'], check=True)
digest = hashlib.sha256(open('/content/mita.bundle', 'rb').read()).hexdigest()
print('BUNDLE_OK' if digest == '$SHA' else 'BUNDLE_BAD ' + digest)
PY
"$COLAB" exec -s "$SESSION" -f "$WORK/join.py" --timeout 240
for script in "$ROOT"/colab/remote/*; do upload "$script" "/content/$(basename "$script")"; done
rm -r "$WORK"
