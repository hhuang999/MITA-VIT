#!/usr/bin/env bash
# Runs LOCALLY. Download finished run folders from a Colab session in 8 MB parts (large single transfers are
# cut off by the network proxy), check the SHA-256, and unpack them into experiments/.
# Usage: colab/local/fetch_runs.sh <session> <run folder name> [...]  (needs /content/pack_run.sh on the VM)
set -euo pipefail
SESSION=${1:?session}; shift
COLAB=${COLAB:-colab}
ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
for RUN in "$@"; do
  # Pack detached on the VM (a long exec call can lose its connection), then poll for the checksum file.
  cat > "$WORK/launch.py" <<PY
import sys
sys.argv = ['launch', '/content/pack_run.sh', '/content/pack_run.log', '$RUN']
exec(open('/content/launch.py').read())
PY
  cat > "$WORK/poll.py" <<PY
import os
run = '$RUN'
done = f'/content/out/{run}.done'
if os.path.exists(done):
    print('SHA', open(done).read().strip())
    print('PARTS', ' '.join(sorted(os.listdir(f'/content/out/{run}'))))
PY
  for wait in 0 10 30; do sleep $wait; timeout 200 "$COLAB" exec -s "$SESSION" -f "$WORK/launch.py" --timeout 120 > /dev/null && break; done
  out=""
  for attempt in $(seq 1 60); do
    sleep 10
    out=$(timeout 200 "$COLAB" exec -s "$SESSION" -f "$WORK/poll.py" --timeout 120 2>/dev/null || true)
    echo "$out" | grep -q '^SHA ' && break
  done
  parts=$(echo "$out" | sed -n 's/^PARTS //p')
  sha=$(echo "$out" | sed -n 's/^SHA //p')
  [ -n "$parts" ] && [ -n "$sha" ] || { echo "pack failed for $RUN" >&2; exit 1; }
  for part in $parts; do
    for wait in 0 2 4 8 16; do
      sleep $wait
      timeout 300 "$COLAB" download -s "$SESSION" "/content/out/$RUN/$part" "$WORK/$part" >/dev/null 2>&1 && [ -s "$WORK/$part" ] && break
    done
    [ -s "$WORK/$part" ] || { echo "download failed: $RUN/$part" >&2; exit 1; }
  done
  cat "$WORK"/part.* > "$WORK/$RUN.tar.gz"
  [ "$(sha256sum "$WORK/$RUN.tar.gz" | cut -d' ' -f1)" = "$sha" ] || { echo "SHA mismatch for $RUN" >&2; exit 1; }
  tar xzf "$WORK/$RUN.tar.gz" -C "$ROOT/experiments"
  rm -f "$WORK"/part.* "$WORK/$RUN.tar.gz"
  echo "FETCHED $RUN"
done
rm -r "$WORK"
