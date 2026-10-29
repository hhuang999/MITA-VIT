#!/usr/bin/env bash
# Runs LOCALLY. Download finished run folders from a Colab session in 8 MB parts (large single transfers are
# cut off by the network proxy), check the SHA-256, and unpack them into experiments/.
# Usage: colab/local/fetch_runs.sh <session> <run folder name> [...]
set -euo pipefail
SESSION=${1:?session}; shift
COLAB=${COLAB:-colab}
ROOT=$(git rev-parse --show-toplevel)
WORK=$(mktemp -d)
for RUN in "$@"; do
  cat > "$WORK/pack.py" <<PY
import hashlib, subprocess
run = '$RUN'
cmd = (f"cd /content/MITA-VIT/experiments && test -f {run}/checks.json && rm -rf /content/out/{run} && "
       f"mkdir -p /content/out/{run} && tar czf /content/out/{run}.tar.gz {run} && "
       f"split -b 8M -d /content/out/{run}.tar.gz /content/out/{run}/part. && ls /content/out/{run}")
listing = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
digest = hashlib.sha256(open(f'/content/out/{run}.tar.gz', 'rb').read()).hexdigest()
print('PARTS', ' '.join(listing.stdout.split()))
print('SHA', digest)
PY
  out=$("$COLAB" exec -s "$SESSION" -f "$WORK/pack.py" --timeout 600)
  parts=$(echo "$out" | sed -n 's/^PARTS //p')
  sha=$(echo "$out" | sed -n 's/^SHA //p')
  [ -n "$parts" ] && [ -n "$sha" ] || { echo "pack failed for $RUN: $out" >&2; exit 1; }
  for part in $parts; do
    for wait in 0 2 4 8 16; do
      sleep $wait
      "$COLAB" download -s "$SESSION" "/content/out/$RUN/$part" "$WORK/$part" >/dev/null 2>&1 && [ -s "$WORK/$part" ] && break
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
