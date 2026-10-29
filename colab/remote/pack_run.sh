#!/usr/bin/env bash
# Runs ON the Colab VM (started detached by colab/local/fetch_runs.sh): pack one run folder into 8 MB parts
# and write its SHA-256 to /content/out/<run>.done. Usage: bash /content/pack_run.sh <run folder name>
run=$1
cd /content/MITA-VIT/experiments
rm -rf /content/out/$run /content/out/$run.tar.gz /content/out/$run.done
mkdir -p /content/out/$run
tar czf /content/out/$run.tar.gz $run && split -b 8M -d /content/out/$run.tar.gz /content/out/$run/part. \
  && sha256sum /content/out/$run.tar.gz | cut -d' ' -f1 > /content/out/$run.done
