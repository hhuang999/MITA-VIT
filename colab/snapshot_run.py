"""Fetch a read-only checkpoint through the contents API while inference runs."""

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run')
parser.add_argument('label')
parser.add_argument('--session', default='mita-rq1-l4-20261027')
args = parser.parse_args()
assert args.run in {'2026-10-27_rq1_colab', '2026-10-27_rq1_colab_k128'}
assert args.label.isalnum()
destination = ROOT / 'data/colab_transfer/checkpoints' / f'{args.run}_{args.label}'
destination.mkdir(parents=True, exist_ok=False)
# Get completed-setting summaries before the append-only prediction file.
# Predictions may include the next partial setting: never treat a snapshot as
# a final result without checking complete settings and exact image pairing.
for name in ['results.csv', 'predictions.csv', 'identity_predictions.csv',
             'config.yaml', 'metadata.json', 'indices.json', 'checks.json', 'log.txt']:
    target = destination / name
    local = '/mnt/' + ROOT.drive[0].lower() + target.as_posix()[2:]
    subprocess.run([sys.executable, str(ROOT / 'colab/cli.py'), 'download',
                    '-s', args.session, f'/content/MITA_vit/experiments/{args.run}/{name}',
                    local], check=True)
print(f'Saved checkpoint {args.label}; formal final-result audit is still required.')
