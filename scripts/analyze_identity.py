"""Summarise identity runs (E3; decision B08): position accuracy and self-similarity per layer and model.

Usage: python scripts/analyze_identity.py RUN [RUN ...]; writes analysis_identity.csv next to the first run's
folder (in experiments/) and prints the table. Chance accuracy is 1/196.
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    table = pd.concat([pd.read_csv(run / 'identity.csv') for run in args.runs])
    output = args.output or args.runs[0].parent / 'analysis_identity.csv'
    table.to_csv(output, index=False)
    print(table.pivot(index='layer', columns='model', values=['position_accuracy', 'self_cosine']).round(3).to_string())
    print('chance', round(1 / 196, 4))


if __name__ == '__main__':
    main()
