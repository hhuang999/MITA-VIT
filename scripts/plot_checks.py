"""Figure: the premise and centre checks. (a) E3: can a linear classifier still read a token's position at
each layer (chance 1/196)? (b) E5: drop of object mass when the same object moves from the centre crop to
the corners, per method (smaller = follows the object). (c) E6: removing object tokens from the reading sets
against removing as many tokens at the same distance from the centre (agreement difference; below 0 = the
object itself matters).
Reads identity.csv, analysis_shift.csv and summary.csv (analyze_matched_deletion.py).

Usage: python scripts/plot_checks.py --identity RUN [RUN ...] --shift RUN [RUN ...] --deletion RUN [RUN ...]
       --output figures/figure11_checks
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mita_probe.plotting import INK, MAIN_METHODS, MODEL_LABELS, QUIET, interval_dots, save

MODEL_STYLES = {'deit': '-', 'dino': '--', 'deit_b': ':'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', type=Path, nargs='+', required=True)
    parser.add_argument('--shift', type=Path, nargs='+', required=True)
    parser.add_argument('--deletion', type=Path, nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(1, 2 + len(args.shift), figsize=(3.8 * (2 + len(args.shift)), 3.4))
    for run in args.identity:
        table = pd.read_csv(run / 'identity.csv').sort_values('layer')
        model = table.model.iloc[0]
        axes[0].plot(table.layer, table.position_accuracy, MODEL_STYLES.get(model, '-'), color=INK, lw=2,
                     marker='o', ms=3, label=MODEL_LABELS.get(model, model))
    axes[0].axhline(1 / 196, color=QUIET, lw=1, ls=':')
    axes[0].set(xlabel='Layer', ylabel='Position read by a linear classifier', ylim=(0, 1.02),
                title='(a) Does token j still mean patch j?', xticks=range(1, 13))
    axes[0].legend(fontsize=7)
    for ax, run in zip(axes[1:-1], args.shift):
        model = json.loads((run / 'metadata.json').read_text())['model']
        table = pd.read_csv(run / 'analysis_shift.csv').query("measure == 'mass_drop'").set_index('method')
        interval_dots(ax, table, MAIN_METHODS)
        ax.axvline(0, color='#c3c2b7', lw=1)
        ax.set(xlabel='Object-mass drop, centre crop to corners', title=f'(b) {MODEL_LABELS.get(model, model)}: object moved')
    rows = []
    for run in args.deletion:
        dataset = json.loads((run / 'metadata.json').read_text()).get('dataset', 'pet')
        summary = pd.read_csv(run / 'summary.csv').query("category == 'foreground'")
        rows += [dict(label=f"{MODEL_LABELS.get(r.model, r.model)}, {dataset}", mean=r.target_minus_matched,
                      low=r.target_minus_matched_low, high=r.target_minus_matched_high) for r in summary.itertuples()]
    rows = pd.DataFrame(rows)
    for y, row in rows.iterrows():
        axes[-1].plot([row['low'], row['high']], [y, y], color=INK, lw=2)
        axes[-1].plot(row['mean'], y, 'o', color=INK, ms=6, mec='white', mew=1.5)
    axes[-1].set_yticks(range(len(rows)), rows.label, fontsize=7)
    axes[-1].invert_yaxis()
    axes[-1].axvline(0, color='#c3c2b7', lw=1)
    axes[-1].grid(axis='y', visible=False)
    axes[-1].set(xlabel='Agreement difference', title='(c) Object vs same distance')
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
