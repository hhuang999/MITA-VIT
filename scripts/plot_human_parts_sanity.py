"""Figures for E7 (ClickMe), E8 (PartImageNet) and the sanity check; one column per model.

human: (row 1) image-specific Spearman with the human map (own - another image's map) per method; (row 2)
the same for the probe at each single layer. parts: (row 1) head enrichment in animal images per method;
(row 2) the model's own reliance, log-probability drop when 4 patches of one part are removed, per animal
part, next to the object and background controls; (row 3) head enrichment of the probe per layer.
sanity: median Spearman with the trained model's map as more of the model is re-initialised.
Reads only the analysis_*.csv tables.

Usage: python scripts/plot_human_parts_sanity.py human RUN [RUN ...] --output figures/figure9_human
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mita_probe.plotting import (BLUE, GRAY, MAIN_METHODS, METHOD_COLORS, METHOD_LABELS, METHOD_STYLES,
                                 MODEL_LABELS, QUIET, band, interval_dots, save)

LEVEL_LABELS = {'head': 'head', 'embed': 'emb'}


def model_of(run):
    return json.loads((run / 'metadata.json').read_text())['model']


def human(runs, output):
    fig, axes = plt.subplots(2, len(runs), figsize=(4 * len(runs), 6.4), squeeze=False)
    for column, run in zip(axes.T, runs):
        model = MODEL_LABELS.get(model_of(run), model_of(run))
        table = pd.read_csv(run / 'analysis_alignment.csv')
        specific = table.query("measure == 'specific_pixel'").set_index('method')
        interval_dots(column[0], specific, MAIN_METHODS)
        column[0].axvline(0, color='#c3c2b7', lw=1)
        column[0].set(title=f'{model}: ClickMe', xlabel='Image-specific Spearman')
        layers = pd.read_csv(run / 'analysis_layers.csv').query("measure == 'specific_pixel'").sort_values('layer')
        band(column[1], layers.layer, layers, BLUE, label='Probe at one layer')
        overall = specific.loc['probe_mass']
        column[1].axhline(overall['mean'], color=BLUE, ls='--', lw=1, label='Probe, layers 9-12')
        column[1].axhline(0, color='#c3c2b7', lw=1)
        column[1].set(xlabel='Layer', ylabel='Image-specific Spearman', xticks=range(1, 13))
        column[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, output)


def parts(runs, output):
    fig, axes = plt.subplots(3, len(runs), figsize=(4.2 * len(runs), 9.6), squeeze=False)
    for column, run in zip(axes.T, runs):
        model = MODEL_LABELS.get(model_of(run), model_of(run))
        table = pd.read_csv(run / 'analysis_parts.csv').query("measure == 'enrichment'")
        animals = table[table.supercategory.isin(['Quadruped', 'Biped', 'Fish', 'Bird', 'Snake', 'Reptile'])]
        heads = animals.query("name == 'Head'")
        # n-weighted mean over animal super-categories (each row is one super-category)
        pooled = heads.assign(w=heads.n).groupby('method').apply(
            lambda g: pd.Series({k: (g[k] * g.w).sum() / g.w.sum() for k in ('mean', 'low', 'high')}),
            include_groups=False)
        interval_dots(column[0], pooled, MAIN_METHODS)
        column[0].axvline(1, color='#c3c2b7', lw=1)
        column[0].set(title=f'{model}: PartImageNet', xlabel='Head enrichment (mass share / area share)')
        reliance = pd.read_csv(run / 'analysis_reliance.csv').query('removed == 4')
        rows = reliance[reliance.supercategory.isin(['Quadruped', 'Bird', 'control'])]
        rows = rows.assign(label=rows.supercategory.where(rows.supercategory != 'control', '') + ' ' + rows.name)
        for y, (_, row) in enumerate(rows.iterrows()):
            color = QUIET if row.supercategory == 'control' else (BLUE if row['name'] == 'Head' else GRAY)
            column[1].plot([row['low'], row['high']], [y, y], color=color, lw=2)
            column[1].plot(row['mean'], y, 'o', color=color, ms=6, mec='white', mew=1.5)
        column[1].set_yticks(range(len(rows)), rows.label.str.strip())
        column[1].invert_yaxis()
        column[1].grid(axis='y', visible=False)
        column[1].set(xlabel='Log-probability drop, 4 patches removed')
        layers = pd.read_csv(run / 'analysis_layers.csv').sort_values('layer')
        band(column[2], layers.layer, layers, BLUE, label='Probe at one layer')
        column[2].axhline(1, color='#c3c2b7', lw=1)
        column[2].set(xlabel='Layer', ylabel='Head enrichment', xticks=range(1, 13))
    fig.tight_layout()
    save(fig, output)


def sanity(runs, output):
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.4), squeeze=False)
    for ax, run in zip(axes[0], runs):
        model = MODEL_LABELS.get(model_of(run), model_of(run))
        table = pd.read_csv(run / 'analysis_sanity.csv')
        steps = table.drop_duplicates('step').sort_values('step')
        for method in MAIN_METHODS:
            curve = table.query('method == @method').sort_values('step')
            if len(curve):
                ax.plot(curve.step, curve['median'], METHOD_STYLES.get(method, '-'), color=METHOD_COLORS[method],
                        lw=2, marker='o', ms=3, label=METHOD_LABELS[method])
        ax.axhline(.5, color='#c3c2b7', lw=1)
        ax.set_xticks(steps.step, [LEVEL_LABELS.get(level, level.removeprefix('block')) for level in steps.level],
                      fontsize=7)
        ax.set(title=f'{model}: weights re-initialised from the top', xlabel='Re-initialised down to (block)',
               ylabel='Median Spearman with trained map', ylim=(-0.45, 1.05))
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc='center right')
    save(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', choices=['human', 'parts', 'sanity'])
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    dict(human=human, parts=parts, sanity=sanity)[args.figure](args.runs, args.output)


if __name__ == '__main__':
    main()
