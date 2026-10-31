"""Figures for S1-S3 (decisions B22-B24); reads only the tables of analyze_supplement.py.

figure15_dense_vs_probe: attention_mass - probe_mass (paired 95% interval) on SRG, same-ring enrichment, ClickMe
image-specific Spearman and head enrichment; the shaded band is the registered 'comparable' margin.
figure16_layer_families: per layer, the probe, dense attention and CLS attention on ClickMe (image-specific
Spearman) and PartImageNet (head enrichment), one column per model.
figure17_high_norm: per layer, the share of each single-layer map on high-norm patches (ImageNet-S) and the
share of patches that are high-norm.

Usage: python scripts/plot_supplement.py [--analysis experiments/2026-10-31_s_analysis] [--output figures]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mita_probe.plotting import BLUE, INK, MODEL_LABELS, ORANGE, QUIET, band, save

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['deit', 'dino', 'deit_b']
FAMILIES = {'probe': (BLUE, '-', 'Probe (top-k sets)'), 'attention': (INK, '--', 'Dense attention'),
            'cls': (ORANGE, ':', 'CLS attention')}
PANELS = [('srg_agree', 'SRG', .05), ('enrichment', 'Same-ring enrichment', .02),
          ('specific_pixel', 'ClickMe image-specific Spearman', .02), ('head_enrichment', 'Head enrichment', .1)]


def dense_vs_probe(table, output):
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4 * len(PANELS), 3.6))
    for ax, (measure, title, margin) in zip(axes, PANELS):
        part = table.query('measure == @measure').reset_index(drop=True)
        ax.axvspan(-margin, margin, color='#e6e5e1', lw=0)
        for y, row in part.iterrows():
            ax.plot([row['low'], row['high']], [y, y], color=INK, lw=2)
            ax.plot(row['difference'], y, 'o', color=INK, ms=5, mec='white')
        ax.set_yticks(range(len(part)), [f"{MODEL_LABELS[r.model]}, {r.dataset}" for r in part.itertuples()],
                      fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color=QUIET, lw=1)
        ax.grid(axis='y', visible=False)
        ax.set(title=title, xlabel='Dense attention - probe')
    fig.tight_layout()
    save(fig, output)


def layer_families(curves, output):
    measures = [('clickme_specific_pixel', 'ClickMe image-specific Spearman'), ('head_enrichment', 'Head enrichment')]
    fig, axes = plt.subplots(len(measures), len(MODELS), figsize=(4.2 * len(MODELS), 6.4), squeeze=False)
    for row, (measure, label) in zip(axes, measures):
        for ax, model in zip(row, MODELS):
            part = curves.query('measure == @measure and model == @model')
            for family, (color, style, name) in FAMILIES.items():
                curve = part.query('family == @family').sort_values('layer')
                if len(curve):
                    band(ax, curve.layer, curve, color, label=name, style=style)
            ax.set(title=f'{MODEL_LABELS[model]}', xlabel='Layer', ylabel=label, xticks=range(1, 13))
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, output)


def high_norm(layers, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for model, style in zip(MODELS, ['-', '--', ':']):
        part = layers.query('model == @model')
        for family, (color, _, name) in FAMILIES.items():
            curve = part.query("family == @family and measure == 'high_share'").sort_values('layer')
            if len(curve):
                band(axes[0], curve.layer, curve, color, label=f'{MODEL_LABELS[model]}, {name}', style=style)
        fraction = part.query("family == 'probe' and measure == 'high_fraction'").sort_values('layer')
        if len(fraction):
            band(axes[1], fraction.layer, fraction, QUIET, label=MODEL_LABELS[model], style=style)
    axes[0].set(xlabel='Layer', ylabel='Share of the map on high-norm patches', xticks=range(1, 13), ylim=(0, 1))
    axes[1].set(xlabel='Layer', ylabel='Share of patches that are high-norm', xticks=range(1, 13))
    axes[0].legend(fontsize=6, ncol=1)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'experiments/2026-10-31_s_analysis')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    dense_vs_probe(pd.read_csv(args.analysis / 's1_comparisons.csv'), args.output / 'figure15_dense_vs_probe')
    layer_families(pd.read_csv(args.analysis / 's2_layers.csv'), args.output / 'figure16_layer_families')
    high_norm(pd.read_csv(args.analysis / 's3_layers.csv'), args.output / 'figure17_high_norm')


if __name__ == '__main__':
    main()
