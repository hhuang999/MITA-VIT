"""Figure 5: DINO ViT-S/16 against DeiT-S (and MiTA-DeiT-S on Pet), from the analysis CSVs only."""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plot_anatomy', ROOT / 'scripts/plot_anatomy.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
plt, band, save = style.plt, style.band, style.save
BLUE, ORANGE, AQUA, INK, QUIET = style.BLUE, style.ORANGE, style.AQUA, style.INK, style.QUIET
MODELS = [('deit', 'DeiT-S', BLUE), ('mita', 'MiTA-DeiT-S', ORANGE), ('dino', 'DINO ViT-S/16', AQUA)]
TITLES = dict(
    a='a. One layer at a time: without summaries,\nDINO loses more (layers 1, 10-12)',
    b='b. Own top-16 holds less attention in DINO\n(layers 1, 10-12): attention is spread out',
    c='c. Pet, union of reading sets (m = k = 25):\nDINO reads the animal most at layer 11',
    d='d. Distinct tokens read\n(union size, of 196)',
    e='e. Matched deletion (MiTA-default plan): removing the animal hurts all three models',
)


def figure5(analysis, deit_analysis, out):
    rq1 = pd.read_csv(analysis / 'compare_rq1_anatomy.csv')
    cover = pd.concat([pd.read_csv(deit_analysis / 'rq1_coverage.csv').assign(model='deit'),
                       pd.read_csv(analysis / 'rq1_coverage.csv').assign(model='dino')])
    union = pd.read_csv(analysis / 'pet_union_composition.csv').query("spec == 'm25_k25_mita'")
    deletion = pd.concat([pd.read_csv(deit_analysis / 'pet_deletion.csv'),
                          pd.read_csv(analysis / 'pet_deletion.csv')]).query("category != 'baseline'")
    layers = np.arange(1, 13)
    fig = plt.figure(figsize=(7.6, 8.6))
    grid = fig.add_gridspec(3, 2)
    a, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    for model, label, color in [MODELS[0], MODELS[2]]:
        q = lambda name, model=model: rq1[(rq1.model == model) & (rq1.quantity == name)].sort_values('layer')
        band(a, layers, q('L4'), color, f'{label}: L4 tokens + summaries', '-')
        band(a, layers, q('L3'), color, f'{label}: L3 tokens only', ':', marker='s')
        band(b, layers, cover.query("model == @model and measure == 'L1_k16_patch'").sort_values('layer'),
             color, f'{label}: own top-16')
    a.set_ylabel('agreement with original top-1')
    b.set_ylabel('attention mass inside set')
    a.legend(fontsize=7, loc='lower right')
    b.legend(fontsize=7, loc='lower center')
    c, d = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    for model, label, color in MODELS:
        frame = union.query('model == @model').sort_values('layer')
        band(c, layers, frame.rename(columns=dict(fg_share='mean', fg_share_low='low', fg_share_high='high')),
             color, label)
        band(d, layers, frame.rename(columns=dict(union_size='mean', union_size_low='low',
                                                  union_size_high='high')), color, label)
    c.set_ylabel('foreground / (fg + bg + high-norm)')
    d.set_ylabel('tokens')
    c.legend(fontsize=7, loc='upper left')
    for axis, key in [(a, 'a'), (b, 'b'), (c, 'c'), (d, 'd')]:
        axis.set_xlabel('layer')
        axis.set_xticks(layers[1::2])
        axis.set_title(TITLES[key], loc='left', color=INK, fontsize=8, wrap=True)
    e = fig.add_subplot(grid[2, :])
    categories = ['foreground', 'background', 'sink']
    width = .26
    for offset, (model, label, color) in zip([-width, 0, width], MODELS):
        frame = deletion.query('model == @model').set_index('category').loc[categories]
        x = np.arange(len(categories)) + offset
        e.bar(x, frame['target_minus_random'], width, color=color, label=label)
        e.errorbar(x, frame['target_minus_random'],
                   [frame['target_minus_random'] - frame['low'], frame['high'] - frame['target_minus_random']],
                   fmt='none', ecolor=INK, elinewidth=.8, capsize=2)
    e.axhline(0, color=QUIET, lw=.8)
    e.set_xticks(range(len(categories)), ['foreground tokens', 'background tokens',
                                          'high-norm tokens (none in DINO)'])
    e.set_ylabel('agreement: target - random')
    e.set_title(TITLES['e'], loc='left', color=INK, fontsize=9)
    e.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout()
    save(fig, out / 'figure5_dino')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'experiments/2026-10-28_v2_dino_analysis')
    parser.add_argument('--deit-analysis', type=Path, default=ROOT / 'experiments/2026-10-28_v2_analysis')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    figure5(args.analysis, args.deit_analysis, args.output)


if __name__ == '__main__':
    main()
