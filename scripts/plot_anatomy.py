"""Figures 2-4 of design v2 from the analysis CSVs only (no model or data access)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, GRAY, INK, QUIET = '#2a78d6', '#eb6834', '#1baf7a', '#8a8986', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': QUIET, 'axes.labelcolor': INK, 'xtick.color': QUIET,
                     'ytick.color': QUIET, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': .6, 'legend.frameon': False})


def band(ax, x, frame, color, label, style='-', marker='o'):
    ax.plot(x, frame['mean'], style, color=color, lw=2, marker=marker, ms=4, label=label)
    ax.fill_between(x, frame['low'], frame['high'], color=color, alpha=.15, lw=0)


def figure2(analysis, noise, out):
    table = pd.read_csv(analysis / 'rq1_anatomy.csv')
    cover = pd.read_csv(analysis / 'rq1_coverage.csv')
    q = lambda name: table.query('quantity == @name').sort_values('layer')
    layers = np.arange(1, 13)
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(7, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1, .8]))
    for name, color, label, style in [('L1_k32', INK, 'L1 per-query top-32', '--'), ('L4', AQUA, 'L4 summaries + shared top-16', '-'),
                                      ('L3', BLUE, 'L3 shared top-16', '-'), ('L2', ORANGE, 'L2 16 summaries only', '-'),
                                      ('rand', GRAY, 'random 16 tokens', ':')]:
        band(a, layers, q(name), color, label, style)
    a.plot(layers, q('drop')['mean'], 'x', color=GRAY, ms=6, label='attention removed')
    a.axhline(noise['deit'], color=QUIET, ls='--', lw=1)
    a.text(12.4, noise['deit'], 'shift-1px\nnoise floor', va='center', fontsize=7.5, color=QUIET)
    a.set_ylabel('agreement with original top-1')
    a.set_title('Replacing one layer at a time (DeiT-S, m = 16 regions, k = 16 tokens)', loc='left', color=INK)
    a.legend(ncol=2, fontsize=7.5, loc='lower right')
    width = .27
    for offset, name, color, label in [(-width, 'phi_R', BLUE, 'routing share (shared tokens)'),
                                       (0, 'phi_C', ORANGE, 'compression share (summaries)'),
                                       (width, 'query_specific', GRAY, 'query-specific (L1 - L4)')]:
        frame = q(name)
        b.bar(layers + offset, frame['mean'], width, color=color, label=label)
        b.errorbar(layers + offset, frame['mean'], [frame['mean'] - frame['low'], frame['high'] - frame['mean']],
                   fmt='none', ecolor=INK, elinewidth=.8, capsize=1.5)
    b.axhline(0, color=QUIET, lw=.8)
    b.set_ylabel('agreement points')
    b.set_title('Shapley split of each layer above the random floor', loc='left', color=INK)
    b.legend(fontsize=7.5, ncol=3, loc='upper left')
    m = lambda name: cover.query('measure == @name').sort_values('layer')
    band(c, layers, m('L3_spatial_k16_patch'), BLUE, 'attention mass inside shared set (patches)')
    band(c, layers, m('L1_k16_patch'), INK, 'inside own top-16 (patches)', '--')
    band(c, layers, m('L3_spatial_k16_union'), AQUA, 'tokens used by any set (union / 197)', marker='s')
    c.set_ylabel('fraction')
    c.set_xlabel('layer')
    c.set_xticks(layers)
    c.legend(fontsize=7.5, ncol=1, loc='best')
    fig.tight_layout()
    save(fig, out / 'figure2_anatomy')


def figure3(analysis, noise, out):
    frontier = pd.read_csv(analysis / 'rq2_frontier.csv')
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    k_lines = [('L1', INK, 'L1 per-query', '--'), ('L4_spatial', AQUA, 'L4 summaries + shared', '-'),
               ('L3_spatial', BLUE, 'L3 shared, region routing', '-'), ('L3_dot', BLUE, 'L3 shared, dot routing', ':'),
               ('rand', GRAY, 'random', ':')]
    for name, color, label, style in k_lines:
        frame = frontier.query('name == @name and m_side == 4').sort_values('k')
        band(a, np.log2(frame['k']), frame.rename(columns={'agree': 'mean'}), color, label, style)
    a.set_xticks(np.log2([4, 8, 16, 32, 64, 128]), [4, 8, 16, 32, 64, 128])
    a.set_xlabel('tokens per set, k (m = 16)')
    a.set_ylabel('all-layer agreement')
    for name, color, label, style in [('L4_spatial_m', AQUA, 'L4', '-'), ('L3_spatial_m', BLUE, 'L3', '-'),
                                      ('L2', ORANGE, 'L2 summaries only', '-')]:
        frame = frontier.query('name == @name').sort_values('m_side')
        if name != 'L2':  # the m = 16 point of the k sweep closes each m line
            main = frontier.query("name == @name.removesuffix('_m') and k == 16 and m_side == 4")
            frame = pd.concat([frame, main]).sort_values('m_side')
        band(b, np.log2(frame['m_side'] ** 2), frame.rename(columns={'agree': 'mean'}), color, label, style)
    b.set_xticks(np.log2([1, 4, 16, 49, 196]), [1, 4, 16, 49, 196])
    b.set_xlabel('regions, m (k = 16)')
    for ax in (a, b):
        ax.axhline(noise['deit'], color=QUIET, ls='--', lw=1)
        ax.axhline(.9, color=QUIET, ls=':', lw=.8)
    a.legend(fontsize=7, loc='lower right')
    b.legend(fontsize=7, loc='lower right')
    fig.suptitle('Replacing all 12 layers at once (DeiT-S, 5,000 ImageNet images)', x=.01, ha='left', color=INK, fontsize=10)
    fig.tight_layout()
    save(fig, out / 'figure3_frontier')


def figure4(analysis, out, examples_dir=None):
    comp = pd.read_csv(analysis / 'pet_composition.csv')
    deletion = pd.read_csv(analysis / 'pet_deletion.csv')
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    layers = np.arange(1, 13)
    for model, color, label in [('mita', ORANGE, 'MiTA-DeiT-S'), ('deit', BLUE, 'DeiT-S')]:
        frame = comp.query("model == @model and spec == 'm25_k25_clean'").sort_values('layer')
        band(a, layers, frame.rename(columns={'fg_share': 'mean', 'fg_share_low': 'low', 'fg_share_high': 'high'}),
             color, f'{label}: foreground share', marker='o')
        sink = frame['sink'] / frame[['sink', 'foreground', 'background']].sum(axis=1)
        a.plot(layers, sink, ':', color=color, lw=1.5, label=f'{label}: high-norm share')
    area = comp.query("spec == 'm25_k25_clean'")['fg_area'].mean()
    a.axhline(area, color=QUIET, ls='--', lw=1)
    a.text(12.3, area, f'foreground\narea {area:.0%}', va='center', fontsize=7.5, color=QUIET)
    a.set_xticks(layers)
    a.set_xlabel('layer')
    a.set_ylabel('share of selected spatial tokens')
    a.set_title('Where the top-25 sets land (Pet test, m = 25)', loc='left', color=INK)
    a.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
    cats = ['foreground', 'background', 'sink']
    x = np.arange(len(cats))
    for offset, model, color, label in [(-.2, 'deit', BLUE, 'DeiT-S'), (.2, 'mita', ORANGE, 'MiTA-DeiT-S')]:
        frame = deletion.query('model == @model').set_index('category').loc[cats]
        b.bar(x + offset, frame['target_minus_random'], .4, color=color, label=label)
        b.errorbar(x + offset, frame['target_minus_random'],
                   [frame['target_minus_random'] - frame['low'], frame['high'] - frame['target_minus_random']],
                   fmt='none', ecolor=INK, elinewidth=.8, capsize=2)
    b.axhline(0, color=QUIET, lw=.8)
    b.set_xticks(x, ['object', 'background', 'high-norm'])
    b.set_ylabel('agreement: removed - matched random')
    b.set_title('Remove one kind from every set', loc='left', color=INK)
    b.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
    fig.tight_layout()
    save(fig, out / 'figure4_object_focus')


def save(fig, stem):
    fig.savefig(f'{stem}.png', dpi=200)
    fig.savefig(f'{stem}.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'experiments/2026-10-28_v2_analysis')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    noise = json.loads((args.analysis / 'checks.json').read_text())['noise_floor']
    figure2(args.analysis, noise, args.output)
    figure3(args.analysis, noise, args.output)
    figure4(args.analysis, args.output)


if __name__ == '__main__':
    main()
