"""Figures 2-4 of design v2 from the analysis CSVs only (no model or data access)."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fastweight_vit.plotting import AQUA, BLUE, GRAY, INK, ORANGE, QUIET, band, plt, save

ROOT = Path(__file__).resolve().parents[1]


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
    a.set_title('a. One layer at a time: shared top-16 sets keep 92-96% of predictions (DeiT-S, m = 16)', loc='left', color=INK, fontsize=9)
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
    b.set_title('b. Routing outweighs compression in layers 1-11; query-specific reading sits in layers 1-3', loc='left', color=INK, fontsize=9)
    b.legend(fontsize=7.5, ncol=1, loc='upper right')
    m = lambda name: cover.query('measure == @name').sort_values('layer')
    band(c, layers, m('L3_spatial_k16_patch'), BLUE, 'attention mass inside shared set (patches)')
    band(c, layers, m('L1_k16_patch'), INK, 'inside own top-16 (patches)', '--')
    band(c, layers, m('L3_spatial_k16_union'), AQUA, 'tokens used by any set (union / 197)', marker='s')
    c.set_ylabel('fraction')
    c.set_title('c. Deep layers read few distinct tokens', loc='left', color=INK, fontsize=9)
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
    fig.suptitle('All 12 layers replaced: sharing costs little against per-query top-k; summaries help only at small k',
                 x=.01, ha='left', color=INK, fontsize=9)
    a.text(np.log2(4), noise['deit'] + .015, 'shift-1px noise floor', fontsize=7, color=QUIET)
    a.text(np.log2(4), .9 - .045, '90%', fontsize=7, color=QUIET)
    fig.tight_layout()
    save(fig, out / 'figure3_frontier')


def figure4(analysis, out, examples_dir=None):
    comp = pd.read_csv(analysis / 'pet_composition.csv')
    union = pd.read_csv(analysis / 'pet_union_composition_post_hoc.csv')
    deletion = pd.read_csv(analysis / 'pet_deletion.csv')
    fig, (a, u, b) = plt.subplots(1, 3, figsize=(7.4, 3.9), gridspec_kw=dict(width_ratios=[1.25, 1, .9]))
    layers = np.arange(1, 13)
    models = [('deit', BLUE, 'DeiT-S'), ('mita', ORANGE, 'MiTA-DeiT-S')]
    area = comp.query("spec == 'm25_k25_clean'")['fg_area'].mean()
    for model, color, label in models:
        frame = comp.query("model == @model and spec == 'm25_k25_clean'").sort_values('layer')
        band(a, layers, frame.rename(columns={'fg_share': 'mean', 'fg_share_low': 'low', 'fg_share_high': 'high'}),
             color, f'{label}')
        frame = union.query("model == @model and spec == 'm25_k25_clean'").sort_values('layer')
        band(a, layers, frame.rename(columns={'fg_share': 'mean', 'fg_share_low': 'low', 'fg_share_high': 'high'}),
             color, None, '--', 's')
        band(u, layers, frame.rename(columns={'union_size': 'mean', 'union_size_low': 'low', 'union_size_high': 'high'}),
             color, None)
    a.axhline(area, color=QUIET, ls=':', lw=1)
    a.text(12.3, area, f'object\narea\n{area:.0%}', va='center', fontsize=7, color=QUIET)
    a.plot([], [], '-', color=QUIET, label='per selection')
    a.plot([], [], '--', color=QUIET, label='once per token')
    a.set_ylabel('object share of read spatial tokens')
    a.set_title('a. Object share', loc='left', color=INK, fontsize=8.5)
    u.set_ylabel('tokens read by any set (of 196)')
    u.set_title('b. Distinct tokens read', loc='left', color=INK, fontsize=8.5)
    for ax in (a, u):
        ax.set_xticks([1, 4, 8, 12])
        ax.set_xlabel('layer')
    cats = ['foreground', 'background', 'sink']
    x = np.arange(len(cats))
    for offset, (model, color, label) in zip([-.2, .2], models):
        frame = deletion.query('model == @model').set_index('category').loc[cats]
        b.bar(x + offset, frame['target_minus_random'], .4, color=color)
        b.errorbar(x + offset, frame['target_minus_random'],
                   [frame['target_minus_random'] - frame['low'], frame['high'] - frame['target_minus_random']],
                   fmt='none', ecolor=INK, elinewidth=.8, capsize=2)
    b.axhline(0, color=QUIET, lw=.8)
    b.set_xticks(x, ['object', 'back-\nground', 'high-\nnorm'])
    b.set_ylabel('agreement change vs matched random')
    b.set_title('c. Remove one kind', loc='left', color=INK, fontsize=8.5)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=7.5, bbox_to_anchor=(.5, 0))
    fig.suptitle('Both models prune deep layers and need the object; DeiT-S sets sit on the object more '
                 '(Pet, m = k = 25)', x=.01, ha='left', fontsize=9, color=INK)
    fig.tight_layout(rect=(0, .07, 1, 1))
    save(fig, out / 'figure4_object_focus')

def figure4_examples(runs_pet, images, out, layers=(4, 8, 12), spec='m25_k25_clean'):
    """MiTA-style pruning view: tokens read by no set at that layer are whitened."""
    from PIL import Image
    examples = pd.read_csv(runs_pet / 'examples.csv', dtype={'tokens': str})
    ids = [2335, 1874, 989]
    fig, axes = plt.subplots(len(ids), 1 + 2 * len(layers), figsize=(7.2, 3.1))
    for row, index in enumerate(ids):
        image = np.asarray(Image.open(images / f'pet_{index}.png').convert('RGB')) / 255.
        axes[row, 0].imshow(image)
        for col, (model, layer) in enumerate([(m, l) for m in ['deit', 'mita'] for l in layers], start=1):
            tokens = examples.query('model == @model and index == @index and spec == @spec and layer == @layer')
            keep = np.array([c == '1' for c in tokens['tokens'].item()]).reshape(14, 14)
            mask = np.kron(keep, np.ones((16, 16)))[..., None]
            axes[row, col].imshow(image * mask + (1 - mask))
            if row == 0:
                axes[row, col].set_title(f"{'DeiT-S' if model == 'deit' else 'MiTA'} L{layer}", fontsize=7.5, color=INK)
    axes[0, 0].set_title('image', fontsize=7.5, color=INK)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle('Tokens read by any of the 25 top-25 sets (white = read by none)', x=.01, ha='left',
                 fontsize=9, color=INK)
    fig.tight_layout(w_pad=.2, h_pad=.2)
    save(fig, out / 'figure4_examples')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=ROOT / 'experiments/2026-10-28_v2_analysis')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    parser.add_argument('--pet-run', type=Path, default=ROOT / 'experiments/2026-10-28_v2_pet')
    parser.add_argument('--images', type=Path,
                        default=ROOT / 'experiments/2026-10-28_v2_colab_environment/example_images')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    noise = json.loads((args.analysis / 'checks.json').read_text())['noise_floor']
    figure2(args.analysis, noise, args.output)
    figure3(args.analysis, noise, args.output)
    figure4(args.analysis, args.output)
    if args.images.exists():
        figure4_examples(args.pet_run, args.images, args.output)


if __name__ == '__main__':
    main()
