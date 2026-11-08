"""v5 figure sketches from the analysis tables (scripts/analyze_v5.py) and the ClickMe gallery.

  python scripts/plot_v5.py <analysis folder> <run prefix> [figure folder]

Figure 2 (read): (a) 12-layer execution agreement against k, reading sets vs random sets of the same size,
dotted line = no attention at all; (b) the first three seeded-random ClickMe images: image, R0 reading of
layers 9-12, human map.
Figure 3 (diagnose): per layer, (a) M1 share of R0, (b) position-probe accuracy (new probe; E3 as dots),
(c) share of R0 on high-norm tokens (S3), (d) R0 ClickMe image-specific Spearman (S2).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fastweight_vit.plotting import plt, save

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'deit': 'DeiT-S', 'deit_b': 'DeiT-B', 'dino': 'DINO ViT-S'}
COLORS = {'deit': '#2f6db5', 'deit_b': '#d9822b', 'dino': '#3a9a5b'}


def figure2(analysis, prefix, out):
    sweep = pd.read_csv(analysis / 'fig2a_ksweep.csv')
    fig = plt.figure(figsize=(11, 3.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])
    ax = fig.add_subplot(grid[0])
    for model, color in COLORS.items():
        for sets, style in (('reading', '-'), ('random', '--')):
            part = sweep[(sweep.model == model) & (sweep.sets == sets)].sort_values('k')
            if part.empty:
                continue
            ax.plot(part.k, part['mean'], style, color=color, marker='o', ms=3,
                    label=f'{LABELS[model]} {"reading sets" if sets == "reading" else "random sets"}')
            ax.fill_between(part.k, part.low, part.high, color=color, alpha=.15, lw=0)
        off = sweep[(sweep.model == model) & (sweep.sets == 'no attention')]
        if len(off):
            ax.axhline(off['mean'].iloc[0], color=color, ls=':', lw=1)
    ax.set_xscale('log', base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128], ['4', '8', '16', '32', '64', '128'])
    ax.set_xlabel('k (neurons read per region)')
    ax.set_ylabel('Top-1 agreement, all 12 layers')
    ax.set_title('(a) How many neurons are enough', fontsize=9)
    ax.legend(fontsize=6, loc='lower right')
    gallery = sorted(Path(f'{prefix}_correct_ablation_clickme_deit').glob('gallery_*.npz'))
    inner = grid[1].subgridspec(3, 3, wspace=.05, hspace=.05)
    if gallery:
        data = [np.load(path) for path in gallery]
        index = np.concatenate([d['index'] for d in data])
        order = np.argsort(index)[:3]
        stacked = {key: np.concatenate([d[key] for d in data])[order] for key in ('image', 'human', 'map_R0')}
        for row in range(len(order)):
            for col, (key, title) in enumerate((('image', 'Image'), ('map_R0', 'R0, layers 9-12 (DeiT-S)'),
                                                 ('human', 'ClickMe human map'))):
                sub = fig.add_subplot(inner[row, col])
                value = stacked[key][row]
                if key == 'map_R0':
                    sub.imshow(stacked['image'][row])
                    sub.imshow(np.kron(value.reshape(14, 14), np.ones((16, 16))), cmap='inferno', alpha=.6)
                else:
                    sub.imshow(value, cmap=None if key == 'image' else 'inferno')
                sub.set_xticks([])
                sub.set_yticks([])
                if row == 0:
                    sub.set_title(title, fontsize=7)
    fig.text(.42, .97, '(b) What the reading looks like (seeded random draw)', fontsize=9)
    fig.tight_layout()
    save(fig, out / 'figure_v5_2_read')


def figure3(analysis, out):
    frame = pd.read_csv(analysis / 'fig3_diagnose.csv')
    panels = [('M1', 'R0', '(a) M1: image-independent share'), ('M2', 'new probe', '(b) M2: position-probe accuracy'),
              ('M3', 'R0 (S3)', '(c) M3: R0 share on high-norm tokens'),
              ('quality', 'R0 (S2)', '(d) Reading quality: ClickMe image-specific')]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True)
    for ax, (panel, variant, title) in zip(axes, panels):
        for model, color in COLORS.items():
            part = frame[(frame.panel == panel) & (frame.variant == variant) & (frame.model == model)].sort_values('layer')
            if len(part):
                ax.plot(part.layer, part['mean'], color=color, marker='o', ms=3, label=LABELS[model])
                ax.fill_between(part.layer, part.low, part.high, color=color, alpha=.15, lw=0)
            if panel == 'M2':
                old = frame[(frame.panel == 'M2') & (frame.variant == 'E3') & (frame.model == model)]
                ax.plot(old.layer, old['mean'], 'x', color=color, ms=4)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('Layer')
        ax.set_xticks([1, 4, 8, 12])
    axes[0].legend(fontsize=7)
    axes[1].text(.02, .02, 'x: E3 (400 training images)', transform=axes[1].transAxes, fontsize=6)
    fig.tight_layout()
    save(fig, out / 'figure_v5_3_diagnose')


def main():
    analysis = ROOT / sys.argv[1]
    prefix = str(ROOT / sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    figure2(analysis, prefix, out)
    figure3(analysis, out)
    print('figures written to', out)


if __name__ == '__main__':
    main()
