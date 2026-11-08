"""Paper figures at print size (X7, decision B46): NeurIPS text width 5.5 in, every font at least 7 pt.

  python scripts/plot_paper_figures.py [figure folder]

- figure_2a: execution agreement against k (reading sets, random sets, no attention), three models
  (experiments/2026-11-05_v5_analysis/fig2a_ksweep.csv). Width 2.2 in, so that 2a and 2b fit side by side.
- figure_2b: the two lowest-index images of the seeded random ClickMe gallery of DeiT-S (B16 rule): image, ClickMe
  human map, R0, C3 and C2 summed over layers 9-12, as in Table 1 (map_* in experiments/2026-11-05_correct_
  ablation_clickme_deit/gallery_*.npz). Each heat map is min-max normalised on its own; one shared colour scale.
  Width 3.3 in; 2a and 2b have the same height. figure_A_2b_layer12: the same at layer 12 alone (appendix).
- figure_3: M1, M2, M3 and R0's image-specific ClickMe alignment per layer (fig3_diagnose.csv). Width 5.5 in.
The script checks the smallest font of every figure and stops if one is below 7 pt.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / 'experiments/2026-11-05_v5_analysis'
GALLERY = ROOT / 'experiments/2026-11-05_correct_ablation_clickme_deit'
LABELS = {'deit': 'DeiT-S', 'deit_b': 'DeiT-B', 'dino': 'DINO'}
COLORS = {'deit': '#2f6db5', 'deit_b': '#d9822b', 'dino': '#3a9a5b'}
MIN_PT = 7
HEIGHT_2 = 1.6  # figures 2a and 2b share this height (side by side, 2.2 + 3.3 = 5.5 in)
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'font.family': 'DejaVu Sans',
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
                     'lines.linewidth': 1.0, 'xtick.major.width': .6, 'ytick.major.width': .6})


def smallest_font(fig):
    sizes = [text.get_fontsize() for text in fig.findobj(matplotlib.text.Text) if text.get_text().strip()]
    return min(sizes)


def save(fig, out, name, tight=True):
    """tight=False keeps the exact figure size (figures 2a and 2b share one height when set side by side)."""
    size = smallest_font(fig)
    if size < MIN_PT:
        raise ValueError(f'{name}: smallest font {size} pt < {MIN_PT} pt')
    for suffix in ('png', 'pdf'):
        if tight:
            fig.savefig(out / f'{name}.{suffix}', dpi=300, bbox_inches='tight', pad_inches=.02)
        else:
            fig.savefig(out / f'{name}.{suffix}', dpi=300)
    width, height = fig.get_size_inches()
    print(f'{name}: {width:.2f} x {height:.2f} in, smallest font {size:.1f} pt')
    plt.close(fig)


def figure_2a(out):
    sweep = pd.read_csv(ANALYSIS / 'fig2a_ksweep.csv')
    fig, ax = plt.subplots(figsize=(2.2, HEIGHT_2))
    for model, color in COLORS.items():
        for sets, style in (('reading', '-'), ('random', '--')):
            part = sweep[(sweep.model == model) & (sweep.sets == sets)].sort_values('k')
            ax.plot(part.k, part['mean'] * 100, style, color=color, marker='o', ms=2,
                    label=LABELS[model] if sets == 'reading' else None)
        off = sweep[(sweep.model == model) & (sweep.sets == 'no attention')]
        ax.axhline(off['mean'].iloc[0] * 100, color=color, ls=':', lw=.8)
    ax.plot([], [], 'k--', label='random sets')
    ax.plot([], [], 'k:', label='no attention')
    ax.set_xscale('log', base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128], ['4', '8', '16', '32', '64', '128'])
    ax.minorticks_off()
    ax.set_xlabel('Units read per region (k)')
    ax.set_ylabel('Top-1\nagreement (%)')
    ax.set_ylim(-3, 100)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2]  # column-major fill: models on the first row, line styles on the second
    fig.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, loc='lower center',
               bbox_to_anchor=(.5, .01), ncol=3, handlelength=1.1, handletextpad=.3, columnspacing=.5,
               labelspacing=.2, borderaxespad=0)
    fig.subplots_adjust(left=.27, right=.97, top=.95, bottom=.44)
    save(fig, out, 'figure_2a', tight=False)


def figure_2b(out, suffix='', name='figure_2b'):
    """suffix '' = layers 9-12 summed (as Table 1; main text); '_l12' = layer 12 alone (appendix)."""
    data = [np.load(path) for path in sorted(GALLERY.glob('gallery_*.npz'))]
    index = np.concatenate([d['index'] for d in data])
    pick = np.argsort(index)[:2]
    maps = [f'map_{reading}{suffix}' for reading in ('R0', 'C3', 'C2')]
    stack = {key: np.concatenate([d[key] for d in data])[pick] for key in ['image', 'human'] + maps}
    columns = [('image', 'Image'), ('human', 'Human')] + list(zip(maps, ('R0', 'C3', 'C2')))
    fig, axes = plt.subplots(2, 5, figsize=(3.3, HEIGHT_2), gridspec_kw=dict(wspace=.04, hspace=.06))
    fig.subplots_adjust(left=.01, right=.86, top=.89, bottom=.04)
    for row in range(2):
        for col, (key, title) in enumerate(columns):
            ax = axes[row, col]
            value = stack[key][row]
            if key == 'image':
                ax.imshow(value)
            else:
                value = value.reshape(14, 14) if value.size == 196 else value
                value = (value - value.min()) / (value.max() - value.min() + 1e-12)
                image = ax.imshow(value, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title, pad=2)
    bar = fig.colorbar(image, cax=fig.add_axes([.885, .12, .025, .66]), ticks=[0, 1])
    bar.ax.tick_params(labelsize=7, width=.5, length=2)
    bar.outline.set_linewidth(.4)
    print(f'{name} images (ClickMe index):', index[pick].tolist())
    save(fig, out, name, tight=False)


def figure_3(out):
    frame = pd.read_csv(ANALYSIS / 'fig3_diagnose.csv')
    panels = [('M1', 'R0', '(a) M1: mean-map share'), ('M2', 'new probe', '(b) M2: probe accuracy'),
              ('M3', 'R0 (S3)', '(c) M3: high-norm share'), ('quality', 'R0 (S2)', '(d) R0 vs. ClickMe')]
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.6), sharex=True)
    for ax, (panel, variant, title) in zip(axes, panels):
        for model, color in COLORS.items():
            part = frame[(frame.panel == panel) & (frame.variant == variant) & (frame.model == model)].sort_values('layer')
            ax.plot(part.layer, part['mean'], color=color, marker='o', ms=1.8, label=LABELS[model])
            ax.fill_between(part.layer, part.low, part.high, color=color, alpha=.18, lw=0)
        ax.set_title(title, pad=3)
        ax.set_xticks([1, 4, 8, 12])
        ax.set_xlabel('Layer', labelpad=1)
    axes[1].set_ylim(0, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc='lower center', ncol=3, bbox_to_anchor=(.5, -.02),
               handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(pad=.2, w_pad=.8, rect=(0, .1, 1, 1))
    save(fig, out, 'figure_3')


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    figure_2a(out)
    figure_2b(out)
    figure_2b(out, '_l12', 'figure_A_2b_layer12')
    figure_3(out)


if __name__ == '__main__':
    main()
