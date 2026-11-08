"""Paper figures from results/ (and the ClickMe gallery of the Table 1 run): text width 5.5 in, every font >= 7 pt.

  python scripts/make_figures.py [--results results] [--outputs outputs] [--figures figures]

- figure_2a: execution agreement against k (reading sets, random sets, no attention), three models (k_sweep.csv).
  Width 2.2 in, so that 2(a) and 2(b) fit side by side at the same height (1.6 in).
- figure_2b: the two lowest-index images of the seeded random ClickMe gallery of DeiT-S (24 images drawn with seed
  0; outputs/table1/clickme/deit/gallery_*.npz): image, ClickMe human map, R0, C3 and C2 summed over layers 9-12,
  as in Table 1. Each heat map is min-max normalised on its own; one shared colour scale. Width 3.3 in.
- figure_A_2b_layer12 (appendix): the same at layer 12 alone.
- figure_3: M1, M2, M3 and R0's image-specific ClickMe alignment per layer (diagnose.csv). Width 5.5 in.
- figure_A_c2 (appendix E): why C2 fails (c2_analysis/errors.csv, shift_bins.csv and the mean predictions in
  outputs/c2_analysis).
The script checks the smallest font of the main-text figures and stops if one is below 7 pt.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUTPUTS = ROOT / 'outputs'
GALLERY = OUTPUTS / 'table1' / 'clickme' / 'deit'
LABELS = {'deit': 'DeiT-S', 'deit_b': 'DeiT-B', 'dino': 'DINO'}
COLORS = {'deit': '#2f6db5', 'deit_b': '#d9822b', 'dino': '#3a9a5b'}
MIN_PT = 7
HEIGHT_2 = 1.6  # figures 2a and 2b share this height (side by side, 2.2 + 3.3 = 5.5 in)
PAPER_STYLE = {'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
               'legend.fontsize': 7, 'font.family': 'DejaVu Sans', 'axes.spines.top': False,
               'axes.spines.right': False, 'axes.linewidth': .6, 'lines.linewidth': 1.0, 'xtick.major.width': .6,
               'ytick.major.width': .6}
APPENDIX_STYLE = {'font.size': 9, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                  'ytick.color': '#52514e', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                  'grid.color': '#e6e5e1', 'grid.linewidth': .6, 'legend.frameon': False}
plt.rcParams.update(PAPER_STYLE)


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
    sweep = pd.read_csv(RESULTS / 'k_sweep.csv')
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
    if not data:
        print(f'{name}: no gallery in {GALLERY}, skipped')
        return
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
    frame = pd.read_csv(RESULTS / 'diagnose.csv')
    panels = [('M1', 'R0', '(a) M1: mean-map share'), ('M2', 'probe', '(b) M2: probe accuracy'),
              ('M3', 'R0', '(c) M3: high-norm share'), ('quality', 'R0', '(d) R0 vs. ClickMe')]
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


def figure_c2(out):
    """Appendix E: probe errors against the matched blur, mean prediction, where C2 moves the scores."""
    errors, bins = RESULTS / 'c2_analysis' / 'errors.csv', RESULTS / 'c2_analysis' / 'shift_bins.csv'
    if not errors.exists():
        print('figure_A_c2: no c2_analysis results, skipped')
        return
    error_table, bin_table = pd.read_csv(errors), pd.read_csv(bins)
    long = {'deit': 'DeiT-S', 'deit_b': 'DeiT-B', 'dino': 'DINO ViT-S'}
    colors = {'deit': '#2f6db5', 'deit_b': '#d9822b', 'dino': '#3a9a5b'}
    with plt.rc_context():
        plt.rcdefaults()  # the appendix figure keeps its own style, not the 7-pt paper style
        plt.rcParams.update(APPENDIX_STYLE)
        fig = plt.figure(figsize=(15, 3.4))
        grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.55, 0.55, 1.1])
        ax = fig.add_subplot(grid[0])
        for model, color in colors.items():
            for source, style, label in (('probe', '-', 'probe'), ('stand_in_blur', '--', 'blur')):
                part = error_table[(error_table.model == model) & (error_table.source == source) &
                                   (error_table.metric == 'distance')].sort_values('layer')
                ax.plot(part.layer, part['mean'], style, color=color, marker='o', ms=3, label=f'{long[model]} {label}')
        ax.set_title('(a) Expected distance error (patches)', fontsize=9)
        ax.set_xlabel('Layer')
        ax.legend(fontsize=6)
        ax = fig.add_subplot(grid[1])
        for model, color in colors.items():
            for source, style in (('probe', '-'), ('stand_in_blur', '--')):
                part = error_table[(error_table.model == model) & (error_table.source == source) &
                                   (error_table.metric == 'far_share')].sort_values('layer')
                ax.plot(part.layer, part['mean'], style, color=color, marker='o', ms=3)
        ax.axhline(.5, color='#999999', lw=.8)
        ax.set_title('(b) Share of errors farther than 3 patches', fontsize=9)
        ax.set_xlabel('Layer')
        for column, model in zip((2, 3), ('deit', 'dino')):
            ax = fig.add_subplot(grid[column])
            values = np.load(OUTPUTS / 'c2_analysis' / model / 'mean_prediction.npz')['layer12'].mean(0) * 196
            image = ax.imshow(values.reshape(14, 14), cmap='viridis')
            ax.set_title(f'(c) {long[model]} L12\nmean prediction x 196', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, fraction=.046)
        ax = fig.add_subplot(grid[4])
        for model, color in colors.items():
            for source, style, label in (('C2', '-', 'C2'), ('stand_in_blur', '--', 'blur')):
                part = bin_table[(bin_table.model == model) & (bin_table.source == source)].sort_values('bin')
                ax.plot(part['bin'], part['mean'], style, color=color, marker='o', ms=3, label=f'{long[model]} {label}')
        ax.axhline(0, color='#999999', lw=.8)
        ax.set_title('(d) Change in score share by distance to centre\n(layers 9-12, ImageNet-S)', fontsize=9)
        ax.set_xlabel('Distance to centre (patches, floor)')
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / 'figure_A_c2.png', dpi=200)
        fig.savefig(out / 'figure_A_c2.pdf')
        plt.close(fig)
    print('figure_A_c2 written')


def main():
    global RESULTS, OUTPUTS, GALLERY
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--outputs', type=Path, default=OUTPUTS)
    parser.add_argument('--figures', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    RESULTS, OUTPUTS = args.results, args.outputs
    GALLERY = OUTPUTS / 'table1' / 'clickme' / 'deit'
    args.figures.mkdir(parents=True, exist_ok=True)
    figure_2a(args.figures)
    figure_2b(args.figures)
    figure_2b(args.figures, '_l12', 'figure_A_2b_layer12')
    figure_3(args.figures)
    figure_c2(args.figures)


if __name__ == '__main__':
    main()
