"""Build the fast-weight schematic, execution plot and fixed-sample gallery."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import json  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from fastweight_vit.gallery import SELECTED, load_gallery  # noqa: E402

OUT = ROOT / 'paper/figures'
OUTPUTS = ROOT / 'outputs'
MIN_PT = 7
for path in ('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
             '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
             '/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf'):
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
FAMILY = 'Liberation Sans' if any(f.name == 'Liberation Sans' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': FAMILY, 'mathtext.fontset': 'stixsans',
                     'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .6, 'lines.linewidth': 1.0,
                     'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5,
                     'ytick.major.size': 2.5, 'legend.frameon': False, 'legend.handlelength': 1.6,
                     'legend.handletextpad': .4, 'legend.columnspacing': .9, 'legend.labelspacing': .25,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})

MODELS = ['deit', 'deit_b', 'dino']
NAME = {'deit': 'DeiT-S', 'deit_b': 'DeiT-B', 'dino': 'DINO ViT-S'}
# Model colours, used in every data figure. The schematic uses other colours (purple, dark red, greys, inferno).
COLOR = {'deit': '#2f6db5', 'deit_b': '#d9822b', 'dino': '#3a9a5b'}
GREY = '#8c8c8c'
FILES = []  # (file, inches) for the manifest printout


def check_and_save(fig, name, pdf_dpi=300):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    smallest = min(t.get_fontsize() for t in texts)
    if smallest < MIN_PT - 1e-6:
        raise ValueError(f'{name}: smallest font {smallest} pt < {MIN_PT} pt')
    tight = fig.get_tightbbox(renderer)  # drawn artists only (tick labels outside the view are not drawn)
    w, h = fig.get_size_inches()
    if tight.x0 < -.01 or tight.y0 < -.01 or tight.x1 > w + .01 or tight.y1 > h + .01:
        raise ValueError(f'{name}: drawn content {tight} exceeds the figure ({w} x {h} in)')
    fig.savefig(OUT / f'{name}.pdf', dpi=pdf_dpi, metadata={'CreationDate': None})
    width, height = fig.get_size_inches()
    FILES.append((name, width, height))
    print(f'{name}: {width:.2f} x {height:.2f} in, smallest font {smallest:.1f} pt')
    plt.close(fig)


def axes_at(fig, left, bottom, width, height):
    """Axes placed in inches (left, bottom, width, height) so that every file keeps its exact size."""
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def fit(fig, ax, pad=.03):
    """Shrink the axes so that its tick labels, axis labels and title stay inside the figure (left, right, top)."""
    w, h = fig.get_size_inches()
    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        box = ax.get_position()
        x0, x1, y1 = box.x0 * w, box.x1 * w, box.y1 * h
        x0 += max(0, pad - tight.x0)
        x1 -= max(0, tight.x1 - (w - pad))
        y1 -= max(0, tight.y1 - (h - pad))
        ax.set_position([x0 / w, box.y0, (x1 - x0) / w, (y1 - box.y0 * h) / h])


def model_handles():
    return [Line2D([], [], color=COLOR[m], marker='o', ms=2.5, lw=1.0) for m in MODELS], [NAME[m] for m in MODELS]


def execution_k_sweep(results):
    sweep = pd.read_csv(results / 'k_sweep.csv')
    fig = plt.figure(figsize=(2.6, 1.9))
    ax = axes_at(fig, .55, .78, 1.98, 1.07)
    for model in MODELS:
        for sets, style in (('reading', '-'), ('random', '--')):
            part = sweep[(sweep.model == model) & (sweep.sets == sets)].sort_values('k')
            ax.plot(part.k, part['mean'] * 100, style, color=COLOR[model], marker='o', ms=2)
        off = sweep[(sweep.model == model) & (sweep.sets == 'no attention')]
        ax.axhline(off['mean'].iloc[0] * 100, color=COLOR[model], ls=':', lw=1.0)
    ax.set_xscale('log', base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128], ['4', '8', '16', '32', '64', '128'])
    ax.minorticks_off()
    ax.set_xlabel('Units read per image region (k)', labelpad=1.5)
    ax.set_ylabel('Top-1 agreement with\nunmodified model (%)')
    ax.set_ylim(-3, 100)
    handles, labels = model_handles()
    styles = [Line2D([], [], color='k', ls='-'), Line2D([], [], color='k', ls='--'), Line2D([], [], color='k', ls=':')]
    fit(fig, ax)
    fig.legend(handles + styles, labels + ['top-k reading sets', 'random sets of the same size', 'attention removed'],
               loc='lower left', bbox_to_anchor=(.0, .0), ncol=2, handlelength=1.6)
    check_and_save(fig, 'execution_k_sweep')


def fast_weight_lens():
    """The compact first-page schematic used by the manuscript."""
    fig, ax = plt.subplots(figsize=(1.93, 1.52))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    for y in (0.24, 0.47, 0.70):
        ax.plot([0.10, 0.48, 0.90], [0.47, y, 0.47], color='#6d8295', lw=0.8, zorder=0)
        ax.add_patch(Circle((0.48, y), 0.071, facecolor='#e4edf5', edgecolor='#2566a5', lw=0.9))
    for x in (0.10, 0.90):
        ax.add_patch(Circle((x, 0.47), 0.075, facecolor='white', edgecolor='0.25', lw=0.8))
    ax.text(0.10, 0.47, r'$q_i^h$', ha='center', va='center', fontsize=10)
    ax.text(0.90, 0.47, r'$o_i^h$', ha='center', va='center', fontsize=10)
    ax.text(0.48, 0.47, r'$a_{ij}^h$', ha='center', va='center', fontsize=9)
    ax.text(0.27, 0.83, r'keys $k_j^h$', ha='center', fontsize=9)
    ax.text(0.76, 0.83, r'$W_O^h v_j^h$', ha='center', fontsize=9)
    ax.text(0.48, 0.10, 'hidden units = tokens', ha='center', fontsize=8)
    ax.text(0.48, 0.023, 'softmax-coupled activations', ha='center', fontsize=7.5)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    check_and_save(fig, 'fast_weight_lens')


def example_maps(data, name, picks, groups_per_row, cell, gap, group_gap):
    """Fixed index order, five map columns per example, no visual selection."""
    columns = [('image', 'image'), ('human', 'human\n(ClickMe)'), ('map_R0', 'received\nattention'),
               ('map_C3', 'contribution\nweighting'), ('map_C2', 'probe\nrelocation')]
    rows = -(-len(picks) // groups_per_row)
    group_w = 5 * cell + 4 * gap
    width = groups_per_row * group_w + (groups_per_row - 1) * group_gap
    height = .27 + rows * cell + (rows - 1) * gap
    fig = plt.figure(figsize=(width, height))
    for n, pick in enumerate(picks):
        r, group = divmod(n, groups_per_row)
        for col, (key, label) in enumerate(columns):
            x = group * (group_w + group_gap) + col * (cell + gap)
            y = height - .27 - (r + 1) * cell - r * gap
            ax = fig.add_axes([x / width, y / height, cell / width, cell / height])
            value = data[key][pick]
            if key == 'image':
                ax.imshow(value)
            else:
                value = value.reshape(14, 14) if value.size == 196 else value
                value = (value - value.min()) / (value.max() - value.min() + 1e-12)
                ax.imshow(value, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.text(.5, 1.06, label, transform=ax.transAxes, ha='center', va='bottom', linespacing=1.25)
    # Keep the gallery's embedded images at the original 200 dpi.
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        check_and_save(fig, name, pdf_dpi=200)


def main():
    global OUT, OUTPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--outputs', type=Path, default=OUTPUTS)
    parser.add_argument('--figures', type=Path, default=ROOT / 'paper/figures')
    args = parser.parse_args()
    OUTPUTS, OUT = args.outputs, args.figures
    OUT.mkdir(parents=True, exist_ok=True)
    fast_weight_lens()
    execution_k_sweep(args.results)
    data = load_gallery(OUTPUTS / 'table1/clickme/deit')
    selection = json.loads((ROOT / 'data/splits/gallery_selection.json').read_text(encoding='utf-8'))
    expected = {'all_indices': data['index'].tolist(), **SELECTED}
    if selection != expected:
        raise ValueError('Saved gallery differs from the fixed sample selection')
    example_maps(data, 'example_maps', [0, 1], 1, .6, .03, 0)
    example_maps(data, 'example_maps_more', [2, 3, 4, 5], 2, .52, .025, .2)


if __name__ == '__main__':
    main()
