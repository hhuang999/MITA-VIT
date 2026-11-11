"""Build the schematic, execution plot and example maps used by the paper."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import json  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402
from fastweight_vit.gallery import SELECTED, load_gallery  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures'
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
                     'pdf.fonttype': 3, 'svg.fonttype': 'none'})

MODELS = ['deit', 'deit_b', 'dino']
NAME = {'deit': 'DeiT-S', 'deit_b': 'DeiT-B', 'dino': 'DINO ViT-S'}
# Model colours, used in every data figure. The schematic uses other colours (purple, dark red, greys, inferno).
COLOR = {'deit': '#2f6db5', 'deit_b': '#d9822b', 'dino': '#3a9a5b'}
GREY = '#8c8c8c'
FILES = []  # (file, inches) for the manifest printout


KEY, VALUE, INK = '#6a3d9a', '#a50f15', '#262626'
HEAT = plt.get_cmap('inferno')


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
    for suffix in ('pdf', 'png'):
        fig.savefig(OUT / f'{name}.{suffix}', dpi=pdf_dpi if suffix == 'pdf' else 300,
                    metadata={'CreationDate': None} if suffix == 'pdf' else None)
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
               loc='lower left', bbox_to_anchor=(.01, .0), ncol=2, handlelength=1.6)
    check_and_save(fig, 'execution_k_sweep')


def gallery_arrays():
    """The two lowest indices of the seeded random ClickMe gallery of DeiT-S (rule fixed in advance)."""
    data = load_gallery(OUTPUTS / 'table1' / 'clickme' / 'deit')
    return {key: value[:2] for key, value in data.items()}


def gallery(name, suffix):
    data = gallery_arrays()
    if data is None:
        print(f'{name}: no gallery outputs, skipped')
        return
    rows = [('image', 'image'), ('human', 'human\n(ClickMe)'), (f'map_R0{suffix}', 'received\nattention'),
            (f'map_C3{suffix}', 'contribution\nweighting'), (f'map_C2{suffix}', 'probe\nrelocation')]
    fig = plt.figure(figsize=(2.8, 1.9))
    cell, pitch, left, top = .335, .355, 1.22, 1.86
    for r, (key, label) in enumerate(rows):
        y = top - (r + 1) * pitch + (pitch - cell) / 2
        for c in range(2):
            ax = axes_at(fig, left + c * (cell + .04), y, cell, cell)
            value = data[key][c]
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
            if c == 0:
                ax.text(-.08, .5, label, transform=ax.transAxes, ha='right', va='center', linespacing=1.05)
    bar_bottom = top - 5 * pitch + (pitch - cell) / 2
    bar = fig.colorbar(image, cax=axes_at(fig, left + 2 * cell + .04 + .1, bar_bottom, .07, 3 * pitch - (pitch - cell)),
                       ticks=[0, 1])
    bar.ax.tick_params(width=.5, length=2)
    bar.outline.set_linewidth(.4)
    bar.set_label('score, min-max\nnormalised per map', labelpad=2)
    check_and_save(fig, name)


def schematic_network():
    fig = plt.figure(figsize=(2.6, 1.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 52)
    ax.set_ylim(0, 38)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((4.5, 33.0), 43, 3.6, boxstyle='round,pad=0.3', fc='#efefef', ec=GREY, lw=.5))
    ax.text(26, 34.8, r'slow weights $W_K, W_V$ write $K, V$ for each image', ha='center', va='center')
    qx, ux, ox, cy = 5.5, 26, 46.5, 16
    ys = [27, 21.5, 16, 10.5, 5]
    acts = [.2, .85, .5, .1, .65]
    for y, a in zip(ys, acts):
        ax.plot([qx + 2.4, ux - 2.0], [cy, y], color=KEY, lw=.7, zorder=1)
        ax.plot([ux + 2.0, ox - 2.4], [y, cy], color=VALUE, lw=.4 + 2.0 * a, alpha=.85, zorder=1,
                solid_capstyle='round')
        ax.add_patch(Circle((ux, y), 2.0, facecolor=HEAT(.15 + .8 * a), edgecolor=INK, lw=.5, zorder=3))
    ax.text(ux, 16, '$j$', ha='center', va='center', color='white', zorder=4)
    for x, label in ((qx, '$q_i$'), (ox, '$o_i$')):
        ax.add_patch(Circle((x, cy), 2.4, facecolor='white', edgecolor=INK, lw=.5, zorder=3))
        ax.text(x, cy, label, ha='center', va='center', zorder=4)
    ax.text(qx, cy - 3.4, 'query', ha='center', va='top', color=GREY)
    ax.text(ox, cy - 3.4, r'output, $W_O$', ha='center', va='top', color=GREY)
    ax.text(14.5, 25.5, 'first layer:\nkeys $k_j$', ha='center', va='bottom', color=KEY, linespacing=1.0)
    ax.text(37.5, 25.5, 'second layer:\nvalues $v_j$', ha='center', va='bottom', color=VALUE, linespacing=1.0)
    ax.text(ux, 29.6, 'hidden units = tokens', ha='center', va='bottom')
    ax.text(ux, .3, r'activation $a_{ij} = \mathrm{softmax}_j(q_i^{\top} k_j / \sqrt{d})$', ha='center',
            va='bottom')
    check_and_save(fig, 'schematic_fast_weight_network')


def example_maps(data, name, picks, groups_per_row, cell, gap, group_gap):
    """v8 layout: fixed index order, five map columns per example, no visual selection."""
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
                ax.text(.5, 1.06, label, transform=ax.transAxes, ha='center', va='bottom', linespacing=1.0)
    # Match the author's v8 PDF rasterisation and embedded font type; PNG remains 300 dpi.
    with plt.rc_context({'pdf.fonttype': 42, 'ps.fonttype': 42}):
        check_and_save(fig, name, pdf_dpi=200)


def main():
    global OUT, OUTPUTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--outputs', type=Path, default=OUTPUTS)
    parser.add_argument('--figures', type=Path, default=ROOT / 'build/rebuilt/figures')
    args = parser.parse_args()
    OUTPUTS, OUT = args.outputs, args.figures
    OUT.mkdir(parents=True, exist_ok=True)
    schematic_network()
    execution_k_sweep(args.results)
    gallery('gallery_layers9_12', '')
    data = load_gallery(OUTPUTS / 'table1/clickme/deit')
    example_maps(data, 'example_maps', [0, 1], 1, .6, .03, 0)
    example_maps(data, 'example_maps_more', [2, 3, 4, 5], 2, .52, .025, .2)
    (OUT / 'gallery_selection.json').write_text(
        json.dumps({'all_indices': data['index'].tolist(), **SELECTED}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
