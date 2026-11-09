"""Paper figures, one panel per file, at print size (PDF and 300-dpi PNG).

  python scripts/make_figures.py [--results results] [--outputs outputs] [--figures figures]

Sources: the summary tables in results/ (built by scripts/make_tables.py) for every line plot; the per-image run
outputs for the two image panels (the ClickMe gallery of DeiT-S, outputs/table1/clickme/deit/gallery_*.npz, and the
probes' mean prediction, outputs/c2_analysis/<model>/mean_prediction.npz). Without those outputs the image figures
are skipped and the committed files are kept. figures/README.md lists every file, its size and its source.

Rules: no figure numbers, panel letters or captions inside a figure; no internal codes (the names follow
paper/naming.md); every panel is a complete figure with its own axis labels and legend; fonts at least 7 pt at the
stated size. The script stops if a font is smaller than 7 pt or if drawn content leaves the figure.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

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


# ---------------------------------------------------------------------------------------------------- helpers
def check_and_save(fig, name):
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
        fig.savefig(OUT / f'{name}.{suffix}', dpi=300, metadata={'CreationDate': None} if suffix == 'pdf' else None)
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


# ---------------------------------------------------------------------------------------------------- execution
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


# ---------------------------------------------------------------------------------------------------- gallery
def gallery_arrays():
    """The two lowest indices of the seeded random ClickMe gallery of DeiT-S (rule fixed in advance)."""
    paths = sorted((OUTPUTS / 'table1' / 'clickme' / 'deit').glob('gallery_*.npz'))
    if not paths:
        return None
    data = [np.load(path) for path in paths]
    index = np.concatenate([d['index'] for d in data])
    pick = np.argsort(index)[:2]
    keys = ['image', 'human', 'map_R0', 'map_C3', 'map_C2', 'map_R0_l12', 'map_C3_l12', 'map_C2_l12']
    return {key: np.concatenate([d[key] for d in data])[pick] for key in keys}


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


# ---------------------------------------------------------------------------------------------------- premises
def premise_panel(results, name, panel, variant, ylabel, ylim=None, yticks=None):
    frame = pd.read_csv(results / 'diagnose.csv')
    fig = plt.figure(figsize=(1.35, 1.5))
    ax = axes_at(fig, .52, .62, .78, .78)
    for model in MODELS:
        part = frame[(frame.panel == panel) & (frame.variant == variant) & (frame.model == model)].sort_values('layer')
        ax.plot(part.layer, part['mean'], color=COLOR[model], marker='o', ms=1.6, lw=.9)
        ax.fill_between(part.layer, part.low, part.high, color=COLOR[model], alpha=.18, lw=0)
    ax.set_xticks([1, 4, 8, 12])
    ax.set_xlabel('Layer', labelpad=1)
    ax.set_ylabel(ylabel, labelpad=2, linespacing=1.05)
    if ylim:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    fit(fig, ax)
    handles, labels = model_handles()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.0, .0), ncol=2, handlelength=1.2,
               columnspacing=.6)
    check_and_save(fig, name)


# ---------------------------------------------------------------------------------------------------- C2 analysis
def c2_lines(results, name, table, metric, sources, ylabel, xlabel, threshold=None, xticks=None):
    frame = pd.read_csv(results / 'c2_analysis' / table)
    fig = plt.figure(figsize=(2.6, 1.9))
    ax = axes_at(fig, .55, .78, 1.95, 1.02)
    for model in MODELS:
        for source, style, _ in sources:
            part = frame[(frame.model == model) & (frame.source == source)]
            if metric:
                part = part[part.metric == metric]
            key = 'layer' if 'layer' in part else 'bin'
            part = part.sort_values(key)
            ax.plot(part[key], part['mean'], style, color=COLOR[model], marker='o', ms=2)
    extra_handles, extra_labels = [], []
    if threshold is not None:
        ax.axhline(threshold, color=GREY, lw=.7)
        extra_handles, extra_labels = [Line2D([], [], color=GREY, lw=.7)], ['pre-registered threshold']
    if metric is None:
        ax.axhline(0, color=GREY, lw=.6)
    ax.set_xlabel(xlabel, labelpad=1.5)
    ax.set_ylabel(ylabel, linespacing=1.05)
    if xticks is not None:
        ax.set_xticks(xticks)
    fit(fig, ax)
    handles, labels = model_handles()
    styles = [Line2D([], [], color='k', ls=style) for _, style, _ in sources] + extra_handles
    fig.legend(handles + styles, labels + [label for _, _, label in sources] + extra_labels,
               loc='lower left', bbox_to_anchor=(.01, .0), ncol=2)
    check_and_save(fig, name)


def c2_mean_prediction(name, model):
    path = OUTPUTS / 'c2_analysis' / model / 'mean_prediction.npz'
    if not path.exists():
        print(f'{name}: no c2_analysis outputs, skipped')
        return
    values = np.load(path)['layer12'].mean(0) * 196  # mean over the 2,000 held-out test images; uniform = 1
    fig = plt.figure(figsize=(1.35, 1.3))
    ax = axes_at(fig, .06, .08, 1.0, 1.0)
    image = ax.imshow(values.reshape(14, 14), cmap='viridis', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f'{NAME[model]}, layer 12', pad=2)
    bar = fig.colorbar(image, cax=axes_at(fig, 1.1, .08, .06, 1.0))
    bar.ax.tick_params(width=.5, length=2)
    bar.outline.set_linewidth(.4)
    bar.set_label('Mean probe prediction\n(uniform = 1)', labelpad=2, linespacing=1.05)
    # the colour bar label sits right of the bar; make room
    fig.set_size_inches(1.35 + .45, 1.3)
    ax.set_position([.06 / 1.8, .08 / 1.3, 1.0 / 1.8, 1.0 / 1.3])
    bar.ax.set_position([1.1 / 1.8, .08 / 1.3, .06 / 1.8, 1.0 / 1.3])
    check_and_save(fig, name)


# ---------------------------------------------------------------------------------------------------- schematic
KEY, VALUE, INK = '#6a3d9a', '#a50f15', '#262626'
HEAT = plt.get_cmap('inferno')


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


def schematic_reading():
    fig = plt.figure(figsize=(1.45, 1.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 29)
    ax.set_ylim(0, 38)
    ax.set_aspect('equal')
    ax.axis('off')
    rng = np.random.default_rng(0)
    n, s, x0, y0 = 7, 2.9, 4.35, 6.2
    yy, xx = np.mgrid[0:n, 0:n]
    heat = np.exp(-((xx - 3.2) ** 2 + (yy - 3.4) ** 2) / 5.0) + .25 * rng.random((n, n))
    heat[1, 5] = 1.35
    heat /= heat.max()
    for r in range(n):
        for c in range(n):
            ax.add_patch(Rectangle((x0 + c * s, y0 + (n - 1 - r) * s), s, s, facecolor=HEAT(.05 + .9 * heat[r, c]),
                                   edgecolor='white', lw=.35))
    ax.add_patch(Rectangle((x0 + 3 * s, y0 + 3 * s), s, s, fill=False, edgecolor='white', lw=1.0))
    ax.text(x0 + 3.5 * s, y0 + 3.5 * s, '$j$', ha='center', va='center', color='white')
    ax.text(14.5, 34.6, 'received attention', ha='center', va='center')
    ax.text(14.5, 30.6, r'$s_j = \sum_h \sum_i a_{ij}$', ha='center', va='center')
    ax.text(14.5, 2.6, 'unit $j$ drawn at patch $j$', ha='center', va='center', color=GREY)
    check_and_save(fig, 'schematic_reading')


def schematic_premises():
    fig = plt.figure(figsize=(5.5, 1.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 30)
    ax.axis('off')
    cols = [1.0, 25.5, 58.0, 81.0]
    for x, head in zip(cols, ['premise of the reading', 'measured by', 'correction', 'stand-in it must beat']):
        ax.text(x, 27.6, head, color=GREY, va='center')
    ax.plot([0.5, 109.5], [25.6, 25.6], color=GREY, lw=.5)
    rows = [('the activations are\nspecific to the image', 'share of variance explained\nby the mean map',
             'mean-map subtraction', 'centre-Gaussian subtraction'),
            ('unit $j$ still stands\nfor patch $j$', 'position-probe accuracy', 'probe relocation', 'matched blur'),
            ('a large activation means\na large contribution', 'share of received attention\non high-norm tokens',
             'contribution weighting', 'zero high-norm tokens')]
    for k, row in enumerate(rows):
        y = 21.0 - 8.2 * k
        if k % 2 == 0:
            ax.add_patch(Rectangle((0.5, y - 4.0), 109, 8.0, facecolor='#f3f3f3', lw=0, zorder=0))
        for x, text in zip(cols, row):
            ax.text(x, y, text, va='center', linespacing=1.1)
    check_and_save(fig, 'schematic_premises')


def main():
    global OUT, OUTPUTS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--outputs', type=Path, default=OUTPUTS)
    parser.add_argument('--figures', type=Path, default=OUT)
    args = parser.parse_args()
    results, OUTPUTS, OUT = args.results, args.outputs, args.figures
    OUT.mkdir(parents=True, exist_ok=True)
    schematic_network()
    schematic_reading()
    schematic_premises()
    execution_k_sweep(results)
    gallery('gallery_layers9_12', '')
    gallery('gallery_layer12', '_l12')
    premise_panel(results, 'premise_mean_map_share', 'M1', 'R0', 'Share of variance\nexplained by\nthe mean map')
    premise_panel(results, 'premise_probe_accuracy', 'M2', 'probe', 'Position-probe\naccuracy', (0, 1.03),
                  [0, .25, .5, .75, 1])
    premise_panel(results, 'premise_high_norm_share', 'M3', 'R0',
                  'Share of received\nattention on\nhigh-norm tokens')
    premise_panel(results, 'received_attention_vs_clickme', 'quality', 'R0',
                  'Received attention:\nimage-specific\nClickMe Spearman')
    probe_vs_blur = [('probe', '-', 'position probe'), ('stand_in_blur', '--', 'matched blur')]
    c2_lines(results, 'c2_analysis_distance_error', 'errors.csv', 'distance', probe_vs_blur,
             'Expected distance from\nthe true patch (patches)', 'Layer', xticks=range(6, 13))
    c2_lines(results, 'c2_analysis_far_error_share', 'errors.csv', 'far_share', probe_vs_blur,
             'Share of error mass\nfarther than 3 patches', 'Layer', threshold=.5, xticks=range(6, 13))
    c2_lines(results, 'c2_analysis_shift_by_distance', 'shift_bins.csv', None,
             [('C2', '-', 'probe relocation'), ('stand_in_blur', '--', 'matched blur')],
             'Change in share of\nthe score, layers 9–12', 'Distance from the image centre (patches)',
             xticks=range(0, 10, 2))
    c2_mean_prediction('c2_analysis_mean_prediction_deit_s', 'deit')
    c2_mean_prediction('c2_analysis_mean_prediction_dino', 'dino')


if __name__ == '__main__':
    main()
