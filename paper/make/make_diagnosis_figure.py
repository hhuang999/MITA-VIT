"""Figure 2 of the paper: the three diagnostics and map quality per layer, in one figure with a shared legend.
The figure carries no panel letters or titles; the paper sets the panel labels in LaTeX above the figure.
Reads results/diagnose.csv only (no number is computed here beyond scaling to per cent).
Usage: python3 -I make_diagnosis_figure.py <repo results dir> <output pdf>"""
import csv, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

results, out = sys.argv[1], sys.argv[2]
for path in ('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
             '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'):
    try: fm.fontManager.addfont(path)
    except Exception: pass
family = 'Liberation Sans' if any(f.name == 'Liberation Sans' for f in fm.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': family, 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .6, 'xtick.major.width': .6,
                     'ytick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
MODELS = ['deit', 'deit_b', 'dino']
NAME = {'deit': 'DeiT-S', 'deit_b': 'DeiT-B', 'dino': 'DINO ViT-S'}
COLOR = {'deit': '#2f6db5', 'deit_b': '#d9822b', 'dino': '#3a9a5b'}
MARK = {'deit': 'o', 'deit_b': 's', 'dino': '^'}
rows = list(csv.DictReader(open(results + '/diagnose.csv')))
PANELS = [('M1', 'R0', '(a) Specificity', 'Variance explained\nby the mean map (%)', 100.0, None),
          ('M2', 'probe', '(b) Location', 'Position-probe\naccuracy (%)', 100.0, (0, 104)),
          ('M3', 'R0', '(c) Contribution', 'Attention on\nhigh-norm tokens (%)', 100.0, None),
          ('quality', 'R0', '(d) Map quality', 'Image-specific\nClickMe alignment', 1.0, None)]
fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.38))
for ax, (panel, variant, title, ylabel, scale, ylim) in zip(axes, PANELS):
    ax.axvspan(8.5, 12.5, color='#000000', alpha=.06, lw=0)
    for m in MODELS:
        part = sorted((r for r in rows if r['panel'] == panel and r['variant'] == variant and r['model'] == m),
                      key=lambda r: int(r['layer']))
        assert len(part) == 12, (panel, m, len(part))
        x = [int(r['layer']) for r in part]
        ax.fill_between(x, [float(r['low']) * scale for r in part], [float(r['high']) * scale for r in part],
                        color=COLOR[m], alpha=.2, lw=0)
        ax.plot(x, [float(r['mean']) * scale for r in part], color=COLOR[m], marker=MARK[m], ms=2.2, lw=1.0)
    ax.set_xticks([1, 4, 8, 12]); ax.set_xlim(.5, 12.5)
    ax.set_xlabel('Layer', labelpad=1)
    ax.set_ylabel(ylabel, labelpad=2, linespacing=1.0)
    if ylim: ax.set_ylim(*ylim)
    if panel == 'quality': ax.axhline(0, color='#8c8c8c', lw=.6, zorder=0)
handles = [Line2D([], [], color=COLOR[m], marker=MARK[m], ms=2.6, lw=1.0) for m in MODELS]
handles.append(Patch(facecolor='#000000', alpha=.10, lw=0))
fig.legend(handles, [NAME[m] for m in MODELS] + ['layers 9–12 (summed in the maps)'], loc='lower center',
           ncol=4, frameon=False, handlelength=1.6, columnspacing=1.6, bbox_to_anchor=(.5, -.02))
fig.subplots_adjust(left=.085, right=.995, top=.96, bottom=.39, wspace=.72)
smallest = min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text())
assert smallest >= 7, smallest
fig.savefig(out)
print('saved', out, 'smallest font', smallest, 'family', family)
