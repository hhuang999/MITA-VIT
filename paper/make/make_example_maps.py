"""Example maps of the paper: Figure 1(b) (two images) and Figure D.1 (four further images).
Reads the stored DeiT-S gallery of the repository (outputs/table1/clickme/deit/gallery_*.npz): 24 ClickMe images
from a seeded random draw. Images are taken in the order of their ClickMe index, without picking: the two smallest
indices for Figure 1(b), the next four for Figure D.1. Maps are layers 9-12, min-max normalised one by one.
No number is computed here. The figures carry no panel letters; the paper sets them in LaTeX.
Usage: python3 -I make_example_maps.py <gallery dir> <main-text pdf> <appendix pdf>"""
import sys
from pathlib import Path
import json
from fastweight_vit.gallery import SELECTED, load_gallery
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

gallery, out_main, out_appendix = sys.argv[1], sys.argv[2], sys.argv[3]
for path in ('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',):
    try: fm.fontManager.addfont(path)
    except Exception: pass
family = 'Liberation Sans' if any(f.name == 'Liberation Sans' for f in fm.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': family, 'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})

COLUMNS = [('image', 'image'), ('human', 'human\n(ClickMe)'), ('map_R0', 'received\nattention'),
           ('map_C3', 'contribution\nweighting'), ('map_C2', 'probe\nrelocation')]
data = load_gallery(gallery)
index = data['index']
order = np.arange(len(index))
arrays = {key: data[key] for key, _ in COLUMNS}
print('ClickMe indices in order:', index[order].tolist())


def draw(ax, key, value):
    if key == 'image':
        ax.imshow(value)
    else:
        value = value.reshape(14, 14) if value.size == 196 else value
        value = (value - value.min()) / (value.max() - value.min() + 1e-12)
        ax.imshow(value, cmap='inferno', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure(out, picks, groups_per_row, cell, gap, group_gap, head):
    rows = -(-len(picks) // groups_per_row)
    group_w = 5 * cell + 4 * gap
    width = groups_per_row * group_w + (groups_per_row - 1) * group_gap
    height = head + rows * cell + (rows - 1) * gap
    fig = plt.figure(figsize=(width, height))
    for n, pick in enumerate(picks):
        r, g = divmod(n, groups_per_row)
        for c, (key, label) in enumerate(COLUMNS):
            x = g * (group_w + group_gap) + c * (cell + gap)
            y = height - head - (r + 1) * cell - r * gap
            ax = fig.add_axes([x / width, y / height, cell / width, cell / height])
            draw(ax, key, arrays[key][pick])
            if r == 0:
                ax.text(.5, 1.06, label, transform=ax.transAxes, ha='center', va='bottom', linespacing=1.0)
    fig.savefig(out)
    fig.savefig(Path(out).with_suffix('.png'), dpi=300)
    plt.close(fig)
    print(out, 'size (in):', round(width, 3), round(height, 3), 'images:', index[order][picks].tolist())


figure(out_main, [0, 1], 1, .6, .03, 0, .27)
figure(out_appendix, [2, 3, 4, 5], 2, .52, .025, .2, .27)
Path(out_main).with_name('gallery_selection.json').write_text(
    json.dumps({'all_indices': index.tolist(), **SELECTED}, indent=2) + '\n', encoding='utf-8')
