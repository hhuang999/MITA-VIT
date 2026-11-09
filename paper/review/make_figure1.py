"""Figure 1 of the paper: attention as a fast-weight network (schematic, no data).

  python paper/review/make_figure1.py   ->  paper/review/figure_1.pdf (+ .png preview)
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.size': 5.6, 'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
BLUE, ORANGE, GREEN, GREY, DARK = '#1f77b4', '#e8710a', '#2ca02c', '#8c8c8c', '#262626'
HEAT = plt.get_cmap('Oranges')

W, H = 110, 40
fig = plt.figure(figsize=(5.5, 5.5 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')


def title(x, text):
    ax.text(x, H - 0.6, text, fontsize=6.2, fontweight='bold', va='top', ha='left')


def unit(x, y, r, face):
    ax.add_patch(Circle((x, y), r, facecolor=face, edgecolor=DARK, lw=0.5, zorder=3))


# (a) one head as a two-layer network written for each image -------------------------------------------
title(0.5, '(a) A head as a per-image network')
ax.add_patch(FancyBboxPatch((3, 31.2), 30, 2.6, boxstyle='round,pad=0.3', fc='#f0f0f0', ec=GREY, lw=0.4))
ax.text(18, 32.5, r'slow $W_K, W_V$ write $K, V$ per image', ha='center', va='center')
qx, ux, ox, cy = 4.5, 18, 31.5, 16
ys = [26, 21, 16, 11, 6]
acts = [0.15, 0.9, 0.45, 0.05, 0.65]
for y, a in zip(ys, acts):
    ax.plot([qx + 1.8, ux - 1.6], [cy, y], color=BLUE, lw=0.5, zorder=1)
    ax.plot([ux + 1.6, ox - 1.8], [y, cy], color=GREEN, lw=0.3 + 2.0 * a, alpha=0.85, zorder=1,
            solid_capstyle='round')
    unit(ux, y, 1.6, HEAT(0.1 + 0.85 * a))
ax.text(ux, 16, '$j$', ha='center', va='center', zorder=4)
unit(qx, cy, 1.8, 'white')
ax.text(qx, cy, '$q_i$', ha='center', va='center', zorder=4)
unit(ox, cy, 1.8, 'white')
ax.text(ox, cy, '$o_i$', ha='center', va='center', zorder=4)
ax.text(qx, cy - 2.6, 'query', ha='center', va='top', color=GREY)
ax.text(ox + 0.8, cy - 3.4, r'via $W_O$', ha='center', va='top', color=GREY)
ax.text(10.2, 25.2, '1st layer:\nkeys $k_j$', ha='center', va='bottom', color=BLUE, linespacing=1.1)
ax.text(25.8, 25.2, '2nd layer:\nvalues $v_j$', ha='center', va='bottom', color=GREEN, linespacing=1.1)
ax.text(ux, 28.6, 'hidden units = tokens', ha='center', va='bottom')
ax.text(ux, 1.2, r'activation $a_{ij}=\mathrm{softmax}_j(q_i^{\top}k_j/\sqrt{d})$', ha='center', va='bottom',
        color=ORANGE)

ax.plot([36.3, 36.3], [2, 36], color='#d9d9d9', lw=0.5)

# (b) the attention map reads the units ------------------------------------------------------------------
title(37.3, '(b) Reading R0')
rng = np.random.default_rng(0)
n, s, x0, y0 = 7, 2.15, 39.6, 7.0
yy, xx = np.mgrid[0:n, 0:n]
heat = np.exp(-((xx - 3.2) ** 2 + (yy - 3.4) ** 2) / 5.0) + 0.25 * rng.random((n, n))
heat[1, 5] = 1.35
heat /= heat.max()
for r in range(n):
    for c in range(n):
        ax.add_patch(Rectangle((x0 + c * s, y0 + (n - 1 - r) * s), s, s, facecolor=HEAT(0.04 + 0.9 * heat[r, c]),
                               edgecolor='white', lw=0.35))
ax.add_patch(Rectangle((x0 + 3 * s, y0 + 3 * s), s, s, fill=False, edgecolor=DARK, lw=0.8))
ax.text(x0 + 3.5 * s, y0 + 3.5 * s, '$j$', ha='center', va='center')
ax.text(x0 + n * s / 2, 26.0, r'$s_j=\sum_h\sum_i a_{ij}$', ha='center', va='center')
ax.text(x0 + n * s / 2, 23.0, 'activation of unit $j$', ha='center', va='center', color=GREY)
ax.text(x0 + n * s / 2, 4.6, 'drawn at patch $j$', ha='center', va='center', color=GREY)

ax.plot([57.6, 57.6], [2, 36], color='#d9d9d9', lw=0.5)

# (c) premises, how they are measured, corrections and stand-ins -----------------------------------------
title(58.6, '(c) Premises and corrections')
cols = [59.0, 75.0, 87.5, 100.6]
for x, h in zip(cols, ['premise', 'measured by', 'correction', 'stand-in']):
    ax.text(x, 33.2, h, color=GREY, va='center')
ax.plot([58.6, 109.5], [31.6, 31.6], color=GREY, lw=0.4)
rows = [
    ('M1  activations are\n       image-specific', 'mean-map\nshare', 'C1  subtract\n       mean map',
     'subtract\ncentre\nGaussian'),
    ('M2  unit $j$ stands\n       for patch $j$', 'position-probe\naccuracy', 'C2  relocate\n       by probe',
     'matched\nblur'),
    ('M3  activation =\n       contribution', 'R0 share on\nhigh-norm\ntokens', 'C3  weight by\n'
     r'       $\Vert W_O v_j\Vert$', 'zero high-\nnorm tokens'),
]
for k, row in enumerate(rows):
    y = 26.2 - 9.4 * k
    ax.add_patch(Rectangle((58.6, y - 4.5), 50.9, 9.0, facecolor='#f4f4f4' if k != 1 else 'white', lw=0, zorder=0))
    for x, text in zip(cols, row):
        ax.text(x, y, text, va='center', linespacing=1.25)
ax.text(84.0, 1.3, 'each correction must beat its stand-in', ha='center', va='bottom', color=GREY)

fig.savefig(OUT / 'figure_1.pdf')
fig.savefig(OUT / 'figure_1.png', dpi=300)
print('wrote', OUT / 'figure_1.pdf')
