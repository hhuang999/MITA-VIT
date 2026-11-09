"""Appendix figure on C2 (paper layout of figure_A_c2 of release/v1, panels relabelled a-e).

  python paper/review/make_figure_c2.py <release results dir> <study experiments dir>
  -> paper/review/figure_A_c2.pdf (+ .png preview)

Data: c2_analysis/errors.csv and c2_analysis/shift_bins.csv (release/v1 results) and the layer-12 mean predictions
(experiments/2026-11-07_explore_c2_<model>/mean_prediction.npz on study/v5).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(__file__).resolve().parent
results, experiments = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({'font.size': 6.5, 'axes.titlesize': 6.8, 'axes.labelsize': 6.5, 'legend.fontsize': 5.6,
                     'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': 0.6})
LONG = {'deit': 'DeiT-S', 'deit_b': 'DeiT-B', 'dino': 'DINO'}
COLORS = {'deit': '#2f6db5', 'deit_b': '#d9822b', 'dino': '#3a9a5b'}
errors = pd.read_csv(results / 'c2_analysis' / 'errors.csv')
bins = pd.read_csv(results / 'c2_analysis' / 'shift_bins.csv')

fig = plt.figure(figsize=(5.5, 3.6))
outer = fig.add_gridspec(2, 1, hspace=0.6)
top = outer[0].subgridspec(1, 2, wspace=0.25)
bottom = outer[1].subgridspec(1, 3, wspace=0.55, width_ratios=[1, 1, 1.25])


def lines(ax, table, metric, sources):
    for model, color in COLORS.items():
        for source, style, label in sources:
            part = table[(table.model == model) & (table.source == source)]
            if metric:
                part = part[part.metric == metric]
            key = 'layer' if 'layer' in part else 'bin'
            part = part.sort_values(key)
            ax.plot(part[key], part['mean'], style, color=color, marker='o', ms=2, lw=0.9,
                    label=f'{LONG[model]} {label}')


ax = fig.add_subplot(top[0])
lines(ax, errors, 'distance', [('probe', '-', 'probe'), ('stand_in_blur', '--', 'blur')])
ax.set_title('(a) Expected distance error (patches)')
ax.set_xlabel('Layer')
ax = fig.add_subplot(top[1])
lines(ax, errors, 'far_share', [('probe', '-', 'probe'), ('stand_in_blur', '--', 'blur')])
ax.axhline(.5, color='#999999', lw=.6)
ax.set_title('(b) Share of error mass farther than 3 patches')
ax.set_xlabel('Layer')
ax.legend(ncol=1, frameon=False, handlelength=2.2, loc='upper left', fontsize=5.2, labelspacing=0.25)
for k, (model, letter) in enumerate((('deit', 'c'), ('dino', 'd'))):
    ax = fig.add_subplot(bottom[k])
    values = np.load(experiments / f'2026-11-07_explore_c2_{model}' / 'mean_prediction.npz')['layer12'].mean(0) * 196
    image = ax.imshow(values.reshape(14, 14), cmap='viridis')
    ax.set_title(f'({letter}) {LONG[model]}, layer 12:\nmean prediction / uniform')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=.046, pad=0.04)
    bar.ax.tick_params(labelsize=5.5)
ax = fig.add_subplot(bottom[2])
lines(ax, bins, None, [('C2', '-', 'C2'), ('stand_in_blur', '--', 'blur')])
ax.axhline(0, color='#999999', lw=.6)
ax.set_title('(e) Change in score share\nby distance to centre')
ax.set_xlabel('Distance to centre (patches)')
ax.set_xticks(range(0, 10, 2))
fig.savefig(OUT / 'figure_A_c2.pdf', bbox_inches='tight')
fig.savefig(OUT / 'figure_A_c2.png', dpi=250, bbox_inches='tight')
print('wrote', OUT / 'figure_A_c2.pdf')
