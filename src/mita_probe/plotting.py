"""Shared figure style: one palette, light grid, 95% bands; figures are saved as PNG and PDF."""

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, GRAY, INK, QUIET = '#2a78d6', '#eb6834', '#1baf7a', '#8a8986', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': QUIET, 'axes.labelcolor': INK, 'xtick.color': QUIET,
                     'ytick.color': QUIET, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': .6, 'legend.frameon': False})


def band(ax, x, frame, color, label=None, style='-', marker='o'):
    """Line for frame['mean'] with a shaded frame['low']..frame['high'] band."""
    ax.plot(x, frame['mean'], style, color=color, lw=2, marker=marker, ms=4, label=label)
    ax.fill_between(x, frame['low'], frame['high'], color=color, alpha=.15, lw=0)


def save(fig, stem):
    fig.savefig(f'{stem}.png', dpi=200)
    fig.savefig(f'{stem}.pdf')
    plt.close(fig)
