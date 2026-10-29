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


# One fixed colour per method in every figure (categorical slots 1-6, validated for colour-vision deficiency);
# references are neutral and told apart by line style.
METHOD_COLORS = {'probe_mass': BLUE, 'cls_attention': ORANGE, 'rollout': AQUA, 'gae': '#eda100',
                 'gradcam': '#e87ba4', 'integrated_gradients': '#008300', 'centre': GRAY, 'random': '#b5b4ae',
                 'mask': INK, 'mean_human': INK}
METHOD_STYLES = {'centre': '--', 'random': ':', 'mask': '-.', 'mean_human': '-.'}
METHOD_LABELS = {'probe_mass': 'Probe (ours)', 'cls_attention': 'CLS attention', 'rollout': 'Rollout',
                 'gae': 'GAE', 'gradcam': 'Grad-CAM', 'integrated_gradients': 'Integrated Gradients',
                 'centre': 'Centre (reference)', 'random': 'Random (reference)', 'mask': 'Object mask (reference)',
                 'mean_human': 'Mean human map (reference)'}
MAIN_METHODS = list(METHOD_LABELS)
MODEL_LABELS = {'deit': 'DeiT-S', 'dino': 'DINO ViT-S', 'deit_b': 'DeiT-B', 'mita': 'MiTA-DeiT-S'}


def interval_dots(ax, frame, order, value='mean'):
    """Horizontal dot-and-whisker rows (frame indexed by method: mean, low, high), top to bottom in `order`."""
    rows = [m for m in order if m in frame.index]
    for y, method in enumerate(rows):
        row = frame.loc[method]
        color = METHOD_COLORS.get(method, QUIET)
        ax.plot([row['low'], row['high']], [y, y], color=color, lw=2, solid_capstyle='round')
        ax.plot(row[value], y, 'o', color=color, ms=6, mec='white', mew=1.5)
    ax.set_yticks(range(len(rows)), [METHOD_LABELS.get(m, m) for m in rows])
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
