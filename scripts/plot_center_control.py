"""Figure 6: probe union reads against the centre prior (decision A14), from the control CSV only."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plot_anatomy', ROOT / 'scripts/plot_anatomy.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
plt, band, save = style.plt, style.band, style.save
MODELS = [('deit', 'DeiT-S', style.BLUE), ('dino', 'DINO ViT-S/16', style.AQUA), ('mita', 'MiTA-DeiT-S', style.ORANGE)]


def main():
    table = pd.read_csv(ROOT / 'experiments/2026-10-28_center_control/center_control_by_layer.csv')
    table = table.query("spec == 'm25_k25_mita'")
    layers = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)
    for axis, (model, label, color) in zip(axes, MODELS):
        frame = table.query('model == @model').sort_values('layer')
        axis.plot(layers, frame['probe'], '-o', color=color, lw=2, ms=4, label='probe reads')
        axis.plot(layers, frame['centre'], '--', color=style.QUIET, lw=1.5, label='same number, nearest centre')
        axis.axhline(.512, color=style.GRAY, lw=.8, ls=':')
        axis.set_title(label, loc='left', color=style.INK, fontsize=9)
        axis.set_xlabel('layer')
        axis.set_xticks(layers[1::2])
    axes[0].set_ylabel('foreground share of tokens read')
    axes[0].text(1.5, .53, 'foreground area 51%', fontsize=7, color=style.QUIET)
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Deep reads look object-like mainly because pets sit in the middle; only DINO layer 11 '
                 'beats the centre', x=.01, ha='left', fontsize=9, color=style.INK)
    fig.tight_layout()
    save(fig, ROOT / 'figures/figure6_center_control')


if __name__ == '__main__':
    main()
