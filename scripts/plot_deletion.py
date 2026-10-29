"""Figure 7: matched deletion of foreground tokens with random and centre-matched controls (A15)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('plot_anatomy', ROOT / 'scripts/plot_anatomy.py')
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)
plt, save = style.plt, style.save
MODELS = [('deit', 'DeiT-S'), ('dino', 'DINO ViT-S/16'), ('mita', 'MiTA-DeiT-S')]
CONDITIONS = [('baseline', 'no deletion', '#c9c7c2'), ('random', 'random tokens removed', style.GRAY),
              ('centre_matched', 'equally central tokens removed', style.BLUE),
              ('target', 'animal tokens removed', style.ORANGE)]


def main():
    table = pd.read_csv(ROOT / 'experiments/2026-10-29_center_matched_deletion/summary.csv')
    fg = table.query("category == 'foreground'").set_index('model')
    base = table.query("category == 'baseline'").set_index('model')['target']
    fig, axis = plt.subplots(figsize=(7.4, 3.3))
    width = .2
    x = np.arange(len(MODELS))
    for offset, (key, label, color) in zip(np.arange(4) * width - 1.5 * width, CONDITIONS):
        values = [100 * (base[m] if key == 'baseline' else fg.loc[m, key]) for m, _ in MODELS]
        bars = axis.bar(x + offset, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f'{value:.0f}', ha='center',
                      fontsize=7, color=style.QUIET)
    axis.set_xticks(x, [label for _, label in MODELS])
    axis.set_ylabel('agreement with original prediction (%)')
    axis.set_ylim(0, 110)
    axis.legend(fontsize=7, ncol=2, loc='upper left')
    axis.set_title('Removing animal tokens hurts far more than removing equally central tokens '
                   '(Pet, all 12 layers)', loc='left', color=style.INK, fontsize=9)
    fig.tight_layout()
    save(fig, ROOT / 'figures/figure7_deletion')


if __name__ == '__main__':
    main()
