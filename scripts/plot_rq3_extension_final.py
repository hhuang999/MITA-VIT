"""Prepare final RQ3 figures from result CSV and verified existing figures."""

from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'


def main():
    destinations = {
        'rq3_peak_locations': EXP / '2026-10-26_rq3_diagnosis/A/peak_locations',
        'rq3_norm_histograms': EXP / '2026-10-26_rq3_diagnosis/A/norm_histograms',
        'rq3_layer_composition': EXP / '2026-10-26_rq3_diagnosis/B1/layer_composition',
        'rq3_register_comparison': EXP / '2026-10-26_rq3_register_comparison/register_comparison',
    }
    for name, source in destinations.items():
        for ext in ['png', 'pdf']:
            shutil.copy2(source.with_suffix('.'+ext), ROOT / f'figures/{name}_2026-10-26.{ext}')
    frame = pd.read_csv(EXP / '2026-10-26_rq3_causal/B3/causal_summary.csv')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
    names = ['baseline', 'sink', 'foreground', 'background', 'random']
    labels = ['Baseline', 'Remove\nhigh norm', 'Remove\nforeground', 'Remove\nbackground', 'Matched\nrandom']
    for i, k in enumerate([16, 32]):
        for j, scope in enumerate(['last4', 'all12']):
            f = frame[(frame.k == k) & (frame.scope == scope) & (frame.support == 'common') & (frame.metric == 'agreement')].set_index('method').loc[names]
            ax = axes[i, j]
            y = f['mean'].to_numpy()*100
            ax.bar(np.arange(5), y, yerr=np.stack([f['mean']-f.low, f.high-f['mean']])*100,
                   color=['#888888', '#c44e52', '#55a868', '#4c72b0', '#8172b2'], capsize=3)
            ax.set_xticks(np.arange(5), labels, fontsize=9)
            for x, value in enumerate(y):
                ax.text(x, value+3, f'{value:.1f}', ha='center', fontsize=9)
            scope_label = 'layers 9-12' if scope == 'last4' else 'all 12 layers'
            ax.set_title(f'k = {k}, {scope_label}\nCommon eligible images: {int(f.n.iloc[0]):,}')
            ax.set_ylabel('Prediction agreement (%)')
            ax.set_ylim(0, 102)
            ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT / f'figures/rq3_causal_2026-10-26.{ext}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
