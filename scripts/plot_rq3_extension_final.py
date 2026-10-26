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
        'rq3_register_comparison': EXP / '2026-10-26_rq3_register_comparison/register_comparison',
    }
    for name, source in destinations.items():
        for ext in ['png', 'pdf']:
            shutil.copy2(source.with_suffix('.'+ext), ROOT / f'figures/{name}_2026-10-26.{ext}')
    # Show the actual fixed random heatmap, shared across layers, in B1.
    composition = pd.read_csv(EXP / '2026-10-26_rq3_diagnosis/B1/composition.csv')
    random = pd.read_csv(EXP / '2026-10-26_rq3_diagnosis/A/heatmaps.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for row, k in enumerate([16, 32]):
        frame = composition[(composition.k == k) & (composition.threshold == 3)]
        names = ['foreground', 'background', 'sink', 'cls', 'unknown']
        axes[row, 0].stackplot(range(1, 13), *[frame[frame.metric == name]['mean']*100 for name in names], labels=names)
        axes[row, 0].set_ylim(0, 100); axes[row, 0].set_title(f'Reading composition, k={k}')
        axes[row, 0].legend(fontsize=8, ncol=3, loc='upper left')
        for metric, color, linestyle in [('fg_mass', '#4c72b0', ':'), ('pointing', '#dd8452', '--')]:
            f = frame[frame.metric == metric]
            axes[row, 1].plot(f.layer, f['mean']*100, color=color, label=metric)
            axes[row, 1].fill_between(f.layer, f.low*100, f.high*100, alpha=.2, color=color)
            r = random[(random.k == k) & (random.method == 'random') & (random.metric == metric)].iloc[0]
            axes[row, 1].axhline(r['mean']*100, linestyle=linestyle, color=color, label=f'Random {metric}')
            axes[row, 1].axhspan(r.low*100, r.high*100, color=color, alpha=.05)
        axes[row, 1].set_ylim(0, 100); axes[row, 1].set_title(f'Layer localization, k={k}')
        axes[row, 1].legend(fontsize=8)
        for ax in axes[row]:
            ax.set_xlabel('Layer'); ax.set_ylabel('Percent')
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT / f'figures/rq3_layer_composition_2026-10-26.{ext}', dpi=180)
    plt.close(fig)
    # Initial shared xlim started at 1, hiding small DINOv2 norms. Keep those
    # initial figures, and add complete-range plots from the unchanged CSV bins.
    for alias in ['dino', 'register']:
        folder = EXP / f'2026-10-26_rq3_{alias}/A'
        hist = pd.read_csv(folder / 'norm_histogram.csv')
        occupied = hist[hist['mean'] > 0]
        limits = (occupied.bin_left.min()/1.2, occupied.bin_right.max()*1.2)
        fig, axes = plt.subplots(3, 4, figsize=(12, 8), layout='constrained')
        for layer, ax in enumerate(axes.flat, 1):
            f = hist[hist.layer == layer]
            x = np.sqrt(f.bin_left*f.bin_right)
            ax.plot(x, f['mean']); ax.fill_between(x, f.low, f.high, alpha=.2)
            ax.set_xscale('log'); ax.set_xlim(*limits); ax.set_title(f'Block {layer} input')
        fig.supxlabel('Patch L2 norm (log; all occupied bins shown)')
        fig.supylabel('Fraction of patches per bin')
        for ext in ['png', 'pdf']:
            fig.savefig(folder / f'norm_histograms_fullrange.{ext}', dpi=170, bbox_inches='tight')
        plt.close(fig)
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
