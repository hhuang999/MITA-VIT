"""Figure: the probe against the baselines on one dataset (E4), one row per model.

(a) agreement with the original top-1 as the most important patches are removed first (lower = more
faithful), (b) SRG on agreement, the registered primary faithfulness measure, (c) ring-matched object
enrichment, the registered primary plausibility measure (comparison at equal distance from the centre).
Reads only the analysis_*.csv tables written by analyze_explanations.py.

Usage: python scripts/plot_explanations.py RUN [RUN ...] --output figures/figure8_explanations
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from fastweight_vit.plotting import (MAIN_METHODS, METHOD_COLORS, METHOD_LABELS, METHOD_STYLES, MODEL_LABELS,
                                 interval_dots, save)

CURVE_METHODS = ['probe_mass', 'cls_attention', 'rollout', 'gae', 'gradcam', 'integrated_gradients', 'centre',
                 'random']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(len(args.runs), 3, figsize=(11, 3.2 * len(args.runs)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    for row, run in zip(axes, args.runs):
        meta = json.loads((run / 'metadata.json').read_text())
        model = MODEL_LABELS.get(meta['model'], meta['model'])
        curves = pd.read_csv(run / 'analysis_curves.csv').query("condition == 'drop_top'")
        for method in CURVE_METHODS:
            curve = curves.query('method == @method').sort_values('budget')
            if len(curve):
                x = [0] + (curve.budget / 196 * 100).tolist()
                row[0].plot(x, [1] + curve.agree.tolist(), METHOD_STYLES.get(method, '-'), lw=2,
                            color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        row[0].set(xlabel='Patches removed, most important first (%)', ylabel='Agreement with original top-1',
                   title=f'{model}: removal curves', ylim=(0, 1.02))
        faith = pd.read_csv(run / 'analysis_faithfulness.csv').query("measure == 'srg_agree'").set_index('method')
        interval_dots(row[1], faith, MAIN_METHODS)
        row[1].axvline(0, color='#c3c2b7', lw=1)
        row[1].set(xlabel='SRG on agreement (higher = more faithful)', title='Faithfulness')
        plaus = run / 'analysis_plausibility.csv'
        if plaus.exists():
            enrich = pd.read_csv(plaus).query("measure == 'enrichment'").set_index('method')
            interval_dots(row[2], enrich, MAIN_METHODS)
            row[2].axvline(0, color='#c3c2b7', lw=1)
            row[2].set(xlabel='Object enrichment (same-distance rings)', title='Plausibility beyond centre')
        else:
            row[2].set_visible(False)
    axes[0, 0].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    save(fig, args.output)


if __name__ == '__main__':
    main()
