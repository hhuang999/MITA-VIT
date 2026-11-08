"""CPU analysis of one explanations run (E4; decisions B06, B11): plausibility, faithfulness, SaCo and cost.

Writes into the run folder:
- analysis_plausibility.csv: every measure per method (mean and 95% interval), plus auroc - auroc_shuffled
  and the centre-adjusted score (method - centre) / (mask - centre) of mass, auroc and precision;
- analysis_faithfulness.csv: SRG on agreement (registered primary; budget 0 counts as agreement 1) and on
  p(top-1), insertion area, and the fixed-budget (probe size) conditions, per method;
- analysis_curves.csv: mean agreement / p(top-1) per method, condition and budget (for the figures);
- analysis_comparisons.csv: the registered primary comparisons, probe_mass against CLS attention, Rollout
  and GAE on SRG and on ring-matched enrichment (paired bootstrap), plus every method against random;
- analysis_strata.csv: enrichment and mass by object-area and object-distance tertile;
- analysis_saco.csv and analysis_cost.csv (milliseconds per image, median over batches);
- per_image_faithfulness.csv.gz: the per-image SRG, insertion and fixed-budget values behind the tables
  (small enough for Git; the full faithfulness.csv can stay in the run's raw/ folder).
Images with a missing value for any method are left out of that measure's paired tables.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fastweight_vit.results import load
from fastweight_vit.stats import interval, paired, ratio_interval, resample_indices, summarise, trapezoid

PRIMARY = 'probe_mass'
COMPARED = ['cls_attention', 'rollout', 'gae']
REFERENCES = {'centre', 'random', 'mask'}


def wide(frame, value):
    """One row per image, one column per method, complete cases only."""
    return frame.pivot(index='index', columns='method', values=value).dropna()


def plausibility(rows, run):
    out = []
    for measure in ['mass', 'pointing', 'auroc', 'auroc_shuffled', 'precision', 'enrichment']:
        table = wide(rows, measure)
        out += summarise(table, resample_indices(len(table)), measure=measure, n=len(table))
    gap = wide(rows, 'auroc') - wide(rows, 'auroc_shuffled')
    gap = gap.dropna()
    out += summarise(gap, resample_indices(len(gap)), measure='auroc_minus_shuffled', n=len(gap))
    for measure in ['mass', 'auroc', 'precision']:
        table = wide(rows, measure)
        if not {'centre', 'mask'} <= set(table.columns):
            continue
        draws = resample_indices(len(table))
        for method in table.columns:
            mean, low, high = ratio_interval(table[method] - table['centre'], table['mask'] - table['centre'], draws)
            out.append(dict(measure=f'{measure}_centre_adjusted', n=len(table), quantity=method, mean=mean, low=low,
                            high=high))
    frame = pd.DataFrame(out).rename(columns={'quantity': 'method'})
    frame.to_csv(run / 'analysis_plausibility.csv', index=False)
    return frame


def strata(rows, run):
    geometry = rows.drop_duplicates('index').set_index('index')[['object_area', 'object_distance']].dropna()
    out = []
    for column in ['object_area', 'object_distance']:
        bins = pd.qcut(geometry[column].rank(method='first'), 3, labels=['low', 'middle', 'high'])
        for measure in ['enrichment', 'mass']:
            table = wide(rows, measure)
            for level in ['low', 'middle', 'high']:
                subset = table.loc[table.index.intersection(bins.index[bins == level])]
                draws = resample_indices(len(subset))
                for row in summarise(subset, draws):
                    out.append(dict(stratum=column, level=level, measure=measure, n=len(subset),
                                    method=row['quantity'], mean=row['mean'], low=row['low'], high=row['high'],
                                    range_low=float(geometry.loc[bins == level, column].min()),
                                    range_high=float(geometry.loc[bins == level, column].max())))
    pd.DataFrame(out).to_csv(run / 'analysis_strata.csv', index=False)


def faithfulness(rows, run):
    curves = rows.query("condition in ['keep_top', 'drop_top', 'drop_bottom']")
    curves.groupby(['method', 'condition', 'budget'])[['agree', 'p_target', 'kl']].mean().reset_index().to_csv(
        run / 'analysis_curves.csv', index=False)
    budgets = sorted(curves.budget.unique())
    x = np.array([0] + budgets) / 196
    per_image = {}
    for value in ['agree', 'p_target']:
        table = curves.pivot_table(index=['method', 'index'], columns=['condition', 'budget'], values=value)
        drop = {c: table[c][budgets].to_numpy() for c in ('drop_top', 'drop_bottom')}
        if value == 'agree':
            start = np.ones((len(table), 1))
            lif, mif = (trapezoid(x, np.hstack([start, drop[c]])) for c in ('drop_bottom', 'drop_top'))
        else:
            lif, mif = (trapezoid(x[1:], drop[c]) for c in ('drop_bottom', 'drop_top'))
        per_image[f'srg_{value}'] = pd.Series(lif - mif, index=table.index)
        per_image[f'insertion_{value}'] = pd.Series(trapezoid(x[1:], table['keep_top'][budgets].to_numpy()),
                                                    index=table.index)
    fixed = rows.query("condition in ['keep_probe_k', 'drop_probe_k']").pivot_table(
        index=['method', 'index'], columns='condition', values='agree')
    for condition in fixed.columns:
        per_image[f'{condition}_agree'] = fixed[condition]
    per_image = pd.DataFrame(per_image)
    per_image.round(5).to_csv(run / 'per_image_faithfulness.csv.gz', compression='gzip')
    out = []
    for quantity in per_image.columns:
        table = per_image[quantity].unstack('method').dropna()
        out += summarise(table, resample_indices(len(table)), measure=quantity, n=len(table))
    frame = pd.DataFrame(out).rename(columns={'quantity': 'method'})
    frame.to_csv(run / 'analysis_faithfulness.csv', index=False)
    return per_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    meta = json.loads((args.run / 'metadata.json').read_text())
    labels = dict(model=meta['model'], dataset=meta['dataset'])
    checks, comparisons = dict(labels), []

    faith, duplicates = load(args.run, 'faithfulness.csv', keys=['index', 'method', 'condition', 'budget'])
    checks['faithfulness_duplicates'] = duplicates
    per_image = faithfulness(faith, args.run)
    for quantity in ['srg_agree', 'srg_p_target']:
        table = per_image[quantity].unstack('method').dropna()
        draws = resample_indices(len(table))
        comparisons += paired(table, PRIMARY, COMPARED, draws, measure=quantity, n=len(table), **labels)
        comparisons += [row for method in table.columns if method != 'random'
                        for row in paired(table, method, ['random'], draws, measure=quantity, n=len(table), **labels)]

    if (args.run / 'plausibility.csv').exists() or (args.run / 'plausibility.csv.gz').exists():
        plaus, duplicates = load(args.run, 'plausibility.csv', keys=['index', 'method'])
        checks['plausibility_duplicates'] = duplicates
        plausibility(plaus, args.run)
        strata(plaus, args.run)
        table = wide(plaus, 'enrichment')
        comparisons += paired(table, PRIMARY, COMPARED + ['centre'], resample_indices(len(table)),
                              measure='enrichment', n=len(table), **labels)

    saco, _ = load(args.run, 'saco.csv', keys=['index', 'method'])
    table = saco.pivot(index='index', columns='method', values='saco').dropna()
    pd.DataFrame(summarise(table, resample_indices(len(table)), measure='saco', n=len(table))).rename(
        columns={'quantity': 'method'}).to_csv(args.run / 'analysis_saco.csv', index=False)

    cost = pd.read_csv(args.run / 'cost.csv')
    cost = cost[cost.start > cost.start.min()]  # the first batch includes warm-up
    per_image = (cost.seconds / cost.images * 1000).groupby(cost.method).median()
    per_image.rename('ms_per_image').reset_index().to_csv(args.run / 'analysis_cost.csv', index=False)

    pd.DataFrame(comparisons).to_csv(args.run / 'analysis_comparisons.csv', index=False)
    (args.run / 'analysis_checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(pd.DataFrame(comparisons).query("other != 'random'").round(4).to_string())


if __name__ == '__main__':
    main()
