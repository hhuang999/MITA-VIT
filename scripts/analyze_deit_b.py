"""CPU analysis of the DeiT-B/16 anatomy runs (decision B09): RQ1 single-layer and RQ2 main-budget tables,
in the same form as the DeiT-S and DINO analyses, plus a side-by-side table of the three models.

Usage: python scripts/analyze_deit_b.py [--prefix experiments/2026-10-29_v3_deit_b_]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mita_probe.results import load, rows_of, seed_mean
from mita_probe.stats import anatomy_table, interval, ratio_interval, resample_indices

ROOT = Path(__file__).resolve().parents[1]
OTHERS = {'deit': ROOT / 'experiments/2026-10-28_v2_analysis', 'dino': ROOT / 'experiments/2026-10-28_v2_dino_analysis'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default=str(ROOT / 'experiments/2026-10-29_v3_deit_b_'))
    args = parser.parse_args()
    run = lambda stage: Path(f'{args.prefix}{stage}')
    out = run('analysis')
    out.mkdir(exist_ok=True)
    checks = json.loads((run('repro') / 'checks.json').read_text())

    rq1, checks['rq1_duplicates'] = load(run('rq1'), 'predictions.csv')
    n = rq1['index'].nunique()
    draws = resample_indices(n)
    rows = []
    for layer, frame in rq1.groupby('layer'):
        table = seed_mean(frame, ['setting'])
        assert len(table) == n and not table.isna().any().any()
        summary = anatomy_table(table, draws, upper='L1_k32')
        summary.loc['sharing_index_k16'] = ratio_interval(table['L4'] - table['rand'],
                                                          table['L1_k16'] - table['rand'], draws)
        rows += rows_of(summary, layer=layer)
    rq1_table = pd.DataFrame(rows)
    rq1_table.to_csv(out / 'rq1_anatomy.csv', index=False)

    rq2, checks['rq2_duplicates'] = load(run('rq2'), 'predictions.csv')
    specs = pd.read_csv(run('rq2') / 'settings.csv')
    rq2 = rq2.merge(specs, on='setting_id', validate='many_to_one')
    rq2['label'] = rq2['name'] + np.where(rq2['name'] == 'L1', '_k' + rq2['k'].astype(int).astype(str), '')
    per_image = rq2.groupby(['label', 'index'])['agree'].mean().unstack('label').sort_index()
    assert len(per_image) == n and not per_image.isna().any().any()
    pd.DataFrame([dict(setting=name, **dict(zip(['agree', 'low', 'high'], interval(per_image[name], draws))))
                  for name in per_image]).to_csv(out / 'rq2_main_budget.csv', index=False)
    main = per_image.rename(columns=dict(L3_spatial='L3', L4_spatial='L4'))
    summary = anatomy_table(main[['rand', 'randC', 'L3', 'L4', 'L2', 'L1_k32', 'L1_k16']], draws, upper='L1_k32')
    all_layers = pd.DataFrame(rows_of(summary, layer='all'))
    all_layers.to_csv(out / 'rq2_anatomy_all_layers.csv', index=False)

    compare = [all_layers.assign(model='deit_b')]
    for model, directory in OTHERS.items():
        path = directory / ('rq2_anatomy_all_layers.csv' if model == 'deit' else 'compare_rq2_anatomy_all_layers.csv')
        if path.exists():
            frame = pd.read_csv(path)
            compare.append(frame.query('model == @model') if 'model' in frame else frame.assign(model=model))
    compare = pd.concat(compare)
    compare.to_csv(out / 'compare_rq2_anatomy_all_layers.csv', index=False)
    (out / 'checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(checks))
    print(compare.pivot(index='quantity', columns='model', values='mean').round(3).to_string())
    layers = rq1_table[rq1_table.quantity.isin(['L4', 'rand', 'L1_k32', 'sharing_index_k16'])]
    print(layers.pivot(index='layer', columns='quantity', values='mean').round(3).to_string())


if __name__ == '__main__':
    main()
