"""Verification layer 2: compare the tables of a full rerun (fresh clone) with the tables of the original runs.

  python docs/release/compare_layer2.py <original results> <rerun results> <report.csv>

Tolerances (docs/handoff 02, section 6.2): means of readings and metrics 1e-4; readings that use the position probe
(C2, its stand-in blur, R_all and the other order, the combinations and variants with C2, D2) 2e-3; execution
agreement 0.1 percentage points (1e-3); retrained probe accuracy 0.5 points (5e-3); every verdict identical.
Writes one row per table and tolerance class: rows compared, largest absolute difference of the means, rows over the
tolerance, verdict mismatches.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROBE = re.compile(r'C2|R_all|blur|D2|MLP|conf')
TABLES = [
    # file, keys, value columns, verdict column, tolerance class
    ('k_sweep.csv', ['model', 'sets', 'k'], ['mean'], None, 'execution'),
    ('k_sweep_tests.csv', ['model', 'test', 'k'], ['mean'], 'holds', 'execution'),
    ('single_layer.csv', ['model', 'test', 'layer'], ['mean'], None, 'execution'),
    ('diagnose.csv', ['model', 'panel', 'variant', 'layer'], ['mean'], None, 'panel'),
    ('diagnose_tests.csv', ['model', 'test'], ['value'], 'holds', 'mean'),
    ('probe_accuracy.csv', ['model', 'layer'], ['position_accuracy'], None, 'probe'),
    ('radial.csv', ['model', 'geometry', 'variant', 'layer'], ['radial_share'], None, 'by_name'),
    ('value_norm.csv', ['config', 'model', 'layer'], ['value_ratio'], None, 'mean'),
    ('table1_long.csv', ['model', 'metric', 'method'], ['mean'], None, 'by_name'),
    ('correct_tests.csv', ['model', 'metric', 'test', 'a', 'b'], ['mean'], 'verdict', 'by_name'),
    ('last_layer.csv', ['model', 'metric', 'reading'], ['recovered'], None, 'by_name'),
    ('order.csv', ['model', 'metric', 'test', 'a', 'b', 'method'], ['mean'], 'verdict', 'by_name'),
    ('variants.csv', ['variant', 'metric', 'test'], ['mean'], 'verdict', 'mean'),
    ('sanity.csv', ['model', 'level', 'method'], ['median'], 'passes', 'by_name'),
    ('negative_controls.csv', ['model', 'metric', 'test'], ['mean'], 'verdict', 'by_name'),
    ('predictions.csv', ['prediction', 'model', 'detail'], [], 'holds', 'mean'),
    ('part_reliance.csv', ['model', 'test'], ['mean'], 'verdict', 'mean'),
    ('map_similarity.csv', ['model'], ['median'], None, 'mean'),
    ('numbers.csv', ['model', 'quantity'], ['value'], None, 'probe'),
    ('c2_analysis/errors.csv', ['model', 'layer', 'source', 'metric'], ['mean'], None, 'probe'),
    ('c2_analysis/shift.csv', ['model', 'source', 'quantity'], ['mean'], None, 'probe'),
    ('c2_analysis/hypotheses.csv', ['hypothesis'], [], 'verdict', 'probe'),
    ('supplementary/combinations.csv', ['model', 'metric', 'test', 'a', 'b'], ['mean'], 'verdict', 'by_name'),
    ('supplementary/c3_mechanism.csv', ['model', 'metric', 'test'], ['mean'], 'verdict', 'mean'),
    ('supplementary/dose.csv', ['model', 'metric', 'improvement'], ['spearman'], None, 'mean'),
    ('supplementary/mlp_probe.csv', ['model', 'layer', 'probe', 'quantity'], ['mean'], None, 'probe'),
    ('supplementary/c2_variants.csv', ['model', 'metric', 'test'], ['mean'], 'verdict', 'probe'),
    ('supplementary/gae_gap.csv', ['model', 'metric'], ['share_closed'], None, 'mean'),
    ('supplementary/branches.csv', ['item'], [], 'branch', 'mean'),
    ('d2/tests.csv', ['model', 'metric', 'test', 'method'], ['mean'], 'verdict', 'probe'),
    ('d2/branch.csv', ['item'], [], 'branch', 'probe'),
]
TOLERANCE = {'mean': 1e-4, 'probe': 2e-3, 'execution': 1e-3}


def row_class(kind, row, keys):
    if kind != 'by_name' and kind != 'panel':
        return kind
    if kind == 'panel':
        return 'probe' if row['panel'] == 'M2' else 'mean'
    label = ' '.join(str(row[key]) for key in keys)
    return 'probe' if PROBE.search(label) else 'mean'


def main():
    original, rerun, report = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows = []
    for name, keys, values, verdict, kind in TABLES:
        if not (original / name).exists() or not (rerun / name).exists():
            rows.append(dict(table=name, tolerance_class='-', rows=0, note='missing'))
            continue
        old, new = pd.read_csv(original / name), pd.read_csv(rerun / name)
        keys = [key for key in keys if key in old and key in new]
        for frame in (old, new):
            for key in keys:
                frame[key] = frame[key].astype(str)
        merged = old.merge(new, on=keys, how='outer', suffixes=('_old', '_new'), indicator=True)
        both = merged[merged._merge == 'both'].copy()
        both['class'] = [row_class(kind, row, keys) for _, row in both.iterrows()]
        for cls, part in both.groupby('class'):
            out = dict(table=name, tolerance_class=cls, rows=len(part),
                       only_one_side=int((merged._merge != 'both').sum()))
            if values:
                diff = pd.concat([(part[f'{v}_old'].astype(float) - part[f'{v}_new'].astype(float)).abs()
                                  for v in values], axis=1).max(axis=1)
                out['max_abs_difference'] = float(np.nanmax(diff.to_numpy())) if len(diff) else 0.
                out['over_tolerance'] = int((diff > TOLERANCE[cls]).sum())
            if verdict:
                a, b = part[f'{verdict}_old'], part[f'{verdict}_new']
                out['verdict_mismatches'] = int(((a.astype(str) != b.astype(str)) & ~(a.isna() & b.isna())).sum())
            rows.append(out)
    frame = pd.DataFrame(rows)
    frame.to_csv(report, index=False)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
