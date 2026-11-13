"""Verification layer 1: compare the tables rebuilt from the saved study/v5 outputs with the study/v5 tables.

  python docs/release/compare_layer1.py <study/v5 checkout> <rebuilt results> <report.csv>

Each pair: the study/v5 table, the release table, the key columns that identify a row, the value columns, and
the renames that map study/v5 labels to release labels. Writes one row per pair: rows matched, rows only on one
side, the largest absolute difference over the value columns.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RENAME = {'blur': 'stand_in_blur', 'no_high': 'stand_in_no_high', 'centre_sub': 'stand_in_centre',
          'blurMLP': 'stand_in_blur_MLP', 'correction - control': 'correction - stand-in',
          'control - R0': 'stand-in - R0', 'new probe': 'probe', 'R0 (S3)': 'R0', 'R0 (S2)': 'R0',
          'probe - blur': 'probe - stand_in_blur', 'no_high - R0': 'stand_in_no_high - R0',
          'C3 - no_high': 'C3 - stand_in_no_high',
          'C2MLP - blurMLP': 'C2MLP - stand_in_blur_MLP', 'X1-A': 'A', 'X1-B': 'B', 'X3-A': 'A', 'X3-B': 'B',
          'M-weak': 'weak', 'H-global does not hold': 'does not hold', 'H-global holds': 'holds'}


def label(value):
    if isinstance(value, str):
        value = RENAME.get(value, value)
        value = re.sub(r'^no_high_l(\d+)$', r'stand_in_no_high_l\1', value)
        value = value.replace(' vs control:', ' vs stand-in:')
    return value


def relabel(frame):
    return frame.apply(lambda column: column.map(label) if not pd.api.types.is_numeric_dtype(column) else column)


def compare(name, old, new, keys, values, old_filter=None, new_filter=None):
    old, new = relabel(old), new.copy()
    if old_filter is not None:
        old = old_filter(old)
    if new_filter is not None:
        new = new_filter(new)
    for frame in (old, new):
        for key in keys:
            frame[key] = frame[key].astype(str)
    merged = old.merge(new, on=keys, how='outer', suffixes=('_old', '_new'), indicator=True)
    both = merged[merged._merge == 'both']
    worst = 0.
    for value in values:
        a, b = both[f'{value}_old'], both[f'{value}_new']
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            mismatch = ((a.astype(str) != b.astype(str)) & ~(a.isna() & b.isna())).sum()
            worst = max(worst, float(mismatch))
        else:
            diff = (a.astype(float) - b.astype(float)).abs()
            nan_mismatch = (a.isna() != b.isna()).sum()
            worst = max(worst, float(np.nanmax(diff.to_numpy())) if len(diff) else 0., float(nan_mismatch))
    return dict(table=name, rows=len(both), only_old=int((merged._merge == 'left_only').sum()),
                only_new=int((merged._merge == 'right_only').sum()), max_abs_difference=worst)


def main():
    study, results, report = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    a = study / 'experiments/2026-11-05_v5_analysis'
    e = study / 'experiments/2026-11-07_explore_c2_analysis'
    s = study / 'experiments/2026-11-07_v5_supp_analysis'
    r = lambda path: pd.read_csv(path)  # noqa: E731
    stat = ['n', 'mean', 'low', 'high']
    rows = [
        compare('k sweep (figure 2a)', r(a / 'fig2a_ksweep.csv'), r(results / 'k_sweep.csv'), ['model', 'sets', 'k'], stat),
        compare('k sweep tests (P1)', r(a / 'p1_tests.csv'), r(results / 'k_sweep_tests.csv'), ['model', 'test', 'k'],
                stat + ['holds']),
        compare('diagnose (figure 3)', r(a / 'fig3_diagnose.csv'), r(results / 'diagnose.csv'),
                ['model', 'panel', 'variant', 'layer'], stat, old_filter=lambda f: f[f.variant != 'E3']),
        compare('Table 1 (long)', r(a / 'table1_long.csv'), r(results / 'table1_long.csv'), ['model', 'metric', 'method'],
                stat),
        compare('correction tests', r(a / 'correct_tests.csv'), r(results / 'correct_tests.csv'),
                ['model', 'metric', 'test', 'a', 'b'], stat + ['verdict']),
        compare('last-layer recovery', r(a / 'a6_last_layer.csv'), r(results / 'last_layer.csv'),
                ['model', 'metric', 'reading'], ['best_layer', 'drop', 'recovered', 'low', 'high']),
        compare('combination order', r(a / 'a7_order.csv'), r(results / 'order.csv'), ['model', 'metric', 'a', 'b'],
                stat + ['verdict'], new_filter=lambda f: f[f.test == 'order']),
        compare('change the model', r(a / 'variants.csv'), r(results / 'variants.csv'), ['variant', 'metric', 'test'],
                stat),
        compare('sanity', r(a / 'sanity.csv'), r(results / 'sanity.csv'), ['model', 'level', 'method'],
                ['median', 'passes']),
        compare('predictions', r(a / 'predictions.csv'), r(results / 'predictions.csv'), ['prediction', 'model', 'detail'],
                ['holds']),
        compare('single layer (appendix B)', r(a / 'a2_single_layer.csv'), r(results / 'single_layer.csv'),
                ['model', 'test', 'layer'], ['mean', 'low', 'high']),
        compare('top-k map similarity', r(study / 'experiments/2026-10-31_s_analysis/s1_similarity.csv'),
                r(results / 'map_similarity.csv'), ['model'], ['median', 'q25', 'q75'],
                old_filter=lambda f: f[f.dataset == 'imagenet_s']),
        compare('probe accuracy', pd.concat([r(study / f'experiments/2026-11-05_diagnose_probe_{m}/identity.csv')
                                             for m in ('deit', 'deit_b', 'dino')]),
                r(results / 'probe_accuracy.csv'), ['model', 'layer'],
                ['position_accuracy', 'error_distance', 'top_probability', 'displacement', 'sigma']),
        compare('C2 analysis: errors', r(e / 'errors.csv'), r(results / 'c2_analysis/errors.csv'),
                ['model', 'layer', 'source', 'metric'], stat),
        compare('C2 analysis: mean prediction', r(e / 'mean_prediction.csv'), r(results / 'c2_analysis/mean_prediction.csv'),
                ['model', 'layer', 'quantity'], stat),
        compare('C2 analysis: shift', r(e / 'shift.csv'), r(results / 'c2_analysis/shift.csv'),
                ['model', 'source', 'quantity'], stat),
        compare('C2 analysis: shift bins', r(e / 'shift_bins.csv'), r(results / 'c2_analysis/shift_bins.csv'),
                ['model', 'source', 'bin'], stat + ['r0_share']),
        compare('C2 analysis: hypotheses', r(e / 'hypotheses.csv').reset_index(),
                r(results / 'c2_analysis/hypotheses.csv').reset_index(), ['index'], ['verdict', 'detail']),
    ]
    for old, new, keys, values in (
            ('x1.csv', 'combinations.csv', ['model', 'metric', 'test', 'a', 'b'], stat + ['verdict']),
            ('x2.csv', 'c3_mechanism.csv', ['model', 'metric', 'test'], stat + ['verdict']),
            ('x2_seeds.csv', 'c3_mechanism_seeds.csv', ['model', 'metric', 'seed'], ['mean', 'c3_minus']),
            ('x3_layers.csv', 'dose_layers.csv', ['model', 'metric', 'test', 'layer'], stat + ['verdict']),
            ('x3_dose.csv', 'dose.csv', ['model', 'metric', 'improvement'], ['n', 'spearman', 'low', 'high', 'mean_dose']),
            ('x4_layers.csv', 'globalisation.csv', ['model', 'layer', 'quantity'], stat),
            ('x5_probe.csv', 'mlp_probe.csv', ['model', 'layer', 'probe', 'quantity'], stat),
            ('x5.csv', 'c2_variants.csv', ['model', 'metric', 'test'], stat + ['verdict']),
            ('x6.csv', 'gae_gap.csv', ['model', 'metric'], ['n', 'c3_gain', 'gae_gap', 'share_closed', 'low', 'high']),
            ('x2_head_weights.csv', 'head_weights.csv', ['model', 'layer', 'head'], ['mean_length'])):
        rows.append(compare(f'supplementary: {new}', r(s / old), r(results / 'supplementary' / new), keys, values))
    d = study / 'experiments/2026-11-08_explore_d2_analysis'
    if d.exists():
        rows.append(compare('D2: tests', r(d / 'tests.csv'), r(results / 'd2/tests.csv'),
                            ['model', 'metric', 'test', 'method'], stat + ['verdict']))
        rows.append(compare('D2: branch', r(d / 'branch.csv').assign(branch=lambda f: f.branch.str[-1]),
                            r(results / 'd2/branch.csv'), ['item'], ['branch', 'higher', 'lower']))
    old_md = (a / 'table1.md').read_text()
    for before, after in RENAME.items():
        old_md = old_md.replace(f'| {before} |', f'| {after} |')
    rows.append(dict(table='Table 1 (markdown)', rows=old_md.count('\n'), only_old=0, only_new=0,
                     max_abs_difference=float(old_md != (results / 'table1.md').read_text())))
    frame = pd.DataFrame(rows)
    frame.to_csv(report, index=False)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
