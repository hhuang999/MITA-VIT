"""v5 analysis (decisions B34-B40): figure 2 and 3 data, Table 1, the predictions P1-P6, appendix A6/A7, sanity.

  python scripts/analyze_v5.py <run prefix> <output folder>
  e.g. python scripts/analyze_v5.py experiments/2026-11-06 experiments/2026-11-06_v5_analysis

Runs are found by name: <prefix>_read_ksweep_<model>, <prefix>_diagnose_{probe,m1}_<model>,
<prefix>_correct_{ablation,order}_{imagenet_s,clickme}_<model>, <prefix>_correct_sanity_<model>,
<prefix>_correct_register_<dataset>_deit, <prefix>_correct_mita_<dataset>_mita. Missing runs are skipped.
Earlier results: S1/S2/S3 (2026-10-31_s_*), E3 (2026-10-29_v3_e3_identity.csv), T7 (2026-11-03_ab_t7_*).

Statistics (B10): one value per image, paired bootstrap (1,000 resamples, seed 0), 95% percentile intervals.
Verdicts (B22 rule, bound per metric): 'comparable' if the interval contains 0 or |mean| < bound; otherwise
'higher' / 'lower'. Bounds: SRG 0.05, enrichment 0.02, ClickMe image-specific Spearman 0.02.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mita_probe.stats import interval, ratio_interval, resample_indices, trapezoid  # noqa: E402

MODELS = ['deit', 'deit_b', 'dino']
SUPERVISED = ['deit', 'deit_b']
METRICS = {'srg': 0.05, 'enrichment': 0.02, 'clickme': 0.02}
KS = [4, 8, 16, 32, 64, 128]
READINGS = ['R0', 'C1', 'C2', 'C3', 'R_all']
CONTROLS = {'C2': 'blur', 'C3': 'no_high', 'C1': 'centre_sub'}
REFERENCES = ['random', 'centre', 'mean_map', 'gae']


def verdict(mean, low, high, bound):
    if (low <= 0 <= high) or abs(mean) < bound:
        return 'comparable'
    return 'higher' if low > 0 else 'lower'


def summary(values, **labels):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    mean, low, high = interval(values, resample_indices(len(values)))
    return dict(labels, n=len(values), mean=mean, low=low, high=high)


def paired(table, a, b, metric, **labels):
    pair = table[[a, b]].dropna()
    row = summary(pair[a] - pair[b], a=a, b=b, metric=metric, **labels)
    row['verdict'] = verdict(row['mean'], row['low'], row['high'], METRICS[metric])
    return row


def find(prefix, name):
    path = Path(f'{prefix}_{name}')
    return path if path.exists() else None


def read(path, name):
    for candidate in (path / name, path / f'{name}.gz', path / 'raw' / f'{name}.gz'):
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


# --- RQ1: k sweep (figure 2a, P1) ------------------------------------------------------------------------

def ksweep(prefix, out):
    rows, tests = [], []
    for model in MODELS:
        run = find(prefix, f'read_ksweep_{model}')
        if run is None:
            continue
        predictions, settings = read(run, 'predictions.csv'), pd.read_csv(run / 'settings.csv')
        table = predictions.merge(settings[['setting_id', 'name', 'k', 'seed']], on='setting_id')
        reading = table[table.name == 'L3_spatial'].pivot_table(index='index', columns='k', values='agree')
        random = table[table.name == 'rand'].pivot_table(index='index', columns='k', values='agree')  # seed mean
        for k in reading.columns:
            rows.append(summary(reading[k], model=model, sets='reading', k=k))
            rows.append(summary(random[k], model=model, sets='random', k=k))
            diff = summary(reading[k] - random[k], model=model, test='reading - random', k=k)
            tests.append(diff | dict(holds=diff['low'] > 0))
        ks = sorted(reading.columns)
        for small, large in zip(ks, ks[1:]):
            diff = summary(reading[large] - reading[small], model=model, test=f'k{large} - k{small}', k=large)
            tests.append(diff | dict(holds=diff['low'] > 0))
    for model in MODELS:
        off = ROOT / f'experiments/2026-11-03_ab_t7_{model}/attention_off.csv'
        if off.exists():
            table = pd.read_csv(off)
            rows.append(summary(table[table.layers == 'all'].agree, model=model, sets='no attention', k=0))
    pd.DataFrame(rows).to_csv(out / 'fig2a_ksweep.csv', index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(out / 'p1_tests.csv', index=False)
    return tests


# --- RQ2: diagnose (figure 3, P2) ------------------------------------------------------------------------

def diagnose(prefix, out):
    rows = []
    for model in MODELS:
        m1 = find(prefix, f'diagnose_m1_{model}')
        if m1 is not None:
            table = read(m1, 'm1.csv')
            for (variant, layer), part in table.groupby(['variant', 'layer']):
                draws = resample_indices(len(part))
                share, low, high = ratio_interval(part.residual.to_numpy(), part.total.to_numpy(), draws)
                rows.append(dict(model=model, panel='M1', variant=variant, layer=layer, n=len(part),
                                 mean=1 - share, low=1 - high, high=1 - low))
        probe = find(prefix, f'diagnose_probe_{model}')
        if probe is not None:
            table = pd.read_csv(probe / 'identity_images.csv')
            for layer, part in table.groupby('layer'):
                rows.append(summary(part.position_accuracy, model=model, panel='M2', variant='new probe', layer=layer))
    e3 = ROOT / 'experiments/2026-10-29_v3_e3_identity.csv'
    if e3.exists():
        for _, row in pd.read_csv(e3).iterrows():
            rows.append(dict(model=row['model'], panel='M2', variant='E3', layer=row['layer'], n=100,
                             mean=row['position_accuracy'], low=np.nan, high=np.nan))
    s3 = pd.read_csv(ROOT / 'experiments/2026-10-31_s_analysis/s3_layers.csv')
    s3 = s3[(s3.family == 'attention') & (s3.measure == 'high_share')]
    rows += [dict(model=r.model, panel='M3', variant='R0 (S3)', layer=r.layer, n=r.n, mean=r['mean'], low=r.low,
                  high=r.high) for _, r in s3.iterrows()]
    s2 = pd.read_csv(ROOT / 'experiments/2026-10-31_s_analysis/s2_layers.csv')
    s2 = s2[(s2.family == 'attention') & (s2.measure == 'clickme_specific_pixel')]
    rows += [dict(model=r.model, panel='quality', variant='R0 (S2)', layer=r.layer, n=r.n, mean=r['mean'],
                  low=r.low, high=r.high) for _, r in s2.iterrows()]
    frame = pd.DataFrame(rows).sort_values(['panel', 'model', 'variant', 'layer'])
    frame.to_csv(out / 'fig3_diagnose.csv', index=False)
    return frame


def p2(frame):
    out = []
    for model in MODELS:
        m2 = frame[(frame.panel == 'M2') & (frame.model == model) & (frame.variant == 'new probe')]
        if len(m2) == 12:
            rho = pd.Series(m2['mean'].to_numpy()).rank().corr(pd.Series(m2.layer.to_numpy()).rank())
            last, first = m2[m2.layer == 12].iloc[0], m2[m2.layer == 1].iloc[0]
            out.append(dict(model=model, test='M2 deepens: accuracy L12 < L1 and Spearman(layer, accuracy) <= -0.8',
                            value=rho, holds=bool(last['high'] < first['low'] and rho <= -0.8)))
        m3 = frame[(frame.panel == 'M3') & (frame.model == model)]
        if len(m3):
            if model == 'dino':
                out.append(dict(model=model, test='M3: DINO high-norm share upper bound < 0.01 at every layer',
                                value=float(m3.high.max()), holds=bool(m3.high.max() < .01)))
            else:
                peak = int(m3.loc[m3['mean'].idxmax(), 'layer'])
                out.append(dict(model=model, test='M3: largest high-norm share in layers 6-12', value=peak,
                                holds=6 <= peak <= 12))
    return pd.DataFrame(out)


# --- RQ3: Table 1, P3-P6, A6, A7 -------------------------------------------------------------------------

def per_image(run_imagenet, run_clickme):
    """{metric: DataFrame index x method} of per-image values."""
    out = {}
    if run_imagenet is not None:
        plaus = read(run_imagenet, 'plausibility.csv')
        out['enrichment'] = plaus.pivot_table(index='index', columns='method', values='enrichment')
        faith = read(run_imagenet, 'faithfulness.csv')
        if faith is not None and len(faith):
            budgets = sorted(faith.budget.unique())
            x = np.array([0] + budgets) / 196
            table = faith.pivot_table(index=['method', 'index'], columns=['condition', 'budget'], values='agree')
            start = np.ones((len(table), 1))
            lif, mif = (trapezoid(x, np.hstack([start, table[c][budgets].to_numpy()]))
                        for c in ('drop_bottom', 'drop_top'))
            out['srg'] = pd.Series(lif - mif, index=table.index).unstack('method')
    if run_clickme is not None:
        aligned = read(run_clickme, 'alignment.csv')
        aligned['specific'] = aligned.spearman_pixel - aligned.spearman_pixel_shuffled
        out['clickme'] = aligned.pivot_table(index='index', columns='method', values='specific')
    return out


def correct(prefix, out):
    table1, tests, recovered, order_rows, repro = [], [], [], [], []
    values = {}
    for model in MODELS:
        data = per_image(find(prefix, f'correct_ablation_imagenet_s_{model}'),
                         find(prefix, f'correct_ablation_clickme_{model}'))
        values[model] = data
        for metric, table in data.items():
            for method in table.columns:
                table1.append(summary(table[method], model=model, metric=metric, method=method))
            for reading in READINGS[1:]:
                if reading in table:
                    tests.append(paired(table, reading, 'R0', metric, model=model, test='correction - R0'))
            for reading, control in CONTROLS.items():
                if {reading, control} <= set(table):
                    tests.append(paired(table, reading, control, metric, model=model, test='correction - control'))
                    tests.append(paired(table, control, 'R0', metric, model=model, test='control - R0'))
            for name in ('C3',):
                if model == 'dino' and name in table:
                    tests.append(paired(table, name, 'R0', metric, model=model, test='P4a'))
            if {'C2_early', 'R0_early'} <= set(table):
                tests.append(paired(table, 'C2_early', 'R0_early', metric, model=model, test='P4b'))
            # A6: share of the last-layer drop each correction recovers (descriptive)
            layers = [f'R0_l{layer}' for layer in (8, 9, 10, 11)]
            if set(layers + ['R0_l12']) <= set(table) and model in SUPERVISED:
                best = max(layers, key=lambda name: table[name].mean())
                for reading in READINGS[1:]:
                    name = f'{reading}_l12'
                    if name not in table:
                        continue
                    pair = table[[best, 'R0_l12', name]].dropna()
                    gain, drop = (pair[name] - pair['R0_l12']).to_numpy(), (pair[best] - pair['R0_l12']).to_numpy()
                    share, low, high = ratio_interval(gain, drop, resample_indices(len(pair)))
                    recovered.append(dict(model=model, metric=metric, reading=reading, best_layer=best,
                                          drop=float(drop.mean()), recovered=share, low=low, high=high))
        order = per_image(find(prefix, f'correct_order_imagenet_s_{model}'), find(prefix, f'correct_order_clickme_{model}'))
        for metric, table in order.items():
            for name in ('R_all', 'R_all_alt'):
                if name in table:
                    order_rows.append(paired(table, name, 'R0', metric, model=model, test='order'))
            if {'R_all_alt', 'R_all'} <= set(table):
                order_rows.append(paired(table, 'R_all_alt', 'R_all', metric, model=model, test='order'))
        repro += reproduction(model, data)
    table1 = pd.DataFrame(table1)
    tests = pd.DataFrame(tests)
    table1.to_csv(out / 'table1_long.csv', index=False)
    tests.to_csv(out / 'correct_tests.csv', index=False)
    pd.DataFrame(recovered).to_csv(out / 'a6_last_layer.csv', index=False)
    pd.DataFrame(order_rows).to_csv(out / 'a7_order.csv', index=False)
    pd.DataFrame(repro).to_csv(out / 'reproduction.csv', index=False)
    variants = variant_rows(prefix, values)
    pd.DataFrame(variants).to_csv(out / 'variants.csv', index=False)
    return table1, tests, pd.DataFrame(variants)


def reproduction(model, data):
    """New R0 against S1's attention_mass on the same images (|difference of means| < 0.01, B22 (7) style)."""
    rows = []
    s1 = ROOT / f'experiments/2026-10-31_s_e4_imagenet_s_{model}'
    if 'enrichment' in data and s1.exists():
        old = pd.read_csv(s1 / 'plausibility.csv').query("method == 'attention_mass'").set_index('index').enrichment
        new = data['enrichment']['R0']
        common = new.index.intersection(old.index)
        rows.append(dict(model=model, metric='enrichment', n=len(common), new=new[common].mean(), old=old[common].mean()))
    if 'srg' in data and s1.exists():
        old = pd.read_csv(s1 / 'per_image_faithfulness.csv.gz').query("method == 'attention_mass'").set_index('index').srg_agree
        new = data['srg']['R0']
        common = new.index.intersection(old.index)
        rows.append(dict(model=model, metric='srg', n=len(common), new=new[common].mean(), old=old[common].mean()))
    s2 = ROOT / f'experiments/2026-10-31_s_e7_clickme_{model}/raw/alignment.csv.gz'
    if 'clickme' in data and s2.exists():
        old = pd.read_csv(s2).query("method == 'attention_mass'")
        old = (old.spearman_pixel - old.spearman_pixel_shuffled).groupby(old['index']).mean()
        new = data['clickme']['R0']
        common = new.index.intersection(old.index)
        rows.append(dict(model=model, metric='clickme', n=len(common), new=new[common].mean(), old=old[common].mean()))
    for row in rows:
        row['difference'] = row['new'] - row['old']
        row['holds'] = abs(row['difference']) < 0.01
    return rows


def variant_rows(prefix, values):
    """'+register' (DeiT-S) and 'top-k training' (MiTA-DeiT-S) on the Table 1 protocol, paired with DeiT-S R0."""
    out = []
    for variant, model in (('register', 'deit'), ('mita', 'mita')):
        data = per_image(find(prefix, f'correct_{variant}_imagenet_s_{model}'),
                         find(prefix, f'correct_{variant}_clickme_{model}'))
        for metric, table in data.items():
            if metric == 'srg':
                continue
            out.append(summary(table['R0'], variant=variant, metric=metric, test='value'))
            base = values.get('deit', {}).get(metric)
            if base is not None:
                joined = pd.concat([table['R0'].rename('variant'), base['R0'].rename('R0')], axis=1)
                row = paired(joined, 'variant', 'R0', metric, variant=variant, test='variant - DeiT-S R0')
                row['holds_p6'] = row['verdict'] != 'higher'
                out.append(row)
    return out


def sanity(prefix, out):
    rows = []
    for model in MODELS:
        run = find(prefix, f'correct_sanity_{model}')
        if run is None:
            continue
        table = read(run, 'sanity.csv')
        for (level, method), part in table.groupby(['level', 'method'], sort=False):
            rows.append(dict(model=model, level=level, method=method, median=part.spearman.median(),
                             passes=part.spearman.median() < 0.5 if level == 'embed' else None))
    pd.DataFrame(rows).to_csv(out / 'sanity.csv', index=False)
    return pd.DataFrame(rows)


def judge(tests, variants, p1, p2_table, out):
    """Predictions P1-P6 with their verdicts."""
    ledger = []
    if len(p1):
        for model, part in p1.groupby('model'):
            ledger.append(dict(prediction='P1', model=model, holds=bool(part.holds.all()),
                               detail=f'{int(part.holds.sum())}/{len(part)} tests'))
    for _, row in p2_table.iterrows():
        ledger.append(dict(prediction='P2', model=row.model, holds=bool(row.holds), detail=row.test))
    if len(tests):
        p3 = tests[(tests.test == 'correction - R0') & (tests.a == 'R_all') & tests.model.isin(SUPERVISED)
                   & tests.metric.isin(['enrichment', 'clickme'])]
        higher = int((p3.verdict == 'higher').sum())
        case = 'holds' if higher == 4 else ('partial' if higher else 'fails')
        ledger.append(dict(prediction='P3', model='deit, deit_b', holds=case == 'holds',
                           detail=f'{higher}/{len(p3)} cells higher; case: {case}'))
        for test in ('P4a', 'P4b'):
            part = tests[tests.test == test]
            if len(part):
                ledger.append(dict(prediction=test, model=','.join(sorted(part.model.unique())),
                                   holds=bool((part.verdict == 'comparable').all()),
                                   detail=f'{int((part.verdict == "comparable").sum())}/{len(part)} comparable'))
    if len(variants) and 'holds_p6' in variants:
        part = variants.dropna(subset=['holds_p6'])
        ledger.append(dict(prediction='P6', model='deit+register, mita', holds=bool(part.holds_p6.all()),
                           detail=f'{int(part.holds_p6.sum())}/{len(part)} not higher'))
    frame = pd.DataFrame(ledger)
    frame.to_csv(out / 'predictions.csv', index=False)
    return frame


def markdown_table(table1, out):
    if not len(table1):
        return
    rows = []
    order = READINGS + list(CONTROLS.values()) + REFERENCES
    for method in order:
        line = [method]
        for model in MODELS:
            for metric in ('srg', 'enrichment', 'clickme'):
                hit = table1[(table1.model == model) & (table1.metric == metric) & (table1.method == method)]
                line.append('' if hit.empty else f"{hit['mean'].iloc[0]:.3f} [{hit.low.iloc[0]:.3f}, {hit.high.iloc[0]:.3f}]")
        rows.append(line)
    header = ['map'] + [f'{model} {metric}' for model in MODELS for metric in ('SRG', 'enrichment', 'ClickMe')]
    text = '| ' + ' | '.join(header) + ' |\n|' + ' --- |' * len(header) + '\n'
    text += '\n'.join('| ' + ' | '.join(line) + ' |' for line in rows) + '\n'
    (out / 'table1.md').write_text(text)
    wide = table1.pivot_table(index='method', columns=['model', 'metric'], values=['mean', 'low', 'high'])
    wide.to_csv(out / 'table1.csv')


def main():
    prefix, out = sys.argv[1], Path(sys.argv[2])
    prefix = str(ROOT / prefix) if not Path(prefix).is_absolute() else prefix
    out = ROOT / out if not out.is_absolute() else out
    out.mkdir(parents=True, exist_ok=True)
    p1 = ksweep(prefix, out)
    frame = diagnose(prefix, out)
    p2_table = p2(frame)
    table1, tests, variants = correct(prefix, out)
    sanity(prefix, out)
    markdown_table(table1, out)
    ledger = judge(tests, variants, p1, p2_table, out)
    print(ledger.to_string(index=False))
    print('written to', out)


if __name__ == '__main__':
    main()
