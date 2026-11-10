"""Every table and number of the paper from the run outputs (CPU, a few minutes).

  python scripts/make_tables.py [--outputs outputs] [--results results]

Reads outputs/<config name>/[<dataset>/]<model>/ as written by scripts/run.py and writes results/:

| file | result |
| --- | --- |
| k_sweep.csv, k_sweep_tests.csv | execution test, k sweep (figures/execution_k_sweep; prediction P1) |
| diagnose.csv, diagnose_tests.csv | per-layer premises (P2); probe_accuracy.csv, radial.csv, value_norm.csv |
| table1.csv, table1.md, table1_long.csv, correct_tests.csv | main results table, paired differences (P3, P5, P7) |
| last_layer.csv, order.csv, variants.csv | last-layer recovery, the other order of the corrections, "change the model" (P6) |
| sanity.csv, negative_controls.csv, predictions.csv | sanity check, negative controls (P4), every registered prediction with its verdict |
| single_layer.csv, part_reliance.csv, map_similarity.csv | single-layer checks, head vs body patches, top-k map similarity |
| c2_analysis/*.csv, supplementary/*.csv, d2/*.csv | exploratory: why C2 fails, combinations, probability discount |
| numbers.csv | single numbers quoted in the text |

Statistics (docs/preregistration.md): one value per image, paired bootstrap over images (1,000 resamples, seed 0),
95% percentile intervals. Verdict per comparison: 'comparable' if the interval contains 0 or |mean| < the margin
of the metric; otherwise 'higher' / 'lower'. Margins: SRG 0.05, enrichment 0.02, ClickMe 0.02. Missing runs are
skipped.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fastweight_vit.data.partimagenet import ANIMALS, PART, SUPERCATEGORY
from fastweight_vit.stats import interval, ratio_interval, resample_indices, trapezoid

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['deit', 'deit_b', 'dino']
SUPERVISED = ['deit', 'deit_b']
MARGINS = {'srg': 0.05, 'enrichment': 0.02, 'clickme': 0.02}
READINGS = ['R0', 'C1', 'C2', 'C3', 'R_all']
STAND_INS = {'C2': 'stand_in_blur', 'C3': 'stand_in_no_high', 'C1': 'stand_in_centre'}
REFERENCES = ['random', 'centre', 'mean_map', 'gae']
DEEP = [9, 10, 11, 12]
TWO = ['enrichment', 'clickme']


# --- shared -----------------------------------------------------------------------------------------------

class Runs:
    """Run folders under the outputs root: runs.get(config, model, dataset) -> Path or None."""

    def __init__(self, root):
        self.root = Path(root)

    def get(self, config, model, dataset=None):
        path = self.root / config / dataset / model if dataset else self.root / config / model
        return path if path.exists() else None


def read(path, name):
    if path is None:
        return None
    for candidate in (path / name, path / f'{name}.gz'):
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


def verdict(mean, low, high, margin):
    if (low <= 0 <= high) or abs(mean) < margin:
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
    row['verdict'] = verdict(row['mean'], row['low'], row['high'], MARGINS[metric])
    return row


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def per_image(run_imagenet, run_clickme):
    """{metric: DataFrame index x method} of per-image values (SRG, enrichment, ClickMe image-specific)."""
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


def write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    frame.to_csv(path, index=False)
    return frame


# --- execution --------------------------------------------------------------------------------------------

def k_sweep(runs, out):
    rows, tests = [], []
    for model in MODELS:
        run = runs.get('k_sweep', model)
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
        table = read(runs.get('attention_off', model), 'attention_off.csv')
        if table is not None:
            rows.append(summary(table[table.layers.astype(str) == 'all'].agree, model=model, sets='no attention', k=0))
    write(rows, out / 'k_sweep.csv')
    return write(tests, out / 'k_sweep_tests.csv')


def single_layer(runs, out):
    """One layer at a time: restricted to its reading sets, or its attention switched off (single-layer checks)."""
    rows = []
    for model in MODELS:
        table = read(runs.get('single_layer', model), 'predictions.csv')
        if table is not None:
            for layer, part in table.groupby('layer'):
                rows.append(summary(part.agree, model=model, layer=layer, test='restricted to reading sets'))
        off = read(runs.get('attention_off', model), 'attention_off.csv')
        if off is not None:
            for layers, part in off.groupby(off.layers.astype(str)):
                rows.append(summary(part.agree, model=model, layer=0 if layers == 'all' else int(layers),
                                    test='attention off'))
    if rows:
        write(pd.DataFrame(rows).sort_values(['model', 'test', 'layer']), out / 'single_layer.csv')


HEADS = [p for p, (s, name) in enumerate(zip(SUPERCATEGORY, PART)) if s in ANIMALS and name == 'Head']
BODIES = [p for p, (s, name) in enumerate(zip(SUPERCATEGORY, PART)) if s in ANIMALS and name == 'Body']


def part_reliance(runs, out, removed=4):
    """Head - body: log-probability drop of the top-1 class when `removed` random head vs body patches go."""
    rows = []
    for model in MODELS:
        reliance = read(runs.get('part_reliance', model), 'reliance.csv')
        if reliance is None:
            continue
        reliance = reliance.assign(log_drop=np.log(reliance.p_original) - np.log(reliance.p_target.clip(lower=1e-12)))
        drop = reliance.groupby(['removed', 'index', 'condition', 'part'])[['log_drop', 'agree']].mean().reset_index()
        main = drop[(drop.removed == removed) & (drop.condition == 'part')]
        head = main[main.part.isin(HEADS)].set_index('index').log_drop
        body = main[main.part.isin(BODIES)].set_index('index').log_drop
        both = pd.DataFrame(dict(head=head, body=body)).dropna()
        diff = summary(both['head'] - both['body'], model=model, measure=f'log-probability drop, {removed} patches',
                       test='head - body')
        rows.append(diff | dict(head=both['head'].mean(), body=both['body'].mean(),
                                verdict='higher' if diff['low'] > 0 else 'lower' if diff['high'] < 0 else 'uncertain'))
    if rows:
        write(rows, out / 'part_reliance.csv')


def map_similarity(runs, out):
    rows = []
    for model in MODELS:
        table = read(runs.get('map_similarity', model), 'similarity.csv')
        if table is not None:
            rho = table.spearman.to_numpy()
            rows.append(dict(model=model, n=len(rho), median=float(np.median(rho)), q25=float(np.quantile(rho, .25)),
                             q75=float(np.quantile(rho, .75))))
    if rows:
        write(rows, out / 'map_similarity.csv')


# --- diagnose: per-layer premises -------------------------------------------------------------------------

def diagnose(runs, out):
    rows = []
    for model in MODELS:
        m1 = read(runs.get('diagnose_mean_maps', model), 'm1.csv')
        if m1 is not None:
            for (variant, layer), part in m1.groupby(['variant', 'layer']):
                draws = resample_indices(len(part))
                share, low, high = ratio_interval(part.residual.to_numpy(), part.total.to_numpy(), draws)
                rows.append(dict(model=model, panel='M1', variant=variant, layer=layer, n=len(part),
                                 mean=1 - share, low=1 - high, high=1 - low))
        probe = read(runs.get('diagnose_probe', model), 'probe_accuracy_images.csv')
        if probe is not None:
            for layer, part in probe.groupby('layer'):
                rows.append(summary(part.position_accuracy, model=model, panel='M2', variant='probe', layer=layer))
        high = read(runs.get('high_norm', model), 'high_norm.csv')
        if high is not None:
            for layer, part in high.groupby('layer'):
                rows.append(summary(part.high_share.dropna(), model=model, panel='M3', variant='R0', layer=layer))
        aligned = read(runs.get('clickme_layers', model, 'clickme'), 'alignment.csv')
        if aligned is not None:
            aligned = aligned.assign(specific=aligned.spearman_pixel - aligned.spearman_pixel_shuffled)
            for name, part in aligned.groupby('method'):
                rows.append(summary(part.specific.dropna(), model=model, panel='quality', variant='R0',
                                    layer=int(name.removeprefix('R0_l'))))
    if not rows:
        return pd.DataFrame()
    frame = write(pd.DataFrame(rows).sort_values(['panel', 'model', 'variant', 'layer']), out / 'diagnose.csv')
    tests = []
    for model in MODELS:
        m2 = frame[(frame.panel == 'M2') & (frame.model == model)]
        if len(m2) == 12:
            rho = pd.Series(m2['mean'].to_numpy()).rank().corr(pd.Series(m2.layer.to_numpy()).rank())
            last, first = m2[m2.layer == 12].iloc[0], m2[m2.layer == 1].iloc[0]
            tests.append(dict(model=model, test='M2 deepens: accuracy L12 < L1 and Spearman(layer, accuracy) <= -0.8',
                              value=rho, holds=bool(last['high'] < first['low'] and rho <= -0.8)))
        m3 = frame[(frame.panel == 'M3') & (frame.model == model)]
        if len(m3):
            if model == 'dino':
                tests.append(dict(model=model, test='M3: DINO high-norm share upper bound < 0.01 at every layer',
                                  value=float(m3.high.max()), holds=bool(m3.high.max() < .01)))
            else:
                peak = int(m3.loc[m3['mean'].idxmax(), 'layer'])
                tests.append(dict(model=model, test='M3: largest high-norm share in layers 6-12', value=peak,
                                  holds=6 <= peak <= 12))
    write(tests, out / 'diagnose_tests.csv')
    return pd.DataFrame(tests)


def diagnose_extras(runs, out):
    """Probe accuracy per layer, ring-explained share of the mean maps, value-norm ratio of high-norm tokens."""
    for config, name, target in (('diagnose_probe', 'probe_accuracy.csv', 'probe_accuracy.csv'),
                                 ('diagnose_mean_maps', 'radial.csv', 'radial.csv')):
        tables = [read(runs.get(config, model), name) for model in MODELS]
        tables = [table for table in tables if table is not None]
        if tables:
            write(pd.concat(tables), out / target)
    rows = []
    for config in ('value_norm', 'high_norm'):
        for model in MODELS:
            table = read(runs.get(config, model), 'high_norm.csv')
            if table is None or 'value_ratio_sum' not in table:
                continue
            by_layer = table.groupby('layer')[['value_ratio_sum', 'value_ratio_n']].sum()
            ratio = by_layer.value_ratio_sum / by_layer.value_ratio_n.replace(0, np.nan)
            rows += [dict(config=config, model=model, layer=layer, value_ratio=value) for layer, value in ratio.items()]
            rows.append(dict(config=config, model=model, layer='9-12 mean', value_ratio=float(ratio.loc[DEEP].mean())))
    if rows:
        write(rows, out / 'value_norm.csv')


# --- correct: main results table --------------------------------------------------------------------------

def p4_rows(model, metric, table):
    """Negative controls: DINO, C3 - R0 on all three metrics (P4a); C2 - R0 at layers 1-4 on enrichment and ClickMe
    (P4b)."""
    rows = []
    if model == 'dino' and {'C3', 'R0'} <= set(table):
        rows.append(paired(table, 'C3', 'R0', metric, model=model, test='P4a'))
    if metric != 'srg' and {'C2_early', 'R0_early'} <= set(table):
        rows.append(paired(table, 'C2_early', 'R0_early', metric, model=model, test='P4b'))
    return rows


def correct(runs, out):
    table1, tests, recovered, order_rows, negative = [], [], [], [], []
    values = {}
    for model in MODELS:
        data = per_image(runs.get('table1', model, 'imagenet_s'), runs.get('table1', model, 'clickme'))
        values[model] = data
        for metric, table in data.items():
            for method in table.columns:
                table1.append(summary(table[method], model=model, metric=metric, method=method))
            for reading in READINGS[1:]:
                if reading in table:
                    tests.append(paired(table, reading, 'R0', metric, model=model, test='correction - R0',
                                        primary=reading == 'R_all'))
            for reading, stand_in in STAND_INS.items():
                if {reading, stand_in} <= set(table):
                    tests.append(paired(table, reading, stand_in, metric, model=model, test='correction - stand-in',
                                        primary=True))
                    tests.append(paired(table, stand_in, 'R0', metric, model=model, test='stand-in - R0'))
            controls = p4_rows(model, metric, table)
            tests += controls
            negative += controls
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
        order = per_image(runs.get('order', model, 'imagenet_s'), runs.get('order', model, 'clickme'))
        for metric, table in order.items():
            for name in ('R0', 'R_all', 'R_all_alt'):
                if name in table:
                    order_rows.append(summary(table[name], model=model, metric=metric, method=name, test='value'))
            for name in ('R_all', 'R_all_alt'):
                if name in table:
                    order_rows.append(paired(table, name, 'R0', metric, model=model, test='order'))
            if {'R_all_alt', 'R_all'} <= set(table):
                order_rows.append(paired(table, 'R_all_alt', 'R_all', metric, model=model, test='order'))
    table1, tests = pd.DataFrame(table1), pd.DataFrame(tests)
    write(table1, out / 'table1_long.csv')
    write(tests, out / 'correct_tests.csv')
    write(recovered, out / 'last_layer.csv')
    write(order_rows, out / 'order.csv')
    write(negative, out / 'negative_controls.csv')
    variants = variant_rows(runs, values)
    write(variants, out / 'variants.csv')
    return table1, tests, pd.DataFrame(variants), values


def variant_rows(runs, values):
    """'+register' (DeiT-S) and 'top-k training' (MiTA-DeiT-S) on the main-table protocol, paired with DeiT-S R0."""
    out = []
    for variant, config, model in (('register', 'table1_register', 'deit'), ('mita', 'table1_mita', 'mita')):
        data = per_image(runs.get(config, model, 'imagenet_s'), runs.get(config, model, 'clickme'))
        for metric, table in data.items():
            out.append(summary(table['R0'], variant=variant, metric=metric, test='value'))
            base = values.get('deit', {}).get(metric)
            if base is not None:
                joined = pd.concat([table['R0'].rename('variant'), base['R0'].rename('R0')], axis=1)
                row = paired(joined, 'variant', 'R0', metric, variant=variant, test='variant - DeiT-S R0')
                row['holds_p6'] = row['verdict'] != 'higher' if metric != 'srg' else None
                out.append(row)
    return out


def sanity(runs, out):
    rows = []
    for model in MODELS:
        table = read(runs.get('sanity', model), 'sanity.csv')
        if table is None:
            continue
        for (level, method), part in table.groupby(['level', 'method'], sort=False):
            rows.append(dict(model=model, level=level, method=method, median=part.spearman.median(),
                             passes=part.spearman.median() < 0.5 if level == 'embed' else None))
    if rows:
        write(rows, out / 'sanity.csv')


def judge(tests, variants, k_tests, diagnose_tests, out):
    """Registered predictions P1-P7 with their verdicts (docs/preregistration.md)."""
    ledger = []
    if len(k_tests):
        for model, part in k_tests.groupby('model'):
            ledger.append(dict(prediction='P1', model=model, holds=bool(part.holds.all()),
                               detail=f'{int(part.holds.sum())}/{len(part)} tests'))
    for _, row in diagnose_tests.iterrows():
        ledger.append(dict(prediction='P2', model=row.model, holds=bool(row.holds), detail=row.test))
    if len(tests):
        p3 = tests[(tests.test == 'correction - R0') & (tests.a == 'R_all') & tests.model.isin(SUPERVISED)
                   & tests.metric.isin(TWO)]
        higher = int((p3.verdict == 'higher').sum())
        case = 'holds' if higher == 4 else ('partial' if higher else 'fails')
        ledger.append(dict(prediction='P3', model='deit, deit_b', holds=case == 'holds',
                           detail=f'{higher}/{len(p3)} cells higher; case: {case}'))
        primary = tests[(tests.test == 'correction - stand-in')]
        for (model, a), part in primary.groupby(['model', 'a']):
            ledger.append(dict(prediction='P7 (primary, no direction)', model=model, holds=None,
                               detail=f'{a} vs stand-in: ' + ', '.join(f'{r.metric} {r.verdict}' for r in part.itertuples())))
        for test in ('P4a', 'P4b'):
            part = tests[tests.test == test]
            if len(part):
                ledger.append(dict(prediction=test, model=','.join(sorted(part.model.unique())),
                                   holds=bool((part.verdict == 'comparable').all()),
                                   detail=f'{int((part.verdict == "comparable").sum())}/{len(part)} comparable'))
    if len(variants) and 'holds_p6' in variants:
        part = variants.dropna(subset=['holds_p6'])
        part = part.assign(holds_p6=part.holds_p6.astype(bool))
        ledger.append(dict(prediction='P6', model='deit+register, mita', holds=bool(part.holds_p6.all()),
                           detail=f'{int(part.holds_p6.sum())}/{len(part)} not higher'))
    return write(ledger, out / 'predictions.csv')


def markdown_table(table1, variants, out):
    if not len(table1):
        return
    rows = []
    for method in READINGS + list(STAND_INS.values()) + REFERENCES:
        line = [method]
        for model in MODELS:
            for metric in ('srg', 'enrichment', 'clickme'):
                hit = table1[(table1.model == model) & (table1.metric == metric) & (table1.method == method)]
                line.append('' if hit.empty else f"{hit['mean'].iloc[0]:.3f} [{hit.low.iloc[0]:.3f}, {hit.high.iloc[0]:.3f}]")
        rows.append(line)
    if len(variants):
        for variant, label in (('register', '+register (DeiT-S)'), ('mita', 'top-k training (MiTA-DeiT-S)')):
            line = [label]
            for model in MODELS:
                for metric in ('srg', 'enrichment', 'clickme'):
                    hit = variants[(variants.variant == variant) & (variants.metric == metric) & (variants.test == 'value')]
                    if model != 'deit' or hit.empty:
                        line.append('n/a (no token removal for MiTA)' if model == 'deit' and variant == 'mita'
                                    and metric == 'srg' else '')
                    else:
                        line.append(f"{hit['mean'].iloc[0]:.3f} [{hit.low.iloc[0]:.3f}, {hit.high.iloc[0]:.3f}]")
            rows.append(line)
    header = ['map'] + [f'{model} {metric}' for model in MODELS for metric in ('SRG', 'enrichment', 'ClickMe')]
    text = '| ' + ' | '.join(header) + ' |\n|' + ' --- |' * len(header) + '\n'
    text += '\n'.join('| ' + ' | '.join(line) + ' |' for line in rows) + '\n'
    (out / 'table1.md').write_text(text)
    wide = table1.pivot_table(index='method', columns=['model', 'metric'], values=['mean', 'low', 'high'])
    wide.to_csv(out / 'table1.csv')


def numbers(runs, out):
    """Single numbers of the text that are not in a table above."""
    rows = []
    for model in MODELS:
        aligned = read(runs.get('table1', model, 'clickme'), 'alignment.csv')
        if aligned is not None:
            centre = aligned[aligned.method == 'centre'].spearman_pixel
            rows.append(dict(model=model, quantity='centre Gaussian vs ClickMe, Spearman (pixel, not image-specific)',
                             value=float(centre.mean()), n=int(centre.count())))
        probe = read(runs.get('diagnose_probe', model), 'probe_accuracy.csv')
        if probe is not None:
            last = probe[probe.layer == 12].iloc[0]
            rows.append(dict(model=model, quantity='layer-12 probe: mean top probability', value=float(last.top_probability),
                             n=None))
        held = runs.get('diagnose_probe', model)
        if held is not None and (held / 'heldout.json').exists():
            info = json.loads((held / 'heldout.json').read_text())
            rows.append(dict(model=model, quantity='held-out candidates skipped as ClickMe near-duplicates',
                             value=len(info['clickme_duplicates']), n=len(info['rows'])))
    if rows:
        write(rows, out / 'numbers.csv')


# --- why C2 fails (exploratory) ---------------------------------------------------------------------------

ERROR_METRICS = ['exact', 'top5', 'distance', 'near3x3', 'far3', 'far_share']
GRID = 14


def closeness():
    rows, cols = np.divmod(np.arange(GRID * GRID), GRID)
    return -np.hypot(rows - (GRID - 1) / 2, cols - (GRID - 1) / 2)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def c2_analysis(runs, out):
    errors, predictions, shifts, bins = [], [], [], []
    near = closeness()
    centre = (np.abs(np.divmod(np.arange(GRID * GRID), GRID)[0] - 6.5) < 3) & \
             (np.abs(np.divmod(np.arange(GRID * GRID), GRID)[1] - 6.5) < 3)
    for model in MODELS:
        run = runs.get('c2_analysis', model)
        if run is None:
            continue
        table = read(run, 'probe_errors.csv')
        for (layer, source), part in table.groupby(['layer', 'source']):
            for metric in ERROR_METRICS:
                errors.append(summary(part[metric], model=model, layer=layer, source=source, metric=metric))
        for layer, part in table.groupby('layer'):
            wide = part.pivot_table(index='row', columns='source', values=ERROR_METRICS)
            for metric in ERROR_METRICS:
                errors.append(summary(wide[metric]['probe'] - wide[metric]['stand_in_blur'], model=model, layer=layer,
                                      source='probe - stand_in_blur', metric=metric))
        per_image = np.load(run / 'mean_prediction.npz')
        means = np.load(runs.get('diagnose_mean_maps', model) / 'means_crop.npz')
        for key in per_image.files:
            layer = int(key.removeprefix('layer'))
            values = per_image[key].astype(np.float64)
            base = means[f'R0_layer{layer}'].astype(np.float64)

            def stats(distribution):
                return dict(max_ratio=distribution.max() * 196,
                            kl_uniform=float((distribution * np.log(distribution * 196)).sum()),
                            centre_mass=float(distribution[centre].sum()),
                            pearson_mean_map=pearson(distribution, base), spearman_mean_map=spearman(distribution, base),
                            pearson_centre=pearson(distribution, near), spearman_centre=spearman(distribution, near))

            point = stats(values.mean(0))
            boot = pd.DataFrame([stats(values[draw].mean(0)) for draw in resample_indices(len(values))])
            for name, value in point.items():
                low, high = np.quantile(boot[name], [.025, .975])
                predictions.append(dict(model=model, layer=layer, quantity=name, n=len(values), mean=value, low=low,
                                        high=high))
        shift = read(run, 'shift.csv')
        for source, part in shift.groupby('source'):
            shifts.append(summary(part.object_r0 - part.object_moved, model=model, source=source,
                                  quantity='object share moved to background'))
            shifts.append(summary(part.object_r0, model=model, source=source, quantity='object share R0'))
            shifts.append(summary(part.object_moved, model=model, source=source, quantity='object share after'))
            for name in ('pearson_mean', 'spearman_mean', 'pearson_centre', 'spearman_centre'):
                shifts.append(summary(part[name], model=model, source=source, quantity=f'{name} of (moved - R0)'))
            for b in range(10):
                bins.append(summary(part[f'bin{b}_moved'] - part[f'bin{b}_r0'], model=model, source=source, bin=b,
                                    r0_share=part[f'bin{b}_r0'].mean()))
    if not errors:
        return
    folder = out / 'c2_analysis'
    error_table = write(errors, folder / 'errors.csv')
    prediction_table = write(predictions, folder / 'mean_prediction.csv')
    shift_table = write(shifts, folder / 'shift.csv')
    write(bins, folder / 'shift_bins.csv')
    write(c2_hypotheses(error_table, prediction_table, shift_table), folder / 'hypotheses.csv')


def c2_hypotheses(error_table, prediction_table, shift_table):
    """Descriptive verdicts registered for this analysis (layers 9-12, all three models)."""
    far = error_table[(error_table.source == 'probe') & (error_table.metric == 'far_share') &
                      error_table.layer.isin(DEEP)]
    above, below = (far.low > .5).all(), (far.high < .5).all()
    out = [dict(hypothesis='probe errors are mostly far (> 3 patches)',
                verdict='supported' if above else 'not supported' if below else 'partly supported',
                detail=f'far share of errors, layers 9-12: {far["mean"].min():.2f}-{far["mean"].max():.2f}')]
    pred = prediction_table[prediction_table.layer.isin(DEEP)].pivot_table(
        index=['model', 'layer'], columns='quantity', values=['mean', 'low'])
    meets = (pred[('mean', 'max_ratio')] >= 2) & (pred[('mean', 'pearson_mean_map')] >= .5) & \
            (pred[('low', 'pearson_mean_map')] > 0)
    by_model = meets.groupby(level='model').all()
    verdict = 'supported' if by_model.all() else 'not supported' if not meets.any() else 'partly supported'
    side = shift_table[(shift_table.source == 'C2') & (shift_table.quantity == 'pearson_mean of (moved - R0)')]
    out.append(dict(hypothesis='the probe carries an image-independent position preference', verdict=verdict,
                    detail=f'{int(meets.sum())}/{len(meets)} model-layer cells meet both conditions; '
                           f'side evidence corr(C2 - R0, mean map): ' +
                           ', '.join(f'{r.model} {r.mean:.2f} [{r.low:.2f}, {r.high:.2f}]' for r in side.itertuples())))
    return out


# --- supplementary analyses (exploratory) -----------------------------------------------------------------

SEEDS = [f'C3shuffle_s{seed}' for seed in range(5)]


def supplementary_data(runs):
    data = {}
    for model in MODELS:
        tables = per_image(runs.get('supplementary_eval', model, 'imagenet_s'),
                           runs.get('supplementary_eval', model, 'clickme'))
        if tables:
            for table in tables.values():
                if set(SEEDS) <= set(table):
                    table['C3shuffle'] = table[SEEDS].mean(axis=1)
            data[model] = tables
    return data


def cells(rows, test):
    frame = pd.DataFrame(rows)
    return frame[(frame.test == test) & frame.model.isin(SUPERVISED) & frame.metric.isin(TWO)]


def combinations(data):
    rows = []
    for model, tables in data.items():
        for metric, table in tables.items():
            names = ['C1C3'] if metric == 'srg' else ['C1C3', 'C2C3', 'C1C2']
            for name in names:
                rows.append(paired(table, name, 'R0', metric, model=model, test='combination - R0'))
            rows.append(paired(table, 'C1C3', 'C3', metric, model=model, test='C1C3 - C3'))
            if metric != 'srg':
                rows.append(paired(table, 'C2C3', 'C3', metric, model=model, test='C2C3 - C3'))
            rows.append(paired(table, 'C1C3', 'C3_centre', metric, model=model, test='C1C3 - (C3 - centre)'))
    part = cells(rows, 'C1C3 - C3')
    branch = 'A' if (part.verdict == 'higher').sum() >= 2 and not (part.verdict == 'lower').any() else 'B'
    return pd.DataFrame(rows), branch


def c3_mechanism(data):
    rows, seeds = [], []
    for model, tables in data.items():
        for metric, table in tables.items():
            if metric == 'srg':
                continue
            for other in ('C3v', 'C3shuffle', 'C3headmean'):
                rows.append(paired(table, 'C3', other, metric, model=model, test=f'C3 - {other}'))
            for name in SEEDS:
                seeds.append(dict(model=model, metric=metric, seed=name[-1], mean=table[name].mean(),
                                  c3_minus=(table['C3'] - table[name]).mean()))
    shuffle, headmean = cells(rows, 'C3 - C3shuffle'), cells(rows, 'C3 - C3headmean')
    expected = all(((part.model == model) & (part.verdict == 'higher')).any() for part in (shuffle, headmean)
                   for model in SUPERVISED) and not ((shuffle.verdict == 'lower').any() or
                                                     (headmean.verdict == 'lower').any())
    head = (headmean.verdict == 'comparable').all() and len(headmean) == 4
    branch = 'expected' if expected else 'head weighting' if head else 'other'
    v = cells(rows, 'C3 - C3v').verdict.value_counts().to_dict()
    return pd.DataFrame(rows), pd.DataFrame(seeds), branch, v


def dose(runs, data):
    layer_rows, dose_rows = [], []
    for model, tables in data.items():
        for metric, table in tables.items():
            if metric == 'srg':
                continue
            for layer in list(range(6, 13)) + ['9-12']:
                suffix = '' if layer == '9-12' else f'_l{layer}'
                for name in ('C3', 'stand_in_no_high'):
                    row = paired(table, f'{name}{suffix}', f'R0{suffix}', metric, model=model, test=f'{name} - R0')
                    layer_rows.append(row | dict(layer=layer))
        if model not in SUPERVISED:
            continue
        for metric, dataset in (('enrichment', 'imagenet_s'), ('clickme', 'clickme')):
            run = runs.get('supplementary_eval', model, dataset)
            if run is None or metric not in tables:
                continue
            doses = pd.read_csv(run / 'dose.csv').drop_duplicates('index').set_index('index').dose
            table = tables[metric].join(doses, how='inner')
            for name in ('R0', 'stand_in_no_high'):
                part = table[['dose', 'C3', name]].dropna()
                gain = (part.C3 - part[name]).to_numpy()
                value = spearman(part.dose.to_numpy(), gain)
                boot = [spearman(part.dose.to_numpy()[d], gain[d]) for d in resample_indices(len(part))]
                low, high = np.quantile(boot, [.025, .975])
                dose_rows.append(dict(model=model, metric=metric, improvement=f'C3 - {name}', n=len(part),
                                      spearman=value, low=low, high=high, mean_dose=part.dose.mean()))
    doses = pd.DataFrame(dose_rows)
    clickme = doses[(doses.metric == 'clickme') & (doses.improvement == 'C3 - R0')] if len(doses) else doses
    branch = 'A' if len(clickme) == 2 and (clickme.low > 0).all() else 'B'
    return pd.DataFrame(layer_rows), doses, branch


def globalisation(runs):
    rows = []
    for model in MODELS:
        table = read(runs.get('supplementary_heldout', model), 'global.csv')
        if table is None:
            continue
        for layer, part in table.groupby('layer'):
            for name in ('rho_exact', 'rho_entropy', 'rho_move', 'rho_exact_ring', 'rho_entropy_ring',
                         'rho_move_ring', 'mean_g'):
                rows.append(summary(part[name], model=model, layer=layer, quantity=name))
            for name in ('exact', 'entropy', 'move'):
                for q in range(1, 6):
                    rows.append(dict(model=model, layer=layer, quantity=f'{name}_q{q}', n=len(part),
                                     mean=part[f'{name}_q{q}'].mean(), low=np.nan, high=np.nan))
    frame = pd.DataFrame(rows)
    meets = {}
    for model in frame.model.unique() if len(frame) else []:
        ok = True
        for layer in (10, 11, 12):
            def get(q):
                return frame[(frame.model == model) & (frame.layer == layer) & (frame.quantity == q)].iloc[0]
            acc, ent = get('rho_exact'), get('rho_entropy')
            acc_ring, ent_ring = get('rho_exact_ring'), get('rho_entropy_ring')
            ok &= bool(acc.high < 0 and ent.low > 0 and acc_ring['mean'] < 0 and ent_ring['mean'] > 0)
        meets[model] = ok
    verdict = 'holds' if sum(meets.values()) >= 2 else 'does not hold'
    return frame, verdict, meets


def mlp_probe(runs, data):
    probe_rows = []
    for model in MODELS:
        table = read(runs.get('supplementary_heldout', model), 'mlp_probe.csv')
        if table is None:
            continue
        for (layer, probe), part in table.groupby(['layer', 'probe']):
            for name in ('exact', 'top5', 'distance'):
                probe_rows.append(summary(part[name], model=model, layer=layer, probe=probe, quantity=name))
    probes = pd.DataFrame(probe_rows)
    gains = {}
    for model in probes.model.unique() if len(probes) else []:
        at = probes[(probes.model == model) & (probes.layer == 12) & (probes.quantity == 'exact')].set_index('probe')
        gains[model] = float(at.loc['mlp', 'mean'] - at.loc['linear', 'mean'])
    rows = []
    for model, tables in data.items():
        for metric, table in tables.items():
            if metric != 'srg':
                for name in ('C2MLP', 'C2conf'):
                    rows.append(paired(table, name, 'R0', metric, model=model, test=f'{name} - R0'))
                rows.append(paired(table, 'C2MLP', 'stand_in_blur_MLP', metric, model=model,
                                   test='C2MLP - stand_in_blur_MLP'))
                rows.append(paired(table, 'C3C2MLP', 'R0', metric, model=model, test='C3C2MLP - R0'))
                rows.append(paired(table, 'C3C2MLP', 'C3', metric, model=model, test='C3C2MLP - C3'))
            elif 'C2MLP' in table:
                rows.append(paired(table, 'C2MLP', 'R0', metric, model=model, test='C2MLP - R0'))
    frame = pd.DataFrame(rows)
    mlp = frame[(frame.test == 'C2MLP - R0') & frame.metric.isin(TWO)]
    lower = int((mlp.verdict == 'lower').sum())
    strong = any(gain > 0.10 for gain in gains.values())
    if not strong:
        branch = 'weak'
    elif (mlp.verdict != 'lower').sum() >= 4:
        branch = 'strong, C2 rescued'
    elif lower > 2:
        branch = 'strong, C2 still fails'
    else:
        branch = 'strong, neither rule (3 lower)'
    conf = frame[(frame.test == 'C2conf - R0') & frame.metric.isin(TWO)]
    conf_rule = int((conf.verdict != 'lower').sum()) >= 4
    return probes, frame, branch, gains, conf_rule


def gae_gap(values):
    """Share of the R0 - GAE gap that C3 closes (supervised models)."""
    rows = []
    for model in SUPERVISED:
        for metric, table in values.get(model, {}).items():
            part = table[['R0', 'C3', 'gae']].dropna()
            ratio, low, high = ratio_interval((part.C3 - part.R0).to_numpy(), (part.gae - part.R0).to_numpy(),
                                              resample_indices(len(part)))
            rows.append(dict(model=model, metric=metric, n=len(part), c3_gain=(part.C3 - part.R0).mean(),
                             gae_gap=(part.gae - part.R0).mean(), share_closed=ratio, low=low, high=high))
    return pd.DataFrame(rows)


def supplementary(runs, values, out):
    folder = out / 'supplementary'
    data = supplementary_data(runs)
    branches = []
    if data:
        frame, branch = combinations(data)
        write(frame, folder / 'combinations.csv')
        branches.append(dict(item='combinations (C1 + C3)', branch=branch))
        frame, seeds, branch, v = c3_mechanism(data)
        write(frame, folder / 'c3_mechanism.csv')
        write(seeds, folder / 'c3_mechanism_seeds.csv')
        branches.append(dict(item='C3 mechanism', branch=branch, detail=f'C3 - C3v verdicts (supervised cells): {v}'))
        layers, doses, branch = dose(runs, data)
        write(layers, folder / 'dose_layers.csv')
        write(doses, folder / 'dose.csv')
        branches.append(dict(item='dose-response', branch=branch))
        probes, frame, branch, gains, conf_rule = mlp_probe(runs, data)
        write(probes, folder / 'mlp_probe.csv')
        write(frame, folder / 'c2_variants.csv')
        branches.append(dict(item='MLP probe', branch=branch, detail=f'MLP - linear at layer 12: {gains}; '
                                                                   f'C2-confident not lower in >= 4 cells: {conf_rule}'))
        weights = [read(runs.get('supplementary_eval', model, 'imagenet_s'), 'head_weights.csv') for model in data]
        weights = [table for table in weights if table is not None]
        if weights:
            write(pd.concat(weights), folder / 'head_weights.csv')
    frame, verdict, meets = globalisation(runs)
    if len(frame):
        write(frame, folder / 'globalisation.csv')
        branches.append(dict(item='globalisation', branch=verdict, detail=f'models meeting the rule at layers 10-12: '
                                                                          f'{meets}'))
    gap = gae_gap(values)
    if len(gap):
        write(gap, folder / 'gae_gap.csv')
    if branches:
        write(branches, folder / 'branches.csv')


# --- discount without relocation (exploratory, registered before running) ---------------------------------

D2_SEEDS = [f'D2_shuffle_s{seed}' for seed in range(5)]


def d2(runs, values, out):
    """D2 - R0, D2 - D2_shuffle (seed mean per image), D2 - C2 on enrichment and ClickMe; branch rule; C3D2 - C3
    (reported only under branch A); R0, C2 and C3 of the same run against the main results table (consistency)."""
    rows, checks = [], []
    for model in MODELS:
        data = per_image(runs.get('d2', model, 'imagenet_s'), runs.get('d2', model, 'clickme'))
        for metric, table in data.items():
            if metric == 'srg' or 'D2' not in table:
                continue
            table['D2_shuffle'] = table[D2_SEEDS].mean(axis=1)
            for name in ('R0', 'C2', 'C3', 'D2', 'D2_shuffle', 'C3D2'):
                rows.append(summary(table[name], model=model, metric=metric, method=name, test='value'))
            for other in ('R0', 'D2_shuffle', 'C2'):
                rows.append(paired(table, 'D2', other, metric, model=model, test=f'D2 - {other}'))
            rows.append(paired(table, 'C3D2', 'C3', metric, model=model, test='C3D2 - C3'))
            reference = values.get(model, {}).get(metric)
            if reference is not None:
                for name in ('R0', 'C2', 'C3'):
                    both = pd.concat([table[name].rename('new'), reference[name].rename('table1')], axis=1).dropna()
                    checks.append(dict(model=model, metric=metric, method=name, n=len(both),
                                       max_abs_difference=float((both.new - both.table1).abs().max())))
    if not rows:
        return
    frame = write(rows, out / 'd2' / 'tests.csv')
    if checks:
        write(checks, out / 'd2' / 'consistency.csv')
    gain = frame[frame.test == 'D2 - R0']
    higher = gain[gain.verdict == 'higher']
    against = frame[frame.test == 'D2 - D2_shuffle'].set_index(['model', 'metric']).verdict
    shuffle_ok = all(against.get((r.model, r.metric)) == 'higher' for r in higher.itertuples())
    branch = 'A' if len(higher) >= 2 and not (gain.verdict == 'lower').any() and shuffle_ok else 'B'
    write([dict(item='D2', branch=branch, cells=len(gain), higher=len(higher),
                lower=int((gain.verdict == 'lower').sum()), shuffle_higher_in_those_cells=shuffle_ok)],
          out / 'd2' / 'branch.csv')



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outputs', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    runs, out = Runs(args.outputs), args.results
    out.mkdir(parents=True, exist_ok=True)
    k_tests = k_sweep(runs, out)
    single_layer(runs, out)
    part_reliance(runs, out)
    map_similarity(runs, out)
    diagnose_tests = diagnose(runs, out)
    diagnose_extras(runs, out)
    table1, tests, variants, values = correct(runs, out)
    sanity(runs, out)
    markdown_table(table1, variants, out)
    numbers(runs, out)
    c2_analysis(runs, out)
    supplementary(runs, values, out)
    d2(runs, values, out)
    ledger = judge(tests, variants, k_tests, diagnose_tests, out)
    print(ledger.to_string(index=False))
    print('written to', out)


if __name__ == '__main__':
    main()
