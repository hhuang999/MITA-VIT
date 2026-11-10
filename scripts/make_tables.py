"""Rebuild the result tables used by the final paper from saved per-image outputs.

CPU only. See docs/reproduce.md for the paper-to-file map and statistical rules.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fastweight_vit.stats import interval, ratio_interval, resample_indices, trapezoid

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['deit', 'deit_b', 'dino']
SUPERVISED = ['deit', 'deit_b']
MARGINS = {'srg': 0.05, 'enrichment': 0.02, 'clickme': 0.02}
READINGS = ['R0', 'C1', 'C2', 'C3', 'R_all']
STAND_INS = {'C2': 'stand_in_blur', 'C3': 'stand_in_no_high', 'C1': 'stand_in_centre'}
DEEP = [9, 10, 11, 12]
TWO = ['enrichment', 'clickme']
ERROR_METRICS = ['exact', 'top5', 'distance', 'near3x3', 'far3', 'far_share']
GRID = 14
SEEDS = [f'C3shuffle_s{seed}' for seed in range(5)]


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
    return frame


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
    table1, tests, recovered, negative = [], [], [], []
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
    table1, tests = pd.DataFrame(table1), pd.DataFrame(tests)
    write(table1, out / 'table1_long.csv')
    write(tests, out / 'correct_tests.csv')
    write(recovered, out / 'last_layer.csv')
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


def closeness():
    rows, cols = np.divmod(np.arange(GRID * GRID), GRID)
    return -np.hypot(rows - (GRID - 1) / 2, cols - (GRID - 1) / 2)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def c2_analysis(runs, out):
    errors, predictions = [], []
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
    if not errors:
        return
    folder = out / 'c2_analysis'
    error_table = write(errors, folder / 'errors.csv')
    prediction_table = write(predictions, folder / 'mean_prediction.csv')


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


def supplementary(runs, out):
    folder = out / 'supplementary'
    data = supplementary_data(runs)
    if data:
        frame, _, _, _ = c3_mechanism(data)
        write(frame, folder / 'c3_mechanism.csv')
        _, doses, _ = dose(runs, data)
        write(doses, folder / 'dose.csv')
        probes, comparisons, _, _, _ = mlp_probe(runs, data)
        write(probes, folder / 'mlp_probe.csv')
        write(comparisons, folder / 'c2_variants.csv')
    frame, _, _ = globalisation(runs)
    if len(frame):
        write(frame, folder / 'globalisation.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--results', type=Path, default=ROOT / 'build/rebuilt/results')
    args = parser.parse_args()
    runs, out = Runs(args.outputs), args.results
    out.mkdir(parents=True, exist_ok=True)
    k_sweep(runs, out)
    diagnose(runs, out)
    diagnose_extras(runs, out)
    correct(runs, out)
    numbers(runs, out)
    c2_analysis(runs, out)
    supplementary(runs, out)
    print('written to', out)


if __name__ == '__main__':
    main()
