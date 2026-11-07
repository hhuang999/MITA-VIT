"""Phase-1 supplementary analyses X1-X6 (decision B46; exploratory, post hoc). Changes no registered verdict.

  python scripts/analyze_supp.py <run prefix> <output folder>
  e.g. python scripts/analyze_supp.py experiments/2026-11-07 experiments/2026-11-07_v5_supp_analysis

Reads <prefix>_supp_eval_{imagenet_s,clickme}_<model> and <prefix>_supp_heldout_<model> (X1-X5) and the v5 formal
correct_ablation runs (X6). Statistics and verdicts as B22 (analyze_v5.paired). Writes x1.csv ... x6.csv,
x2_seeds.csv, x4_layers.csv, x5_probe.csv, x2_head_weights.csv and branches.csv (the branch each item took).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_v5 import find, paired, per_image, summary  # noqa: E402
from mita_probe.stats import ratio_interval, resample_indices  # noqa: E402

MODELS = ['deit', 'deit_b', 'dino']
SUPERVISED = ['deit', 'deit_b']
TWO = ['enrichment', 'clickme']
SEEDS = [f'C3shuffle_s{seed}' for seed in range(5)]


def load(prefix):
    data = {}
    for model in MODELS:
        tables = per_image(find(prefix, f'supp_eval_imagenet_s_{model}'), find(prefix, f'supp_eval_clickme_{model}'))
        if 'clickme' in tables or 'enrichment' in tables:
            for metric, table in tables.items():
                if set(SEEDS) <= set(table):
                    table['C3shuffle'] = table[SEEDS].mean(1)
            data[model] = tables
    return data


def cells(rows, test):
    frame = pd.DataFrame(rows)
    return frame[(frame.test == test) & frame.model.isin(SUPERVISED) & frame.metric.isin(TWO)]


def x1(data):
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
    branch = 'X1-A' if (part.verdict == 'higher').sum() >= 2 and not (part.verdict == 'lower').any() else 'X1-B'
    return pd.DataFrame(rows), branch


def x2(data):
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
    frame = pd.DataFrame(rows)
    shuffle, headmean = cells(rows, 'C3 - C3shuffle'), cells(rows, 'C3 - C3headmean')
    expected = all(((part.model == model) & (part.verdict == 'higher')).any() for part in (shuffle, headmean)
                   for model in SUPERVISED) and not ((shuffle.verdict == 'lower').any() or
                                                     (headmean.verdict == 'lower').any())
    head = (headmean.verdict == 'comparable').all() and len(headmean) == 4
    branch = 'expected' if expected else 'head weighting' if head else 'other'
    v = cells(rows, 'C3 - C3v').verdict.value_counts().to_dict()
    return frame, pd.DataFrame(seeds), branch, v


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def x3(prefix, data):
    layer_rows, dose_rows = [], []
    for model, tables in data.items():
        for metric, table in tables.items():
            if metric == 'srg':
                continue
            for layer in list(range(6, 13)) + ['9-12']:
                suffix = '' if layer == '9-12' else f'_l{layer}'
                for name in ('C3', 'no_high'):
                    row = paired(table, f'{name}{suffix}', f'R0{suffix}', metric, model=model, test=f'{name} - R0')
                    layer_rows.append(row | dict(layer=layer))
        if model not in SUPERVISED:
            continue
        for metric, dataset in (('enrichment', 'imagenet_s'), ('clickme', 'clickme')):
            run = find(prefix, f'supp_eval_{dataset}_{model}')
            if run is None or metric not in tables:
                continue
            dose = pd.read_csv(run / 'dose.csv').drop_duplicates('index').set_index('index').dose
            table = tables[metric].join(dose, how='inner')
            for name in ('R0', 'no_high'):
                part = table[['dose', 'C3', name]].dropna()
                gain = (part.C3 - part[name]).to_numpy()
                value = spearman(part.dose.to_numpy(), gain)
                draws = resample_indices(len(part))
                boot = [spearman(part.dose.to_numpy()[d], gain[d]) for d in draws]
                low, high = np.quantile(boot, [.025, .975])
                dose_rows.append(dict(model=model, metric=metric, improvement=f'C3 - {name}', n=len(part),
                                      spearman=value, low=low, high=high, mean_dose=part.dose.mean()))
    doses = pd.DataFrame(dose_rows)
    clickme = doses[(doses.metric == 'clickme') & (doses.improvement == 'C3 - R0')]
    branch = 'X3-A' if len(clickme) == 2 and (clickme.low > 0).all() else 'X3-B'
    return pd.DataFrame(layer_rows), doses, branch


def x4(prefix):
    rows = []
    for model in MODELS:
        run = find(prefix, f'supp_heldout_{model}')
        if run is None:
            continue
        table = pd.read_csv(run / 'global.csv')
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
            get = lambda q: frame[(frame.model == model) & (frame.layer == layer) & (frame.quantity == q)].iloc[0]
            acc, ent = get('rho_exact'), get('rho_entropy')
            acc_ring, ent_ring = get('rho_exact_ring'), get('rho_entropy_ring')
            ok &= bool(acc.high < 0 and ent.low > 0 and acc_ring['mean'] < 0 and ent_ring['mean'] > 0)
        meets[model] = ok
    verdict = 'H-global holds' if sum(meets.values()) >= 2 else 'H-global does not hold'
    return frame, verdict, meets


def x5(prefix, data):
    probe_rows = []
    for model in MODELS:
        run = find(prefix, f'supp_heldout_{model}')
        if run is None:
            continue
        table = pd.read_csv(run / 'mlp_probe.csv')
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
                rows.append(paired(table, 'C2MLP', 'blurMLP', metric, model=model, test='C2MLP - blurMLP'))
                rows.append(paired(table, 'C3C2MLP', 'R0', metric, model=model, test='C3C2MLP - R0'))
                rows.append(paired(table, 'C3C2MLP', 'C3', metric, model=model, test='C3C2MLP - C3'))
            elif 'C2MLP' in table:
                rows.append(paired(table, 'C2MLP', 'R0', metric, model=model, test='C2MLP - R0'))
    frame = pd.DataFrame(rows)
    mlp = frame[(frame.test == 'C2MLP - R0') & frame.metric.isin(TWO)]
    lower = int((mlp.verdict == 'lower').sum())
    strong = any(gain > 0.10 for gain in gains.values())
    if not strong:
        branch = 'M-weak'
    elif (mlp.verdict != 'lower').sum() >= 4:
        branch = 'version C'
    elif lower > 2:
        branch = 'M-strong, C2 still fails'
    else:
        branch = 'M-strong, neither rule (3 lower)'
    conf = frame[(frame.test == 'C2conf - R0') & frame.metric.isin(TWO)]
    conf_rule = int((conf.verdict != 'lower').sum()) >= 4
    return probes, frame, branch, gains, conf_rule


def x6():
    rows = []
    for model in SUPERVISED:
        tables = per_image(find(str(ROOT / 'experiments/2026-11-05'), f'correct_ablation_imagenet_s_{model}'),
                           find(str(ROOT / 'experiments/2026-11-05'), f'correct_ablation_clickme_{model}'))
        for metric, table in tables.items():
            part = table[['R0', 'C3', 'gae']].dropna()
            ratio, low, high = ratio_interval((part.C3 - part.R0).to_numpy(), (part.gae - part.R0).to_numpy(),
                                              resample_indices(len(part)))
            rows.append(dict(model=model, metric=metric, n=len(part), c3_gain=(part.C3 - part.R0).mean(),
                             gae_gap=(part.gae - part.R0).mean(), share_closed=ratio, low=low, high=high))
    return pd.DataFrame(rows)


def main():
    prefix = str(ROOT / sys.argv[1])
    out = ROOT / sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    data = load(prefix)
    branches = []
    if data:
        frame, branch = x1(data)
        frame.to_csv(out / 'x1.csv', index=False)
        branches.append(dict(item='X1', branch=branch))
        frame, seeds, branch, v = x2(data)
        frame.to_csv(out / 'x2.csv', index=False)
        seeds.to_csv(out / 'x2_seeds.csv', index=False)
        branches.append(dict(item='X2', branch=branch, detail=f'C3 - C3v verdicts (supervised cells): {v}'))
        layers, doses, branch = x3(prefix, data)
        layers.to_csv(out / 'x3_layers.csv', index=False)
        doses.to_csv(out / 'x3_dose.csv', index=False)
        branches.append(dict(item='X3', branch=branch))
        probes, frame, branch, gains, conf_rule = x5(prefix, data)
        probes.to_csv(out / 'x5_probe.csv', index=False)
        frame.to_csv(out / 'x5.csv', index=False)
        branches.append(dict(item='X5', branch=branch, detail=f'MLP - linear at layer 12: {gains}; '
                                                              f'C2-confident not lower in >= 4 cells: {conf_rule}'))
        weights = [pd.read_csv(find(prefix, f'supp_eval_imagenet_s_{m}') / 'head_weights.csv') for m in data
                   if find(prefix, f'supp_eval_imagenet_s_{m}') is not None]
        if weights:
            pd.concat(weights).to_csv(out / 'x2_head_weights.csv', index=False)
    frame, verdict, meets = x4(prefix)
    frame.to_csv(out / 'x4_layers.csv', index=False)
    branches.append(dict(item='X4', branch=verdict, detail=f'models meeting the rule at layers 10-12: {meets}'))
    x6().to_csv(out / 'x6.csv', index=False)
    pd.DataFrame(branches).to_csv(out / 'branches.csv', index=False)
    print(pd.DataFrame(branches).to_string(index=False))


if __name__ == '__main__':
    main()
