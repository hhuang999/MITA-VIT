"""Analyse the attention-budget runs (decisions B27-B31) from their result files only (CPU).

  python scripts/analyze_budget.py pilot  --prefix experiments/2026-11-03_ab_   # T0, T1, T2, T7 -> pilot tables
  python scripts/analyze_budget.py formal --prefix experiments/<date>_ab_      # T3-T6 -> judgements

Paired comparisons recompute the deep (layers 9-12) enrichment from each run's raw/batch_*.npz, excluding the
union of both models' high-norm patches in those layers (B28). Output: experiments/<prefix>analysis_<stage>/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mita_probe import plausibility
from mita_probe.budget import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('deit', 'mita', 'deit_t', 'mita_t', 'deit3_s', 'deit_b', 'dino')
SUPERVISED = ('deit', 'mita', 'deit_t', 'mita_t', 'deit3_s', 'deit_b')
PAIRS = (('mita', 'deit'), ('mita_t', 'deit_t'))
DEEP = (9, 10, 11, 12)
Z_DECISION = 2.807  # two-sided 99.5% (Bonferroni over 10 judged comparisons)
MARGIN = 0.03


def load_raw(run):
    parts = [np.load(path) for path in sorted((run / 'raw').glob('batch_*.npz'))]
    index = np.concatenate([p['index'] for p in parts])
    high = np.unpackbits(np.concatenate([p['high'] for p in parts]), axis=-1)[..., :196].astype(bool)
    return dict(index=index, scores=np.concatenate([p['deep_scores'] for p in parts]), high=high,
                foreground=np.concatenate([p['foreground'] for p in parts]),
                valid=np.concatenate([p['valid'] for p in parts]))


def deep_exclusion(raw):
    return raw['high'][:, [layer - 1 for layer in DEEP]].any(1)


def enrichment(raw, exclude, chunk=512):
    """Same-distance-ring enrichment of the deep scores with `exclude` patches marked invalid."""
    out = []
    for start in range(0, len(raw['index']), chunk):
        part = slice(start, start + chunk)
        fg = torch.from_numpy(raw['foreground'][part])
        valid = torch.from_numpy(raw['valid'][part] & ~exclude[part])
        values = plausibility.measures(torch.from_numpy(raw['scores'][part]), fg, valid, fg.roll(1, 0),
                                       generator=torch.Generator().manual_seed(start))
        out.append(values['enrichment'])
    return np.concatenate(out)


def paired(raw_a, raw_b):
    """Per-image enrichment difference a - b on the images both runs hold, both excluding the union."""
    common, ia, ib = np.intersect1d(raw_a['index'], raw_b['index'], return_indices=True)
    take = lambda raw, idx: {key: value[idx] for key, value in raw.items()}  # noqa: E731
    a, b = take(raw_a, ia), take(raw_b, ib)
    if not (a['foreground'] == b['foreground']).all():
        raise ValueError('Paired runs must cut the images into the same patches (uniform preprocessing)')
    union = deep_exclusion(a) | deep_exclusion(b)
    return common, enrichment(a, union) - enrichment(b, union)


def bootstrap(values, resamples, seed=0, level=.95):
    values = values[~np.isnan(values)]
    draws = np.random.default_rng(seed).integers(0, len(values), (resamples, len(values)))
    means = values[draws].mean(1)
    tail = (1 - level) / 2
    return float(values.mean()), float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail)), len(values)


def runs(prefix, tag):
    found = {}
    for model in MODELS:
        path = ROOT / f'{prefix}{tag}_{model}'
        if path.exists():
            found[model] = path
    return found


def pilot(prefix, out):
    t1 = runs(prefix, 't1')
    raw = {model: load_raw(path) for model, path in t1.items()}
    rows = []
    for model, path in t1.items():
        budget = pd.read_csv(path / 'budget.csv')
        allq = budget[budget['query'] == 'all']
        presence = (allq.assign(has=allq.share_high > 0).groupby('layer').has.mean())
        shares = allq.groupby('layer')[[f'share_{c}' for c in CATEGORIES]].mean()
        high = raw[model]['high']
        for layer in range(1, 13):
            rows.append(dict(model=model, layer=layer, images_with_high=float(high[:, layer - 1].any(1).mean()),
                             high_patch_fraction=float(high[:, layer - 1].mean()),
                             presence_from_budget=float(presence.get(layer, np.nan)),
                             **{c: float(shares.loc[layer, f'share_{c}']) for c in CATEGORIES}))
    pd.DataFrame(rows).to_csv(out / 'pilot_budget.csv', index=False)

    rows = []
    for model, path in t1.items():
        table = pd.read_csv(path / 'enrichment.csv')
        for scope in ['deep'] + [str(layer) for layer in range(1, 13)]:
            for exclude in ('high', 'none'):
                values = table[(table.scope.astype(str) == scope) & (table.exclude == exclude)].enrichment.to_numpy()
                mean, low, high_, n = bootstrap(values, 1000)
                rows.append(dict(model=model, scope=scope, exclude=exclude, mean=mean, low=low, high=high_, n=n,
                                 sd=float(np.nanstd(values, ddof=1))))
    pd.DataFrame(rows).to_csv(out / 'pilot_enrichment.csv', index=False)

    formal_n = 12419 - 500
    rows = []
    comparisons = [('H1', 'dino', model) for model in SUPERVISED] + [('H2', a, b) for a, b in PAIRS] + \
        [('ref', 'deit3_s', 'deit')]
    for hypothesis, a, b in comparisons:
        if a not in raw or b not in raw:
            continue
        _, diff = paired(raw[a], raw[b])
        mean, low, high_, n = bootstrap(diff, 1000)
        sd = float(np.nanstd(diff, ddof=1))
        rows.append(dict(hypothesis=hypothesis, a=a, b=b, n=n, mean=mean, low=low, high=high_, sd=sd,
                         predicted_half_width=Z_DECISION * sd / np.sqrt(formal_n)))
    pd.DataFrame(rows).to_csv(out / 'pilot_precision.csv', index=False)

    rows = []
    for model, path in t1.items():
        heads = pd.read_csv(path / 'heads.csv').groupby(['layer', 'head']).sum(numeric_only=True)
        for (layer, head), r in heads.iterrows():
            rows.append(dict(model=model, layer=layer, head=head, high_share=r.high_share / r.images,
                             dominant=r.dominant / r.images,
                             value_ratio=r.value_ratio / r.value_ratio_n if r.value_ratio_n else np.nan,
                             rank=r['rank'] / r.images,
                             rank_dominant=r.rank_dominant / r.rank_dominant_n if r.rank_dominant_n else np.nan))
    pd.DataFrame(rows).to_csv(out / 'pilot_heads.csv', index=False)

    rows = []
    for model, path in t1.items():
        selected = pd.read_csv(path / 'selected.csv').groupby('layer').mean(numeric_only=True)
        for layer, r in selected.iterrows():
            rows.append(dict(model=model, layer=layer, fraction=r.fraction, object_fraction=r.object_fraction,
                             enrichment=r.enrichment))
    pd.DataFrame(rows).to_csv(out / 'pilot_selected.csv', index=False)

    rows = []
    for model, path in runs(prefix, 't2').items():
        gate = json.loads((path / 'gate.json').read_text())
        row = dict(model=model, first_high_layer=gate['first_high_layer'], found=gate['found'], n=gate['n'],
                   neurons=json.dumps(gate['neurons']))
        if (path / 'gate.csv').exists():
            g = pd.read_csv(path / 'gate.csv')
            for n, part in g.groupby('n'):
                found = part[part.condition == 'found'].iloc[0]
                row[f'n{n}_reduction'] = found.reduction
                row[f'n{n}_agreement'] = found.agreement
                row[f'n{n}_random_reduction'] = part[part.condition == 'random'].reduction.mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / 'pilot_gate.csv', index=False)

    rows = []
    for model, path in runs(prefix, 't0').items():
        p = pd.read_csv(path / 'predictions.csv')
        rows.append(dict(model=model, n=len(p), own=float((p.own == p.label).mean() * 100),
                         uniform=float((p.uniform == p.label).mean() * 100),
                         change=float(((p.uniform == p.label).mean() - (p.own == p.label).mean()) * 100)))
    pd.DataFrame(rows).to_csv(out / 'model_check.csv', index=False)

    rows = []
    for model, path in runs(prefix, 't7').items():
        off = pd.read_csv(path / 'attention_off.csv')
        for layers, part in off.groupby(off.layers.astype(str)):
            mean, low, high_, n = bootstrap(part.agree.to_numpy() * 100., 1000)
            rows.append(dict(model=model, layers=layers, agreement=mean, low=low, high=high_, n=n))
    pd.DataFrame(rows).to_csv(out / 'attention_off.csv', index=False)


def verdict(hypothesis, low, high):
    """B28 rules for one comparison's 99.5% interval."""
    if hypothesis == 'H1':
        return 'holds' if low > MARGIN else ('fails' if high < MARGIN else 'uncertain')
    return 'holds' if high < MARGIN else ('fails' if low > MARGIN else 'uncertain')


def overall(hypothesis, verdicts):
    if not verdicts:
        return 'untestable'
    if 'fails' in verdicts:
        return 'fails'
    return 'holds' if all(v == 'holds' for v in verdicts) else 'uncertain'


def effect(drop, agree):
    """B31: 'none' (upper < 0.05 nats and agreement >= 99%), 'effect' (lower > 0.05), else 'unclear'."""
    mean, low, high, _ = bootstrap(drop, 1000)
    if high < .05 and agree >= 99:
        return 'none', mean, low, high
    return ('effect' if low > .05 else 'unclear'), mean, low, high


def formal(prefix, out):
    t3 = runs(prefix, 't3')
    raw = {model: load_raw(path) for model, path in t3.items()}
    judged = []
    for hypothesis, pairs in (('H1', [('dino', m) for m in SUPERVISED]), ('H2', list(PAIRS))):
        for a, b in pairs:
            if a in raw and b in raw:
                _, diff = paired(raw[a], raw[b])
                mean, low, high, n = bootstrap(diff, 10000, level=.995)
                judged.append(dict(hypothesis=hypothesis, a=a, b=b, n=n, mean=mean, low=low, high=high,
                                   verdict=verdict(hypothesis, low, high)))
    if 'deit3_s' in raw and 'deit' in raw:
        _, diff = paired(raw['deit3_s'], raw['deit'])
        mean, low, high, n = bootstrap(diff, 1000)
        judged.append(dict(hypothesis='reference', a='deit3_s', b='deit', n=n, mean=mean, low=low, high=high,
                           verdict='descriptive (95%)'))

    t6 = runs(prefix, 't6')
    t2 = runs(prefix, 't2')
    h4 = {}
    for model in ('deit', 'deit_b'):
        gate = json.loads((t2[model] / 'gate.json').read_text()) if model in t2 else {'found': False}
        if not gate['found'] or model not in t6:
            judged.append(dict(hypothesis='H6', a=f'{model}:R2', b=f'{model}:R0', verdict='untestable'))
            continue
        deep = pd.read_csv(t6[model] / 'deep.csv')
        deep = deep[deep.exclude == 'high'].pivot(index='index', columns='condition', values='enrichment')
        diff = (deep['R2'] - deep['R0']).to_numpy()
        mean, low, high, n = bootstrap(diff, 10000, level=.995)
        per_layer = pd.read_csv(t6[model] / 'register.csv')
        later = per_layer[per_layer.layer >= gate['first_high_layer']]
        counts = later.groupby('condition').high_count.sum()
        accuracy = pd.read_csv(t6[model] / 'accuracy.csv')
        acc = accuracy.assign(ok=accuracy.prediction == accuracy.label).groupby('condition').ok.mean() * 100
        reduction = 1 - counts['R2'] / max(counts['R0'], 1)
        h4[model] = dict(reduction=float(reduction), accuracy_R0=float(acc['R0']), accuracy_R2=float(acc['R2']),
                         passed=bool(reduction >= .8 and acc['R0'] - acc['R2'] <= 1))
        judged.append(dict(hypothesis='H6', a=f'{model}:R2', b=f'{model}:R0', n=n, mean=mean, low=low, high=high,
                           verdict=verdict('H6', low, high) if h4[model]['passed'] else 'untestable (H4)'))
        rows = []
        for condition in sorted(per_layer.condition.unique()):
            row = dict(model=model, condition=condition, accuracy=float(acc[condition]),
                       high_reduction=float(1 - counts[condition] / max(counts['R0'], 1)))
            for exclude in ('high', 'none'):
                values = pd.read_csv(t6[model] / 'deep.csv')
                part = values[(values.condition == condition) & (values.exclude == exclude)]
                row[f'enrichment_{exclude}'] = float(part.enrichment.mean())
                row['agreement'] = float(part.agree.mean() * 100)
            shares = per_layer[per_layer.condition == condition].groupby('layer')[[f'share_{c}' for c in CATEGORIES]]
            means = shares.mean()
            for layer in (11, 12):
                for c in ('register', 'high', 'object', 'background', 'cls'):
                    row[f'L{layer}_{c}'] = float(means.loc[layer, f'share_{c}'])
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f'register_{model}.csv', index=False)
    table = pd.DataFrame(judged)
    table.to_csv(out / 'judgements.csv', index=False)
    summary = {h: overall(h, [v for v in table[table.hypothesis == h].verdict if v in ('holds', 'fails', 'uncertain')])
               for h in ('H1', 'H2', 'H6')}
    summary['H4'] = h4
    write = lambda name, rows: pd.DataFrame(rows).to_csv(out / name, index=False)  # noqa: E731

    rows, curves = [], []
    for model, path in t3.items():
        budget = pd.read_csv(path / 'budget.csv')
        for query, part in budget.groupby('query'):
            means = part.groupby('layer')[[f'share_{c}' for c in CATEGORIES]].mean()
            for layer, r in means.iterrows():
                rows.append(dict(model=model, query=query, layer=layer, **{c: r[f'share_{c}'] for c in CATEGORIES}))
        enrichment = pd.read_csv(path / 'enrichment.csv')
        for (scope, exclude), part in enrichment.groupby([enrichment.scope.astype(str), 'exclude']):
            mean, low, high, n = bootstrap(part.enrichment.to_numpy(), 1000)
            curves.append(dict(model=model, scope=scope, exclude=exclude, mean=mean, low=low, high=high, n=n,
                               mass=float(part.mass.mean())))
    write('budget_layers.csv', rows)
    write('enrichment.csv', curves)

    rows = []
    for model, path in t3.items():
        selected = pd.read_csv(path / 'selected.csv').groupby('layer').mean(numeric_only=True)
        rows += [dict(model=model, layer=layer, fraction=r.fraction, object_fraction=r.object_fraction,
                      enrichment=r.enrichment) for layer, r in selected.iterrows()]
    write('selected.csv', rows)

    heads = {}
    for model, path in t3.items():
        h = pd.read_csv(path / 'heads.csv').groupby(['layer', 'head']).sum(numeric_only=True).reset_index()
        deep = h[h.layer.isin(DEEP)]
        by_layer = deep.groupby('layer').sum(numeric_only=True)
        ratio = (by_layer.value_ratio / by_layer.value_ratio_n.replace(0, np.nan)).mean()
        # B31 compares dominant heads with the other heads of the same layer, then averages over layers
        # (weighted by dominant cases): effective rank falls with depth for every head.
        other_n = by_layer.images - by_layer.rank_dominant_n
        dom = by_layer.rank_dominant / by_layer.rank_dominant_n.replace(0, np.nan)
        other = (by_layer['rank'] - by_layer.rank_dominant) / other_n.replace(0, np.nan)
        weights = by_layer.rank_dominant_n
        ratio_rank = float((dom / other * weights).sum() / weights.sum()) if weights.sum() else float('nan')
        heads[model] = dict(value_ratio=float(ratio), rank_ratio=ratio_rank, dominant_cases=int(weights.sum()))

    rows, kinds = [], {}
    for model, path in runs(prefix, 't4').items():
        use = pd.read_csv(path / 'use.csv')
        c2 = use[use.condition.str.startswith('c2_s')].groupby('index')[['drop', 'agree']].mean()
        wide = use.pivot(index='index', columns='condition', values='drop')
        wide['c2'] = c2['drop']
        agree = use.pivot(index='index', columns='condition', values='agree') * 100
        agree['c2'] = c2['agree'] * 100
        stats = {}
        for condition in ['high', 'c1', 'c2', 'mass', 'zero', 'mean']:
            values = wide[condition].to_numpy()
            if condition == 'mass':
                feasible = use[use.condition == 'mass'].set_index('index').feasible.astype(bool)
                values = wide.loc[feasible[feasible].index, 'mass'].to_numpy()
            kind, mean, low, high = effect(values, float(agree[condition].mean()))
            stats[condition] = kind
            share = use[use.condition == (condition if condition != 'c2' else 'c2_s0')].removed_share.mean()
            rows.append(dict(model=model, condition=condition, drop=mean, low=low, high=high,
                             agreement=float(agree[condition].mean()), removed_share=float(share), effect=kind,
                             feasible=float(use[use.condition == 'mass'].feasible.mean()) if condition == 'mass'
                             else 1.0))
        comparison = {}
        for control in ('c1', 'c2'):
            mean, low, high, _ = bootstrap((wide['high'] - wide[control]).to_numpy(), 1000)
            comparison[control] = (mean, low, high)
            rows.append(dict(model=model, condition=f'high-{control}', drop=mean, low=low, high=high))
        used = 'not used' if all(v[2] < 0 for v in comparison.values()) else \
            ('used' if all(v[1] > 0 for v in comparison.values()) else 'uncertain')
        d = heads.get(model, {})
        small_value = d.get('value_ratio', np.nan) < .5
        low_rank = d.get('rank_ratio', np.nan) <= .8
        if stats['zero'] == 'none' and small_value:
            kind = 'no-op'
        elif stats['zero'] == 'effect' and stats['mean'] == 'none':
            kind = 'constant bias'
        elif stats['mean'] == 'effect' and low_rank:
            kind = 'broadcast'
        else:
            kind = 'uncertain'
        kinds[model] = dict(use=used, sink_type=kind, small_value=bool(small_value), low_rank=bool(low_rank),
                            **{f'effect_{k}': v for k, v in stats.items()}, **d)
    write('sink_use.csv', rows)
    summary['sinks'] = kinds
    summary['heads'] = heads

    pairs = table[table.hypothesis == 'H2']
    definitive = summary['H2'] in ('holds', 'fails')
    same_direction = len(pairs) == 2 and (np.sign(pairs['mean']).nunique() == 1)
    causal = any(v.get('passed') for v in h4.values())
    use_clear = any(v['use'] in ('used', 'not used') for v in kinds.values())
    summary['story'] = dict(h2_definitive=definitive, same_direction=bool(same_direction), rq3_causal=causal,
                            use_clear=use_clear,
                            main_story='new study' if definitive and same_direction and (causal or use_clear)
                            else 'existing paper + new section')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(table.to_string())
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['pilot', 'formal'])
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    out = ROOT / f'{args.prefix}analysis_{args.stage}'
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'pilot':
        pilot(args.prefix, out)
    else:
        formal(args.prefix, out)
    print('written', out)


if __name__ == '__main__':
    main()
