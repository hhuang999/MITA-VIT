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
        raise SystemExit('formal analysis is added after B31 is registered')
    print('written', out)


if __name__ == '__main__':
    main()
