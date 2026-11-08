"""CPU analysis of the supplementary experiments S1-S3 (decisions B22-B24).

Reads the supplementary runs (<prefix>e4_<dataset>_<model>, <prefix>e7_clickme_<model>, <prefix>e8_parts_<model>,
<prefix>layers_<model>) and the v3 runs they are checked against, and writes into <prefix>analysis/:
- s1_comparisons.csv: attention_mass - probe_mass from the same run, paired bootstrap (B10), on SRG and
  same-ring enrichment (E4), ClickMe image-specific Spearman, animal head enrichment and part agreement
  (reliance from the v3 runs), with both maps' means;
- s1_similarity.csv: per-image Spearman between the two maps on ImageNet-S (saved scores);
- s1_reproduction.csv: probe_mass SRG in the new run against the v3 run (different GPU);
- s2_layers.csv: ClickMe image-specific Spearman and head enrichment per model, family and layer;
- s2_patterns.csv: per model, measure and family, Pearson r with the probe's layer curve and layer 12 minus
  the best of layers 8-11 (paired), plus the new probe curve's largest difference from v3;
- s3_layers.csv: per model, family and layer: high-norm patch fraction, map share on them, object mass share and
  same-ring enrichment (ImageNet-S); s3_tests.csv: the registered paired tests;
- predictions.csv: every registered prediction of B22-B24 with its values and verdict.
The per-run E4 tables (analysis_faithfulness.csv, per_image_faithfulness.csv.gz) are written into each E4 run.

Usage: python scripts/analyze_supplement.py [--prefix experiments/2026-10-31_s_]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_explanations import faithfulness as e4_faithfulness  # noqa: E402
from analyze_human_alignment import per_image as clickme_per_image  # noqa: E402
from analyze_parts import HEADS, agreement, part_drops  # noqa: E402

from fastweight_vit.metrics import spearman  # noqa: E402
from fastweight_vit.results import load  # noqa: E402
from fastweight_vit.stats import interval, paired, resample_indices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / 'experiments/2026-10-29_v3_'
MODELS = ['deit', 'dino', 'deit_b']
E4 = [(d, m) for d in ['imagenet_s', 'hard_imagenet'] for m in MODELS] + [
    (d, m) for d in ['pet', 'imagenet'] for m in ['deit', 'dino']]
MARGIN = {'srg_agree': 0.05, 'enrichment': 0.02, 'specific_pixel': 0.02, 'head_enrichment': 0.1}  # B22
PEAK = {'deit': 9, 'deit_b': 6}  # v3 ClickMe peak layer of the probe (B24)
LAYER_MAP = re.compile(r'^(probe|attention|cls)_layer(\d+)$')
DENSE, PROBE = 'attention_mass', 'probe_mass'


def compare(table, measure, **labels):
    """attention_mass - probe_mass on a per-image table, with both means and the B22 verdicts."""
    draws = resample_indices(len(table))
    row = paired(table, DENSE, [PROBE], draws, measure=measure, n=len(table), **labels)[0]
    margin = MARGIN.get(measure)
    comparable = None if margin is None else bool(row['low'] <= 0 <= row['high'] or abs(row['difference']) < margin)
    return dict(row, probe=table[PROBE].mean(), attention=table[DENSE].mean(), comparable=comparable)


def described(values, **labels):
    mean, low, high = interval(np.asarray(values, float), resample_indices(len(values)))
    return dict(labels, n=len(values), mean=mean, low=low, high=high)


def layer_tables(per_image, measure, model):
    """per_image: [image x method] with single-layer maps. Curves per family, and the registered pattern tests."""
    curves, patterns = [], []
    columns = {name: LAYER_MAP.match(name).groups() for name in per_image if LAYER_MAP.match(name)}
    means = {}
    for name, (family, layer) in columns.items():
        values = per_image[name].dropna()
        curves.append(described(values, model=model, measure=measure, family=family, layer=int(layer)))
        means.setdefault(family, {})[int(layer)] = values.mean()
    probe = pd.Series(means['probe']).sort_index()
    for family, curve in means.items():
        curve = pd.Series(curve).sort_index()
        best = int(curve.loc[8:11].idxmax())
        last = (per_image[f'{family}_layer12'] - per_image[f'{family}_layer{best}']).dropna()
        patterns.append(dict(described(last, model=model, measure=measure, family=family), best_layer=best,
                             pearson_with_probe=float(np.corrcoef(curve, probe)[0, 1])))
    return curves, patterns


def s1_e4(prefix):
    comparisons, reproduction, similarity = [], [], []
    for dataset, model in E4:
        run = Path(f'{prefix}e4_{dataset}_{model}')
        if not run.exists():
            print('missing', run.name)
            continue
        labels = dict(dataset=dataset, model=model)
        faith, _ = load(run, 'faithfulness.csv', keys=['index', 'method', 'condition', 'budget'])
        per_image = e4_faithfulness(faith, run)
        srg = per_image['srg_agree'].unstack('method').dropna()
        comparisons.append(compare(srg, 'srg_agree', **labels))
        if (run / 'plausibility.csv').exists() or (run / 'plausibility.csv.gz').exists() or (
                run / 'raw' / 'plausibility.csv.gz').exists():
            plaus, _ = load(run, 'plausibility.csv', keys=['index', 'method'])
            comparisons.append(compare(plaus.pivot(index='index', columns='method', values='enrichment').dropna(),
                                       'enrichment', **labels))
        old = pd.read_csv(Path(f'{V3}e4_{dataset}_{model}') / 'per_image_faithfulness.csv.gz', index_col=[0, 1])
        both = pd.DataFrame(dict(v3=old.loc[PROBE, 'srg_agree'], new=srg[PROBE])).dropna()
        reproduction.append(dict(labels, n=len(both), v3=both.v3.mean(), new=both.new.mean(),
                                 difference=both.new.mean() - both.v3.mean(),
                                 identical_images=float((both.v3 - both.new).abs().lt(1e-4).mean())))
        scores = sorted((run / 'raw' / 'scores').glob('*.npz')) or sorted((run / 'scores').glob('*.npz'))
        if dataset == 'imagenet_s' and scores:
            rho = []
            for path in scores:
                arrays = np.load(path)
                rho += spearman(torch.from_numpy(arrays[PROBE].astype(np.float64)),
                                torch.from_numpy(arrays[DENSE].astype(np.float64))).tolist()
            rho = np.asarray(rho)
            similarity.append(dict(labels, n=len(rho), median=float(np.median(rho)),
                                   q25=float(np.quantile(rho, .25)), q75=float(np.quantile(rho, .75))))
    return comparisons, reproduction, similarity


def s1_s2_clickme(prefix):
    comparisons, curves, patterns = [], [], []
    for model in MODELS:
        run = Path(f'{prefix}e7_clickme_{model}')
        if not run.exists():
            print('missing', run.name)
            continue
        rows, _ = load(run, 'alignment.csv', keys=['index', 'method'])
        table = clickme_per_image(rows).pivot(index='index', columns='method', values='specific_pixel')
        comparisons.append(compare(table[[PROBE, DENSE]].dropna(), 'specific_pixel', dataset='clickme', model=model))
        model_curves, model_patterns = layer_tables(table, 'clickme_specific_pixel', model)
        old = pd.read_csv(Path(f'{V3}e7_clickme_{model}') / 'analysis_layers.csv').query(
            "measure == 'specific_pixel'").set_index('layer')['mean']
        new = pd.Series({c['layer']: c['mean'] for c in model_curves if c['family'] == 'probe'})
        for pattern in model_patterns:
            pattern['probe_v3_max_difference'] = float((new - old).abs().max())
        curves += model_curves
        patterns += model_patterns
    return comparisons, curves, patterns


def s1_s2_parts(prefix):
    comparisons, curves, patterns = [], [], []
    for model in MODELS:
        run = Path(f'{prefix}e8_parts_{model}')
        if not run.exists():
            print('missing', run.name)
            continue
        parts, _ = load(run, 'parts.csv', keys=['index', 'method', 'part'])
        heads = parts[parts.part.isin(HEADS)].pivot(index='index', columns='method', values='enrichment')
        comparisons.append(compare(heads[[PROBE, DENSE]].dropna(), 'head_enrichment', dataset='partimagenet',
                                   model=model))
        reliance, _ = load(Path(f'{V3}e8_parts_{model}'), 'reliance.csv',
                           keys=['index', 'condition', 'part', 'seed', 'removed'])
        drop = part_drops(reliance)
        wide, chance = agreement(parts[parts.method.isin([PROBE, DENSE])],
                                 drop[(drop.removed == 4) & (drop.condition == 'part')])
        comparisons.append(dict(compare(wide, 'agreement', dataset='partimagenet', model=model),
                                chance=float(chance.mean())))
        model_curves, model_patterns = layer_tables(heads, 'head_enrichment', model)
        old = pd.read_csv(Path(f'{V3}e8_parts_{model}') / 'analysis_layers.csv').set_index('layer')['mean']
        new = pd.Series({c['layer']: c['mean'] for c in model_curves if c['family'] == 'probe'})
        for pattern in model_patterns:
            pattern['probe_v3_max_difference'] = float((new - old).abs().max())
        curves += model_curves
        patterns += model_patterns
    return comparisons, curves, patterns


def s3(prefix):
    layers, tests, per_model = [], [], {}
    for model in MODELS:
        run = Path(f'{prefix}layers_{model}')
        if not run.exists():
            print('missing', run.name)
            continue
        rows, _ = load(run, 'layer_reading.csv', keys=['index', 'family', 'layer'])
        per_model[model] = rows
        for (family, layer), group in rows.groupby(['family', 'layer']):
            for value in ['high_fraction', 'high_share', 'mass', 'enrichment']:
                layers.append(described(group[value].dropna(), model=model, family=family, layer=layer, measure=value))
    for model, peak in PEAK.items():
        if model in per_model:
            share = per_model[model].query("family == 'probe'").pivot(index='index', columns='layer',
                                                                     values='high_share')
            tests.append(described((share[12] - share[peak]).dropna(), test=f'layer 12 - layer {peak}', model=model))
    if {'deit', 'dino'} <= set(per_model):
        last = {m: per_model[m].query("family == 'probe' and layer == 12").set_index('index').high_share
                for m in ['deit', 'dino']}
        tests.append(described((last['dino'] - last['deit']).dropna(), test='layer 12: dino - deit', model='dino'))
    return layers, tests


def judge(results):
    return 'holds' if all(results) else 'does not hold' if not any(results) else 'partly holds'


def predictions(s1, reproduction, similarity, patterns, tests):
    s1 = pd.DataFrame(s1)
    out = []

    def add(key, text, results, evidence):
        out.append(dict(decision=key, prediction=text, verdict=judge(results) if results else 'no data',
                        evidence=evidence))

    def listing(frame, column='difference'):
        return '; '.join(f"{r.model}/{r.dataset} {r[column]:+.3f} [{r.low:+.3f}, {r.high:+.3f}]"
                         for _, r in frame.iterrows())

    for number, measure, text in [(1, 'srg_agree', 'SRG comparable (dense - probe)'),
                                  (2, 'enrichment', 'Same-ring enrichment comparable'),
                                  (3, 'specific_pixel', 'ClickMe image-specific Spearman comparable')]:
        part = s1.query('measure == @measure') if len(s1) else s1
        add(f'B22 ({number})', text, list(part.comparable) if len(part) else [], listing(part) if len(part) else '')
    part = s1.query("measure == 'head_enrichment'") if len(s1) else s1
    add('B22 (4)', 'Head enrichment higher for the probe than for dense attention',
        list(part.verdict == 'lower') if len(part) else [], listing(part) if len(part) else '')
    part = s1.query("measure == 'agreement'") if len(s1) else s1
    out.append(dict(decision='B22 (5)', prediction='Part agreement: no direction', verdict='not predicted',
                    evidence=listing(part) if len(part) else ''))
    sim = pd.DataFrame(similarity)
    add('B22 (6)', 'Median per-image Spearman between the maps > 0.5 (ImageNet-S)',
        list(sim['median'] > .5) if len(sim) else [],
        '; '.join(f'{r.model} {r.median:.2f} [{r.q25:.2f}, {r.q75:.2f}]' for r in sim.itertuples()))
    rep = pd.DataFrame(reproduction)
    add('B22 (7)', 'New-run probe SRG within 0.01 of v3', list(rep.difference.abs() < .01) if len(rep) else [],
        '; '.join(f'{r.model}/{r.dataset} {r.difference:+.4f}' for r in rep.itertuples()))
    pat = pd.DataFrame(patterns)
    dense = pat.query("family == 'attention'") if len(pat) else pat
    add('B23 (1)', 'Dense single-layer curve matches the probe curve (Pearson r >= 0.8)',
        list(dense.pearson_with_probe >= .8) if len(dense) else [],
        '; '.join(f'{r.model}/{r.measure} r={r.pearson_with_probe:.2f}' for r in dense.itertuples()))
    supervised = dense[dense.model.isin(['deit', 'deit_b'])] if len(dense) else dense
    add('B23 (2)', 'Dense attention also drops at layer 12 (DeiT-S, DeiT-B; layer 12 - best of 8-11 < 0)',
        list(supervised.high < 0) if len(supervised) else [],
        '; '.join(f'{r.model}/{r.measure} {r.mean:+.3f} [{r.low:+.3f}, {r.high:+.3f}] vs layer {r.best_layer}'
                  for r in supervised.itertuples()))
    cls = pat.query("family == 'cls'") if len(pat) else pat
    out.append(dict(decision='B23 (3)', prediction='CLS attention per layer: no prediction', verdict='not predicted',
                    evidence='; '.join(f'{r.model}/{r.measure} r={r.pearson_with_probe:.2f}, last {r.mean:+.3f}'
                                       for r in cls.itertuples())))
    tests = pd.DataFrame(tests)
    rise = tests[tests.test.str.startswith('layer 12 -')] if len(tests) else tests
    add('B24 (1)', 'DeiT-S, DeiT-B: probe high-norm share higher at layer 12 than at the ClickMe peak layer',
        list(rise.low > 0) if len(rise) else [],
        '; '.join(f'{r.model} {r.test}: {r.mean:+.3f} [{r.low:+.3f}, {r.high:+.3f}]' for r in rise.itertuples()))
    gap = tests[tests.test.str.contains('dino - deit')] if len(tests) else tests
    add('B24 (2)', 'Layer 12: DINO probe high-norm share lower than DeiT-S', list(gap.high < 0) if len(gap) else [],
        '; '.join(f'{r.mean:+.3f} [{r.low:+.3f}, {r.high:+.3f}]' for r in gap.itertuples()))
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default=str(ROOT / 'experiments/2026-10-31_s_'))
    args = parser.parse_args()
    out = Path(f'{args.prefix}analysis')
    out.mkdir(exist_ok=True)
    s1, reproduction, similarity = s1_e4(args.prefix)
    clickme, clickme_curves, clickme_patterns = s1_s2_clickme(args.prefix)
    parts, parts_curves, parts_patterns = s1_s2_parts(args.prefix)
    s1 += clickme + parts
    layers, tests = s3(args.prefix)
    tables = dict(s1_comparisons=s1, s1_reproduction=reproduction, s1_similarity=similarity,
                  s2_layers=clickme_curves + parts_curves, s2_patterns=clickme_patterns + parts_patterns,
                  s3_layers=layers, s3_tests=tests)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
    verdicts = predictions(s1, reproduction, similarity, clickme_patterns + parts_patterns, tests)
    verdicts.to_csv(out / 'predictions.csv', index=False)
    with pd.option_context('display.max_colwidth', 200, 'display.width', 250):
        print(verdicts.to_string())


if __name__ == '__main__':
    main()
