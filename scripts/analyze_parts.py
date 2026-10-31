"""CPU analysis of one PartImageNet parts run (E8; decision B13).

Animal super-categories (Quadruped, Biped, Fish, Bird, Snake, Reptile) carry the registered comparisons:
(a) head enrichment, probe_mass against the centre reference (paired by image);
(b) the model's own reliance, head against body removal (log-probability drop of the original top-1, r = 4,
    seeds averaged first; paired by image);
(c) explanation-reliance agreement: on images with at least two removable parts, does the part a method
    enriches most equal the part whose removal hurts most? Paired against the centre and random references;
    chance = mean of 1 / number of removable parts.
Writes analysis_parts.csv (enrichment, mass, AUROC per method and part), analysis_reliance.csv (log drop per
part and control), analysis_agreement.csv, analysis_comparisons.csv and analysis_layers.csv (head
enrichment of the probe at each single layer).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mita_probe.data.partimagenet import ANIMALS, PART, SUPERCATEGORY
from mita_probe.results import load
from mita_probe.stats import interval, paired, resample_indices, summarise

PRIMARY = 'probe_mass'
REMOVED = 4
HEADS = [p for p, (s, name) in enumerate(zip(SUPERCATEGORY, PART)) if s in ANIMALS and name == 'Head']
BODIES = [p for p, (s, name) in enumerate(zip(SUPERCATEGORY, PART)) if s in ANIMALS and name == 'Body']


def describe(frame, value, by, **labels):
    out = []
    for key, group in frame.groupby(by):
        values = group[value].dropna().to_numpy()
        if len(values):
            mean, low, high = interval(values, resample_indices(len(values)))
            out.append(dict(labels, **dict(zip(by, key if isinstance(key, tuple) else (key,))), n=len(values),
                            mean=mean, low=low, high=high))
    return out


def part_drops(reliance):
    """Log-probability drop and agreement per removal count, image, condition and part (seeds averaged)."""
    reliance = reliance.assign(log_drop=np.log(reliance.p_original) - np.log(reliance.p_target.clip(lower=1e-12)))
    return reliance.groupby(['removed', 'index', 'condition', 'part'])[['log_drop', 'agree']].mean().reset_index()


def agreement(parts, main_drop):
    """Per image with at least two removable parts: does each map's most enriched removable part equal the part
    whose removal hurts most? Returns the match table [image x method] and each image's chance level."""
    removable = main_drop.groupby('index').filter(lambda g: len(g) >= 2)
    causal = removable.loc[removable.groupby('index').log_drop.idxmax(), ['index', 'part']].set_index('index').part
    count = removable.groupby('index').size()
    candidates = parts.merge(removable[['index', 'part']], on=['index', 'part'])
    top = candidates.loc[candidates.dropna(subset=['enrichment']).groupby(['index', 'method']).enrichment.idxmax()]
    match = top.assign(match=(top.part.to_numpy() == causal.reindex(top['index']).to_numpy()).astype(float))
    wide = match.pivot(index='index', columns='method', values='match').dropna()
    return wide, (1 / count.loc[wide.index]).to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    meta = json.loads((args.run / 'metadata.json').read_text())
    labels = dict(model=meta['model'])
    parts, part_duplicates = load(args.run, 'parts.csv', keys=['index', 'method', 'part'])
    parts = parts.assign(supercategory=parts.part.map(dict(enumerate(SUPERCATEGORY))),
                         name=parts.part.map(dict(enumerate(PART))))
    out = []
    for value in ['enrichment', 'mass', 'area', 'auroc']:
        out += [row | dict(measure=value) for row in describe(parts, value, ['method', 'supercategory', 'name'],
                                                              **labels)]
    pd.DataFrame(out).to_csv(args.run / 'analysis_parts.csv', index=False)

    heads = parts[parts.part.isin(HEADS)]
    layers = heads[heads.method.str.startswith('probe_layer')].assign(
        layer=lambda f: f.method.str.removeprefix('probe_layer').astype(int))
    pd.DataFrame(describe(layers, 'enrichment', ['layer'], **labels)).to_csv(args.run / 'analysis_layers.csv',
                                                                            index=False)
    comparisons = []
    wide = heads.pivot(index='index', columns='method', values='enrichment').dropna()
    comparisons += paired(wide, PRIMARY, ['centre', 'random', 'cls_attention', 'rollout', 'gae'],
                          resample_indices(len(wide)), measure='head_enrichment', n=len(wide), **labels)

    reliance, reliance_duplicates = load(args.run, 'reliance.csv', keys=['index', 'condition', 'part', 'seed',
                                                                         'removed'])
    drop = part_drops(reliance)
    named = drop.assign(supercategory=drop.part.map(dict(enumerate(SUPERCATEGORY))).fillna('control'),
                        name=drop.part.map(dict(enumerate(PART))).fillna(drop.condition))
    pd.DataFrame(describe(named, 'log_drop', ['removed', 'supercategory', 'name'], **labels)).to_csv(
        args.run / 'analysis_reliance.csv', index=False)

    main_drop = drop[(drop.removed == REMOVED) & (drop.condition == 'part')]
    head = main_drop[main_drop.part.isin(HEADS)].set_index('index').log_drop
    body = main_drop[main_drop.part.isin(BODIES)].set_index('index').log_drop
    both = pd.DataFrame(dict(head=head, body=body)).dropna()
    comparisons += paired(both, 'head', ['body'], resample_indices(len(both)), measure='reliance_log_drop_r4',
                          n=len(both), **labels)

    # (c) agreement between each map's most enriched part and the part whose removal hurts most.
    wide, chance = agreement(parts, main_drop)
    rows = summarise(wide, resample_indices(len(wide)), measure='agreement', n=len(wide), **labels)
    rows.append(dict(labels, measure='agreement', n=len(wide), quantity='chance',
                     **dict(zip(('mean', 'low', 'high'), interval(chance, resample_indices(len(chance)))))))
    pd.DataFrame(rows).rename(columns={'quantity': 'method'}).to_csv(args.run / 'analysis_agreement.csv',
                                                                     index=False)
    comparisons += [row for method in ['probe_mass', 'gae', 'cls_attention', 'rollout'] if method in wide
                    for row in paired(wide, method, ['centre', 'random'], resample_indices(len(wide)),
                                      measure='agreement', n=len(wide), **labels)]
    pd.DataFrame(comparisons).to_csv(args.run / 'analysis_comparisons.csv', index=False)
    checks = dict(labels, part_duplicates=part_duplicates, reliance_duplicates=reliance_duplicates,
                  images=int(parts['index'].nunique()), agreement_images=len(wide))
    (args.run / 'analysis_checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(checks))
    print(pd.DataFrame(comparisons).round(4).to_string())


if __name__ == '__main__':
    main()
