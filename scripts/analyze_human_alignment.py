"""CPU analysis of one ClickMe human-alignment run (E7; decision B12).

Writes into the run folder:
- analysis_alignment.csv: per method, the pixel and patch Spearman (mean, 95% interval), the shuffled-map
  control, the image-specific part (own - shuffled) and the Harmonization alignment score (Spearman / ceiling);
- analysis_comparisons.csv: the registered comparisons on the image-specific pixel Spearman, probe_mass
  against CLS attention, Rollout and GAE, and every method against the mean human map on the raw Spearman;
- analysis_layers.csv: the probe at each single layer (for the figure);
- analysis_by_correct.csv: the same per method, split by whether the model's top-1 is the label.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from mita_probe.data.clickme import HUMAN_CEILING
from mita_probe.results import load
from mita_probe.stats import paired, resample_indices, summarise

PRIMARY = 'probe_mass'
COMPARED = ['cls_attention', 'rollout', 'gae']


def per_image(rows):
    rows = rows.assign(specific_pixel=rows.spearman_pixel - rows.spearman_pixel_shuffled,
                       specific_token=rows.spearman_token - rows.spearman_token_shuffled,
                       alignment=rows.spearman_pixel / HUMAN_CEILING)
    return rows


def table(rows, value):
    return rows.pivot(index='index', columns='method', values=value).dropna()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    meta = json.loads((args.run / 'metadata.json').read_text())
    labels = dict(model=meta['model'])
    rows, duplicates = load(args.run, 'alignment.csv', keys=['index', 'method'])
    rows = per_image(rows)
    out, comparisons = [], []
    for value in ['spearman_pixel', 'spearman_pixel_shuffled', 'specific_pixel', 'alignment', 'spearman_token',
                  'spearman_token_shuffled', 'specific_token']:
        wide = table(rows, value)
        out += summarise(wide, resample_indices(len(wide)), measure=value, n=len(wide), **labels)
    frame = pd.DataFrame(out).rename(columns={'quantity': 'method'})
    frame.to_csv(args.run / 'analysis_alignment.csv', index=False)
    frame[frame.method.str.startswith('probe_layer')].assign(
        layer=lambda f: f.method.str.removeprefix('probe_layer').astype(int)).sort_values(['measure', 'layer']).to_csv(
        args.run / 'analysis_layers.csv', index=False)

    wide = table(rows, 'specific_pixel')
    draws = resample_indices(len(wide))
    comparisons += paired(wide, PRIMARY, COMPARED, draws, measure='specific_pixel', n=len(wide), **labels)
    raw = table(rows, 'spearman_pixel')
    comparisons += [row for method in raw.columns if method != 'mean_human'
                    for row in paired(raw, method, ['mean_human'], resample_indices(len(raw)),
                                      measure='spearman_pixel', n=len(raw), **labels)]
    pd.DataFrame(comparisons).to_csv(args.run / 'analysis_comparisons.csv', index=False)

    split = []
    correct = rows.drop_duplicates('index').set_index('index').eval('label == predicted')
    for flag, name in [(True, 'correct'), (False, 'wrong')]:
        subset = rows[rows['index'].isin(correct.index[correct == flag])]
        wide = table(subset, 'specific_pixel')
        if len(wide):
            split += summarise(wide, resample_indices(len(wide)), measure='specific_pixel', subset=name, n=len(wide),
                               **labels)
    pd.DataFrame(split).rename(columns={'quantity': 'method'}).to_csv(args.run / 'analysis_by_correct.csv',
                                                                       index=False)
    checks = dict(labels, duplicates=duplicates, images=int(rows['index'].nunique()), top1=float(correct.mean()))
    (args.run / 'analysis_checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(checks))
    show = frame.query("measure in ['spearman_pixel', 'specific_pixel'] and not method.str.startswith('probe_layer')")
    print(show.pivot(index='method', columns='measure', values='mean').round(3).to_string())
    print(pd.DataFrame(comparisons).query("measure == 'specific_pixel'").round(4).to_string())


if __name__ == '__main__':
    main()
