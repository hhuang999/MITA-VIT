"""CPU analysis of one shifted-crop run (E5; registered prediction 4 of experiment plan v3).

For each image with the centre crop and at least one corner crop kept: the drop of each measure from the
centre crop to the mean of its corner crops. Registered comparison: the probe's object-mass drop against the
centre reference's (paired by image; the prediction is a smaller drop for the probe). Also reported: the
object's distance from the frame centre in each crop (to show it moved) and the drop of every measure for
every method. Writes analysis_shift.csv and analysis_comparisons.csv.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from fastweight_vit.results import load
from fastweight_vit.stats import paired, resample_indices, summarise

MEASURES = ['mass', 'pointing', 'auroc', 'precision', 'enrichment', 'object_distance']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    meta = json.loads((args.run / 'metadata.json').read_text())
    labels = dict(model=meta['model'])
    rows, duplicates = load(args.run, 'crops.csv', keys=['index', 'dy', 'dx', 'method'])
    rows = rows.assign(position=lambda f: (f.dy == 0) & (f.dx == 0)).replace({'position': {True: 'centre',
                                                                                          False: 'corner'}})
    means = rows.groupby(['index', 'method', 'position'])[MEASURES].mean().unstack('position')
    usable = means.dropna(subset=[('mass', 'centre'), ('mass', 'corner')]).index
    out, comparisons = [], []
    for measure in MEASURES:
        drop = (means[(measure, 'centre')] - means[(measure, 'corner')]).loc[usable].unstack('method').dropna()
        out += summarise(drop, resample_indices(len(drop)), measure=f'{measure}_drop', n=len(drop), **labels)
        for position in ['centre', 'corner']:
            level = means[(measure, position)].loc[usable].unstack('method').dropna()
            out += summarise(level, resample_indices(len(level)), measure=f'{measure}_{position}', n=len(level),
                             **labels)
        if measure == 'mass':
            others = [m for m in ['centre', 'cls_attention', 'rollout', 'gae'] if m in drop]
            comparisons += paired(drop, 'probe_mass', others, resample_indices(len(drop)), measure='mass_drop',
                                  n=len(drop), **labels)
    pd.DataFrame(out).rename(columns={'quantity': 'method'}).to_csv(args.run / 'analysis_shift.csv', index=False)
    pd.DataFrame(comparisons).to_csv(args.run / 'analysis_comparisons.csv', index=False)
    print(json.dumps(dict(labels, duplicates=duplicates, images=len(usable.get_level_values(0).unique()))))
    print(pd.DataFrame(comparisons).round(4).to_string())


if __name__ == '__main__':
    main()
