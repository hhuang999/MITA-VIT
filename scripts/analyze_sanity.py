"""CPU analysis of one weight-randomisation sanity run (decision B14).

Per level (cumulative re-initialisation from the head down) and method: median and mean Spearman with the
trained model's map (mean with a 95% interval), and the share of images whose top-1 is unchanged. Verdict
at level block1 (head and all 12 blocks re-initialised): 'pass' when the median Spearman is below 0.5.
Writes analysis_sanity.csv and analysis_verdicts.csv.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from fastweight_vit.results import load
from fastweight_vit.stats import interval, resample_indices

THRESHOLD = 0.5
VERDICT_LEVEL = 'block1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    meta = json.loads((args.run / 'metadata.json').read_text())
    rows, duplicates = load(args.run, 'sanity.csv', keys=['index', 'level', 'method'])
    order = {level: n for n, level in enumerate(meta['levels'])}
    out = []
    for (level, method), group in rows.groupby(['level', 'method']):
        values = group.spearman.dropna().to_numpy()
        mean, low, high = interval(values, resample_indices(len(values))) if len(values) else (float('nan'),) * 3
        out.append(dict(model=meta['model'], level=level, step=order[level], method=method, n=len(values),
                        median=float(pd.Series(values).median()), mean=mean, low=low, high=high,
                        top1_kept=group.top1_kept.mean()))
    table = pd.DataFrame(out).sort_values(['method', 'step'])
    table.to_csv(args.run / 'analysis_sanity.csv', index=False)
    verdicts = table.query('level == @VERDICT_LEVEL').assign(
        verdict=lambda f: f['median'].map(lambda m: 'pass' if m < THRESHOLD else 'fail'))
    verdicts.to_csv(args.run / 'analysis_verdicts.csv', index=False)
    print(json.dumps(dict(model=meta['model'], duplicates=duplicates)))
    print(table.pivot(index='method', columns='step', values='median').round(2).to_string())
    print(verdicts[['method', 'median', 'verdict']].to_string())


if __name__ == '__main__':
    main()
