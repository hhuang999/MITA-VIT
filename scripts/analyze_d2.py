"""D2 analysis (decision B50; exploratory, post hoc): discount without relocation.

  python scripts/analyze_d2.py <run prefix> <output folder>
  e.g. python scripts/analyze_d2.py experiments/2026-11-08 experiments/2026-11-08_explore_d2_analysis

Reads <prefix>_explore_d2_{imagenet_s,clickme}_<model>. Per model and metric (enrichment, ClickMe): D2 - R0,
D2 - D2_shuffle (per image, mean over seeds 0-4), D2 - C2, and C3D2 - C3 (reported only under branch D-A). Statistics
and verdicts as B22 (analyze_v5.paired). Branch rule of B50. R0, C2 and C3 of the same runs are checked against the
formal correct_ablation runs (consistency.csv). Writes tests.csv, consistency.csv, branch.csv.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_v5 import ROOT, find, paired, per_image, summary  # noqa: E402

MODELS = ['deit', 'deit_b', 'dino']
SEEDS = [f'D2_shuffle_s{seed}' for seed in range(5)]


def main():
    prefix, out = str(ROOT / sys.argv[1]), ROOT / sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    rows, checks = [], []
    for model in MODELS:
        data = per_image(find(prefix, f'explore_d2_imagenet_s_{model}'), find(prefix, f'explore_d2_clickme_{model}'))
        formal = per_image(find(str(ROOT / 'experiments/2026-11-05'), f'correct_ablation_imagenet_s_{model}'),
                           find(str(ROOT / 'experiments/2026-11-05'), f'correct_ablation_clickme_{model}'))
        for metric, table in data.items():
            if metric == 'srg':
                continue
            table['D2_shuffle'] = table[SEEDS].mean(axis=1)
            for name in ('R0', 'C2', 'C3', 'D2', 'D2_shuffle', 'C3D2'):
                rows.append(summary(table[name], model=model, metric=metric, method=name, test='value'))
            for other in ('R0', 'D2_shuffle', 'C2'):
                rows.append(paired(table, 'D2', other, metric, model=model, test=f'D2 - {other}'))
            rows.append(paired(table, 'C3D2', 'C3', metric, model=model, test='C3D2 - C3'))
            for name in ('R0', 'C2', 'C3'):
                both = pd.concat([table[name].rename('new'), formal[metric][name].rename('formal')], axis=1).dropna()
                checks.append(dict(model=model, metric=metric, method=name, n=len(both),
                                   max_abs_difference=float((both.new - both.formal).abs().max())))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'tests.csv', index=False)
    pd.DataFrame(checks).to_csv(out / 'consistency.csv', index=False)
    gain = frame[frame.test == 'D2 - R0']
    higher = gain[gain.verdict == 'higher']
    against = frame[frame.test == 'D2 - D2_shuffle'].set_index(['model', 'metric']).verdict
    shuffle_ok = all(against.get((r.model, r.metric)) == 'higher' for r in higher.itertuples())
    branch = 'D-A' if len(higher) >= 2 and not (gain.verdict == 'lower').any() and shuffle_ok else 'D-B'
    pd.DataFrame([dict(item='D2', branch=branch, cells=len(gain), higher=len(higher),
                       lower=int((gain.verdict == 'lower').sum()), shuffle_higher_in_those_cells=shuffle_ok)]).to_csv(
        out / 'branch.csv', index=False)
    print(frame[frame.test != 'value'][['model', 'metric', 'test', 'mean', 'low', 'high', 'verdict']].to_string(index=False))
    print('branch:', branch)


if __name__ == '__main__':
    main()
