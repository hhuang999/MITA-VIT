"""CPU analysis of the centre-matched deletion (decision A15) on any dataset (E6); on Pet, also its
consistency with the V2/DINO runs."""

import argparse
import json
from pathlib import Path

import pandas as pd

from fastweight_vit.results import load
from fastweight_vit.stats import interval, ratio_interval, resample_indices

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = {'deit': '2026-10-28_v2_pet', 'mita': '2026-10-28_v2_pet', 'dino': '2026-10-28_v2_dino_pet'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'experiments/2026-10-29_center_matched_deletion')
    args = parser.parse_args()
    rows, duplicates = load(args.run, 'deletion.csv')
    checks = dict(duplicates=duplicates)
    summary = []
    for (model, category), frame in rows.query("kind != 'baseline'").groupby(['model', 'category']):
        target = frame.query("kind == 'target'").set_index('index')
        eligible = target.index[target.eligible == 1].sort_values()
        mean = lambda kind: frame.query('kind == @kind').groupby('index')['agree'].mean().loc[eligible]
        t, r, c = target.loc[eligible, 'agree'], mean('random'), mean('centre_matched')
        draws = resample_indices(len(eligible))
        row = dict(model=model, category=category, n=len(eligible), target=t.mean(), random=r.mean(),
                   centre_matched=c.mean(), mean_removed_per_set=target.loc[eligible, 'mean_removed_per_set'].mean())
        for name, diff in [('target_minus_random', t - r), ('target_minus_matched', t - c),
                           ('matched_minus_random', c - r)]:
            row[name], row[f'{name}_low'], row[f'{name}_high'] = interval(diff, draws)
        row['remaining'], row['remaining_low'], row['remaining_high'] = ratio_interval(t - c, t - r, draws)
        primary = category == 'foreground' and model in {'deit', 'dino'}
        low, high = row['target_minus_matched_low'], row['target_minus_matched_high']
        row['verdict'] = ('not explained by centre' if high < 0 else 'explained by centre' if low > 0
                          else 'uncertain') if primary else ''
        summary.append(row)
    for model, frame in rows.query("kind == 'baseline'").groupby('model'):
        summary.append(dict(model=model, category='baseline', n=len(frame), target=frame.agree.mean()))
    table = pd.DataFrame(summary)
    table.to_csv(args.run / 'summary.csv', index=False)

    # The recomputed target and random conditions should repeat the earlier Pet runs image by image.
    keys = ['index', 'category', 'kind', 'seed']
    dataset = json.loads((args.run / 'metadata.json').read_text()).get('dataset', 'pet')
    for model, directory in (ORIGINAL.items() if dataset == 'pet' else []):
        old, _ = load(ROOT / 'experiments' / directory, 'deletion.csv')
        old = old.query("model == @model and kind in ['target', 'random', 'baseline']").set_index(keys)
        new = rows.query("model == @model and kind in ['target', 'random', 'baseline']").set_index(keys)
        joined = new[['agree', 'kl']].merge(old[['agree', 'kl']], left_index=True, right_index=True,
                                             suffixes=('', '_old'))
        checks[f'{model}_compared_rows'] = len(joined)
        checks[f'{model}_agree_mismatches'] = int((joined.agree != joined.agree_old).sum())
        checks[f'{model}_max_kl_difference'] = float((joined.kl - joined.kl_old).abs().max())
    (args.run / 'analysis_checks.json').write_text(json.dumps(checks, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(checks, indent=2))
    columns = ['model', 'category', 'n', 'target', 'random', 'centre_matched', 'target_minus_random',
               'target_minus_matched', 'target_minus_matched_low', 'target_minus_matched_high', 'remaining',
               'remaining_low', 'remaining_high', 'verdict']
    print(table[columns].round(4).to_string())


if __name__ == '__main__':
    main()
