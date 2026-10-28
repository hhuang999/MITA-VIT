"""CPU analysis of the DINO ViT-S/16 supplement (decision A13): tables, DeiT-S comparison, D1-D6."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd

from mita_probe.anatomy_analysis import anatomy_table, interval, ratio_interval, resample_indices

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('analyze_anatomy', ROOT / 'scripts/analyze_anatomy.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
load, rows_of, seed_mean = base.load, base.rows_of, base.seed_mean
SPECS = ['m16_k16_clean', 'm25_k25_clean', 'm25_k25_mita']


def fg_share(frame):
    return frame['foreground'] / frame[['foreground', 'background', 'sink']].sum(axis=1).where(
        lambda total: total > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--prefix', default='2026-10-28_v2_dino_')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/2026-10-28_v2_dino_analysis')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    run = lambda stage: args.runs / f'{args.prefix}{stage}'
    deit_run = lambda stage: args.runs / f'2026-10-28_v2_{stage}'
    checks, value = {}, {}

    repro, checks['repro_duplicates'] = load(run('repro'), 'predictions.csv')
    checks['top1'], checks['repro_images'] = float(repro.correct.mean() * 100), int(len(repro))

    noise, checks['noise_duplicates'] = load(run('noise'), 'predictions.csv')
    noise = noise.sort_values('index')
    n = noise['index'].nunique()
    draws = resample_indices(n)
    floor = interval(noise.agree, draws)
    pd.DataFrame([dict(model='dino', n=n, agree=floor[0], low=floor[1], high=floor[2],
                       kl=noise.kl.mean())]).to_csv(out / 'noise_floor.csv', index=False)
    checks['noise_floor'] = floor[0]

    # RQ1: single-layer anatomy, per-image tables kept for the layer-band contrast (D2).
    rq1, checks['rq1_duplicates'] = load(run('rq1'), 'predictions.csv')
    anatomy_rows, specific = [], {}
    for layer, frame in rq1.groupby('layer'):
        table = seed_mean(frame, ['setting'])
        assert len(table) == n and not table.isna().any().any()
        summary = anatomy_table(table, draws, upper='L1_k32')
        summary.loc['sharing_index_k16'] = ratio_interval(table['L4'] - table['rand'],
                                                          table['L1_k16'] - table['rand'], draws)
        anatomy_rows += rows_of(summary, layer=layer)
        specific[layer] = table['L1_k32'] - table['L4']
    rq1_table = pd.DataFrame(anatomy_rows)
    rq1_table.to_csv(out / 'rq1_anatomy.csv', index=False)
    shallow = pd.concat([specific[l] for l in range(1, 4)], axis=1).mean(axis=1)
    deep = pd.concat([specific[l] for l in range(9, 13)], axis=1).mean(axis=1)
    value['D2'] = dict(zip(['shallow_minus_deep', 'low', 'high'], interval(shallow - deep, draws)),
                       shallow=shallow.mean(), deep=deep.mean())

    cover, _ = load(run('rq1'), 'coverage.csv')
    coverage_rows = []
    for layer, frame in cover.groupby('layer'):
        frame = frame.sort_values('index')
        for measure in [c for c in cover if c not in {'index', 'layer'}]:
            mean, low, high = interval(frame[measure], draws)
            coverage_rows.append(dict(layer=layer, measure=measure, mean=mean, low=low, high=high))
    pd.DataFrame(coverage_rows).to_csv(out / 'rq1_coverage.csv', index=False)

    # RQ2: main budget, all layers at once.
    rq2, checks['rq2_duplicates'] = load(run('rq2'), 'predictions.csv')
    specs = pd.read_csv(run('rq2') / 'settings.csv')
    rq2 = rq2.merge(specs, on='setting_id', validate='many_to_one')
    rq2['label'] = rq2['name'] + np.where(rq2['name'] == 'L1', '_k' + rq2['k'].astype(int).astype(str), '')
    per_image = rq2.groupby(['label', 'index'])['agree'].mean().unstack('label').sort_index()
    assert len(per_image) == n and not per_image.isna().any().any()
    main_rows = [dict(setting=name, **dict(zip(['agree', 'low', 'high'], interval(per_image[name], draws))))
                 for name in per_image]
    pd.DataFrame(main_rows).to_csv(out / 'rq2_main_budget.csv', index=False)
    main = per_image.rename(columns=dict(L3_spatial='L3', L4_spatial='L4'))
    all_summary = anatomy_table(main[['rand', 'randC', 'L3', 'L4', 'L2', 'L1_k32', 'L1_k16']], draws,
                                upper='L1_k32')
    pd.DataFrame(rows_of(all_summary, layer='all')).to_csv(out / 'rq2_anatomy_all_layers.csv', index=False)
    phi_c = .5 * ((main.randC - main.rand) + (main.L4 - main.L3))
    phi_r = .5 * ((main.L3 - main.rand) + (main.L4 - main.randC))
    value['D1'] = dict(zip(['phi_r_minus_phi_c', 'low', 'high'], interval(phi_r - phi_c, draws)),
                       sharing_index=all_summary.loc['sharing_index'].tolist())

    # Pet: composition (preregistered), union (preregistered for DINO), deletion.
    comp, _ = load(run('pet'), 'composition.csv')
    deit_comp, _ = load(deit_run('pet'), 'composition.csv')
    comp = pd.concat([deit_comp, comp])
    comp['sink_share'] = comp['sink'] / comp[['sink', 'foreground', 'background']].sum(axis=1).where(
        lambda total: total > 0)
    comp['fg_share'] = fg_share(comp)
    pet_ids = np.sort(comp.query("model == 'dino'")['index'].unique())
    pet_draws = resample_indices(len(pet_ids))
    comp_rows = []
    for (model, name, layer), frame in comp.groupby(['model', 'spec', 'layer']):
        frame = frame.set_index('index').loc[pet_ids]
        row = dict(model=model, spec=name, layer=layer, sink_share=frame['sink_share'].mean())
        row['fg_share'], row['fg_share_low'], row['fg_share_high'] = interval(frame['fg_share'].fillna(0),
                                                                              pet_draws)
        comp_rows.append(row)
    composition = pd.DataFrame(comp_rows)
    composition.to_csv(out / 'pet_composition.csv', index=False)
    deep_sink = composition.query('layer >= 9').groupby(['spec', 'model'])['sink_share'].mean().unstack()
    value['D6'] = deep_sink.to_dict(orient='index')

    union, _ = load(run('pet'), 'union_composition.csv')
    deit_union, _ = load(deit_run('pet_union'), 'union_composition.csv')
    union = pd.concat([deit_union, union])
    union['fg_share'] = fg_share(union)
    union_rows = []
    for (model, name, layer), frame in union.groupby(['model', 'spec', 'layer']):
        frame = frame.set_index('index').loc[pet_ids]
        share, size = interval(frame['fg_share'].fillna(0), pet_draws), interval(frame['union_size'], pet_draws)
        union_rows.append(dict(model=model, spec=name, layer=layer, fg_share=share[0], fg_share_low=share[1],
                               fg_share_high=share[2], union_size=size[0], union_size_low=size[1],
                               union_size_high=size[2], sink=frame['sink'].mean()))
    pd.DataFrame(union_rows).to_csv(out / 'pet_union_composition.csv', index=False)
    deep_fg = (union.query("spec == 'm25_k25_mita' and layer >= 9").assign(fg_share=lambda f: f.fg_share.fillna(0))
               .groupby(['model', 'index'])['fg_share'].mean().unstack('model').loc[pet_ids])
    value['D3'] = dict(zip(['dino_minus_deit', 'low', 'high'], interval(deep_fg['dino'] - deep_fg['deit'], pet_draws)),
                       dino=deep_fg['dino'].mean(), deit=deep_fg['deit'].mean(), mita=deep_fg['mita'].mean())
    layer12 = union.query("model == 'dino' and spec == 'm25_k25_mita' and layer == 12").set_index('index')
    value['D4'] = dict(zip(['union_size', 'low', 'high'], interval(layer12.loc[pet_ids, 'union_size'], pet_draws)))

    deletion, _ = load(run('pet'), 'deletion.csv')
    deletion_rows = []
    for category, frame in deletion.query("kind != 'baseline'").groupby('category'):
        target = frame.query("kind == 'target'").set_index('index')
        random = frame.query("kind == 'random'").groupby('index')['agree'].mean()
        eligible = target.index[target.eligible == 1]
        diff = (target.loc[eligible, 'agree'] - random.loc[eligible]).sort_index()
        mean, low, high = interval(diff, resample_indices(len(diff)))
        deletion_rows.append(dict(model='dino', category=category, n=len(diff),
                                  target=target.loc[eligible, 'agree'].mean(), random=random.loc[eligible].mean(),
                                  target_minus_random=mean, low=low, high=high))
    baseline = deletion.query("kind == 'baseline'")
    deletion_rows.append(dict(model='dino', category='baseline', n=len(baseline), target=baseline.agree.mean()))
    deletion_table = pd.DataFrame(deletion_rows)
    deletion_table.to_csv(out / 'pet_deletion.csv', index=False)
    effect = deletion_table.set_index('category')
    value['D5'] = dict(foreground=effect.loc['foreground', ['target_minus_random', 'low', 'high']].tolist(),
                       background=effect.loc['background', ['target_minus_random', 'low', 'high']].tolist())

    # DeiT-S reference values from the main V2 analysis, same images and procedures.
    deit_all = pd.read_csv(deit_run('analysis') / 'rq2_anatomy_all_layers.csv').assign(model='deit')
    dino_all = pd.DataFrame(rows_of(all_summary, layer='all')).assign(model='dino')
    pd.concat([deit_all, dino_all]).to_csv(out / 'compare_rq2_anatomy_all_layers.csv', index=False)
    deit_rq1 = pd.read_csv(deit_run('analysis') / 'rq1_anatomy.csv').assign(model='deit')
    pd.concat([deit_rq1, rq1_table.assign(model='dino')]).to_csv(out / 'compare_rq1_anatomy.csv', index=False)

    verdict = lambda low, high, threshold=0: 'supported' if low > threshold else (
        'not supported' if high < threshold else 'uncertain')
    holds = {
        'D1': 'supported' if value['D1']['low'] > 0 and value['D1']['sharing_index'][1] >= .8 else (
            'not supported' if value['D1']['high'] < 0 or value['D1']['sharing_index'][2] < .8 else 'uncertain'),
        'D2': verdict(value['D2']['low'], value['D2']['high']),
        'D3': verdict(value['D3']['low'], value['D3']['high']),
        'D4': 'supported' if value['D4']['high'] <= 100 else ('not supported' if value['D4']['low'] > 100
                                                               else 'uncertain'),
        'D5': ('not supported' if value['D5']['foreground'][1] > -.20
               or abs(value['D5']['background'][0]) >= abs(value['D5']['foreground'][0])
               else 'supported' if value['D5']['foreground'][2] < -.20 else 'uncertain'),
        'D6': 'supported' if all(row['dino'] < row['deit'] for row in value['D6'].values()) else 'not supported',
    }
    predictions = {key: dict(value=value[key], verdict=holds[key]) for key in holds}
    write = lambda name, obj: (out / name).write_text(json.dumps(obj, indent=2, default=float) + '\n',
                                                      encoding='utf-8')
    write('predictions_check.json', predictions)
    write('checks.json', checks)
    print(json.dumps(checks, indent=2, default=float))
    print(json.dumps(predictions, indent=2, default=float))


if __name__ == '__main__':
    main()
