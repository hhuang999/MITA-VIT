"""CPU analysis of the V2 stages: tables, v1 regression and the frozen predictions P1-P10."""

import argparse
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import pandas as pd

from fastweight_vit.results import load, rows_of, seed_mean
from fastweight_vit.stats import anatomy_table, interval, ratio_interval, resample_indices

ROOT = Path(__file__).resolve().parents[1]
V1_SUMMARY = ROOT / 'experiments/2026-10-27_rq1_colab_combined/summary.csv'
V1_RQ2 = ROOT / 'experiments/2026-10-26_rq2_imagenet/results.csv'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'experiments')
    parser.add_argument('--prefix', default='2026-10-28_v2_')
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/2026-10-28_v2_analysis')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    run = lambda stage: args.runs / f'{args.prefix}{stage}'
    checks, predictions = {}, {}

    noise, checks['noise_duplicates'] = load(run('noise'), 'predictions.csv')
    n = noise['index'].nunique()
    draws = resample_indices(n)
    noise_rows = []
    for model, frame in noise.groupby('model'):
        frame = frame.sort_values('index')
        noise_rows.append(dict(model=model, n=len(frame), **dict(zip(['agree', 'low', 'high'],
                          interval(frame.agree, draws))), kl=frame.kl.mean()))
    noise_table = pd.DataFrame(noise_rows)
    noise_table.to_csv(out / 'noise_floor.csv', index=False)
    floor = noise_table.set_index('model')['agree']

    # RQ1: single-layer anatomy.
    rq1, checks['rq1_duplicates'] = load(run('rq1'), 'predictions.csv')
    anatomy_rows, kl_rows = [], []
    for layer, frame in rq1.groupby('layer'):
        table = seed_mean(frame, ['setting'])
        assert len(table) == n and not table.isna().any().any()
        summary = anatomy_table(table, draws, upper='L1_k32')
        sharing_k16 = ratio_interval(table['L4'] - table['rand'], table['L1_k16'] - table['rand'], draws)
        summary.loc['sharing_index_k16'] = sharing_k16
        anatomy_rows += rows_of(summary, layer=layer)
        kl = frame.groupby('setting')['kl'].mean()
        kl_rows += [dict(layer=layer, setting=s, kl=v) for s, v in kl.items()]
    rq1_table = pd.DataFrame(anatomy_rows)
    rq1_table.to_csv(out / 'rq1_anatomy.csv', index=False)
    pd.DataFrame(kl_rows).to_csv(out / 'rq1_kl.csv', index=False)
    value = lambda layer, quantity: rq1_table.query('layer == @layer and quantity == @quantity')['mean'].item()

    cover, checks['coverage_duplicates'] = load(run('rq1'), 'coverage.csv')
    measures = [c for c in cover if c not in {'index', 'layer'}]
    coverage_rows = []
    for layer, frame in cover.groupby('layer'):
        frame = frame.sort_values('index')
        for measure in measures:
            mean, low, high = interval(frame[measure], draws)
            coverage_rows.append(dict(layer=layer, measure=measure, mean=mean, low=low, high=high))
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(out / 'rq1_coverage.csv', index=False)

    # RQ2: all-layer frontier.
    rq2, checks['rq2_duplicates'] = load(run('rq2'), 'predictions.csv')
    specs = pd.read_csv(run('rq2') / 'settings.csv')
    rq2 = rq2.merge(specs, on='setting_id', validate='many_to_one')
    group = ['name', 'mode', 'm_side', 'k', 'routing', 'cls', 'scoring', 'summary_scale', 'landmark_mode']
    rq2['correct'] = (rq2['executed'] == rq2['target']).astype(float)
    per_image = rq2.groupby(group + ['index'], dropna=False).agg(
        agree=('agree', 'mean'), kl=('kl', 'mean'), correct=('correct', 'mean')).reset_index()
    frontier_rows = []
    for key, frame in per_image.groupby(group, dropna=False):
        frame = frame.sort_values('index')
        assert len(frame) == n
        mean, low, high = interval(frame.agree, draws)
        frontier_rows.append(dict(zip(group, key), agree=mean, low=low, high=high, kl=frame.kl.mean(),
                                  accuracy=frame.correct.mean()))
    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(out / 'rq2_frontier.csv', index=False)

    config_k = sorted(specs['k'].dropna().unique().astype(int))
    config_k = [k for k in config_k if k != 25]

    def all_layer(name, k=16, m_side=4):
        frame = per_image.query('name == @name and k == @k and m_side == @m_side').sort_values('index')
        assert len(frame) == n, (name, k, m_side)
        return frame.set_index('index')['agree']

    main = pd.DataFrame(dict(rand=all_layer('rand'), randC=all_layer('randC'), L3=all_layer('L3_spatial'),
                             L4=all_layer('L4_spatial'), L2=all_layer('L2'), L1_k32=all_layer('L1', k=32),
                             L1_k16=all_layer('L1')))
    all_summary = anatomy_table(main, draws, upper='L1_k32')
    pd.DataFrame(rows_of(all_summary, layer='all')).to_csv(out / 'rq2_anatomy_all_layers.csv', index=False)

    def paired(name_a, name_b, k=16, m_side=4, k_b=None, m_b=None):
        a = all_layer(name_a, k, m_side)
        b = all_layer(name_b, k if k_b is None else k_b, m_side if m_b is None else m_b)
        return interval(a - b, draws)

    contrasts = [('sharing cost: L3 region - L1 per-query', 'L3_spatial', 'L1', {}),
                 ('readability cost: L3 region - L3 dot', 'L3_spatial', 'L3_dot', {}),
                 ('CLS own - CLS routed (L3)', 'L3_spatial', 'ablate_cls_routed_L3', {}),
                 ('CLS own - CLS routed (L4)', 'L4_spatial', 'ablate_cls_routed_L4', {}),
                 ('attention - query scoring (L3)', 'ablate_attention_scoring_L3', 'L3_spatial', {}),
                 ('attention - query scoring (L4)', 'ablate_attention_scoring_L4', 'L4_spatial', {}),
                 ('standard - MiTA summary scale (L2)', 'ablate_standard_scale_L2', 'L2', {}),
                 ('standard - MiTA summary scale (L4)', 'ablate_standard_scale_L4', 'L4_spatial', {}),
                 ('MiTA default on DeiT: region/own - dot/routed', 'mita_default_spatial', 'mita_default_dot',
                  dict(k=25, m_side=5))]
    contrasts += [(f'summaries added: L4 - L3 (k={k})', 'L4_spatial', 'L3_spatial', dict(k=k))
                  for k in config_k]
    contrast_rows = []
    for label, a, b, kw in contrasts:
        mean, low, high = paired(a, b, **kw)
        contrast_rows.append(dict(contrast=label, difference=mean, low=low, high=high))
    pd.DataFrame(contrast_rows).to_csv(out / 'rq2_contrasts.csv', index=False)

    # v1 regression: L3, dot routing, CLS routed must reproduce the v1 grid line exactly.
    v1 = pd.read_csv(V1_SUMMARY).query("method == 'grid'").set_index('k')['agree']
    regression = frontier.query("name == 'v1_regression'").set_index('k')['agree']
    checks['v1_regression_max_abs_diff'] = float((regression - v1.loc[regression.index]).abs().max())

    # RQ1 layer-12 CLS check against v1 RQ2 (k = 8, layer 12).
    layer12 = rq1.query('layer == 12')
    v1_layer12 = pd.read_csv(V1_RQ2).query('k == 8 and layer == 12').iloc[0]
    routed = layer12.query("setting == 'L3_k8_v1_cls_routed'")
    own = layer12.query("setting == 'L3_k8_cls_own'")
    checks['v1_layer12_k8_agree_diff'] = float(routed.agree.mean() - v1_layer12['agree'])
    checks['v1_layer12_k8_kl_diff'] = float(routed.kl.mean() - v1_layer12['kl'])

    # MiTA reproduction and all-layer anatomy of MiTA-DeiT-S.
    repro, _ = load(run('mita_repro'), 'predictions.csv')
    checks['mita_top1'] = float(repro.correct.mean() * 100)
    checks['mita_images'] = int(len(repro))
    mita, checks['mita_duplicates'] = load(run('mita_anatomy'), 'predictions.csv')
    mita_table = seed_mean(mita, ['setting']).rename(columns={'L4_frozen': 'L4'})
    mita_summary = anatomy_table(mita_table, draws, upper='L1_k25')
    pd.DataFrame(rows_of(mita_summary, model='mita')).to_csv(out / 'mita_anatomy.csv', index=False)
    mita_l3_l2 = interval(mita_table['L3'] - mita_table['L2'], draws)

    # Pet: composition and matched deletion.
    comp, _ = load(run('pet'), 'composition.csv')
    comp['spatial'] = comp[['sink', 'foreground', 'background']].sum(axis=1)
    comp['fg_share'] = comp['foreground'] / comp['spatial'].where(comp['spatial'] > 0)
    pet_ids = np.sort(comp['index'].unique())
    pet_draws = resample_indices(len(pet_ids))
    comp_rows = []
    for (model, spec, layer), frame in comp.groupby(['model', 'spec', 'layer']):
        frame = frame.set_index('index').loc[pet_ids]
        row = dict(model=model, spec=spec, layer=layer)
        for column in ['cls', 'sink', 'foreground', 'background', 'unknown', 'fg_area']:
            row[column] = frame[column].mean()
        row['fg_share'], row['fg_share_low'], row['fg_share_high'] = interval(frame['fg_share'].fillna(0), pet_draws)
        comp_rows.append(row)
    composition = pd.DataFrame(comp_rows)
    composition.to_csv(out / 'pet_composition.csv', index=False)

    deletion, _ = load(run('pet'), 'deletion.csv')
    deletion_rows = []
    for (model, category), frame in deletion.query("kind != 'baseline'").groupby(['model', 'category']):
        target = frame.query("kind == 'target'").set_index('index')
        random = frame.query("kind == 'random'").groupby('index')['agree'].mean()
        eligible = target.index[target.eligible == 1]
        diff = (target.loc[eligible, 'agree'] - random.loc[eligible]).sort_index()
        local = resample_indices(len(diff))
        mean, low, high = interval(diff, local)
        deletion_rows.append(dict(model=model, category=category, n=len(diff),
                                  target=target.loc[eligible, 'agree'].mean(), random=random.loc[eligible].mean(),
                                  target_minus_random=mean, low=low, high=high))
    for model, frame in deletion.query("kind == 'baseline'").groupby('model'):
        deletion_rows.append(dict(model=model, category='baseline', n=len(frame), target=frame.agree.mean()))
    pd.DataFrame(deletion_rows).to_csv(out / 'pet_deletion.csv', index=False)

    # Post-hoc (decision A12): union-based composition, each token counted once per layer.
    union_dir = run('pet_union')
    if any(union_dir.glob('union_composition.csv*')):
        union_comp, _ = load(union_dir, 'union_composition.csv')
        union_comp['fg_share'] = union_comp['foreground'] / union_comp[['foreground', 'background', 'sink']].sum(axis=1)
        union_rows = []
        for (model, spec, layer), frame in union_comp.groupby(['model', 'spec', 'layer']):
            frame = frame.set_index('index').loc[pet_ids]
            mean, low, high = interval(frame['fg_share'].fillna(0), pet_draws)
            size, size_low, size_high = interval(frame['union_size'], pet_draws)
            union_rows.append(dict(model=model, spec=spec, layer=layer, fg_share=mean, fg_share_low=low,
                                   fg_share_high=high, union_size=size, union_size_low=size_low,
                                   union_size_high=size_high, sink=frame['sink'].mean()))
        pd.DataFrame(union_rows).to_csv(out / 'pet_union_composition_post_hoc.csv', index=False)

    # Frozen predictions (design v2 section 8).
    mean_of = lambda layers, quantity: np.mean([value(l, quantity) for l in layers])
    union = coverage.query("measure == 'L3_spatial_k16_union'").set_index('layer')['mean']
    patch_cov = coverage.query("measure == 'L3_spatial_k16_patch'").set_index('layer')['mean']
    l3_single = rq1_table.query("quantity == 'L3'").set_index('layer')['mean']
    get = lambda name, **kw: frontier.query(' and '.join(f'{k} == {v!r}' for k, v in
                                                          dict(name=name, **kw).items()))['agree'].item()
    deep = lambda model: composition.query("model == @model and spec == 'm25_k25_clean' and layer >= 9")
    predictions = {
        'P1': dict(value=dict(si_1_3=mean_of(range(1, 4), 'sharing_index'), si_4_12=mean_of(range(4, 13), 'sharing_index'),
                              union_1_3=union.loc[1:3].mean(), union_4_12=union.loc[4:12].mean())),
        'P2': dict(value=dict(phi_c_9_12=mean_of(range(9, 13), 'phi_C'), phi_r_9_12=mean_of(range(9, 13), 'phi_R'))),
        'P3': dict(value=dict(spatial_minus_dot_L3=get('L3_spatial', k=16) - get('L3_dot', k=16),
                              spatial_minus_dot_L4=get('L4_spatial', k=16) - get('L4_dot', k=16))),
        'P4': dict(value=dict(kl_routed=routed.kl.mean(), kl_own=own.kl.mean(),
                              agree_routed=routed.agree.mean(), agree_own=own.agree.mean())),
        'P5': dict(value=dict(l4_minus_l3_k16=get('L4_spatial', k=16) - get('L3_spatial', k=16),
                              l4_minus_l3_k128=get('L4_spatial', k=128) - get('L3_spatial', k=128))),
        'P6': dict(value=dict(L1_k16=get('L1', k=16))),
        'P7': dict(value=dict(coverage_rank_layer1=int(patch_cov.rank().loc[1]),
                              agreement_rank_layer1=int(l3_single.rank(ascending=False).loc[1]))),
        'P8': dict(value=dict(fg_share_mita=deep('mita').fg_share.mean(), fg_share_deit=deep('deit').fg_share.mean(),
                              sink_mita=deep('mita').sink.mean() / deep('mita')[['sink', 'foreground', 'background']].sum(axis=1).mean())),
        'P9': dict(value=dict(l3_minus_l2=mita_l3_l2[0], low=mita_l3_l2[1], high=mita_l3_l2[2])),
        'P10': dict(value=dict(attention_minus_query_L3=get('ablate_attention_scoring_L3') - get('L3_spatial', k=16))),
    }
    v = {key: item['value'] for key, item in predictions.items()}
    holds = {
        'P1': v['P1']['si_1_3'] < v['P1']['si_4_12'] and v['P1']['union_1_3'] > v['P1']['union_4_12'],
        'P2': v['P2']['phi_c_9_12'] > v['P2']['phi_r_9_12'],
        'P3': v['P3']['spatial_minus_dot_L3'] >= -0.03,
        'P4': v['P4']['kl_own'] < 0.15,
        'P5': v['P5']['l4_minus_l3_k16'] >= 0.10 and v['P5']['l4_minus_l3_k128'] < 0.02,
        'P6': v['P6']['L1_k16'] > 0.90,
        'P7': v['P7']['coverage_rank_layer1'] <= 3 and v['P7']['agreement_rank_layer1'] <= 3,
        'P8': v['P8']['fg_share_mita'] - v['P8']['fg_share_deit'] >= 0.15 and v['P8']['sink_mita'] < 0.02,
        'P9': v['P9']['low'] > 0,
        'P10': v['P10']['attention_minus_query_L3'] >= 0.02,
    }
    for key in predictions:
        predictions[key]['holds'] = bool(holds[key])
    checks['noise_floor'] = floor.to_dict()
    write = lambda name, obj: (out / name).write_text(json.dumps(obj, indent=2, default=float) + '\n', encoding='utf-8')
    write('predictions_check.json', predictions)
    write('checks.json', checks)
    print(json.dumps(checks, indent=2, default=float))
    print(json.dumps({k: p['holds'] for k, p in predictions.items()}, indent=2))


if __name__ == '__main__':
    main()
