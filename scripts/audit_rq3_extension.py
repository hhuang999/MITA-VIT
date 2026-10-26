"""Audit saved RQ3 extension artifacts without another model forward."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mita_probe.io import write_json
from mita_probe.rq3_extension import bootstrap, sink_tokens, numpy_heatmap_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    source = args.source
    checks = {}

    def check(name, passed):
        checks[name] = bool(passed)
        if not passed:
            raise AssertionError(name)

    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    original_checks = json.loads((source / 'checks.json').read_text(encoding='utf-8'))
    check('all_3669_images', metadata['n_images'] == 3669)
    check('observe_exact_identity', original_checks['max_observe_error'] == 0)
    if (source / 'raw/observations.npz').exists():
        filename = source / 'raw/observations.npz'
        data = dict(np.load(filename))
        prefix = metadata['num_prefix_tokens']
        n, layers, patches = data['norms'].shape
        check('12_blocks', layers == 12)
        check('finite_positive_norms', np.isfinite(data['norms']).all() and (data['norms'] > 0).all())
        check('spatial_shape', data['foreground'].shape == (3669, patches))
        check('prefix_shape', data['counts'].shape == (3669, 2, 12, patches+prefix))
        check('finite_cls', np.isfinite(data['cls']).all() and (data['cls'] >= 0).all())
        for ki, k in enumerate([16, 32]):
            check(f'count_conservation_{k}', (data['counts'][:, ki].sum(-1) == 96*k).all())
        for threshold in [2, 3, 5]:
            sink = data['norms'] > threshold*np.median(data['norms'], -1, keepdims=True)
            torch.set_num_threads(2)
            check(f'numpy_torch_sink_definition_{threshold}', np.array_equal(sink,
                  sink_tokens(torch.from_numpy(data['norms']), threshold).numpy()))
        if (source / 'A/heatmap_per_image.csv').exists():
            metrics = pd.read_csv(source / 'A/heatmap_per_image.csv')
            for ki, k in enumerate([16, 32]):
                hm = data['counts'][:, ki, 8:12, prefix:].mean(1)
                mass, hit, _ = numpy_heatmap_metrics(hm, data['foreground'], data['valid'])
                for key, values in [('fg_mass', mass), ('pointing', hit)]:
                    frame = metrics[(metrics.k == k) & (metrics.method == 'selection') & (metrics.metric == key)]
                    check(f'heatmap_recalculation_{k}_{key}', np.allclose(frame.value, values))
            f = pd.read_csv(source / 'A/peak_frequencies.csv')
            check('peak_frequencies_sum_one', np.allclose(f.groupby(['k', 'method'])['mean'].sum(), 1))
        if (source / 'B1/layer_per_image.csv').exists():
            f = pd.read_csv(source / 'B1/layer_per_image.csv')
            composition = f[~f.metric.isin(['fg_mass', 'pointing'])]
            check('per_image_composition_conservation', np.allclose(composition.groupby(['index', 'k', 'layer']).value.sum(), 1))
            check('all_layer_values_finite', np.isfinite(f.value).all())
            check('no_duplicate_composition', not f.duplicated(['index', 'k', 'layer', 'metric']).any())
        if (source / 'B2/heatmap_per_image.csv').exists():
            f = pd.read_csv(source / 'B2/heatmap_per_image.csv')
            check('b2_full_coverage', len(f) == 3669*2*6*2)
            check('b2_finite', np.isfinite(f.value).all())
        if (source / 'B4/groups.csv').exists():
            f = pd.read_csv(source / 'B4/groups.csv')
            check('eccentricity_recorded_all_images', len(f) == 3669)
            check('missing_eccentricity_exactly_no_fg', np.array_equal(f.group.eq(-1), ~data['foreground'].any(-1)))
    else:
        filename = source / 'predictions.csv'
        f = pd.read_csv(filename)
        check('36_settings', len(f) == 3669*36)
        check('k197_exact', original_checks['identity_error'] == 0)
        check('no_duplicate_predictions', not f.duplicated(['index', 'k', 'scope', 'method', 'seed']).any())
        check('same_reference_prediction', f.groupby('index').original.nunique().eq(1).all())
        check('only_known_infeasibility', f[f.eligible == 0].reason.eq('fewer_than_k_non_target_tokens').all())
        check('feasibility_nan_matches', f.agree.isna().eq(f.eligible == 0).all())
        for (k, scope), g in f.groupby(['k', 'scope']):
            dose = g.groupby(['index', 'method']).mean_removed_per_set.mean().unstack()
            check(f'matched_random_dose_{k}_{scope}', np.allclose(dose.random, dose.sink))
            check(f'all_baseline_eligible_{k}_{scope}', g[g.method == 'baseline'].eligible.eq(1).all())
            for method in ['sink', 'foreground', 'background']:
                p = g[g.method == method]
                check(f'dose_bounded_{k}_{scope}_{method}', p.mean_removed_per_set.between(0, k).all())
        observations = dict(np.load(source.parent / '2026-10-26_rq3_diagnosis/raw/observations.npz'))
        old = observations['original']
        check('reference_same_as_observation', np.array_equal(f.groupby('index').original.first().to_numpy(), old))
        sinks = observations['norms'] > 3*np.median(observations['norms'], -1, keepdims=True)
        fg, valid = observations['foreground'], observations['valid']
        for (k, scope, method), group in f[f.method.isin(['sink', 'foreground', 'background'])].groupby(['k', 'scope', 'method']):
            layers = np.arange(8, 12) if scope == 'last4' else np.arange(12)
            category = (sinks[:, layers] if method == 'sink' else
                        (np.broadcast_to((fg & valid)[:, None], (3669, len(layers), 196)) if method == 'foreground'
                         else (~fg & valid)[:, None] & ~sinks[:, layers]))
            eligible = (197-category.sum(-1) >= k).all(-1)
            counts = observations['counts'][:, 0 if k == 16 else 1, layers, 1:]
            dose = (counts*category).sum(-1).mean(-1)/96
            group = group.sort_values('index')
            check(f'independent_feasibility_{k}_{scope}_{method}', np.array_equal(group.eligible, eligible))
            check(f'independent_dose_{k}_{scope}_{method}', np.allclose(group.mean_removed_per_set, dose, atol=2e-6))
    check('raw_sha256_matches', hashlib.sha256(filename.read_bytes()).hexdigest() == original_checks['output_sha256'])
    write_json(source / 'audit.json', dict(passed=len(checks), checks=checks))
    print(f'{source}: {len(checks)} audit checks passed')


if __name__ == '__main__':
    main()
