"""Image-paired summaries and figures for the preregistered RQ3 extension."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from mita_probe.io import write_json
from mita_probe.rq3_extension import numpy_heatmap_metrics, bootstrap

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
torch.set_num_threads(2)


class Intervals:
    def __init__(self, n):
        rng = np.random.default_rng(0)
        indices = rng.integers(n, size=(1000, n))
        self.weights = np.stack([np.bincount(row, minlength=n) for row in indices]).astype('float32')

    def compute(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        finite = np.isfinite(values)
        numerator = torch.from_numpy(self.weights) @ torch.from_numpy(np.nan_to_num(values).astype('float32'))
        denominator = torch.from_numpy(self.weights) @ torch.from_numpy(finite.astype('float32'))
        means = (numerator / denominator).numpy()
        with np.errstate(invalid='ignore', divide='ignore'):
            average = np.nansum(values, 0) / finite.sum(0)
        return [dict(mean=float(average[j]), low=float(np.nanquantile(means[:, j], .025)),
                     high=float(np.nanquantile(means[:, j], .975)), n=int(finite[:, j].sum()))
                for j in range(values.shape[1])]


def savefig(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def load_data(source):
    data = dict(np.load(source / 'raw/observations.npz'))
    data['metadata'] = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    data['prefix'] = data['metadata']['num_prefix_tokens']
    data['grid'] = int(np.sqrt(data['foreground'].shape[1]))
    data['sink'] = {t: data['norms'] > t*np.median(data['norms'], -1, keepdims=True)
                    for t in [2, 3, 5]}
    data['ci'] = Intervals(len(data['foreground']))
    coordinate = np.arange(data['grid']) - (data['grid']-1)/2
    gaussian = np.exp(-(coordinate[:, None]**2+coordinate[None, :]**2)/(2*(data['grid']/4)**2))
    data['gaussian'] = np.broadcast_to(gaussian.flatten(), data['foreground'].shape)
    data['random'] = np.random.default_rng(0).random(data['foreground'].shape)
    data['area'] = (data['foreground'] & data['valid']).sum(-1) / data['valid'].sum(-1)
    data['selection'] = data['counts'][:, :, 8:12, data['prefix']:].mean(2)
    return data


def heatmaps(data, ki, removed=False, threshold=3):
    maps = {'selection': (data['selection'][:, ki], data['valid']),
            'cls': (data['cls'], data['valid']), 'gaussian': (data['gaussian'], data['valid']),
            'random': (data['random'], data['valid'])}
    if removed:
        maps['selection_no_sink'] = (data['selection'][:, ki], data['valid'] & ~data['sink'][threshold][:, 8:12].any(1))
        maps['cls_no_sink'] = (data['cls'], data['valid'] & ~data['sink'][threshold][:, 11])
    return maps


def metrics_table(data, removed=False, threshold=3, group=None):
    rows, per_image = [], []
    for ki, k in enumerate([16, 32]):
        for method, (heatmap, valid) in heatmaps(data, ki, removed, threshold).items():
            mass, hit, _ = numpy_heatmap_metrics(heatmap, data['foreground'], valid)
            for metric, values in [('fg_mass', mass), ('pointing', hit)]:
                per_image.extend(dict(index=i, k=k, method=method, metric=metric, value=v)
                                 for i, v in enumerate(values))
                groups = [('all', np.ones(len(values), dtype=bool))] if group is None else [
                    (name, group == j) for j, name in enumerate(['centered', 'middle', 'offcenter'])]
                for name, mask in groups:
                    rows.append(dict(k=k, threshold=threshold, method=method, metric=metric,
                                     group=name, **bootstrap(values[mask])))
    return pd.DataFrame(rows), pd.DataFrame(per_image)


def stage_a(source, output, data):
    n, grid, ci = len(data['foreground']), data['grid'], data['ci']
    rows, frequency, top, norms, histogram = [], [], [], [], []
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for ki, k in enumerate([16, 32]):
        for mi, (method, hm) in enumerate([('selection', data['selection'][:, ki]), ('cls', data['cls'])]):
            for transposed in [False, True]:
                fg, valid = data['foreground'], data['valid']
                if transposed:
                    fg = fg.reshape(n, grid, grid).transpose(0, 2, 1).reshape(n, -1)
                    valid = valid.reshape(n, grid, grid).transpose(0, 2, 1).reshape(n, -1)
                mass, hit, peak = numpy_heatmap_metrics(hm, fg, valid)
                for metric, values in [('fg_mass', mass), ('pointing', hit)]:
                    rows.append(dict(k=k, method=method, transposed=transposed, metric=metric, **bootstrap(values)))
            peak = numpy_heatmap_metrics(hm, data['foreground'], data['valid'])[2]
            indicator = peak[:, None] == np.arange(grid*grid)
            intervals = ci.compute(indicator)
            for token, interval in enumerate(intervals):
                frequency.append(dict(k=k, method=method, token=token, row=token//grid,
                                      col=token % grid, count=int(indicator[:, token].sum()), **interval))
            order = np.argsort(-indicator.sum(0), kind='stable')[:10]
            top.extend(dict(k=k, method=method, rank=r+1, token=int(token), row=int(token//grid),
                            col=int(token % grid), **intervals[token]) for r, token in enumerate(order))
            ax = axes[ki, mi]
            im = ax.imshow(indicator.mean(0).reshape(grid, grid)*100, cmap='magma')
            ax.set_title(f'{method}, k={k}'); fig.colorbar(im, ax=ax, label='% images')
        axes[ki, 2].axis('off')
        axes[ki, 2].text(0, .5, 'Rows and columns are zero-based.\nPeaks selected over valid patches.\nNo heatmap transpose is applied.')
    savefig(fig, output, 'peak_locations')
    for threshold, sink in data['sink'].items():
        for ki, k in enumerate([16, 32]):
            for li in range(12):
                hm = data['counts'][:, ki, li, data['prefix']:]
                peak = numpy_heatmap_metrics(hm, data['foreground'], data['valid'])[2]
                values = {'count': sink[:, li].sum(-1), 'has_sink': sink[:, li].any(-1),
                          'peak_is_sink': sink[np.arange(n), li, peak]}
                for metric, vector in values.items():
                    norms.append(dict(threshold=threshold, k=k, layer=str(li+1), method='selection',
                                      metric=metric, **bootstrap(vector)))
            for method, hm in [('selection', data['selection'][:, ki]), ('cls', data['cls'])]:
                mask = sink[:, 8:12].any(1) if method == 'selection' else sink[:, 11]
                peak = numpy_heatmap_metrics(hm, data['foreground'], data['valid'])[2]
                for metric, vector in {'count': mask.sum(-1), 'has_sink': mask.any(-1),
                                        'peak_is_sink': mask[np.arange(n), peak]}.items():
                    norms.append(dict(threshold=threshold, k=k, layer='9-12' if method == 'selection' else '12',
                                      method=method, metric=metric, **bootstrap(vector)))
    bins = np.logspace(-3, 5, 161)
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), constrained_layout=True)
    for li, ax in enumerate(axes.flat):
        hists = np.stack([np.histogram(row, bins)[0] for row in data['norms'][:, li]]) / (grid*grid)
        intervals = ci.compute(hists)
        x = np.sqrt(bins[:-1]*bins[1:])
        ax.plot(x, [v['mean'] for v in intervals]); ax.set_xscale('log')
        ax.fill_between(x, [v['low'] for v in intervals], [v['high'] for v in intervals], alpha=.2)
        ax.set_title(f'Block {li+1} input'); ax.set_xlim(1, 1000)
        for bi, interval in enumerate(intervals):
            histogram.append(dict(layer=li+1, bin_left=bins[bi], bin_right=bins[bi+1], **interval))
    fig.supxlabel('Patch L2 norm (log scale; fixed bins)'); fig.supylabel('Fraction of patches per bin')
    savefig(fig, output, 'norm_histograms')
    pd.DataFrame(rows).to_csv(output / 'orientation.csv', index=False)
    pd.DataFrame(frequency).to_csv(output / 'peak_frequencies.csv', index=False)
    pd.DataFrame(top).to_csv(output / 'peak_top10.csv', index=False)
    pd.DataFrame(norms).to_csv(output / 'sink_summary.csv', index=False)
    pd.DataFrame(histogram).to_csv(output / 'norm_histogram.csv', index=False)
    table, per_image = metrics_table(data)
    table.to_csv(output / 'heatmaps.csv', index=False)
    per_image.to_csv(output / 'heatmap_per_image.csv', index=False)
    pd.DataFrame([dict(metric='fg_area', **bootstrap(data['area']))]).to_csv(output / 'chance.csv', index=False)
    examples = [i for i in data['metadata']['example_indices'] if i < n]
    if examples:
        fig, axes = plt.subplots(len(examples), 4, figsize=(11, 2.6*len(examples)), squeeze=False)
        for ri, index in enumerate(examples):
            image = Image.open(source / f'example_{index}.png')
            for col, label in enumerate(['Image', 'Token foreground', 'Selection 9-12', 'CLS']):
                ax = axes[ri, col]; ax.imshow(image); ax.axis('off')
                if col:
                    hm = [data['foreground'], data['selection'][:, 0], data['cls']][col-1][index]
                    ax.imshow(hm.reshape(grid, grid), extent=(-.5, 223.5, 223.5, -.5), alpha=.5,
                              cmap='cool' if col == 1 else 'magma', interpolation='nearest')
                ax.set_title(f'{label} | {index}')
        savefig(fig, output, 'five_overlays')
    check = dict(n_images=n, histogram_outside_bins=int(((data['norms'] < bins[0]) | (data['norms'] > bins[-1])).sum()),
                 orientation='row-major, same mask geometry; transpose is diagnostic only',
                 b2_enabled=bool(data['sink'][3][:, 8:12].any((1, 2)).mean() >= .1))
    if data['prefix'] == 1 and grid == 14 and n == 3669:
        old = pd.read_csv(source.parent / '2026-10-26_rq3_pet/heatmap_predictions.csv')
        max_difference = 0.
        for k in [16, 32]:
            for method, legacy in [('selection', 'selection'), ('cls', 'cls_attention'), ('gaussian', 'center_gaussian')]:
                frame = old[(old.k == k) & (old.method == legacy)].sort_values('index')
                for metric, legacy_metric in [('fg_mass', 'fg_mass'), ('pointing', 'pointing')]:
                    new = per_image[(per_image.k == k) & (per_image.method == method) & (per_image.metric == metric)].sort_values('index')
                    max_difference = max(max_difference, float(np.max(np.abs(frame[legacy_metric].to_numpy()-new.value.to_numpy()))))
        check['legacy_max_metric_difference'] = max_difference
        assert max_difference < 1e-5
    write_json(output / 'checks.json', check)


def stage_b1(source, output, data):
    rows, per_image = [], []
    prefix, fg, valid = data['prefix'], data['foreground'], data['valid']
    for threshold, sink in data['sink'].items():
        for ki, k in enumerate([16, 32]):
            for li in range(12):
                counts = data['counts'][:, ki, li].astype(float)
                spatial = counts[:, prefix:]
                categories = {'foreground': fg & valid & ~sink[:, li],
                              'background': ~fg & valid & ~sink[:, li], 'sink': sink[:, li],
                              'unknown': ~valid & ~sink[:, li]}
                values = {name: (spatial*mask).sum(-1)/(96*k) for name, mask in categories.items()}
                values['cls'] = counts[:, 0]/(96*k)
                for register in range(1, prefix):
                    values[f'register_{register}'] = counts[:, register]/(96*k)
                assert np.allclose(sum(values.values()), 1)
                mass, hit, _ = numpy_heatmap_metrics(spatial, fg, valid)
                values.update(fg_mass=mass, pointing=hit)
                for metric, vector in values.items():
                    rows.append(dict(k=k, threshold=threshold, layer=li+1, metric=metric, **bootstrap(vector)))
                    if threshold == 3:
                        per_image.extend(dict(index=i, k=k, layer=li+1, metric=metric, value=v)
                                         for i, v in enumerate(vector))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'composition.csv', index=False)
    pd.DataFrame(per_image).to_csv(output / 'layer_per_image.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ki, k in enumerate([16, 32]):
        values = frame[(frame.k == k) & (frame.threshold == 3)]
        names = [m for m in values.metric.unique() if m not in ['fg_mass', 'pointing']]
        axes[ki, 0].stackplot(range(1, 13), *[values[values.metric == m]['mean'] for m in names], labels=names)
        axes[ki, 0].set_title(f'Reading composition, k={k}'); axes[ki, 0].set_ylim(0, 1)
        axes[ki, 0].legend(fontsize=7, ncol=3, loc='upper left')
        for metric in ['fg_mass', 'pointing']:
            f = values[values.metric == metric]
            axes[ki, 1].plot(f.layer, f['mean'], label=metric)
            axes[ki, 1].fill_between(f.layer, f.low, f.high, alpha=.2)
        axes[ki, 1].axhline(data['area'].mean(), color='gray', linestyle='--', label='chance area')
        axes[ki, 1].set_title(f'Per-layer localization, k={k}'); axes[ki, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel('Layer'); ax.set_ylabel('Fraction')
    savefig(fig, output, 'layer_composition')


def stage_b2(source, output, data):
    enabled = data['sink'][3][:, 8:12].any((1, 2)).mean() >= .1
    write_json(output / 'checks.json', dict(enabled=bool(enabled), prevalence=bootstrap(data['sink'][3][:, 8:12].any((1, 2)))))
    if not enabled:
        return
    summaries = []
    for threshold in [2, 3, 5]:
        table, values = metrics_table(data, True, threshold)
        summaries.append(table)
        if threshold == 3:
            values.to_csv(output / 'heatmap_per_image.csv', index=False)
    pd.concat(summaries).to_csv(output / 'heatmaps.csv', index=False)


def stage_b4(source, output, data):
    cuts = np.nanquantile(data['distance'], [1/3, 2/3])
    group = np.searchsorted(cuts, data['distance'], side='left')
    group[~np.isfinite(data['distance'])] = -1
    enabled = data['sink'][3][:, 8:12].any((1, 2)).mean() >= .1
    table, _ = metrics_table(data, enabled, group=group)
    table.to_csv(output / 'stratified_heatmaps.csv', index=False)
    pd.DataFrame(dict(index=np.arange(len(group)), eccentricity=data['distance'], group=group)).to_csv(output / 'groups.csv', index=False)
    write_json(output / 'checks.json', dict(cuts=cuts.tolist(), group_counts=np.bincount(group[group >= 0]).tolist(),
        missing=int((group < 0).sum()), b2_enabled=bool(enabled)))


def stage_b3(source, output):
    frame = pd.read_csv(source / 'predictions.csv')
    rows, comparisons = [], []
    for (k, scope), group in frame.groupby(['k', 'scope']):
        baseline = group[group.method == 'baseline'].set_index('index').agree.sort_index()
        average = group.groupby(['index', 'method']).agree.mean().unstack().sort_index()
        common = average.notna().all(1)
        for method in average.columns:
            values = average[method].to_numpy()
            for support, mask in [('available', np.isfinite(values)), ('common', common.to_numpy())]:
                effect = baseline.to_numpy()[mask] - values[mask]
                rows.append(dict(k=k, scope=scope, method=method, support=support,
                                 metric='agreement', **bootstrap(values[mask])))
                rows.append(dict(k=k, scope=scope, method=method, support=support,
                                 metric='drop_from_baseline', **bootstrap(effect)))
                dose = group[group.method == method].groupby('index').mean_removed_per_set.mean().sort_index()
                rows.append(dict(k=k, scope=scope, method=method, support=support,
                                 metric='removed_per_set', **bootstrap(dose.to_numpy()[mask])))
        comparisons.append(dict(k=k, scope=scope, contrast='random_minus_sink_agreement',
            **bootstrap((average.random - average.sink).to_numpy())))
        for seed, g in group[group.method == 'random'].groupby('seed'):
            rows.append(dict(k=k, scope=scope, method=f'random_seed{seed}', support='available',
                             metric='agreement', **bootstrap(g.agree)))
    summary = pd.DataFrame(rows)
    summary.to_csv(output / 'causal_summary.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output / 'causal_contrasts.csv', index=False)
    frame.groupby(['k', 'scope', 'method', 'seed']).eligible.agg(['sum', 'count']).to_csv(output / 'coverage.csv')
    assert not frame.duplicated(['index', 'k', 'scope', 'method', 'seed']).any()
    assert frame.original.groupby(frame['index']).nunique().eq(1).all()
    assert (frame.agree.notna() == frame.eligible.eq(1)).all()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    names = ['baseline', 'sink', 'foreground', 'background', 'random']
    for ki, k in enumerate([16, 32]):
        for si, scope in enumerate(['last4', 'all12']):
            f = summary[(summary.k == k) & (summary.scope == scope) & (summary.support == 'common') & (summary.metric == 'agreement')].set_index('method').loc[names]
            axes[ki, si].bar(names, f['mean']*100, yerr=np.stack([f['mean']-f.low, f.high-f['mean']])*100,
                             color=['gray', '#c44e52', '#55a868', '#4c72b0', '#8172b2'], capsize=3)
            axes[ki, si].set_title(f'k={k}, {scope}, common N={int(f.n.iloc[0])}')
            axes[ki, si].set_ylabel('Agreement (%)'); axes[ki, si].set_ylim(0, 100)
            axes[ki, si].tick_params(axis='x', rotation=20)
    savefig(fig, output, 'causal_interventions')
    write_json(output / 'checks.json', dict(n_rows=len(frame), n_images=frame['index'].nunique(),
        duplicate_rows=0, missing_agreement_matches_eligibility=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--stage', choices=['A', 'B1', 'B2', 'B3', 'B4'], required=True)
    args = parser.parse_args()
    output = args.source / args.stage
    output.mkdir(exist_ok=False)
    if args.stage == 'B3':
        stage_b3(args.source, output)
    else:
        data = load_data(args.source)
        {'A': stage_a, 'B1': stage_b1, 'B2': stage_b2, 'B4': stage_b4}[args.stage](args.source, output, data)
    (output / 'log.txt').write_text(f'Completed {args.stage}; source={args.source}; image bootstrap=1000, seed=0.\n', encoding='utf-8')
    print(f'Completed {args.stage}: {output}')


if __name__ == '__main__':
    main()
