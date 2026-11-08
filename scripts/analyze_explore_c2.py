"""E-C2 analysis (decision B45; exploratory, post hoc): why relocation (C2) fails. Changes no registered verdict.

  python scripts/analyze_explore_c2.py <run prefix> <output folder> [figure folder]
  e.g. python scripts/analyze_explore_c2.py experiments/2026-11-07_explore_c2 experiments/2026-11-07_explore_c2_analysis

Reads <prefix>_<model>/{probe_errors.csv, mean_prediction.npz, shift.csv} and the v5 mean maps
(experiments/2026-11-05_diagnose_m1_<model>/means_crop.npz). Statistics as B10: per-image values, bootstrap over
images (1,000 resamples, seed 0), 95% percentile intervals. Writes errors.csv, mean_prediction.csv, shift.csv,
shift_bins.csv, hypotheses.csv and the appendix figure figure_v5_A_c2.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastweight_vit.stats import interval, resample_indices  # noqa: E402

MODELS = ['deit', 'deit_b', 'dino']
LABELS = {'deit': 'DeiT-S', 'deit_b': 'DeiT-B', 'dino': 'DINO ViT-S'}
COLORS = {'deit': '#2f6db5', 'deit_b': '#d9822b', 'dino': '#3a9a5b'}
METRICS = ['exact', 'top5', 'distance', 'near3x3', 'far3', 'far_share']
DEEP = [9, 10, 11, 12]
GRID = 14


def table(path):
    """Read a CSV that may have been gzip-compressed before committing (scripts/compress_tables.sh)."""
    path = Path(path)
    return pd.read_csv(path if path.exists() else path.with_name(path.name + '.gz'))


def closeness():
    rows, cols = np.divmod(np.arange(GRID * GRID), GRID)
    return -np.hypot(rows - (GRID - 1) / 2, cols - (GRID - 1) / 2)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def spearman(a, b):
    return pearson(pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy())


def summary(values, **labels):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    mean, low, high = interval(values, resample_indices(len(values)))
    return dict(labels, n=len(values), mean=mean, low=low, high=high)


def errors(prefix):
    rows = []
    for model in MODELS:
        fixed = Path(f'{prefix}_probefix_{model}/probe_errors.csv')   # B47: rerun after the float32 fix
        table = pd.read_csv(fixed) if fixed.exists() else globals()['table'](f'{prefix}_{model}/probe_errors.csv')
        for (layer, source), part in table.groupby(['layer', 'source']):
            for metric in METRICS:
                rows.append(summary(part[metric], model=model, layer=layer, source=source, metric=metric))
        for layer, part in table.groupby('layer'):
            wide = part.pivot_table(index='row', columns='source', values=METRICS)
            for metric in METRICS:
                rows.append(summary(wide[metric]['probe'] - wide[metric]['blur'], model=model, layer=layer,
                                    source='probe - blur', metric=metric))
    return pd.DataFrame(rows)


def mean_prediction(prefix):
    rows = []
    near = closeness()
    centre = (np.abs(np.divmod(np.arange(GRID * GRID), GRID)[0] - 6.5) < 3) & \
             (np.abs(np.divmod(np.arange(GRID * GRID), GRID)[1] - 6.5) < 3)
    for model in MODELS:
        per_image = np.load(f'{prefix}_{model}/mean_prediction.npz')
        means = np.load(ROOT / f'experiments/2026-11-05_diagnose_m1_{model}/means_crop.npz')
        for key in per_image.files:
            layer = int(key.removeprefix('layer'))
            values = per_image[key].astype(np.float64)                       # [images, 196]
            base = means[f'R0_layer{layer}'].astype(np.float64)

            def stats(distribution):
                return dict(max_ratio=distribution.max() * 196,
                            kl_uniform=float((distribution * np.log(distribution * 196)).sum()),
                            centre_mass=float(distribution[centre].sum()),
                            pearson_mean_map=pearson(distribution, base), spearman_mean_map=spearman(distribution, base),
                            pearson_centre=pearson(distribution, near), spearman_centre=spearman(distribution, near))

            point = stats(values.mean(0))
            draws = resample_indices(len(values))
            boot = pd.DataFrame([stats(values[draw].mean(0)) for draw in draws])
            for name, value in point.items():
                low, high = np.quantile(boot[name], [.025, .975])
                rows.append(dict(model=model, layer=layer, quantity=name, n=len(values), mean=value, low=low,
                                 high=high))
    return pd.DataFrame(rows)


def shift(prefix):
    rows, bins = [], []
    for model in MODELS:
        table = globals()['table'](f'{prefix}_{model}/shift.csv')
        for source, part in table.groupby('source'):
            rows.append(summary(part.object_r0 - part.object_moved, model=model, source=source,
                                quantity='object share moved to background'))
            rows.append(summary(part.object_r0, model=model, source=source, quantity='object share R0'))
            rows.append(summary(part.object_moved, model=model, source=source, quantity='object share after'))
            for name in ('pearson_mean', 'spearman_mean', 'pearson_centre', 'spearman_centre'):
                rows.append(summary(part[name], model=model, source=source, quantity=f'{name} of (moved - R0)'))
            for b in range(10):
                bins.append(summary(part[f'bin{b}_moved'] - part[f'bin{b}_r0'], model=model, source=source, bin=b,
                                    r0_share=part[f'bin{b}_r0'].mean()))
    return pd.DataFrame(rows), pd.DataFrame(bins)


def hypotheses(error_table, prediction_table, shift_table):
    """Descriptive verdicts of B45 (layers 9-12, all three models)."""
    out = []
    far = error_table[(error_table.source == 'probe') & (error_table.metric == 'far_share') &
                      error_table.layer.isin(DEEP)]
    above, below = (far.low > .5).all(), (far.high < .5).all()
    out.append(dict(hypothesis='H-a: probe errors are mostly far (> 3 patches)',
                    verdict='supported' if above else 'not supported' if below else 'partly supported',
                    detail=f'far share of errors, layers 9-12: {far["mean"].min():.2f}-{far["mean"].max():.2f}'))
    pred = prediction_table[prediction_table.layer.isin(DEEP)].pivot_table(
        index=['model', 'layer'], columns='quantity', values=['mean', 'low'])
    meets = (pred[('mean', 'max_ratio')] >= 2) & (pred[('mean', 'pearson_mean_map')] >= .5) & \
            (pred[('low', 'pearson_mean_map')] > 0)
    by_model = meets.groupby(level='model').all()
    verdict = 'supported' if by_model.all() else 'not supported' if not meets.any() else 'partly supported'
    side = shift_table[(shift_table.source == 'C2') & (shift_table.quantity == 'pearson_mean of (moved - R0)')]
    out.append(dict(hypothesis='H-b: the probe carries an image-independent position preference',
                    verdict=verdict,
                    detail=f'{int(meets.sum())}/{len(meets)} model-layer cells meet both conditions; '
                           f'side evidence corr(C2 - R0, mean map): ' +
                           ', '.join(f'{r.model} {r.mean:.2f} [{r.low:.2f}, {r.high:.2f}]' for r in side.itertuples())))
    return pd.DataFrame(out)


def figure(prefix, error_table, bin_table, out):
    from fastweight_vit.plotting import plt, save
    fig = plt.figure(figsize=(15, 3.4))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.55, 0.55, 1.1])
    ax = fig.add_subplot(grid[0])
    for model, color in COLORS.items():
        for source, style in (('probe', '-'), ('blur', '--')):
            part = error_table[(error_table.model == model) & (error_table.source == source) &
                               (error_table.metric == 'distance')].sort_values('layer')
            ax.plot(part.layer, part['mean'], style, color=color, marker='o', ms=3,
                    label=f'{LABELS[model]} {source}')
    ax.set_title('(a) Expected distance error (patches)', fontsize=9)
    ax.set_xlabel('Layer')
    ax.legend(fontsize=6)
    ax = fig.add_subplot(grid[1])
    for model, color in COLORS.items():
        for source, style in (('probe', '-'), ('blur', '--')):
            part = error_table[(error_table.model == model) & (error_table.source == source) &
                               (error_table.metric == 'far_share')].sort_values('layer')
            ax.plot(part.layer, part['mean'], style, color=color, marker='o', ms=3)
    ax.axhline(.5, color='#999999', lw=.8)
    ax.set_title('(b) Share of errors farther than 3 patches', fontsize=9)
    ax.set_xlabel('Layer')
    for column, model in zip((2, 3), ('deit', 'dino')):
        ax = fig.add_subplot(grid[column])
        values = np.load(f'{prefix}_{model}/mean_prediction.npz')['layer12'].mean(0) * 196
        image = ax.imshow(values.reshape(GRID, GRID), cmap='viridis')
        ax.set_title(f'(c) {LABELS[model]} L12\nmean prediction x 196', fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=.046)
    ax = fig.add_subplot(grid[4])
    for model, color in COLORS.items():
        for source, style in (('C2', '-'), ('blur', '--')):
            part = bin_table[(bin_table.model == model) & (bin_table.source == source)].sort_values('bin')
            ax.plot(part['bin'], part['mean'], style, color=color, marker='o', ms=3,
                    label=f'{LABELS[model]} {source}')
    ax.axhline(0, color='#999999', lw=.8)
    ax.set_title('(d) Change in score share by distance to centre\n(layers 9-12, ImageNet-S)', fontsize=9)
    ax.set_xlabel('Distance to centre (patches, floor)')
    ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, out / 'figure_v5_A_c2')


def main():
    prefix = str(ROOT / sys.argv[1])
    out = ROOT / sys.argv[2]
    figures = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    error_table = errors(prefix)
    prediction_table = mean_prediction(prefix)
    shift_table, bin_table = shift(prefix)
    error_table.to_csv(out / 'errors.csv', index=False)
    prediction_table.to_csv(out / 'mean_prediction.csv', index=False)
    shift_table.to_csv(out / 'shift.csv', index=False)
    bin_table.to_csv(out / 'shift_bins.csv', index=False)
    verdicts = hypotheses(error_table, prediction_table, shift_table)
    verdicts.to_csv(out / 'hypotheses.csv', index=False)
    figure(prefix, error_table, bin_table, figures)
    print(verdicts.to_string(index=False))


if __name__ == '__main__':
    main()
