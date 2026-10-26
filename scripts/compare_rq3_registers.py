"""Compare pinned DINOv2 checkpoints using saved, paired image observations."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mita_probe.io import write_json
from mita_probe.rq3_extension import bootstrap

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'


def main():
    output = EXP / '2026-10-26_rq3_register_comparison'
    output.mkdir(exist_ok=False)
    folders = {name: EXP / f'2026-10-26_rq3_{name}' for name in ['dino', 'register']}
    heat, composition, norm, per_image = {}, {}, {}, {}
    for name, folder in folders.items():
        heat[name] = pd.read_csv(folder / 'A/heatmaps.csv')
        composition[name] = pd.read_csv(folder / 'B1/composition.csv')
        norm[name] = pd.read_csv(folder / 'A/sink_summary.csv', dtype={'layer': str})
        per_image[name] = pd.read_csv(folder / 'A/heatmap_per_image.csv').set_index(['index', 'k', 'method', 'metric']).value
    merged = pd.DataFrame(per_image)
    assert merged.notna().all().all() and len(merged) == 3669*2*4*2
    contrast = []
    for (k, method, metric), frame in merged.reset_index().groupby(['k', 'method', 'metric']):
        contrast.append(dict(k=k, method=method, metric=metric, contrast='register_minus_no_register',
                             **bootstrap(frame.register - frame.dino)))
    pd.DataFrame(contrast).to_csv(output / 'heatmap_contrasts.csv', index=False)
    for name, datasets in [('heatmaps', heat), ('composition', composition), ('sink_summary', norm)]:
        pd.concat([frame.assign(model=model) for model, frame in datasets.items()]).to_csv(output / f'{name}.csv', index=False)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, k in enumerate([16, 32]):
        for model, color in [('dino', '#4c72b0'), ('register', '#c44e52')]:
            frame = norm[model]
            f = frame[(frame.k == k) & (frame.threshold == 3) & (frame.method == 'selection') & (frame.metric == 'count') & (frame.layer != '9-12')]
            x = f.layer.astype(int)
            axes[row, 0].plot(x, f['mean'], label=model, color=color)
            axes[row, 0].fill_between(x, f.low, f.high, color=color, alpha=.2)
            c = composition[model]
            for metric, style in [('sink', '-'), ('cls', ':' )]:
                f = c[(c.k == k) & (c.threshold == 3) & (c.metric == metric)]
                axes[row, 1].plot(f.layer, f['mean']*100, style, color=color, label=f'{model} {metric}')
                axes[row, 1].fill_between(f.layer, f.low*100, f.high*100, color=color, alpha=.12)
            if model == 'register':
                # Summed per-image register fractions are bootstrapped together.
                p = pd.read_csv(folders[model] / 'B1/layer_per_image.csv')
                p = p[(p.k == k) & p.metric.str.startswith('register_')].groupby(['index', 'layer']).value.sum()
                reg = pd.DataFrame([dict(layer=l, **bootstrap(p.xs(l, level='layer'))) for l in range(1, 13)])
                reg.assign(k=k).to_csv(output / f'register_total_k{k}.csv', index=False)
                axes[row, 1].plot(reg.layer, reg['mean']*100, '--', color=color, label='register total')
                axes[row, 1].fill_between(reg.layer, reg.low*100, reg.high*100, color=color, alpha=.12)
        for mi, method in enumerate(['selection', 'cls', 'gaussian', 'random']):
            for model, shift, color in [('dino', -.18, '#4c72b0'), ('register', .18, '#c44e52')]:
                f = heat[model]
                r = f[(f.k == k) & (f.method == method) & (f.metric == 'pointing')].iloc[0]
                axes[row, 2].bar(mi+shift, r['mean']*100, width=.35, color=color,
                    yerr=np.array([[r['mean']-r.low], [r.high-r['mean']]])*100, capsize=3)
        axes[row, 2].set_xticks(range(4), ['selection', 'CLS', 'Gaussian', 'random'])
        axes[row, 2].set_ylim(0, 100)
        axes[row, 0].set_title(f'k={k}: high-norm patches per image')
        axes[row, 1].set_title(f'k={k}: reading allocation (%)')
        axes[row, 2].set_title(f'k={k}: pointing game (%)')
        axes[row, 0].legend(); axes[row, 1].legend(fontsize=7)
    for ext in ['png', 'pdf']:
        fig.savefig(output / f'register_comparison.{ext}', dpi=170, bbox_inches='tight')
    plt.close(fig)
    metadata = {name: json.loads((folder / 'metadata.json').read_text(encoding='utf-8')) for name, folder in folders.items()}
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'checks.json', dict(paired_images=3669, same_geometry=metadata['dino']['preprocessing'] == metadata['register']['preprocessing'],
        prefixes=[metadata[name]['num_prefix_tokens'] for name in ['dino', 'register']]))
    (output / 'log.txt').write_text('B5 comparison completed from saved CSV; bootstrap=1000, seed=0.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
