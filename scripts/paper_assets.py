"""Tables and figures of the course paper (paper/), built only from result files.

- paper/tables/main.tex: ImageNet-S (ClickMe, PartImageNet) comparison of attention readouts and gradient methods
  on SRG, same-ring enrichment, ClickMe image-specific Spearman and animal head enrichment, three models;
- paper/tables/srg_all.tex (appendix): SRG for every dataset and model, with dense attention;
- paper/figures/execution.pdf: an example image with single-layer probe maps, the execution k sweep and
  single-layer agreement;
- paper/figures/layers.pdf: ClickMe and head enrichment per layer for three readouts, and high-norm reading.

Usage: python scripts/paper_assets.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from mita_probe.plotting import BLUE, GRAY, INK, MODEL_LABELS, ORANGE, QUIET

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments'
V3 = EXP / '2026-10-29_v3_'
SUP = EXP / '2026-10-31_s_analysis'
OUT = ROOT / 'paper'
MODELS = ['deit', 'dino', 'deit_b']
ANIMALS = ['Quadruped', 'Biped', 'Fish', 'Bird', 'Snake', 'Reptile']
ROWS = [('probe_mass', 'Probe (top-$k$ sets)'), ('attention_mass', 'Dense attention received'),
        ('cls_attention', 'CLS attention'), ('rollout', 'Rollout'), ('gae', 'GAE$^\\dagger$'),
        ('gradcam', 'Grad-CAM$^\\dagger$'), ('integrated_gradients', 'Integrated Gradients$^\\dagger$'),
        ('centre', 'Centre Gaussian (reference)')]
ATTENTION = ['probe_mass', 'attention_mass', 'cls_attention', 'rollout']
FAMILIES = {'probe': (BLUE, '-', 'Probe (top-$k$)'), 'attention': (INK, '--', 'Dense attention'),
            'cls': (ORANGE, ':', 'CLS attention')}


def band(ax, x, frame, color, label=None, style='-'):
    """Thin line with markers for frame['mean'] and a shaded 95% band (sized for a one-column paper figure)."""
    ax.plot(x, frame['mean'], style, color=color, lw=1.1, marker='o', ms=1.8, label=label)
    ax.fill_between(x, frame['low'], frame['high'], color=color, alpha=.15, lw=0)


def v3_values():
    """{(measure, model): Series indexed by method} from the v3 analysis tables."""
    out = {}
    for model in MODELS:
        faith = pd.read_csv(Path(f'{V3}e4_imagenet_s_{model}') / 'analysis_faithfulness.csv')
        out[('srg', model)] = faith.query("measure == 'srg_agree'").set_index('method')['mean']
        plaus = pd.read_csv(Path(f'{V3}e4_imagenet_s_{model}') / 'analysis_plausibility.csv')
        out[('ring', model)] = plaus.query("measure == 'enrichment'").set_index('method')['mean']
        human = pd.read_csv(Path(f'{V3}e7_clickme_{model}') / 'analysis_alignment.csv')
        out[('human', model)] = human.query("measure == 'specific_pixel'").set_index('method')['mean']
        parts = pd.read_csv(Path(f'{V3}e8_parts_{model}') / 'analysis_parts.csv').query("measure == 'enrichment'")
        heads = parts[parts.supercategory.isin(ANIMALS) & (parts.name == 'Head')]
        # The n-weighted mean over animal super-categories is the mean over animal images.
        out[('head', model)] = heads.groupby('method').apply(lambda g: (g['mean'] * g.n).sum() / g.n.sum(),
                                                             include_groups=False)
    dense = pd.read_csv(SUP / 's1_comparisons.csv')
    names = {'srg_agree': 'srg', 'enrichment': 'ring', 'specific_pixel': 'human', 'head_enrichment': 'head'}
    for row in dense[dense.measure.isin(names) & dense.dataset.isin(['imagenet_s', 'clickme', 'partimagenet'])].itertuples():
        out[(names[row.measure], row.model)]['attention_mass'] = row.attention
    return out


def main_table(values):
    measures = [('srg', 'SRG $\\uparrow$'), ('ring', 'Same-ring enrichment $\\uparrow$'),
                ('human', 'ClickMe image-specific $\\rho$ $\\uparrow$'), ('head', 'Head enrichment')]
    lines = ['\\begin{tabular}{l' + 'ccc' * len(measures) + '}', '\\toprule',
             ' & ' + ' & '.join(f'\\multicolumn{{3}}{{c}}{{{title}}}' for _, title in measures) + ' \\\\',
             ' '.join(f'\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(measures))),
             'Method & ' + ' & '.join(['S', 'D', 'B'] * len(measures)) + ' \\\\', '\\midrule']
    best = {(m, model): max(values[(m, model)][a] for a in ATTENTION) for m, _ in measures for model in MODELS}
    for method, label in ROWS:
        cells = []
        for measure, _ in measures:
            for model in MODELS:
                value = values[(measure, model)].get(method, np.nan)
                text = f'{value:.2f}' if measure == 'head' else f'{value:.3f}'
                text = text.lstrip('-') if float(text) == 0 else text
                cells.append(f'\\textbf{{{text}}}' if method in ATTENTION and value == best[(measure, model)] else text)
        if method == 'gae':
            lines.append('\\midrule')
        if method == 'centre':
            lines.append('\\midrule')
        lines.append(f'{label} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    (OUT / 'tables' / 'main.tex').write_text('\n'.join(lines) + '\n')


def srg_table():
    dense = pd.read_csv(SUP / 's1_comparisons.csv').query("measure == 'srg_agree'")
    combos = [(d, m) for d in ['imagenet_s', 'hard_imagenet'] for m in MODELS] + [
        (d, m) for d in ['pet', 'imagenet'] for m in ['deit', 'dino']]
    names = {'imagenet_s': 'ImageNet-S', 'hard_imagenet': 'Hard ImageNet', 'pet': 'Pet', 'imagenet': 'ImageNet'}
    short = {'deit': 'S', 'dino': 'D', 'deit_b': 'B'}
    lines = ['\\begin{tabular}{l' + 'c' * len(combos) + '}', '\\toprule',
             'Method & ' + ' & '.join(f'{names[d]} {short[m]}' for d, m in combos) + ' \\\\', '\\midrule']
    table = {}
    for d, m in combos:
        faith = pd.read_csv(Path(f'{V3}e4_{d}_{m}') / 'analysis_faithfulness.csv').query("measure == 'srg_agree'")
        table[(d, m)] = faith.set_index('method')['mean']
        table[(d, m)]['attention_mass'] = dense.query('dataset == @d and model == @m').attention.item()
    for method, label in ROWS + [('random', 'Random (reference)')]:
        lines.append(f'{label} & ' + ' & '.join(f'{table[c][method]:.3f}' for c in combos) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    (OUT / 'tables' / 'srg_all.tex').write_text('\n'.join(lines) + '\n')


def execution_figure():
    gallery = np.load(Path(f'{V3}gallery_deit') / 'gallery_imagenet_s.npz')
    sweep = pd.read_csv(EXP / '2026-10-27_rq1_colab_combined' / 'summary.csv')
    main = pd.read_csv(EXP / '2026-10-29_v3_deit_b_analysis' / 'compare_rq2_anatomy_all_layers.csv')
    single = {'deit': EXP / '2026-10-28_v2_analysis', 'dino': EXP / '2026-10-28_v2_dino_analysis',
              'deit_b': EXP / '2026-10-29_v3_deit_b_analysis'}
    fig = plt.figure(figsize=(6.75, 1.95))
    grid = GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 0.25, 2.3, 0.35, 2.3], wspace=0.08, hspace=0.32)
    n = 3
    panels = [('Image', None), ('Layer 6', 5), ('Layer 9', 8), ('Layer 12', 11)]
    for i, (title, layer) in enumerate(panels):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        ax.imshow(gallery['rgb'][n])
        if layer is not None:
            ax.imshow(np.kron(gallery['probe_layers'][n, layer].reshape(14, 14), np.ones((16, 16))), cmap='magma',
                      alpha=.7)
        ax.set_title(title, fontsize=5, pad=1.5)
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
    ax = fig.add_subplot(grid[:, 3])
    for method, color, label in [('grid', BLUE, 'Probe sets, $k$ sweep (DeiT-S)'), ('random', GRAY, 'Random sets, $k$ sweep')]:
        part = sweep.query('method == @method').sort_values('k')
        ax.plot(part.k, part.agree, 'o-', color=color, lw=1.2, ms=2, label=label)
        ax.fill_between(part.k, part.agree_ci_low, part.agree_ci_high, color=color, alpha=.15, lw=0)
    for model, marker in [('deit', 'o'), ('dino', 's'), ('deit_b', '^')]:
        row = main.query("model == @model and quantity in ['L3', 'rand']").set_index('quantity')['mean']
        ax.plot(16, row['L3'], marker, color=INK, ms=3.5, mec='white', mew=0.5, label=f'Main setting, {MODEL_LABELS[model]}')
        ax.plot(16, row['rand'], marker, color=GRAY, ms=3.5, mec='white', mew=0.5)
    ax.set(xscale='log', xticks=[4, 8, 16, 32, 64, 128], xticklabels=[4, 8, 16, 32, 64, 128], ylim=(0, 1),
           xlabel='$k$ tokens read per region', ylabel='Top-1 agreement', title='(b) All 12 layers executed')
    ax.legend(fontsize=4.5, loc='lower right')
    ax = fig.add_subplot(grid[:, 5])
    for (model, folder), style in zip(single.items(), ['-', '--', ':']):
        layers = pd.read_csv(folder / 'rq1_anatomy.csv').query("quantity == 'L3'").astype({'layer': int})
        band(ax, layers.sort_values('layer').layer, layers.sort_values('layer'), BLUE, label=MODEL_LABELS[model],
             style=style)
    ax.set(xlabel='Layer executed alone ($k=16$)', ylabel='Top-1 agreement', title='(c) One layer executed',
           xticks=range(1, 13), ylim=(0.7, 1))
    ax.legend(fontsize=4.5, loc='lower left')
    fig.text(0.01, 0.98, '(a) Probe map per layer (DeiT-S)', fontsize=6, va='top')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.2)
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'figures' / 'execution.pdf')
    fig.savefig(OUT / 'figures' / 'execution.png', dpi=300)
    plt.close(fig)


def layers_figure():
    curves = pd.read_csv(SUP / 's2_layers.csv')
    s3 = pd.read_csv(SUP / 's3_layers.csv')
    fig, axes = plt.subplots(2, 4, figsize=(6.75, 2.7))
    for row, (measure, ylabel) in zip(axes, [('clickme_specific_pixel', 'ClickMe specific $\\rho$'),
                                             ('head_enrichment', 'Head enrichment')]):
        for ax, model in zip(row, MODELS):
            for family, (color, style, name) in FAMILIES.items():
                curve = curves.query('measure == @measure and model == @model and family == @family')
                band(ax, curve.sort_values('layer').layer, curve.sort_values('layer'), color, label=name, style=style)
            ax.set(xticks=range(1, 13, 2), ylabel=ylabel if model == 'deit' else None)
            if measure == 'head_enrichment':
                ax.axhline(1, color='#c3c2b7', lw=1)
                ax.set_xlabel('Layer')
            else:
                ax.set_title(MODEL_LABELS[model])
    axes[0, 0].legend(fontsize=4.5, loc='lower right')
    for family, color in [('probe', BLUE), ('attention', INK)]:
        for model, style in [('deit', '-'), ('deit_b', ':')]:
            curve = s3.query("model == @model and family == @family and measure == 'high_share'").sort_values('layer')
            axes[0, 3].plot(curve.layer, curve['mean'], style, color=color, lw=1.1, marker='o', ms=1.8,
                            label=f'{MODEL_LABELS[model]}, {FAMILIES[family][2]}')
    axes[0, 3].set(title='High-norm patches (ImageNet-S)', ylabel='Share of reading', xticks=range(1, 13, 2),
                   ylim=(0, 0.7))
    axes[0, 3].legend(fontsize=4, loc='upper left')
    for model, style in zip(MODELS, ['-', '--', ':']):
        curve = s3.query("model == @model and family == 'probe' and measure == 'high_fraction'").sort_values('layer')
        axes[1, 3].plot(curve.layer, curve['mean'], style, color=QUIET, lw=1.1, marker='o', ms=1.8,
                        label=MODEL_LABELS[model])
    axes[1, 3].set(xlabel='Layer', ylabel='Share of patches', xticks=range(1, 13, 2))
    axes[1, 3].legend(fontsize=4.5)
    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(left=0.075)
    fig.savefig(OUT / 'figures' / 'layers.pdf')
    fig.savefig(OUT / 'figures' / 'layers.png', dpi=300)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size': 5.5, 'axes.titlesize': 6, 'lines.linewidth': 1.2, 'lines.markersize': 2.5,
                         'legend.handlelength': 2.2})
    main_table(v3_values())
    srg_table()
    execution_figure()
    layers_figure()
    print('written:', sorted(str(p.relative_to(ROOT)) for p in OUT.rglob('*.*')))


if __name__ == '__main__':
    main()
