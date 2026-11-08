"""Figures 18-20 of the attention-budget study, from the analysis tables only (scripts/analyze_budget.py).

  python scripts/plot_budget.py experiments/2026-11-03_ab_analysis_formal

18 budget ledger: share of each layer's attention budget on CLS, high-norm, object, background patches;
19 object focus: deep same-ring enrichment per model and the judged paired differences (99.5% intervals);
20 sinks: was the budget on high-norm patches used (T4), and where does it go with a test-time register (T6).
"""

import sys
from pathlib import Path

import pandas as pd

from fastweight_vit.plotting import BLUE, GRAY, INK, ORANGE, QUIET, plt, save

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'deit': 'DeiT-S', 'mita': 'MiTA-DeiT-S', 'deit_t': 'DeiT-T', 'mita_t': 'MiTA-DeiT-T',
          'deit3_s': 'DeiT-III-S', 'deit_b': 'DeiT-B', 'dino': 'DINO ViT-S'}
ORDER = ['deit', 'mita', 'deit_t', 'mita_t', 'deit3_s', 'deit_b', 'dino']
PARTS = [('object', BLUE, 'Object'), ('background', '#c9c8c3', 'Background'), ('high', ORANGE, 'High-norm'),
         ('cls', INK, 'CLS'), ('register', '#1baf7a', 'Register'), ('unknown', '#eeede9', 'Boundary')]


def ledger(table, out):
    table = table[table['query'] == 'all']
    fig, axes = plt.subplots(1, 7, figsize=(14, 2.6), sharey=True)
    for ax, model in zip(axes, ORDER):
        part = table[table.model == model].sort_values('layer')
        if part.empty:
            ax.set_visible(False)
            continue
        bottom = 0
        for key, color, label in PARTS[:4] + PARTS[5:]:
            ax.bar(part.layer, part[key], bottom=bottom, color=color, width=.85, label=label)
            bottom = bottom + part[key].to_numpy()
        ax.set_title(LABELS[model])
        ax.set_xticks([1, 6, 12])
        ax.grid(False)
    axes[0].set_ylabel('Share of attention budget')
    axes[3].set_xlabel('Layer')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .86))
    save(fig, out / 'figure18_budget_ledger')


def focus(enrichment, judgements, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw=dict(width_ratios=[1, 1.3]))
    deep = enrichment[enrichment.scope == 'deep']
    for y, model in enumerate(ORDER):
        for exclude, face in (('high', BLUE), ('none', 'white')):
            row = deep[(deep.model == model) & (deep.exclude == exclude)]
            if row.empty:
                continue
            row = row.iloc[0]
            left.plot([row['low'], row['high']], [y, y], color=BLUE, lw=2)
            left.plot(row['mean'], y, 'o', color=BLUE, mfc=face, ms=6)
    left.set_yticks(range(len(ORDER)), [LABELS[m] for m in ORDER])
    left.invert_yaxis()
    left.axvline(0, color=QUIET, lw=.8)
    left.set_xlabel('Same-ring object enrichment, layers 9-12')
    left.set_title('Filled: high-norm patches excluded; open: all patches', fontsize=8)
    judged = judgements[judgements.hypothesis.isin(['H1', 'H2', 'reference'])].reset_index(drop=True)
    for y, row in judged.iterrows():
        color = GRAY if row.hypothesis == 'reference' else (BLUE if row.hypothesis == 'H1' else ORANGE)
        right.plot([row['low'], row['high']], [y, y], color=color, lw=2)
        right.plot(row['mean'], y, 'o', color=color, ms=6, mec='white')
    right.set_yticks(range(len(judged)), [f'{row.hypothesis}: {LABELS[row.a]} - {LABELS[row.b]}'
                                          for _, row in judged.iterrows()])
    right.invert_yaxis()
    right.axvline(0, color=QUIET, lw=.8)
    right.axvline(.03, color=QUIET, lw=.8, ls='--')
    right.set_xlabel('Paired difference (99.5% interval; reference 95%)')
    fig.tight_layout()
    save(fig, out / 'figure19_object_focus')


def sinks(analysis, out):
    path = analysis / 'sink_use.csv'
    use = pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()
    registers = sorted(analysis.glob('register_*.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    names = {'high': 'High-norm not read', 'c1': 'C1: as many top ordinary', 'c2': 'C2: as many, same rings',
             'mass': 'As much budget (secondary)', 'zero': 'High-norm values = 0', 'mean': 'High-norm values = mean'}
    shown = [c for c in names if not use.empty and c in set(use.condition)]
    for offset, (model, color) in enumerate((('deit', BLUE), ('deit_b', ORANGE))):
        part = use[use.model == model].set_index('condition') if not use.empty else pd.DataFrame()
        for y, condition in enumerate(shown):
            if condition not in part.index:
                continue
            row = part.loc[condition]
            yy = y + (offset - .5) * .3
            axes[0].plot([row['low'], row['high']], [yy, yy], color=color, lw=2)
            axes[0].plot(row['drop'], yy, 'o', color=color, ms=5, mec='white', label=LABELS[model] if y == 0 else None)
    axes[0].set_yticks(range(len(shown)), [names[c] for c in shown])
    axes[0].invert_yaxis()
    axes[0].axvline(0, color=QUIET, lw=.8)
    axes[0].set_xlabel('Drop in log-probability of the top-1 class (nats)')
    axes[0].legend(loc='lower right')
    if registers:
        table = pd.read_csv(registers[0])
        keep = table[table.condition.isin(['R0', 'R1', 'R2', 'R3_s0', 'R4', 'R5'])].set_index('condition')
        bottom = 0
        for key, color, label in PARTS[:5]:
            values = keep[f'L11_{key}']
            axes[1].barh(range(len(keep)), values, left=bottom, color=color, label=label)
            bottom = bottom + values.to_numpy()
        labels = {'R0': 'R0 original', 'R1': 'R1 + register only', 'R2': 'R2 found units moved',
                  'R3_s0': 'R3 random units', 'R4': 'R4 matched units', 'R5': 'R5 units zeroed'}
        axes[1].set_yticks(range(len(keep)), [labels[c] for c in keep.index])
        axes[1].invert_yaxis()
        axes[1].set_xlabel(f'Layer-11 budget, {LABELS[registers[0].stem.split("_", 1)[1]]}')
        axes[1].legend(loc='center left', fontsize=7, bbox_to_anchor=(1.0, .5))
        axes[1].grid(False)
    else:
        axes[1].set_visible(False)
    fig.tight_layout()
    save(fig, out / 'figure20_sinks')


def main():
    analysis = ROOT / sys.argv[1] if not Path(sys.argv[1]).is_absolute() else Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / 'figures'
    ledger(pd.read_csv(analysis / 'budget_layers.csv'), out)
    focus(pd.read_csv(analysis / 'enrichment.csv'), pd.read_csv(analysis / 'judgements.csv'), out)
    sinks(analysis, out)
    print('figures 18-20 written to', out)


if __name__ == '__main__':
    main()
