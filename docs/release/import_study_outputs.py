"""Verification layer 1 (not part of the package): lay out the saved study/v5 run outputs as release outputs.

  python docs/release/import_study_outputs.py <study/v5 checkout> <outputs folder>

Copies (or converts) each study/v5 run into outputs/<config>/[<dataset>/]<model>/ with the release names, so that
scripts/make_tables.py and scripts/make_figures.py can rebuild every table and figure from the saved outputs:
- method names: blur, no_high, centre_sub -> stand_in_blur, stand_in_no_high, stand_in_centre (also inside the
  supplementary names: no_high_l<l> -> stand_in_no_high_l<l>, blurMLP -> stand_in_blur_MLP);
- diagnose: identity.csv / identity_images.csv -> probe_accuracy.csv / probe_accuracy_images.csv;
- figure 3(c): the 'attention' family of the layer-reading runs -> high_norm.csv (high_share; dense attention mass
  is R0 up to a positive factor, so the share is the same);
- figure 3(d): attention_layer<l> of the ClickMe alignment runs -> R0_l<l>;
- value-norm ratio: heads.csv of the budget runs (sums per batch, layer and head) -> high_norm.csv rows per batch
  and layer (value_ratio_sum, value_ratio_n), which make_tables sums the same way;
- top-k map similarity: recomputed from the saved float16 scores (probe_mass vs attention_mass), as before.
Some per-image tables exist only in the untracked raw/ folders of the original checkout; pass that checkout's
experiments folder with --raw.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fastweight_vit.metrics import spearman  # noqa: E402

MODELS = ['deit', 'deit_b', 'dino']
RENAME = {'blur': 'stand_in_blur', 'no_high': 'stand_in_no_high', 'centre_sub': 'stand_in_centre',
          'blurMLP': 'stand_in_blur_MLP'}


def renamed(value):
    if value in RENAME:
        return RENAME[value]
    match = re.fullmatch(r'no_high_l(\d+)', str(value))
    return f'stand_in_no_high_l{match.group(1)}' if match else value


class Source:
    def __init__(self, experiments, raw):
        self.experiments, self.raw = Path(experiments), Path(raw) if raw else None

    def find(self, run, name):
        for base in [self.experiments] + ([self.raw] if self.raw else []):
            for candidate in (base / run / name, base / run / f'{name}.gz', base / run / 'raw' / f'{name}.gz',
                              base / run / 'raw' / name):
                if candidate.exists():
                    return candidate
        return None

    def table(self, run, name):
        path = self.find(run, name)
        return None if path is None else pd.read_csv(path)


def place(out, config, model, dataset=None):
    folder = out / config / dataset / model if dataset else out / config / model
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'checks.json').write_text('{"complete": true, "imported": true}\n')
    return folder


def copy(source, run, name, folder, target=None, transform=None):
    path = source.find(run, name)
    if path is None:
        print('missing', run, name)
        return
    if transform is None and path.suffix != '.gz':
        shutil.copy(path, folder / (target or path.name))
        return
    table = pd.read_csv(path)
    if transform is not None:
        table = transform(table)
    table.to_csv(folder / f'{target or name}.gz', index=False, compression='gzip')


def methods(table):
    for column in ('method', 'source'):
        if column in table:
            table[column] = table[column].map(renamed)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('study', type=Path)
    parser.add_argument('outputs', type=Path)
    parser.add_argument('--raw', type=Path, help='experiments folder holding the untracked raw/ tables')
    args = parser.parse_args()
    source, out = Source(args.study / 'experiments', args.raw), args.outputs
    rq1 = {'deit': '2026-10-28_v2_rq1', 'dino': '2026-10-28_v2_dino_rq1', 'deit_b': '2026-10-29_v3_deit_b_rq1'}
    for model in MODELS:
        f = place(out, 'fig2a_k_sweep', model)
        copy(source, f'2026-11-05_read_ksweep_{model}', 'predictions.csv', f)
        copy(source, f'2026-11-05_read_ksweep_{model}', 'settings.csv', f)
        copy(source, f'2026-11-03_ab_t7_{model}', 'attention_off.csv', place(out, 'attention_off', model))
        copy(source, rq1[model], 'predictions.csv', place(out, 'single_layer', model),
             transform=lambda t: t[t.setting == 'L3'])
        f = place(out, 'diagnose_probe', model)
        run = f'2026-11-05_diagnose_probe_{model}'
        for name, target in (('identity.csv', 'probe_accuracy.csv'), ('identity_images.csv', 'probe_accuracy_images.csv'),
                             ('heldout.json', None), ('probes.npz', None)):
            copy(source, run, name, f, target)
        f = place(out, 'diagnose_mean_maps', model)
        for name in ('m1.csv', 'radial.csv', 'means_crop.npz', 'means_clickme.npz'):
            copy(source, f'2026-11-05_diagnose_m1_{model}', name, f)
        copy(source, f'2026-10-31_s_layers_{model}', 'layer_reading.csv', place(out, 'high_norm', model), 'high_norm.csv',
             transform=lambda t: t[t.family == 'attention'])
        copy(source, f'2026-10-31_s_e7_clickme_{model}', 'alignment.csv', place(out, 'clickme_layers', model, 'clickme'),
             transform=lambda t: t[t.method.str.startswith('attention_layer')].assign(
                 method=lambda f: 'R0_l' + f.method.str.removeprefix('attention_layer')))
        for dataset in ('imagenet_s', 'clickme'):
            f = place(out, 'table1', model, dataset)
            run = f'2026-11-05_correct_ablation_{dataset}_{model}'
            for name in ('plausibility.csv', 'faithfulness.csv', 'alignment.csv'):
                if source.find(run, name):
                    copy(source, run, name, f, transform=methods)
            for gallery in sorted((args.study / 'experiments' / run).glob('gallery_*.npz')):
                arrays = dict(np.load(gallery))
                arrays = {f'map_{renamed(key[4:])}' if key.startswith('map_') else key: value
                          for key, value in arrays.items()}
                np.savez_compressed(f / gallery.name, **arrays)
            f = place(out, 'order', model, dataset)
            for name in ('plausibility.csv', 'alignment.csv'):
                if source.find(f'2026-11-05_correct_order_{dataset}_{model}', name):
                    copy(source, f'2026-11-05_correct_order_{dataset}_{model}', name, f)
            if source.find(f'2026-11-05_correct_order_{dataset}_{model}', 'faithfulness.csv'):
                copy(source, f'2026-11-05_correct_order_{dataset}_{model}', 'faithfulness.csv', f)
            f = place(out, 'supplementary_eval', model, dataset)
            run = f'2026-11-07_supp_eval_{dataset}_{model}'
            for name in ('plausibility.csv', 'faithfulness.csv', 'alignment.csv', 'dose.csv', 'head_weights.csv'):
                if source.find(run, name):
                    copy(source, run, name, f, transform=methods if name != 'dose.csv' and name != 'head_weights.csv'
                         else None)
        copy(source, f'2026-11-05_correct_sanity_{model}', 'sanity.csv', place(out, 'sanity', model))
        copy(source, f'2026-10-29_v3_e8_parts_{model}', 'reliance.csv', place(out, 'part_reliance', model))
        f = place(out, 'c2_analysis', model)
        copy(source, f'2026-11-07_explore_c2_probefix_{model}', 'probe_errors.csv', f, transform=methods)
        copy(source, f'2026-11-07_explore_c2_{model}', 'mean_prediction.npz', f)
        copy(source, f'2026-11-07_explore_c2_{model}', 'shift.csv', f, transform=methods)
        f = place(out, 'supplementary_heldout', model)
        for name in ('mlp_probe.csv', 'global.csv'):
            copy(source, f'2026-11-07_supp_heldout_{model}', name, f)
        rho = []
        run = args.raw / f'2026-10-31_s_e4_imagenet_s_{model}' if args.raw else None
        scores = sorted((run / 'raw' / 'scores').glob('*.npz')) if run else []
        for path in scores:
            arrays = np.load(path)
            rho += spearman(torch.from_numpy(arrays['probe_mass'].astype(np.float64)),
                            torch.from_numpy(arrays['attention_mass'].astype(np.float64))).tolist()
        if rho:
            pd.DataFrame(dict(model=model, index=range(len(rho)), spearman=rho)).to_csv(
                place(out, 'map_similarity', model) / 'similarity.csv', index=False)
    for model in ('deit', 'deit_b'):
        heads = source.table(f'2026-11-03_ab_t3_{model}', 'heads.csv')
        if heads is not None:
            rows = heads.groupby(['start', 'layer'])[['value_ratio', 'value_ratio_n']].sum().reset_index()
            rows = rows.rename(columns={'start': 'index', 'value_ratio': 'value_ratio_sum'}).assign(model=model)
            rows.to_csv(place(out, 'value_norm', model) / 'high_norm.csv', index=False)
    for dataset in ('imagenet_s', 'clickme'):
        for variant, config, model in (('register', 'table1_register', 'deit'), ('mita', 'table1_mita', 'mita')):
            run = f'2026-11-05_correct_{variant}_{dataset}_{model}'
            f = place(out, config, model, dataset)
            for name in ('plausibility.csv', 'faithfulness.csv', 'alignment.csv'):
                if source.find(run, name):
                    copy(source, run, name, f)
    print('imported into', out)


if __name__ == '__main__':
    main()
