"""Run one paper element: every model (and dataset) its config lists.

  python scripts/run.py --config configs/table1.yaml
  python scripts/run.py --config configs/table1.yaml --model deit --dataset clickme
  python scripts/run.py --config configs/fig2a_k_sweep.yaml --smoke 16 --output outputs/smoke

Each run writes outputs/<config name>/[<dataset>/]<model>/ with the effective config, provenance (git commit,
versions, GPU), log and result files. A run whose checks.json exists is complete and skipped; an interrupted run
continues with --resume. The source tree must be committed: the git commit is stored with every result.
--batch-size and --workers override the config (the saved config.yaml records the values used); the batch size
changes how seeded random streams are split, so keep the config value for reproduction.
"""

import argparse
import copy
import logging
from pathlib import Path
from types import SimpleNamespace

from fastweight_vit.runtime import configure_runtime

ROOT = configure_runtime()

from fastweight_vit.experiments import common, load  # noqa: E402
from fastweight_vit.io import load_config, source_commit  # noqa: E402

DATASET_SECTION = {'correct': 'correct', 'supplementary': 'supplementary'}


def rooted(value, root):
    """Point the paths of earlier runs ('outputs/...') at the chosen outputs root (e.g. outputs/smoke)."""
    if isinstance(value, dict):
        return {key: rooted(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [rooted(item, root) for item in value]
    if isinstance(value, str) and value.startswith('outputs/') and root != 'outputs':
        return f'{root}/{value.removeprefix("outputs/")}'
    return value


def runs(config, model=None, dataset=None):
    """[(model, dataset or None, effective config)] for the models and datasets of a config."""
    models = config['models']
    models = {name: {} for name in models} if isinstance(models, list) else models
    datasets = config.get('datasets') or [None]
    out = []
    for name, overrides in models.items():
        if model and name != model:
            continue
        for data in datasets:
            if dataset and data != dataset:
                continue
            effective = {key: copy.deepcopy(value) for key, value in config.items() if key not in ('models', 'datasets')}
            effective.update(copy.deepcopy(overrides or {}))
            if data is not None:
                effective[DATASET_SECTION[config['experiment']]]['dataset'] = data
            out.append((name, data, effective))
    if not out:
        raise SystemExit(f'Nothing to run for model={model} dataset={dataset}')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', help='run only this model of the config')
    parser.add_argument('--dataset', help='run only this dataset of the config')
    parser.add_argument('--output', type=Path, default=Path('outputs'), help='root folder of the runs')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--smoke', type=int, default=0, help='run on the first N images only')
    parser.add_argument('--resume', action='store_true', help='continue interrupted runs in place')
    args = parser.parse_args()
    config = rooted(load_config(ROOT / args.config), args.output.as_posix().rstrip('/'))
    module = load(config['experiment'])
    stage = config.get('stage')
    if hasattr(module, 'STAGES') and stage not in module.STAGES:
        parser.error(f"{config['experiment']} needs a stage from {module.STAGES}")
    commit = source_commit(ROOT)
    common.DEVICE = args.device
    for model, dataset, effective in runs(config, args.model, args.dataset):
        if args.batch_size:
            effective['batch_size'] = args.batch_size
        if args.workers is not None:
            effective['num_workers'] = args.workers
        output = ROOT / args.output / args.config.stem
        output = (output / dataset if dataset else output) / model
        if (output / 'checks.json').exists():
            print(f'complete, skipped: {output.relative_to(ROOT)}')
            continue
        common.prepare_output(output, effective, resume=args.resume, experiment=config['experiment'], stage=stage,
                              model=model, smoke_images=args.smoke)
        run_args = SimpleNamespace(model=model, stage=stage, smoke=args.smoke, resume=args.resume)
        try:
            module.run(run_args, effective, output, commit)
        except Exception:
            logging.exception('Run failed; keep this folder and continue with --resume where supported')
            raise


if __name__ == '__main__':
    main()
