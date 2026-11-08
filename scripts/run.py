"""Run one experiment and write its results to a new folder under experiments/.

Examples:
  python scripts/run.py anatomy --stage rq1 --model deit --config configs/anatomy_deit.yaml \
      --output experiments/2026-11-02_rq1_deit
  python scripts/run.py matched_deletion --config configs/matched_deletion.yaml --output experiments/...
  python scripts/run.py center_control --device cpu --config configs/center_control.yaml --output ...

Add --smoke N to run on the first N images, and --resume to continue an interrupted run in place.
--batch-size and --workers override the config (the saved config.yaml records the values used).
The source tree must be committed: the git commit is stored with every result.
"""

import argparse
import logging
from pathlib import Path

from fastweight_vit.runtime import configure_runtime

ROOT = configure_runtime()

from fastweight_vit.experiments import EXPERIMENTS, common, load  # noqa: E402
from fastweight_vit.io import load_config, source_commit  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('experiment', choices=sorted(EXPERIMENTS))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', help='anatomy: noise, rq1, rq2, repro, mita_repro, mita_anatomy, pet; '
                        'budget: pilot, formal; register: search, formal; diagnose: probe, m1; supp: heldout, eval')
    parser.add_argument('--model', default='deit', help='model name: deit, deit_b, dino, mita, deit_t, mita_t, deit3_s')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--smoke', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    module = load(args.experiment)
    if hasattr(module, 'STAGES') and args.stage not in module.STAGES:
        parser.error(f'{args.experiment} needs --stage from {module.STAGES}')
    config = load_config(ROOT / args.config)
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.workers is not None:
        config['num_workers'] = args.workers
    commit = source_commit(ROOT)
    common.DEVICE = args.device
    output = ROOT / args.output
    common.prepare_output(output, config, resume=args.resume, experiment=args.experiment, stage=args.stage,
                          model=args.model, smoke_images=args.smoke)
    try:
        module.run(args, config, output, commit)
    except Exception:
        logging.exception('Run failed; keep this folder and continue with --resume where supported')
        raise


if __name__ == '__main__':
    main()
