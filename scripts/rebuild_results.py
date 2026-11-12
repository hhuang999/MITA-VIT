"""Rebuild saved observations into a fresh directory, never into the reference results."""

import argparse
from pathlib import Path

from fastweight_vit.evidence import CONFIGS, rebuild, validate, validate_completed_run
from fastweight_vit.io import load_config


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--destination', type=Path, default=Path('build/rebuilt'))
    args = parser.parse_args()
    if args.outputs.resolve() == (root/'outputs').resolve():
        validate(root)
        print(rebuild(root, args.destination))
    else:
        # New observations have no obligation to equal the preserved results.
        from run import rooted, runs
        for name in CONFIGS:
            if name == 'register':
                continue  # The pilot has a different historical output layout.
            config = rooted(load_config(root/f'configs/{name}.yaml'), args.outputs.as_posix().rstrip('/'))
            for model, dataset, effective in runs(config):
                output = root/args.outputs/name
                output = output/dataset/model if dataset else output/model
                validate_completed_run(output, effective)
        print(rebuild(root,args.destination,outputs=args.outputs,compare_reference=False))


if __name__ == '__main__':
    main()
