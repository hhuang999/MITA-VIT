"""Experiments. Each module exposes `run(args, config, output, commit)` and is started by scripts/run.py.

| name | module | question |
| --- | --- | --- |
| anatomy | anatomy.py | RQ1-RQ3 stages: noise floor, single-layer and all-layer probes, Pet composition |
| matched_deletion | matched_deletion.py | RQ3 necessity: remove foreground/background neurons vs random and centre-matched |
| center_control | center_control.py | RQ3: probe reading sets vs the same number of central patches (CPU) |

To add an experiment, write a module with `run(...)` and list it in EXPERIMENTS.
"""

import importlib

EXPERIMENTS = {
    'anatomy': 'Anatomy stages for RQ1-RQ3 (needs --stage)',
    'matched_deletion': 'Foreground/background removal with random and centre-matched controls',
    'center_control': 'Probe reading sets against the nearest-centre patches (CPU)',
}


def load(name):
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {name}; choose from {sorted(EXPERIMENTS)}')
    return importlib.import_module(f'{__name__}.{name}')
