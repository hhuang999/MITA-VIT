"""Paper experiments. Modules expose run(args, config, output, commit).

Execution, correction and supplementary modules also contain shared historical stages.
Only the configurations in configs/ are part of the paper workflow.
"""

import importlib

EXPERIMENTS = ('execution', 'diagnose', 'correct', 'register_search', 'c2_analysis', 'supplementary')


def load(name):
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {name}; choose from {EXPERIMENTS}')
    return importlib.import_module(f'{__name__}.{name}')
