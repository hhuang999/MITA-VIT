"""Experiments. Each module exposes `run(args, config, output, commit)`; scripts/run.py starts one from a config.

| experiment | module | result |
| --- | --- | --- |
| execution | execution.py | execution test: k sweep (k_sweep, attention_off), single-layer checks (single_layer, attention_off) |
| diagnose | diagnose.py | per-layer premises: position probes (probe), mean maps and M1 (mean_maps), high-norm tokens (high_norm) |
| correct | correct.py | main results table, negative controls, model-side variants, other order, ClickMe gallery, per-layer ClickMe alignment |
| sanity | sanity.py | sanity check: cascading weight randomisation |
| part_reliance | part_reliance.py | part reliance: head vs body patches |
| map_similarity | map_similarity.py | top-k map vs full attention map |
| register_search | register_search.py | the FFN units of the "+register" row (configs/register.yaml) |
| c2_analysis | c2_analysis.py | exploratory: why probe relocation (C2) fails |
| supplementary | supplementary.py | exploratory: combinations, C3 mechanism, dose, globalisation, MLP probe |
"""

import importlib

EXPERIMENTS = ('execution', 'diagnose', 'correct', 'sanity', 'part_reliance', 'map_similarity', 'register_search',
               'c2_analysis', 'supplementary')


def load(name):
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {name}; choose from {EXPERIMENTS}')
    return importlib.import_module(f'{__name__}.{name}')
