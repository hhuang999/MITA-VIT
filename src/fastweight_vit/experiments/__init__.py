"""Experiments. Each module exposes `run(args, config, output, commit)`; scripts/run.py starts one from a config.

| experiment | module | paper element |
| --- | --- | --- |
| execution | execution.py | figure 2(a) (stage k_sweep, attention_off) and appendix B (single_layer, attention_off) |
| diagnose | diagnose.py | figure 3 and section 4.3: position probes (probe), mean maps and M1 (mean_maps), high-norm tokens (high_norm) |
| correct | correct.py | Table 1, sections 4.2 and 4.4, figure 2(b), figure 3(d), appendices C and D |
| sanity | sanity.py | appendix C: cascading weight randomisation |
| part_reliance | part_reliance.py | appendix B: head vs body patches |
| map_similarity | map_similarity.py | appendix B: top-k map vs full attention map |
| register_search | register_search.py | the FFN units of the "+register" row (configs/register.yaml) |
| c2_analysis | c2_analysis.py | appendix E (exploratory): why C2 fails |
| supplementary | supplementary.py | appendices D and E (exploratory): combinations, C3 mechanism, dose, globalisation, MLP probe |
"""

import importlib

EXPERIMENTS = ('execution', 'diagnose', 'correct', 'sanity', 'part_reliance', 'map_similarity', 'register_search',
               'c2_analysis', 'supplementary')


def load(name):
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {name}; choose from {EXPERIMENTS}')
    return importlib.import_module(f'{__name__}.{name}')
