"""Experiments. Each module exposes `run(args, config, output, commit)` and is started by scripts/run.py.

| name | module | question |
| --- | --- | --- |
| anatomy | anatomy.py | RQ1-RQ3 stages: noise floor, single-layer and all-layer probes, Pet composition |
| matched_deletion | matched_deletion.py | RQ3 necessity: remove foreground/background neurons vs random and centre-matched |
| center_control | center_control.py | RQ3: probe reading sets vs the same number of central patches (CPU) |
| explanations | explanations.py | RQ3/RQ4: probe and baseline maps; plausibility, deletion/insertion, SaCo |
| shifted_crops | shifted_crops.py | RQ3: the same object at five places in the frame |
| identity | identity.py | premise: does token j still stand for patch j at each layer? |
| human_alignment | human_alignment.py | RQ3: maps against ClickMe human importance maps |
| parts | parts.py | RQ3: part preference of every map and the model's own part reliance |
| sanity | sanity.py | RQ4: cascading weight randomisation |
| gallery | gallery.py | maps on randomly drawn images, for figures |
| layer_reading | layer_reading.py | per-layer reading: share on high-norm patches, object focus |

To add an experiment, write a module with `run(...)` and list it in EXPERIMENTS.
"""

import importlib

EXPERIMENTS = {
    'anatomy': 'Anatomy stages for RQ1-RQ3 (needs --stage)',
    'matched_deletion': 'Foreground/background removal with random and centre-matched controls',
    'center_control': 'Probe reading sets against the nearest-centre patches (CPU)',
    'explanations': 'Probe and baseline patch maps: plausibility and token-removal faithfulness',
    'shifted_crops': 'Probe and baseline maps when the object is moved off-centre',
    'identity': 'Linear position probe and self-similarity of tokens, per layer',
    'human_alignment': 'Probe and baseline maps against ClickMe human importance maps',
    'parts': 'Part preference of every map and the model\'s own part reliance (PartImageNet)',
    'sanity': 'Cascading weight randomisation: do the maps depend on the trained model?',
    'gallery': 'Every map and the per-layer probe on randomly drawn images, for figures',
    'layer_reading': 'Single-layer probe, dense and CLS attention: share on high-norm patches, object focus',
}


def load(name):
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {name}; choose from {sorted(EXPERIMENTS)}')
    return importlib.import_module(f'{__name__}.{name}')
