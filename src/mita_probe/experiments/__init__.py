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
| model_check | model_check.py | attention budget T0: accuracy under own and uniform preprocessing |
| budget | budget.py | attention budget T1/T3: where the budget goes, object focus without high-norm patches |
| sink_use | sink_use.py | attention budget T4: is the budget on high-norm patches used? sink type |
| register | register.py | attention budget T2/T6: neurons that make high-norm tokens, test-time registers |
| attention_off | attention_off.py | attention budget T7: predictions with no attention at all |
| diagnose | diagnose.py | v5 RQ2: held-out images, position probes (C2), mean maps and the M1 share |
| correct | correct.py | v5 RQ3: lens readings C1-C3, R_all, their controls and references; model variants |
| explore_c2 | explore_c2.py | E-C2 (B45, exploratory): probe error geometry, mean prediction, where C2 moves scores |

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
    'model_check': 'Attention budget T0: ImageNet accuracy under own and uniform preprocessing',
    'budget': 'Attention budget T1/T3: budget shares per layer and object focus (needs --stage pilot|formal)',
    'sink_use': 'Attention budget T4: removing high-norm patches from reading vs same-count controls',
    'register': 'Attention budget T2/T6: register-neuron search and test-time registers (--stage search|formal)',
    'attention_off': 'Attention budget T7: every attention branch set to zero',
    'diagnose': 'v5 diagnose: held-out set, position probes, mean maps, M1 share (--stage probe|m1)',
    'correct': 'v5 correct: lens readings, controls and references on ImageNet-S or ClickMe',
    'explore_c2': 'E-C2 (exploratory): why relocation (C2) fails: probe errors, mean prediction, score shift',
}


def load(name):
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {name}; choose from {sorted(EXPERIMENTS)}')
    return importlib.import_module(f'{__name__}.{name}')
