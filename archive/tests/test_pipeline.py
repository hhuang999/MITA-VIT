"""Every experiment end to end on CPU, from the release configs, with random weights and fake data."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from fastweight_vit.experiments import (EXPERIMENTS, c2_analysis, common, correct, diagnose, execution, load,
                                        map_similarity, part_reliance, register_search, sanity, supplementary)
from fastweight_vit.io import load_config

from fakes import (ROOT, FakeClickMe, FakeFiles, FakeImageNet, FakeImageNetS, FakeMasks, FakeParts, config_for,
                   tiny)

MODULES = (execution, diagnose, correct, sanity, part_reliance, map_similarity, register_search, c2_analysis,
           supplementary)
PREPROCESSING = dict(input_size=(3, 224, 224), interpolation='bicubic', crop_pct=0.875, crop_mode='center',
                     mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


@pytest.fixture
def patched(monkeypatch, tmp_path):
    common.DEVICE = 'cpu'
    model = tiny()

    def build(name, model_, model_name, config):
        if name == 'imagenet':
            return FakeImageNet(4), list(range(4))
        if name == 'partimagenet':
            return FakeParts(3), list(range(3))
        split = config.get('split', 'all')
        return FakeMasks(model_, n=8), dict(all=list(range(6)), pilot=[0, 1, 2], formal=list(range(3, 8)))[split]

    for module in MODULES:
        monkeypatch.setattr(module, 'load_model', lambda name, device: model)
        monkeypatch.setattr(module, 'build', build)
    for module in (diagnose, c2_analysis, supplementary):
        monkeypatch.setattr(module, 'imagenet_files', lambda: FakeFiles())
        monkeypatch.setattr(module, 'preprocessing_of', lambda model, name: PREPROCESSING)
    for module in (diagnose, correct, supplementary):
        monkeypatch.setattr(module, 'ClickMe', FakeClickMe)
    monkeypatch.setattr(diagnose, 'ImageNetS', lambda *a, **k: FakeImageNetS())
    return tmp_path


def run_args(**fields):
    return SimpleNamespace(**({'model': 'deit', 'smoke': 0, 'resume': False, 'stage': None} | fields))


def folder(tmp, name):
    path = tmp / name
    path.mkdir(parents=True)
    return path


def lens_config(name, tmp, **overrides):
    """A config whose lens points at the diagnose outputs of this test."""
    config = config_for(name, **overrides)
    if 'diagnose' in config:
        config['diagnose'].update(probe_images=6, mean_images=4, epochs=2, probe_run=str(tmp / 'probe'))
    if 'correct' in config:
        config['correct'].update(lens_runs=dict(probe=str(tmp / 'probe'), mean_maps=str(tmp / 'mean_maps')),
                                 faithfulness_images=3, budgets=[1, 98])
    return config


def make_lens(tmp):
    config = lens_config('diagnose_probe', tmp)
    diagnose.run(run_args(stage='probe'), config, folder(tmp, 'probe'), 'test')
    diagnose.run(run_args(stage='mean_maps'), lens_config('diagnose_mean_maps', tmp), folder(tmp, 'mean_maps'), 'test')


def test_part_reliance_run_expansion(patched):
    spec = importlib.util.spec_from_file_location('run_script', ROOT / 'scripts' / 'run.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    parts = script.runs(load_config(ROOT / 'configs/part_reliance.yaml'), model='deit_b')
    assert len(parts) == 1 and parts[0][2]['batch_size'] == 64


def test_single_layer_execution(patched):
    tmp = patched
    execution.run(run_args(stage='single_layer'), config_for('single_layer'), folder(tmp, 'one'), 'test')
    single = pd.read_csv(tmp / 'one' / 'predictions.csv')
    assert len(single) == 4 * 12 and set(single.setting) == {'L3'}


def test_other_correction_order(patched):
    tmp = patched
    make_lens(tmp)
    order = lens_config('order', tmp)
    correct.run(run_args(), order, folder(tmp, 'order'), 'test')
    assert set(pd.read_csv(tmp / 'order' / 'plausibility.csv').method) == {'R0', 'R_all', 'R_all_alt'}


def test_probability_discount(patched):
    tmp = patched
    make_lens(tmp)
    d2 = lens_config('d2', tmp)
    d2['correct']['faithfulness_images'] = 0  # as in configs/d2.yaml: no SRG
    correct.run(run_args(), d2, folder(tmp, 'd2'), 'test')
    assert {'D2', 'D2_shuffle_s4', 'C3D2'} <= set(pd.read_csv(tmp / 'd2' / 'plausibility.csv').method)
    assert pd.read_csv(tmp / 'd2' / 'faithfulness.csv').empty


def test_randomised_model(patched):
    tmp = patched
    make_lens(tmp)
    config = lens_config('sanity', tmp)
    config['sanity']['images'] = 4
    sanity.run(run_args(), config, folder(tmp, 'sanity'), 'test')
    table = pd.read_csv(tmp / 'sanity' / 'sanity.csv')
    assert {'R0', 'C1', 'C2', 'C3', 'R_all', 'centre'} == set(table.method)
    assert (table[table.method == 'centre'].spearman > .999).all()


def test_part_reliance_and_map_similarity(patched):
    tmp = patched
    config = config_for('part_reliance')
    part_reliance.run(run_args(), config, folder(tmp, 'parts'), 'test')
    reliance = pd.read_csv(tmp / 'parts' / 'reliance.csv')
    assert set(reliance.condition) == {'part', 'object', 'background'} and set(reliance.removed) == {4, 8}
    assert set(reliance[reliance.condition == 'part'].part) == {0, 1} and set(reliance.seed) == set(range(5))

    map_similarity.run(run_args(), config_for('map_similarity'), folder(tmp, 'similarity'), 'test')
    rho = pd.read_csv(tmp / 'similarity' / 'similarity.csv')
    assert len(rho) == 6 and rho.spearman.between(-1, 1).all()
