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


def test_every_config_names_an_experiment_and_stage():
    for path in sorted((ROOT / 'configs').glob('*.yaml')):
        config = load_config(path)
        assert config['experiment'] in EXPERIMENTS, path.name
        module = load(config['experiment'])
        assert not hasattr(module, 'STAGES') or config.get('stage') in module.STAGES, path.name


def test_run_script_expands_models_and_datasets():
    spec = importlib.util.spec_from_file_location('run_script', ROOT / 'scripts' / 'run.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    table1 = load_config(ROOT / 'configs/table1.yaml')
    expanded = script.runs(table1)
    assert [(model, data) for model, data, _ in expanded] == [
        (m, d) for m in ('deit', 'deit_b', 'dino') for d in ('imagenet_s', 'clickme')]
    assert expanded[1][2]['correct']['dataset'] == 'clickme' and 'models' not in expanded[0][2]
    parts = script.runs(load_config(ROOT / 'configs/part_reliance.yaml'), model='deit_b')
    assert len(parts) == 1 and parts[0][2]['batch_size'] == 64


def test_execution_stages(patched):
    tmp = patched
    config = config_for('fig2a_k_sweep')
    config['sweep']['k_sweep'] = [4, 16]
    execution.run(run_args(stage='k_sweep'), config, folder(tmp, 'k'), 'test')
    predictions = pd.read_csv(tmp / 'k' / 'predictions.csv')
    settings = pd.read_csv(tmp / 'k' / 'settings.csv')
    assert len(settings) == 2 + 2 * 5 and len(predictions) == 4 * len(settings)
    assert set(settings.name) == {'L3_spatial', 'rand'} and json.loads((tmp / 'k' / 'checks.json').read_text())['complete']
    execution.run(run_args(stage='single_layer'), config_for('single_layer'), folder(tmp, 'one'), 'test')
    single = pd.read_csv(tmp / 'one' / 'predictions.csv')
    assert len(single) == 4 * 12 and set(single.setting) == {'L3'}
    execution.run(run_args(stage='attention_off'), config_for('attention_off'), folder(tmp, 'off'), 'test')
    off = pd.read_csv(tmp / 'off' / 'attention_off.csv')
    assert len(off) == 4 * 13 and set(off.layers.astype(str)) == {'all'} | {str(layer) for layer in range(1, 13)}


def test_diagnose_correct_and_sanity(patched):
    tmp = patched
    make_lens(tmp)
    held = json.loads((tmp / 'probe' / 'heldout.json').read_text())
    assert len(held['rows']) == 10 and not set(held['rows']) & {0, 1, 2}
    accuracy = pd.read_csv(tmp / 'probe' / 'probe_accuracy.csv')
    assert list(accuracy.layer) == list(range(1, 13)) and (accuracy.sigma >= 0).all()
    assert len(pd.read_csv(tmp / 'probe' / 'probe_accuracy_images.csv')) == 4 * 12
    m1 = pd.read_csv(tmp / 'mean_maps' / 'm1.csv')
    assert len(m1) == 6 * 12 * 4 and (m1.residual >= 0).all()
    assert (tmp / 'mean_maps' / 'means_crop.npz').exists() and (tmp / 'mean_maps' / 'means_clickme.npz').exists()

    config = lens_config('high_norm', tmp)
    config['diagnose']['norm_ratio'] = 1.2  # random weights: make some patches high-norm
    diagnose.run(run_args(stage='high_norm'), config, folder(tmp, 'high'), 'test')
    high = pd.read_csv(tmp / 'high' / 'high_norm.csv')
    assert len(high) == 6 * 12 and high.high_share.between(0, 1).all() and (high.value_ratio_n >= 0).all()
    assert (high.value_ratio_n > 0).any()

    config = lens_config('table1', tmp)
    config['correct']['gallery_images'] = 2
    correct.run(run_args(), config, folder(tmp, 'imagenet_s'), 'test')
    plaus = pd.read_csv(tmp / 'imagenet_s' / 'plausibility.csv')
    names = [entry['name'] for entry in config['correct']['methods']]
    assert set(plaus.method) == set(names) and len(plaus) == 6 * len(names)
    assert {'stand_in_blur', 'stand_in_no_high', 'stand_in_centre'} <= set(plaus.method)
    faith = pd.read_csv(tmp / 'imagenet_s' / 'faithfulness.csv')
    assert faith['index'].nunique() == 3 and set(faith.condition) == {'drop_top', 'drop_bottom'}
    assert not {'R0_early', 'C2_early'} & set(faith.method)
    assert 'map_R_all' in np.load(next((tmp / 'imagenet_s').glob('gallery_*.npz'))).files

    config['correct']['dataset'] = 'clickme'
    correct.run(run_args(), config, folder(tmp, 'clickme'), 'test')
    assert set(pd.read_csv(tmp / 'clickme' / 'alignment.csv').method) == set(names)
    assert any('image' in np.load(path).files for path in (tmp / 'clickme').glob('gallery_*.npz'))

    order = lens_config('order', tmp)
    correct.run(run_args(), order, folder(tmp, 'order'), 'test')
    assert set(pd.read_csv(tmp / 'order' / 'plausibility.csv').method) == {'R0', 'R_all', 'R_all_alt'}

    layers = lens_config('clickme_layers', tmp)
    layers['correct']['dataset'] = 'clickme'
    correct.run(run_args(), layers, folder(tmp, 'layers'), 'test')
    assert set(pd.read_csv(tmp / 'layers' / 'alignment.csv').method) == {f'R0_l{layer}' for layer in range(1, 13)}

    register = lens_config('table1_register', tmp)
    register['correct'].update(register_units=[[2, 5], [3, 7]])
    correct.run(run_args(), register, folder(tmp, 'register'), 'test')
    assert set(pd.read_csv(tmp / 'register' / 'plausibility.csv').method) == {'R0', 'R0_l12'}
    assert set(pd.read_csv(tmp / 'register' / 'faithfulness.csv').method) == {'R0', 'R0_l12'}

    config = lens_config('sanity', tmp)
    config['sanity']['images'] = 4
    sanity.run(run_args(), config, folder(tmp, 'sanity'), 'test')
    table = pd.read_csv(tmp / 'sanity' / 'sanity.csv')
    assert {'R0', 'C1', 'C2', 'C3', 'R_all', 'centre'} == set(table.method)
    assert (table[table.method == 'centre'].spearman > .999).all()


def test_c2_analysis_and_supplementary(patched):
    tmp = patched
    make_lens(tmp)
    c2_analysis.run(run_args(), lens_config('c2_analysis', tmp), folder(tmp, 'c2'), 'test')
    errors = pd.read_csv(tmp / 'c2' / 'probe_errors.csv')
    assert len(errors) == 4 * 7 * 2 and set(errors.source) == {'probe', 'stand_in_blur'}
    assert (errors[errors.source == 'stand_in_blur'].exact == 1).all()
    prediction = np.load(tmp / 'c2' / 'mean_prediction.npz')
    assert prediction['layer12'].shape == (4, 196) and np.allclose(prediction['layer12'].sum(1), 1, atol=1e-5)
    shift = pd.read_csv(tmp / 'c2' / 'shift.csv')
    assert len(shift) == 6 * 2 and np.allclose(shift[[f'bin{b}_moved' for b in range(10)]].sum(axis=1), 1, atol=1e-6)

    config = lens_config('supplementary_heldout', tmp)
    config['supplementary'].update(heldout_run=str(tmp / 'held'), mlp_hidden=32)
    supplementary.run(run_args(stage='heldout'), config, folder(tmp, 'held'), 'test')
    mlp = pd.read_csv(tmp / 'held' / 'mlp_probe.csv')
    assert set(mlp.probe) == {'linear', 'mlp'} and set(mlp.layer) == set(range(1, 13))
    assert set(pd.read_csv(tmp / 'held' / 'global.csv').layer) == set(range(6, 13))
    config = lens_config('supplementary_eval', tmp)
    config['supplementary'].update(heldout_run=str(tmp / 'held'), mlp_hidden=32)
    supplementary.run(run_args(stage='eval'), config, folder(tmp, 'eval_s'), 'test')
    plaus = pd.read_csv(tmp / 'eval_s' / 'plausibility.csv')
    assert {'C1C3', 'C2C3', 'C1C2', 'C3v', 'C3headmean', 'C3shuffle_s4', 'C2conf', 'C2MLP', 'C3_l6',
            'stand_in_no_high_l12', 'stand_in_blur_MLP', 'C3_centre'} <= set(plaus.method)
    faith = pd.read_csv(tmp / 'eval_s' / 'faithfulness.csv')
    assert set(faith.method) == set(config['supplementary']['srg_maps']) and faith['index'].nunique() == 3
    assert len(pd.read_csv(tmp / 'eval_s' / 'dose.csv')) == 6
    config['supplementary']['dataset'] = 'clickme'
    supplementary.run(run_args(stage='eval'), config, folder(tmp, 'eval_c'), 'test')
    assert 'C2MLP' in set(pd.read_csv(tmp / 'eval_c' / 'alignment.csv').method)


def test_appendix_checks(patched):
    tmp = patched
    config = config_for('part_reliance')
    part_reliance.run(run_args(), config, folder(tmp, 'parts'), 'test')
    reliance = pd.read_csv(tmp / 'parts' / 'reliance.csv')
    assert set(reliance.condition) == {'part', 'object', 'background'} and set(reliance.removed) == {4, 8}
    assert set(reliance[reliance.condition == 'part'].part) == {0, 1} and set(reliance.seed) == set(range(5))

    map_similarity.run(run_args(), config_for('map_similarity'), folder(tmp, 'similarity'), 'test')
    rho = pd.read_csv(tmp / 'similarity' / 'similarity.csv')
    assert len(rho) == 6 and rho.spearman.between(-1, 1).all()

    config = config_for('register')
    config['search']['norm_ratio'] = 1.2
    config['register'].update(counts=[1, 2], seeds=[0, 1], presence_share=0.0)
    register_search.run(run_args(), config, folder(tmp, 'search'), 'test')
    gate = json.loads((tmp / 'search' / 'gate.json').read_text())
    assert {'first_high_layer', 'found', 'neurons'} <= set(gate)


def test_released_register_units_are_the_search_result():
    units = load_config(ROOT / 'configs/table1_register.yaml')['correct']['register_units']
    assert len(units) == 8 and all(layer in (4, 5) for layer, _ in units)
    assert torch.tensor(units).shape == (8, 2)


def test_scripts_do_not_reference_old_paths():
    for path in list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'configs').glob('*.yaml')):
        text = Path(path).read_text()
        assert 'experiments/2026' not in text and 'mita_probe' not in text, path.name
