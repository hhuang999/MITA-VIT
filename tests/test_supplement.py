"""S1-S3 (decisions B22-B24): dense attention mass, single-layer map families, maps-only parts and the
layer_reading experiment, on CPU with random weights and fake data."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from mita_probe import explain
from mita_probe.experiments import common, explanations, human_alignment, layer_reading, parts
from mita_probe.experiments.explanations import layerwise_methods
from mita_probe.interventions import sink_tokens
from mita_probe.io import load_config
from mita_probe.models.deit import load_deit
from mita_probe.probe import resolve

from test_human_parts_sanity import FakeParts, write_shards
from test_new_experiments import FakeMasks

ROOT = Path(__file__).resolve().parents[1]
LAYER_MAPS = {f'{family}_layer{layer}' for family in ('probe', 'attention', 'cls') for layer in range(1, 13)}


def test_attention_mass_is_the_probe_without_top_k():
    torch.manual_seed(0)
    model = load_deit(pretrained=False)
    images = torch.randn(2, 3, 224, 224)
    layers = [9, 10, 11, 12]
    dense = explain.compute('attention', model, images, layers=layers)
    every_token = resolve({'mode': 'L3', 'k': 197}, load_config(ROOT / 'configs/explanations.yaml')['main'])
    torch.testing.assert_close(dense, explain.probe(model, images, every_token, layers), rtol=1e-5, atol=1e-4)
    # Every query's attention sums to 1, so all tokens together receive heads x tokens per layer.
    heads, tokens = model.blocks[0].attn.num_heads, 197
    assert dense.shape == (2, 196) and (dense > 0).all() and (dense.sum(1) < len(layers) * heads * tokens).all()


def test_layerwise_families_keep_the_probe_default():
    assert [m['name'] for m in layerwise_methods(2)] == ['probe_layer1', 'probe_layer2']
    entries = layerwise_methods(12, ('probe', 'attention', 'cls_attention'))
    assert {m['name'] for m in entries} == LAYER_MAPS
    assert all(m['layers'] == [int(m['name'].rsplit('_layer', 1)[1])] for m in entries)
    assert {m['method'] for m in entries} == {'probe', 'attention', 'cls_attention'}


def test_high_norm_uses_the_midpoint_median():
    # Even count: the median is the midpoint of the two middle norms (2 and 4 -> 3), so only 10 > 9 is high.
    norms = torch.tensor([[1., 2., 4., 8.5, 10., 1.], [2., 2., 2., 6., 7., 2.]])
    assert sink_tokens(norms, 3).tolist() == [[False, False, False, False, True, False],
                                              [False, False, False, False, True, False]]


@pytest.fixture
def patched(monkeypatch, tmp_path):
    common.DEVICE = 'cpu'
    torch.manual_seed(0)
    model = load_deit(pretrained=False)
    for module in (explanations, human_alignment, parts, layer_reading):
        monkeypatch.setattr(module, 'load_model', lambda name, device: model)
    write_shards(tmp_path / 'data' / 'clickme')
    monkeypatch.setattr(human_alignment, 'data_root', lambda: tmp_path / 'data')

    def build(name, model_, model_name, config, deit_reference=None):
        dataset = FakeParts(model_) if name == 'partimagenet' else FakeMasks(model_)
        return dataset, list(range(len(dataset)))
    monkeypatch.setattr(explanations.datasets, 'build', build)  # the module all experiments share
    out = tmp_path / 'out'
    out.mkdir()
    return out


def config_for(name):
    config = load_config(ROOT / f'configs/{name}.yaml')
    config.update(batch_size=2, num_workers=0)
    return config


def test_dense_explanations_config(patched):
    explanations.run(SimpleNamespace(model='deit', smoke=2, resume=False), config_for('explanations_dense'),
                     patched, 'test')
    assert set(pd.read_csv(patched / 'faithfulness.csv').method) == {'probe_mass', 'attention_mass'}
    assert set(pd.read_csv(patched / 'plausibility.csv').method) == {'probe_mass', 'attention_mass'}


def test_human_alignment_with_three_layer_families(patched):
    config = config_for('human_alignment_dense')
    config['human_alignment']['shards'] = 2
    human_alignment.run(SimpleNamespace(model='deit', smoke=3, resume=False), config, patched, 'test')
    rows = pd.read_csv(patched / 'alignment.csv')
    methods = {'probe_mass', 'attention_mass', 'mean_human'} | LAYER_MAPS
    assert set(rows.method) == methods and len(rows) == 3 * len(methods)


def test_parts_maps_only(patched):
    parts.run(SimpleNamespace(model='deit', smoke=0, resume=False), config_for('parts_dense'), patched, 'test')
    rows = pd.read_csv(patched / 'parts.csv')
    assert set(rows.method) == {'probe_mass', 'attention_mass'} | LAYER_MAPS
    assert not (patched / 'reliance.csv').exists()


def test_layer_reading_rows(patched):
    layer_reading.run(SimpleNamespace(model='deit', smoke=4, resume=False), config_for('layer_reading'), patched,
                      'test')
    rows = pd.read_csv(patched / 'layer_reading.csv')
    assert len(rows) == 4 * 3 * 12 and set(rows.family) == {'probe', 'attention', 'cls'}
    assert rows.layer.between(1, 12).all() and rows.groupby('family').layer.nunique().eq(12).all()
    for column in ['high_fraction', 'high_share', 'mass']:
        assert rows[column].between(0, 1).all()
    # The fraction of high-norm patches depends on the layer only, not on the map read at that layer.
    assert rows.groupby(['index', 'layer']).high_fraction.nunique().eq(1).all()
