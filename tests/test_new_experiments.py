"""The explanations, shifted-crop and identity experiments end to end on CPU with random weights and fake data."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from fastweight_vit.data import MaskDataset
from fastweight_vit.experiments import common, explanations, identity, shifted_crops
from fastweight_vit.io import load_config
from fastweight_vit.models.deit import load_deit

ROOT = Path(__file__).resolve().parents[1]


class FakeMasks(MaskDataset):
    """Grey images with a bright square object at a seeded random place."""

    def __init__(self, reference, n=6):
        super().__init__(reference)
        self.n = n

    def __len__(self):
        return self.n

    def sample(self, index):
        rng = np.random.default_rng(index)
        top, left = rng.integers(0, 200, 2)
        trimap = np.full((300, 400), 2, dtype=np.uint8)
        trimap[top:top + 100, left:left + 150] = 1
        image = np.full((300, 400, 3), 100, dtype=np.uint8)
        image[trimap == 1] = 200
        return Image.fromarray(image), 0, Image.fromarray(trimap)


@pytest.fixture
def patched(monkeypatch):
    common.DEVICE = 'cpu'
    torch.manual_seed(0)
    model = load_deit(pretrained=False)
    for module in (explanations, identity, shifted_crops):
        monkeypatch.setattr(module, 'load_model', lambda name, device: model)
    images = TensorDataset(torch.randn(6, 3, 224, 224), torch.randint(0, 1000, (6,)))

    def build(name, model_, model_name, config, deit_reference=None):
        if name == 'imagenet':
            return images, list(range(6))
        dataset = FakeMasks(model_)
        return dataset, list(range(len(dataset)))
    monkeypatch.setattr(explanations.datasets, 'build', build)
    return model


def config_for(name):
    config = load_config(ROOT / f'configs/{name}.yaml')
    config.update(batch_size=2, num_workers=0)
    return config


def test_explanations_writes_every_table(patched, tmp_path):
    config = config_for('explanations')
    keep = {'probe_mass', 'probe_count', 'cls_attention', 'gae', 'centre', 'random', 'mask'}
    config['explanations'].update(faithfulness_images=2, budgets=[1, 98], saco_groups=2,
                                  methods=[m for m in config['explanations']['methods'] if m['name'] in keep])
    explanations.run(SimpleNamespace(model='deit', smoke=4, resume=False), config, tmp_path, 'test')
    plaus = pd.read_csv(tmp_path / 'plausibility.csv')
    assert len(plaus) == 4 * len(keep) and set(plaus.method) == keep
    assert (plaus.query("method == 'mask'").auroc == 1).all()
    faith = pd.read_csv(tmp_path / 'faithfulness.csv')
    assert len(faith) == 2 * len(keep) * (3 * 2 + 2) and faith['index'].nunique() == 2
    assert set(faith.condition) == {'keep_top', 'drop_top', 'drop_bottom', 'keep_probe_k', 'drop_probe_k'}
    assert len(pd.read_csv(tmp_path / 'saco.csv')) == 2 * len(keep)
    cost = pd.read_csv(tmp_path / 'cost.csv')
    assert set(cost.method) == keep | {'forward'} and (cost.seconds > 0).all()
    shards = sorted((tmp_path / 'scores').glob('*.npz'))
    assert len(shards) == 2 and np.load(shards[0])['probe_mass'].shape == (2, 196)


def test_explanations_on_imagenet_skips_masks(patched, tmp_path):
    config = config_for('explanations_imagenet')
    config['explanations'].update(faithfulness_images=1, budgets=[4], conditions=['keep_top'], saco_groups=0,
                                  methods=[m for m in config['explanations']['methods']
                                           if m['name'] in {'probe_mass', 'rollout', 'mask'}])
    explanations.run(SimpleNamespace(model='deit', smoke=2, resume=False), config, tmp_path, 'test')
    assert not (tmp_path / 'plausibility.csv').exists()
    assert set(pd.read_csv(tmp_path / 'faithfulness.csv').method) == {'probe_mass', 'rollout'}


def test_shifted_crops_follow_the_moved_window(patched, tmp_path):
    config = config_for('shifted_crops')
    config['shifted_crops'].update(images=3, min_object_kept=0.3,
                                   methods=[{'name': 'centre', 'method': 'centre'},
                                            {'name': 'mask', 'method': 'mask'}])
    shifted_crops.run(SimpleNamespace(model='deit', smoke=0), config, tmp_path, 'test')
    rows = pd.read_csv(tmp_path / 'crops.csv')
    assert set(rows.method) == {'centre', 'mask'} and rows.object_kept.ge(0.3).all()
    assert len(rows.groupby(['dy', 'dx'])) > 1 and (rows.query("method == 'mask'").auroc == 1).all()


def test_identity_rows_per_layer(patched, tmp_path):
    config = config_for('identity')
    config['identity'].update(epochs=2)
    identity.run(SimpleNamespace(model='deit', smoke=4), config, tmp_path, 'test')
    rows = pd.read_csv(tmp_path / 'identity.csv')
    assert list(rows.layer) == list(range(1, 13)) and rows.position_accuracy.between(0, 1).all()
    assert rows.self_cosine.iloc[0] == pytest.approx(1, abs=1e-4)
