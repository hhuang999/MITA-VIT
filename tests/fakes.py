"""Small models and fake datasets for CPU tests (random weights; no downloads, no real data)."""

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset

from fastweight_vit.data import MaskDataset
from fastweight_vit.io import load_config
from fastweight_vit.models.mita import build_mita

ROOT = Path(__file__).resolve().parents[1]


def tiny():
    """DeiT-T with random weights: the same block structure as DeiT-S, 3 heads, width 192."""
    torch.manual_seed(0)
    return timm.create_model('deit_tiny_patch16_224', pretrained=False).eval()


def small_mita():
    """MiTA-DeiT-S built from random DeiT-S weights (the real checkpoint is not needed for structure tests)."""
    torch.manual_seed(0)
    skeleton = timm.create_model('deit_small_patch16_224', pretrained=False)
    return build_mita(skeleton.state_dict(), argparse.Namespace(model='deit_small_patch16_224'))


def config_for(name, **overrides):
    """A release config with the models / datasets lists removed, small batches and no worker processes."""
    config = load_config(ROOT / 'configs' / f'{name}.yaml')
    config.pop('models', None)
    config.pop('datasets', None)
    config.update(batch_size=2, num_workers=0, **overrides)
    return config


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


class FakeImageNet(Dataset):
    def __init__(self, n=4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return torch.randn(3, 224, 224, generator=generator), index % 3


class FakeParts(Dataset):
    """(image, object, valid, part fractions [196, 41], index): left half head (part 0), right half body (part 1)."""

    def __init__(self, n=3):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        fractions = torch.zeros(196, 41)
        columns = torch.arange(196) % 14
        fractions[columns < 5, 0] = 1
        fractions[(columns >= 5) & (columns < 10), 1] = 1
        fractions[columns >= 10, 40] = 1
        obj = columns < 10
        return torch.randn(3, 224, 224, generator=generator), obj, torch.ones(196, dtype=torch.bool), fractions, index


class FakeClickMe(Dataset):
    def __init__(self, *args, n=6, **kwargs):
        self.n = n

    def __len__(self):
        return self.n

    def record(self, index):
        image = Image.fromarray(np.full((256, 256, 3), index * 20, dtype=np.uint8))
        return image, np.zeros((256, 256), np.uint8), index % 3

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return torch.randn(3, 224, 224, generator=generator), torch.rand(224, 224, generator=generator), index % 3, index


class FakeFiles:
    names = [f'ILSVRC2012_val_{i:08d}' for i in range(40)]

    def labels(self):
        return [i % 5 for i in range(40)]

    def image_at(self, row):
        rng = np.random.default_rng(row)
        return Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8))


class FakeImageNetS:
    def __len__(self):
        return 3

    def name(self, index):
        return FakeFiles.names[index]
