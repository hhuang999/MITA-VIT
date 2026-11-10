"""Datasets and fixed lists: ClickMe records, ImageNet-S masks, sampling, splits."""

import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fastweight_vit.data import ImageNetS, subset_indices, token_foreground
from fastweight_vit.data.clickme import ClickMe, gaussian_kernel, parse_example, record_offsets
from fastweight_vit.data.imagenet_s import IGNORE
from fastweight_vit.data.splits import split_indices

from fakes import tiny

ROOT = Path(__file__).resolve().parents[1]


def _varint(value):
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number, payload):
    return _varint(number << 3 | 2) + _varint(len(payload)) + payload


def example(image: bytes, heatmap: bytes, label: int) -> bytes:
    """A tf.train.Example with two BytesList features and one packed Int64List feature."""
    def entry(name, feature):
        return _field(1, _field(1, name.encode()) + _field(2, feature))
    features = (entry('image', _field(1, _field(1, image))) + entry('heatmap', _field(1, _field(1, heatmap)))
                + entry('label', _field(3, _field(1, _varint(label)))))
    return _field(1, features)


def jpeg(array, mode):
    stream = io.BytesIO()
    Image.fromarray(array, mode).save(stream, format='JPEG', quality=95)
    return stream.getvalue()


def write_shards(root, n_shards=2, per_shard=3):
    """Shards whose human map marks a bright square; labels 0, 1, 2, ..."""
    root.mkdir(parents=True, exist_ok=True)
    label = 0
    for shard in range(n_shards):
        with (root / f'val-{shard}.tfrecords').open('wb') as stream:
            for _ in range(per_shard):
                rng = np.random.default_rng(label)
                top, left = rng.integers(0, 160, 2)
                image = np.full((256, 256, 3), 90, dtype=np.uint8)
                image[top:top + 96, left:left + 96] = 220
                heat = np.zeros((256, 256), dtype=np.uint8)
                heat[top:top + 96, left:left + 96] = 200
                record = example(jpeg(image, 'RGB'), jpeg(heat, 'L'), label)
                stream.write(struct.pack('<Q', len(record)) + b'\0' * 4 + record + b'\0' * 4)
                label += 1


def test_example_round_trip_and_offsets(tmp_path):
    record = example(b'abc', b'xyz', 913)
    assert parse_example(record) == {'image': [b'abc'], 'heatmap': [b'xyz'], 'label': [913]}
    write_shards(tmp_path, 1, 4)
    assert len(record_offsets(tmp_path / 'val-0.tfrecords')) == 4


def test_clickme_dataset_geometry(tmp_path):
    write_shards(tmp_path)
    data = ClickMe(tmp_path, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), expected_shards=2)
    assert len(data) == 6
    image, human, label, index = data[4]
    assert image.shape == (3, 224, 224) and human.shape == (224, 224) and (label, index) == (4, 4)
    rng = np.random.default_rng(4)
    top, left = (rng.integers(0, 160, 2) * 224 / 256).astype(int)
    assert human[top + 42, left + 42] > human.mean() * 2  # the square's centre stays the hottest area
    kernel = gaussian_kernel()
    assert kernel.shape == (10, 10) and kernel.sum() == pytest.approx(1) and kernel.argmax() == 4 * 10 + 5


class Stub:
    def image(self, name):
        return Image.new('RGB', (64, 48), (120, 120, 120))


def test_imagenet_s_masks(tmp_path):
    reference = tiny()
    folder = tmp_path / 'ImageNetS919' / 'validation-segmentation'
    for wnid, value in [('n01440764', 1), ('n01443537', 2)]:
        (folder / wnid).mkdir(parents=True)
        ids = np.zeros((48, 64), dtype=np.int64)
        ids[:, 32:] = value  # right half is this image's category
        ids[0, :] = IGNORE
        rgb = np.stack([ids % 256, ids // 256, np.zeros_like(ids)], -1).astype(np.uint8)
        Image.fromarray(rgb).save(folder / wnid / f'ILSVRC2012_val_0000000{value}.png')
    dataset = ImageNetS(tmp_path / 'ImageNetS919', Stub(), reference, expected=2)
    assert dataset.target(0) == 0 and dataset.target(1) == 1 and dataset.name(1) == 'ILSVRC2012_val_00000002'
    _, fg, valid, _ = dataset[1]
    grid, grid_valid = fg.reshape(14, 14), valid.reshape(14, 14)
    assert grid[:, 9:].all() and not grid[:, :5].any() and grid_valid.all()


def test_subset_sampling_is_shared_reproducible_and_without_replacement():
    indices = subset_indices(50000, 5000)
    assert indices == subset_indices(50000, 5000, seed=0)
    assert len(set(indices)) == 5000
    assert indices != subset_indices(50000, 5000, seed=1)


def test_trimap_boundaries_and_half_foreground_threshold():
    mask = torch.full((32, 32), 3)
    mask[:8, :16] = 1
    mask[8:16, :16] = 2
    mask[16:, :16] = 2
    mask[16:, 16:] = 1
    foreground, valid = token_foreground(mask)
    assert foreground.tolist() == [True, False, False, True]
    assert valid.tolist() == [True, False, True, True]


def test_split_is_disjoint_and_fixed():
    pilot, formal = split_indices(1000, 'pilot', 100), split_indices(1000, 'formal', 100)
    assert len(pilot) == 100 and len(formal) == 900 and not set(pilot) & set(formal)
    assert sorted(pilot + formal) == list(range(1000)) and pilot == split_indices(1000, 'pilot', 100)
    assert split_indices(5, 'all') == list(range(5))


def test_fixed_lists_match_their_seeded_draws():
    """data/splits holds the lists the code draws: the 5,000 ImageNet images and the 2,000 SRG images."""
    assert json.loads((ROOT / 'data/splits/imagenet_val_5000.json').read_text()) == subset_indices(50000, 5000, 0)
    rng = np.random.default_rng(0)
    srg = sorted(np.arange(12419)[rng.choice(12419, 2000, replace=False)].tolist())
    assert json.loads((ROOT / 'data/splits/imagenet_s_srg_2000.json').read_text()) == srg
    held = json.loads((ROOT / 'data/splits/heldout_4000.json').read_text())
    assert len(held['rows']) == len(set(held['rows'])) == held['probe_images'] + held['mean_images'] == 4000
