"""ClickMe parsing, PartImageNet parts, the sanity check and the gallery, on CPU with fake data."""

import io
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mita_probe.data import partimagenet
from mita_probe.data.clickme import ClickMe, gaussian_kernel, parse_example, record_offsets
from mita_probe.data.partimagenet import PartImageNet, part_fractions
from mita_probe.experiments import common, gallery, human_alignment, parts, sanity
from mita_probe.experiments.parts import assignment, part_measures
from mita_probe.io import load_config
from mita_probe.metrics import spearman
from mita_probe.models.deit import load_deit

from test_new_experiments import FakeMasks

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


def test_alignment_is_one_for_the_human_map():
    coarse = torch.rand(3, 1, 7, 7, generator=torch.Generator().manual_seed(0))
    human = torch.nn.functional.interpolate(coarse, size=(224, 224), mode='bilinear', align_corners=False)[:, 0]
    values = human_alignment.alignment(human_alignment.per_patch(human), human)
    assert torch.allclose(values['spearman_token'], torch.ones(3, dtype=torch.float64))
    assert (values['spearman_pixel'] > 0.8).all()


def test_spearman_matches_pandas_with_ties():
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 4, (5, 196), generator=g).float()
    b = torch.rand(5, 196, generator=g)
    ours = spearman(a, b).numpy()
    ref = [np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1] for x, y in zip(a.numpy(), b.numpy())]
    assert np.allclose(ours, ref)


def test_part_measures_on_an_indicator_map():
    labels = torch.full((224, 224), 40)
    labels[:64, :64] = 0       # head: 16 patches
    labels[64:160, :96] = 1    # body: 36 patches
    fractions = part_fractions(labels)[None]
    assign = assignment(fractions)
    assert (assign == 0).sum() == 16 and (assign == 1).sum() == 36
    head = (assign == 0).float()
    values = part_measures(head, fractions, assign)
    assert values['mass'][0, 0] == pytest.approx(1) and values['area'][0, 0] == pytest.approx(16 / 52)
    assert values['enrichment'][0, 0] == pytest.approx(52 / 16) and values['auroc'][0, 0] == 1
    assert values['auroc'][0, 1] == 0 and torch.isnan(values['auroc'][0, 2])


class FakeParts(PartImageNet):
    """Grey images, a head square and a body rectangle; two eval splits of two images each."""

    def __init__(self, reference):
        super(PartImageNet, self).__init__(reference)
        self.items = [('val', f'n01440764_{n}') for n in range(4)]

    def labels(self, index):
        labels = np.full((300, 400), 40, dtype=np.uint8)
        labels[40 + 10 * index:120 + 10 * index, 60:140] = 13      # bird head
        labels[120 + 10 * index:260, 60:300] = 14                  # bird body
        return labels

    def sample(self, index):
        labels = self.labels(index)
        image = np.where(labels[..., None] < 40, 200, 90).astype(np.uint8).repeat(3, -1)
        trimap = np.where(labels < 40, 1, 2).astype(np.uint8)
        return Image.fromarray(image), 0, Image.fromarray(trimap)


def test_partimagenet_reads_the_release_layout(tmp_path):
    for split in ('val', 'test'):
        for n in range(2):
            name = f'n01440764_{split}{n}'
            (tmp_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (tmp_path / 'annotations' / split).mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((300, 400, 3), 120, dtype=np.uint8)).save(tmp_path / 'images' / split / f'{name}.JPEG')
            labels = np.full((300, 400), 40, dtype=np.uint8)
            labels[100:200, 100:300] = 1
            Image.fromarray(labels).save(tmp_path / 'annotations' / split / f'{name}.png')
    # An image stored sideways with an EXIF rotation; its mask matches the displayed (rotated) image.
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(np.full((400, 300, 3), 120, dtype=np.uint8)).save(
        tmp_path / 'images' / 'test' / 'n01440764_test1.JPEG', exif=exif)
    torch.manual_seed(0)
    data = PartImageNet(tmp_path, load_deit(pretrained=False), splits=('val', 'test'), expected={'val': 2, 'test': 2})
    assert data.sample(3)[0].size == (400, 300)
    image, foreground, valid, fractions, index = data[3]
    assert image.shape == (3, 224, 224) and fractions.shape == (196, 41) and index == 3
    assert torch.allclose(fractions.sum(-1), torch.ones(196)) and fractions[:, 1].sum() > 0
    assert foreground.sum() > 0 and partimagenet.PART_NAMES[13] == 'Bird Head'


@pytest.fixture
def patched(monkeypatch, tmp_path):
    common.DEVICE = 'cpu'
    torch.manual_seed(0)
    model = load_deit(pretrained=False)
    for module in (human_alignment, parts, sanity, gallery):
        monkeypatch.setattr(module, 'load_model', lambda name, device: model)
    write_shards(tmp_path / 'data' / 'clickme')
    for module in (human_alignment, gallery):
        monkeypatch.setattr(module, 'data_root', lambda: tmp_path / 'data')

    def build(name, model_, model_name, config, deit_reference=None):
        dataset = FakeParts(model_) if name == 'partimagenet' else FakeMasks(model_)
        return dataset, list(range(len(dataset)))
    for module in (parts, sanity, gallery):
        monkeypatch.setattr(module.datasets, 'build', build)
    return model


def config_for(name, **settings):
    config = load_config(ROOT / f'configs/{name}.yaml')
    config.update(batch_size=2, num_workers=0)
    keep = {'probe_mass', 'cls_attention', 'gae', 'centre', 'random'}
    config['explanations']['methods'] = [m for m in config['explanations']['methods'] if m['name'] in keep | {'mask'}]
    config[name].update(settings)
    return config


def test_human_alignment_rows(patched, tmp_path):
    config = config_for('human_alignment', shards=2)
    out = tmp_path / 'out'
    out.mkdir()
    human_alignment.run(SimpleNamespace(model='deit', smoke=5, resume=False), config, out, 'test')
    rows = pd.read_csv(out / 'alignment.csv')
    methods = {'probe_mass', 'cls_attention', 'gae', 'centre', 'random', 'mean_human'} | {
        f'probe_layer{layer}' for layer in range(1, 13)}
    assert set(rows.method) == methods and len(rows) == 5 * len(methods)
    assert (rows.label == rows['index']).all() and rows.spearman_pixel.between(-1, 1).all()
    centre = rows.query("method == 'centre'")
    # A fixed map scores the same against any image's map up to which image it is paired with.
    assert centre.spearman_token.notna().all()


def test_parts_rows_and_reliance(patched, tmp_path):
    config = config_for('parts', layerwise=False, seeds=[0, 1], removed=[4])
    out = tmp_path / 'out'
    out.mkdir()
    parts.run(SimpleNamespace(model='deit', smoke=0, resume=False), config, out, 'test')
    rows = pd.read_csv(out / 'parts.csv')
    assert set(rows.part) == {13, 14} and len(rows) == 4 * 5 * 2
    assert np.allclose(rows.groupby(['index', 'method']).mass.sum(), 1)
    assert np.allclose(rows.groupby(['index', 'method']).area.sum(), 1)
    reliance = pd.read_csv(out / 'reliance.csv')
    assert set(reliance.condition) == {'part', 'object', 'background'} and (reliance.removed == 4).all()
    assert len(reliance) == 4 * 2 * 4 and reliance.p_target.between(0, 1).all()


def test_sanity_levels(patched, tmp_path):
    config = config_for('sanity', images=4)
    out = tmp_path / 'out'
    out.mkdir()
    sanity.run(SimpleNamespace(model='deit', smoke=0), config, out, 'test')
    rows = pd.read_csv(out / 'sanity.csv')
    assert rows.level.nunique() == 14 and len(rows) == 14 * 5 * 4
    assert np.allclose(rows.query("method == 'centre'").spearman, 1)
    head = rows.query("level == 'head'")
    assert np.allclose(head.query("method in ['probe_mass', 'cls_attention']").spearman, 1)
    last = rows.query("level == 'embed' and method == 'probe_mass'").spearman
    assert (last < 0.99).all()


def test_gallery_saves_every_map(patched, tmp_path):
    config = config_for('gallery', images=3, datasets=['imagenet_s', 'clickme', 'partimagenet'])
    config['human_alignment']['shards'] = 2
    out = tmp_path / 'out'
    out.mkdir()
    gallery.run(SimpleNamespace(model='deit'), config, out, 'test')
    clickme = np.load(out / 'gallery_clickme.npz')
    assert clickme['rgb'].shape == (3, 224, 224, 3) and clickme['truth'].shape == (3, 224, 224)
    assert clickme['probe_layers'].shape == (3, 12, 196) and clickme['gae'].shape == (3, 196)
    assert np.load(out / 'gallery_partimagenet.npz')['truth'].shape == (3, 224, 224)
    assert np.load(out / 'gallery_imagenet_s.npz')['truth'].shape == (3, 196)
