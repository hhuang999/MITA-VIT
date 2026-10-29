"""Plausibility measures, the token-identity check and the mask-dataset loaders."""

import numpy as np
import torch
from PIL import Image

from mita_probe.data import HardImageNet, ImageNetS
from mita_probe.data.imagenet_s import IGNORE
from mita_probe.explain import centre
from mita_probe.identity import self_similarity, train_position_probe
from mita_probe.models.deit import load_deit
from mita_probe.plausibility import auroc, geometry, measures

ROWS, COLS = np.divmod(np.arange(196), 14)


def square(top, left, size):
    return torch.from_numpy((ROWS >= top) & (ROWS < top + size) & (COLS >= left) & (COLS < left + size))


def test_object_scores_are_perfect_and_centre_scores_follow_centre_objects():
    fg = torch.stack([square(4, 4, 6), square(0, 0, 5)])
    valid = torch.ones_like(fg)
    perfect = measures(fg.float(), fg, valid, fg.flip(0))
    for key in ['mass', 'pointing', 'auroc', 'precision']:
        np.testing.assert_allclose(perfect[key], 1.0)
    assert (perfect['enrichment'] > 0).all()
    central = measures(centre(2), fg, valid, fg.flip(0))
    assert central['auroc'][0] > 0.95 and central['auroc'][1] < 0.5  # centred object vs corner object
    assert central['enrichment'][0] < 0.05  # at equal distance from the centre the centre map has no edge


def test_auroc_counts_ties_as_half_and_skips_missing_classes():
    scores = torch.tensor([[1., 1., 0., 2.]])
    positive = torch.tensor([[True, False, False, False]])
    valid = torch.ones_like(positive)
    assert auroc(scores, positive, valid)[0] == (0.5 + 1 + 0) / 3
    assert np.isnan(auroc(scores, torch.zeros_like(positive), valid)[0])


def test_geometry_area_and_distance():
    fg = torch.stack([square(6, 6, 2), square(0, 0, 2)])
    area, distance = geometry(fg, torch.ones_like(fg))
    np.testing.assert_allclose(area, 4 / 196)
    assert distance[0] == 0 and distance[1] > 8


def test_position_probe_and_self_similarity():
    torch.manual_seed(0)
    codes = torch.randn(196, 32)
    train = codes + 0.1 * torch.randn(20, 196, 32)
    test = codes + 0.1 * torch.randn(5, 196, 32)
    accuracy, distance = train_position_probe(train, test, epochs=20, batch=512)
    assert accuracy > 0.95 and distance < 0.5
    noise_accuracy, _ = train_position_probe(torch.randn(20, 196, 32), torch.randn(5, 196, 32), epochs=5,
                                             batch=512)
    assert noise_accuracy < 0.05
    same, other = self_similarity(test, test)
    assert torch.allclose(same, torch.ones(5)) and (other < 0.3).all()


class Stub:
    def image(self, name):
        return Image.new('RGB', (64, 48), (120, 120, 120))


def test_imagenet_s_and_hard_imagenet_masks(tmp_path):
    reference = load_deit(pretrained=False)
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
    hard = tmp_path / 'hardImageNet' / 'val'
    hard.mkdir(parents=True)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[:, :32] = 255
    Image.fromarray(mask).save(hard / 'n01440764_ILSVRC2012_val_00000007.JPEG')
    dataset = HardImageNet(tmp_path / 'hardImageNet', Stub(), reference, expected=1)
    _, fg, valid, _ = dataset[0]
    assert dataset.name(0) == 'ILSVRC2012_val_00000007' and dataset.target(0) == 0
    assert fg.reshape(14, 14)[:, :5].all() and not fg.reshape(14, 14)[:, 9:].any()
