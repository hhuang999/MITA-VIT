"""PartImageNet test from the original mixed test file."""

import numpy as np
from PIL import Image
from fastweight_vit.data import PartImageNet, partimagenet
from fakes import tiny


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
    data = PartImageNet(tmp_path, tiny(), splits=('val', 'test'), expected={'val': 2, 'test': 2})
    assert data.sample(3)[0].size == (400, 300)
    image, foreground, valid, fractions, index = data[3]
    assert image.shape == (3, 224, 224) and fractions.shape == (196, 41) and index == 3
    assert torch.allclose(fractions.sum(-1), torch.ones(196)) and fractions[:, 1].sum() > 0
    assert foreground.sum() > 0 and partimagenet.PART_NAMES[13] == 'Bird Head'

