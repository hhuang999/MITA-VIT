"""PartImageNet part masks (He et al., ECCV 2022), PartImageNet_Seg release.

24,080 images of 158 ImageNet classes in 11 super-categories, each with a per-pixel part mask: an 'L' PNG
whose value is the part id 0-39 (PART_NAMES, the COCO category ids of the release) or 40 for background.
The images are ImageNet *training* images (file names n<wnid>_<number>), so the classifiers saw them during
training. Parts are evaluated in the same geometry as the model input: the model's eval Resize +
CenterCrop, applied to the mask with nearest-neighbour resizing.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .imagenet import wnid_index
from .masks import MaskDataset

ARCHIVE_URL = ('https://huggingface.co/datasets/turkeyju/PartImageNet/resolve/'
               'cfa2edf3cf5ffbede2bd1fc934edca0aba99f042/PartImageNet_Seg.zip')
ARCHIVE_SHA256 = '9719778db7a7f589af94de4d7e4a025b832835502df370154f7c0a8b35466090'  # HF LFS object id
SPLITS = ('train', 'val', 'test')
N_IMAGES = {'train': 20466, 'val': 1206, 'test': 2408}
BACKGROUND = 40
PART_NAMES = [
    'Quadruped Head', 'Quadruped Body', 'Quadruped Foot', 'Quadruped Tail',
    'Biped Head', 'Biped Body', 'Biped Hand', 'Biped Foot', 'Biped Tail',
    'Fish Head', 'Fish Body', 'Fish Fin', 'Fish Tail',
    'Bird Head', 'Bird Body', 'Bird Wing', 'Bird Foot', 'Bird Tail',
    'Snake Head', 'Snake Body',
    'Reptile Head', 'Reptile Body', 'Reptile Foot', 'Reptile Tail',
    'Car Body', 'Car Tier', 'Car Side Mirror',
    'Bicycle Body', 'Bicycle Head', 'Bicycle Seat', 'Bicycle Tier',
    'Boat Body', 'Boat Sail',
    'Aeroplane Head', 'Aeroplane Body', 'Aeroplane Engine', 'Aeroplane Wing', 'Aeroplane Tail',
    'Bottle Mouth', 'Bottle Body',
]
SUPERCATEGORY = [name.split(' ')[0] for name in PART_NAMES]
PART = [name.split(' ', 1)[1] for name in PART_NAMES]
ANIMALS = ('Quadruped', 'Biped', 'Fish', 'Bird', 'Snake', 'Reptile')


def part_fractions(labels: torch.Tensor, patch_size=16, n_labels=BACKGROUND + 1):
    """[H, W] part ids -> [patches, n_labels]: the share of each patch's pixels with each id."""
    h, w = labels.shape
    patches = labels.reshape(h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 2, 1, 3)
    onehot = torch.nn.functional.one_hot(patches.reshape(-1, patch_size * patch_size), n_labels)
    return onehot.float().mean(1)


class PartImageNet(MaskDataset):
    def __init__(self, root: Path, reference, *, preprocessing=None, splits=SPLITS, expected=None):
        """root: the extracted folder that holds images/<split>/ and annotations/<split>/."""
        super().__init__(reference, preprocessing=preprocessing)
        self.items = []
        for split in splits:
            names = sorted(path.stem for path in Path(root, 'annotations', split).glob('n*.png'))
            count = expected[split] if expected else N_IMAGES[split]
            if len(names) != count:
                raise ValueError(f'Expected {count} PartImageNet {split} masks, found {len(names)}')
            self.items += [(split, name) for name in names]
        self.root = Path(root)

    def __len__(self):
        return len(self.items)

    def labels(self, index):
        split, name = self.items[index]
        return np.array(Image.open(self.root / 'annotations' / split / f'{name}.png'))

    def sample(self, index):
        split, name = self.items[index]
        image = Image.open(self.root / 'images' / split / f'{name}.JPEG').convert('RGB')
        trimap = np.where(self.labels(index) < BACKGROUND, 1, 2).astype(np.uint8)
        return image, wnid_index(name.split('_')[0]), Image.fromarray(trimap)

    def cropped_labels(self, index):
        """Part ids in the model's input geometry, [224, 224] int64."""
        labels = Image.fromarray(self.labels(index))
        labels = TF.resize(labels, self.resize.size, interpolation=InterpolationMode.NEAREST,
                           max_size=self.resize.max_size)
        return torch.from_numpy(np.array(TF.center_crop(labels, self.crop.size), dtype=np.int64))

    def __getitem__(self, index):
        tensor, _, _, foreground, valid = self.aligned(index)
        return tensor, foreground, valid, part_fractions(self.cropped_labels(index), self.patch_size), index
