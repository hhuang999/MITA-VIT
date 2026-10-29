"""ImageNet-S919 validation masks (Gao et al., TPAMI 2022) on ImageNet validation images.

12,419 validation images of 919 ImageNet classes carry pixel-level semantic masks. A mask is an RGB PNG
whose class id is R + 256 G: 0 = other, k = the k-th category of the 919-class list, 1000 = ignore.
The object is the image's own category (the folder it is filed under, which already merges the five
ImageNet-S class merges); every other annotated pixel counts as background.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .imagenet import ImageNetFiles, wnid_index
from .masks import MaskDataset, trimap_from_labels

ANNOTATION_URL = ('https://github.com/LUSSeg/ImageNet-S/releases/download/ImageNet-S/'
                  'ImageNetS919-5f7f58ae1003d21da9409a8576bf7680.zip')
ANNOTATION_SHA256 = '31de0324c3a36e0817eeab0d704a7241b77ae5e725aeab103fd60c5b78dadf5e'
N_VALIDATION = 12419
IGNORE = 1000


class ImageNetS(MaskDataset):
    def __init__(self, root: Path, imagenet: ImageNetFiles, reference, *, preprocessing=None, expected=N_VALIDATION):
        """root: the extracted ImageNetS919 folder (holds validation-segmentation/<wnid>/<name>.png)."""
        super().__init__(reference, preprocessing=preprocessing)
        self.masks = sorted(Path(root, 'validation-segmentation').glob('*/*.png'))
        if len(self.masks) != expected:
            raise ValueError(f'Expected {expected} ImageNet-S919 validation masks, found {len(self.masks)}')
        self.categories = sorted({path.parent.name for path in self.masks})
        self.imagenet = imagenet

    def __len__(self):
        return len(self.masks)

    def name(self, index):
        return self.masks[index].stem  # ILSVRC2012_val_000xxxxx

    def sample(self, index):
        path = self.masks[index]
        rgb = np.array(Image.open(path).convert('RGB')).astype(np.int64)
        ids = rgb[..., 0] + 256 * rgb[..., 1]
        category = path.parent.name
        trimap = trimap_from_labels(ids, self.categories.index(category) + 1, IGNORE)
        return self.imagenet.image(self.name(index)), wnid_index(category), Image.fromarray(trimap)
