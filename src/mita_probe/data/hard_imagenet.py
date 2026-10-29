"""Hard ImageNet validation masks (Moayeri et al., NeurIPS 2022 D&B) on ImageNet validation images.

750 validation images of 15 classes with strong spurious context (e.g. ski, snorkel, seat belt), whose
objects are smaller and less centred than usual. Masks are stored as JPEG files named
`<wnid>_ILSVRC2012_val_000xxxxx.JPEG`; JPEG compression adds faint noise, so the object is mask >= 128.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .imagenet import ImageNetFiles, wnid_index
from .masks import MaskDataset

ANNOTATION_URL = ('https://app.box.com/index.php?rm=box_download_shared_file&'
                  'shared_name=ca7qlcfsqlfqul9rzgtuqhb2c6pm62qd&file_id=f_972129165893')
ANNOTATION_SHA256 = 'e5ac379bf8cbe4811411b3681808260ad6fb3afb23e408a5c5a4e5819113bcf2'
N_VALIDATION = 750


class HardImageNet(MaskDataset):
    def __init__(self, root: Path, imagenet: ImageNetFiles, reference, *, preprocessing=None, expected=N_VALIDATION):
        """root: the extracted hardImageNet folder (holds val/<wnid>_<name>.JPEG masks)."""
        super().__init__(reference, preprocessing=preprocessing)
        self.masks = sorted(Path(root, 'val').glob('n*_ILSVRC2012_val_*.JPEG'))
        if len(self.masks) != expected:
            raise ValueError(f'Expected {expected} Hard ImageNet validation masks, found {len(self.masks)}')
        self.imagenet = imagenet

    def __len__(self):
        return len(self.masks)

    def name(self, index):
        return self.masks[index].stem.split('_', 1)[1]

    def sample(self, index):
        path = self.masks[index]
        mask = np.array(Image.open(path).convert('L'))
        trimap = np.where(mask >= 128, 1, 2).astype(np.uint8)
        return self.imagenet.image(self.name(index)), wnid_index(path.stem.split('_', 1)[0]), Image.fromarray(trimap)
