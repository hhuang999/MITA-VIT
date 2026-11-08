"""Build an evaluation dataset by name for a given model, with the right preprocessing.

| name | content | preprocessing |
| --- | --- | --- |
| imagenet | the fixed 5,000-image ImageNet validation subset (data/splits), no masks | the model's ImageNet evaluation |
| imagenet_s | ImageNet-S919 validation masks, 12,419 images | the model's ImageNet evaluation |
| partimagenet | PartImageNet_Seg part masks, 24,080 ImageNet training images (`partimagenet_splits` selects
  splits) | the model's ImageNet evaluation |

`preprocessing: uniform` runs every model with models.UNIFORM_PREPROCESSING (Resize 256, crop 0.875; used by the
attention-off check and the register search). `split: pilot|formal` (with `pilot_images`, `split_seed`) keeps one
part of a mask dataset (data.splits; the register search uses the pilot part).

Mask datasets yield (image, object, valid, index); `imagenet` yields (image, label); `partimagenet` adds per-patch
part fractions: (image, object, valid, parts, index).
"""

from ..models import IMAGENET_PREPROCESSING, UNIFORM_PREPROCESSING, data_reference
from ..runtime import data_root, project_path
from .imagenet import ImageNetFiles, ImageNetValidation, experiment_indices
from .imagenet_s import ImageNetS
from .partimagenet import PartImageNet
from .splits import split_indices

NAMES = ('imagenet', 'imagenet_s', 'partimagenet')
_FILES = {}


def imagenet_files():
    """One shared file-name index over the ImageNet validation copy."""
    if 'files' not in _FILES:
        _FILES['files'] = ImageNetFiles(data_root() / 'imagenet_val')
    return _FILES['files']


def build(name, model, model_name, config):
    """Return (dataset, indices)."""
    reference = data_reference(model)
    preprocessing = UNIFORM_PREPROCESSING if config.get('preprocessing') == 'uniform' \
        else IMAGENET_PREPROCESSING[model_name]
    if name == 'imagenet':
        dataset = ImageNetValidation(data_root() / 'imagenet_val', reference, preprocessing=preprocessing)
        return dataset, experiment_indices(len(dataset), config['n_images'], config['subset_seed'],
                                           project_path(config['indices']))
    if name == 'imagenet_s':
        dataset = ImageNetS(data_root() / 'imagenet_s' / 'ImageNetS919', imagenet_files(), reference,
                            preprocessing=preprocessing)
    elif name == 'partimagenet':
        splits = tuple(config.get('partimagenet_splits', ('train', 'val', 'test')))
        dataset = PartImageNet(data_root() / 'partimagenet' / 'PartImageNet', reference, preprocessing=preprocessing,
                               splits=splits)
    else:
        raise ValueError(f'Unknown dataset {name}; choose from {NAMES}')
    split = config.get('split', 'all')
    return dataset, split_indices(len(dataset), split, config.get('pilot_images', 500), config.get('split_seed', 0))
