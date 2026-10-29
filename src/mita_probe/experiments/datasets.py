"""Build an evaluation dataset by name for a given model, with the right preprocessing.

| name | content | preprocessing |
| --- | --- | --- |
| imagenet | the fixed 5,000-image ImageNet validation subset, no masks | the model's ImageNet evaluation |
| pet | Oxford-IIIT Pet test, 3,669 trimaps | DeiT-S geometry for every model (identical object labels) |
| imagenet_s | ImageNet-S919 validation masks, 12,419 images | the model's ImageNet evaluation |
| hard_imagenet | Hard ImageNet validation masks, 750 images | the model's ImageNet evaluation |
| partimagenet | PartImageNet_Seg part masks, 24,080 ImageNet training images | the model's ImageNet evaluation |

Mask datasets yield (image, object, valid, index); `imagenet` yields (image, label) — experiments that
need masks reject it; `partimagenet` adds per-patch part fractions: (image, object, valid, parts, index).
"""

from ..data import (HardImageNet, ImageNetFiles, ImageNetS, ImageNetValidation, PartImageNet, PetSegmentation,
                    experiment_indices)
from ..models import IMAGENET_PREPROCESSING, data_reference
from ..runtime import data_root, project_path

MASK_DATASETS = ('pet', 'imagenet_s', 'hard_imagenet')
NAMES = ('imagenet',) + MASK_DATASETS + ('partimagenet',)
_FILES = {}


def imagenet_files():
    """One shared file-name index over the ImageNet validation copy."""
    if 'files' not in _FILES:
        _FILES['files'] = ImageNetFiles(data_root() / 'imagenet_val')
    return _FILES['files']


def build(name, model, model_name, config, deit_reference=None):
    """Return (dataset, indices). deit_reference: the DeiT-S model whose geometry Pet uses for all models."""
    reference = data_reference(model)
    preprocessing = IMAGENET_PREPROCESSING[model_name]
    if name == 'imagenet':
        dataset = ImageNetValidation(data_root() / 'imagenet_val', reference, preprocessing=preprocessing)
        return dataset, experiment_indices(len(dataset), config['n_images'], config['subset_seed'],
                                           project_path(config['indices']))
    if name == 'pet':
        dataset = PetSegmentation(data_root(), data_reference(deit_reference or model))
    elif name == 'imagenet_s':
        dataset = ImageNetS(data_root() / 'imagenet_s' / 'ImageNetS919', imagenet_files(), reference,
                            preprocessing=preprocessing)
    elif name == 'hard_imagenet':
        dataset = HardImageNet(data_root() / 'hard_imagenet' / 'hardImageNet', imagenet_files(), reference,
                               preprocessing=preprocessing)
    elif name == 'partimagenet':
        dataset = PartImageNet(data_root() / 'partimagenet' / 'PartImageNet', reference, preprocessing=preprocessing)
    else:
        raise ValueError(f'Unknown dataset {name}; choose from {NAMES}')
    return dataset, list(range(len(dataset)))
