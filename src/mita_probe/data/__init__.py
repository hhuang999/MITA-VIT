"""Datasets.

ImageNet validation (`imagenet`) feeds the faithfulness and structure experiments (RQ1, RQ2, RQ4).
Mask datasets feed the plausibility and necessity experiments (RQ3, RQ4). Every mask dataset returns
`(image tensor, object [196] bool, valid [196] bool, index)` (see masks.py); a new one subclasses
MaskDataset, implements `sample`, and is added to MASK_DATASETS in mita_probe.experiments.datasets.
PartImageNet also returns per-patch part fractions; ClickMe (clickme.py) holds human maps instead of masks.
"""

from .hard_imagenet import HardImageNet
from .imagenet import (IMAGENET_REPO, IMAGENET_REVISION, ImageNetFiles, ImageNetValidation, experiment_indices,
                       shifted_transform, subset_indices, wnid_index)
from .imagenet_s import ImageNetS
from .masks import MaskDataset, token_foreground, trimap_from_labels
from .partimagenet import PartImageNet
from .pet import PetSegmentation

__all__ = ['IMAGENET_REPO', 'IMAGENET_REVISION', 'HardImageNet', 'ImageNetFiles', 'ImageNetS', 'ImageNetValidation',
           'MaskDataset', 'PartImageNet', 'PetSegmentation', 'experiment_indices', 'shifted_transform', 'subset_indices',
           'token_foreground', 'trimap_from_labels', 'wnid_index']
