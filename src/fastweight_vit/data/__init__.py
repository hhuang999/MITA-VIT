"""Datasets and fixed image lists.

`imagenet` (the fixed 5,000-image validation subset) feeds the execution checks; ImageNet-S masks feed the
same-ring enrichment and SRG; ClickMe (clickme.py) holds human maps; PartImageNet adds per-patch part
fractions (part reliance). Mask datasets return `(image, object [196] bool, valid [196] bool, index)`.
`build.build` returns a dataset and its image indices for a model, with the model's preprocessing.
"""

from .imagenet import ImageNetFiles, ImageNetValidation, experiment_indices, subset_indices, wnid_index
from .imagenet_s import ImageNetS
from .masks import MaskDataset, token_foreground, trimap_from_labels
from .partimagenet import PartImageNet

__all__ = ['ImageNetFiles', 'ImageNetS', 'ImageNetValidation', 'MaskDataset', 'PartImageNet', 'experiment_indices',
           'subset_indices', 'token_foreground', 'trimap_from_labels', 'wnid_index']
