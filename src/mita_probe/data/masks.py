"""Mask datasets: an image with a per-pixel object mask, both cropped exactly as the model sees them.

A subclass implements `__len__` and `sample(index) -> (RGB PIL image, target class, trimap)`, where the
trimap is a PIL 'L' image with 1 = object, 2 = background, 3 = ignore (boundary or unlabelled). The base
class applies the model's eval Resize + CenterCrop to the image and (nearest-neighbour) to the trimap, then
labels each 16x16 patch: valid when it has at least one non-ignored pixel, object when at least half of its
valid pixels are object. Every dataset returns `(image tensor, object [196] bool, valid [196] bool, index)`.
"""

import numpy as np
import torch
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, InterpolationMode, Resize
from torchvision.transforms import functional as TF


def token_foreground(mask: torch.Tensor, patch_size=16):
    """Ignore label 3; a patch is foreground when at least half valid pixels are 1."""
    if mask.ndim != 2 or mask.shape[0] % patch_size or mask.shape[1] % patch_size:
        raise ValueError("Expected a patch-aligned two-dimensional trimap")
    if not torch.isin(mask, torch.tensor([1, 2, 3], device=mask.device)).all():
        raise ValueError("Trimap must contain only labels 1, 2, 3")
    h, w = mask.shape
    patches = mask.reshape(h // patch_size, patch_size, w // patch_size, patch_size).permute(0, 2, 1, 3)
    valid = patches.ne(3).sum((-1, -2))
    foreground = patches.eq(1).sum((-1, -2))
    return (foreground * 2 >= valid).flatten() & valid.flatten().gt(0), valid.flatten().gt(0)


def trimap_from_labels(labels: np.ndarray, target: int, ignore: int | None = None):
    """Per-pixel class ids -> trimap: the target class is object, `ignore` is ignored, the rest background."""
    trimap = np.full(labels.shape, 2, dtype=np.uint8)
    trimap[labels == target] = 1
    if ignore is not None:
        trimap[labels == ignore] = 3
    return trimap


class MaskDataset(Dataset):
    patch_size = 16

    def __init__(self, reference, *, preprocessing=None):
        """reference: the timm model whose data config defines preprocessing (unless given explicitly)."""
        self.preprocessing = preprocessing or resolve_model_data_config(reference)
        self.transform = create_transform(**self.preprocessing, is_training=False)
        self.resize, self.crop = self.transform.transforms[:2]
        if not isinstance(self.resize, Resize) or not isinstance(self.crop, CenterCrop):
            raise ValueError("Expected an eval transform that starts with Resize then CenterCrop")

    def sample(self, index):
        raise NotImplementedError

    def target(self, index):
        """The ImageNet-1k class index of the annotated object, or -1 when there is none."""
        return self.sample(index)[1]

    def aligned(self, index):
        """(model input, cropped RGB, cropped trimap, object tokens, valid tokens)."""
        image, _, trimap = self.sample(index)
        if image.size != trimap.size:
            raise ValueError("Image and mask dimensions differ")
        rgb = self.crop(self.resize(image))
        mask = TF.resize(trimap, self.resize.size, interpolation=InterpolationMode.NEAREST,
                         max_size=self.resize.max_size)
        mask = torch.from_numpy(np.array(TF.center_crop(mask, self.crop.size), dtype=np.int64))
        foreground, valid = token_foreground(mask, self.patch_size)
        return self.transform(image), rgb, mask, foreground, valid

    def __getitem__(self, index):
        tensor, _, _, foreground, valid = self.aligned(index)
        return tensor, foreground, valid, index
