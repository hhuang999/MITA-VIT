"""Aligned Pet images, trimaps and spatial token labels for RQ3."""

import numpy as np
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import CenterCrop, InterpolationMode, Resize
from torchvision.transforms import functional as TF


def token_foreground(mask: torch.Tensor, patch_size=16):
    """Ignore label 3; a patch is foreground when at least half valid pixels are 1."""
    if mask.ndim != 2 or mask.shape[0] % patch_size or mask.shape[1] % patch_size:
        raise ValueError("Expected a patch-aligned two-dimensional trimap")
    if not torch.isin(mask, torch.tensor([1, 2, 3], device=mask.device)).all():
        raise ValueError("Pet trimap must contain only labels 1, 2, 3")
    h, w = mask.shape
    patches = mask.reshape(h//patch_size, patch_size, w//patch_size, patch_size).permute(0, 2, 1, 3)
    valid = patches.ne(3).sum((-1, -2))
    foreground = patches.eq(1).sum((-1, -2))
    return (foreground * 2 >= valid).flatten() & valid.flatten().gt(0), valid.flatten().gt(0)


class PetSegmentation(Dataset):
    def __init__(self, root, model):
        self.dataset = OxfordIIITPet(root, split="test", target_types="segmentation", download=False)
        self.preprocessing = resolve_model_data_config(model)
        self.transform = create_transform(**self.preprocessing, is_training=False)
        self.resize, self.crop = self.transform.transforms[:2]
        if not isinstance(self.resize, Resize) or not isinstance(self.crop, CenterCrop):
            raise ValueError("Expected the pinned DeiT eval Resize followed by CenterCrop")

    def __len__(self):
        return len(self.dataset)

    def aligned(self, index):
        image, mask = self.dataset[index]
        if image.size != mask.size:
            raise ValueError("Image and segmentation dimensions differ")
        rgb = self.crop(self.resize(image.convert("RGB")))
        mask = TF.resize(mask, self.resize.size, interpolation=InterpolationMode.NEAREST,
                         max_size=self.resize.max_size)
        mask = TF.center_crop(mask, self.crop.size)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        foreground, valid = token_foreground(mask)
        return self.transform(image), rgb, mask, foreground, valid

    def __getitem__(self, index):
        tensor, _, _, foreground, valid = self.aligned(index)
        return tensor, foreground, valid, index


def grid_region_membership(grid=14, regions=4):
    """Exact adaptive-average-pooling windows (overlap for 14 -> 4)."""
    membership = torch.zeros(regions*regions, grid, grid, dtype=torch.bool)
    for y in range(regions):
        for x in range(regions):
            membership[y*regions+x, y*grid//regions:((y+1)*grid+regions-1)//regions,
                       x*grid//regions:((x+1)*grid+regions-1)//regions] = True
    return membership.flatten(1)


def background_regions(foreground, valid):
    membership = grid_region_membership().to(foreground.device)
    count = valid.float() @ membership.float().T
    fg_count = (foreground & valid).float() @ membership.float().T
    return (2 * fg_count < count) & count.gt(0)


def selection_counts(readsets, n_patches=196):
    batch = readsets.shape[0]
    counts = torch.zeros(batch, n_patches+1, device=readsets.device)
    flat = readsets.flatten(1)
    counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=counts.dtype))
    return counts[:, 1:]
