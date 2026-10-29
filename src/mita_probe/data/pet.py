"""Oxford-IIIT Pet test split: images aligned with their trimaps, labelled per 16x16 patch."""

import numpy as np
import torch
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
    def __init__(self, root, model, *, patch_size=16, preprocessing=None):
        self.dataset = OxfordIIITPet(root, split="test", target_types="segmentation", download=False)
        self.patch_size = patch_size
        self.preprocessing = preprocessing or resolve_model_data_config(model)
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
        foreground, valid = token_foreground(mask, self.patch_size)
        return self.transform(image), rgb, mask, foreground, valid

    def __getitem__(self, index):
        tensor, _, _, foreground, valid = self.aligned(index)
        return tensor, foreground, valid, index

