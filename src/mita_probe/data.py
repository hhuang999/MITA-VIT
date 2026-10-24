"""Imagenette validation images with labels in the pretrained ImageNet-1k space."""

from pathlib import Path

import numpy as np
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import Dataset
from torchvision.datasets import Imagenette


IMAGENET_IDS = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}


class Imagenette1k(Dataset):
    def __init__(self, root: Path, model):
        self.preprocessing = resolve_model_data_config(model)
        self.dataset = Imagenette(
            root, split="val", size="160px",
            transform=create_transform(**self.preprocessing, is_training=False),
        )
        self.labels = [IMAGENET_IDS[wnid] for wnid in self.dataset.wnids]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, self.labels[target]


def subset_indices(n_total: int, n_images: int, seed: int = 0) -> list[int]:
    return np.random.default_rng(seed).choice(n_total, n_images, replace=False).tolist()
