"""Imagenette validation images with labels in the pretrained ImageNet-1k space."""

from pathlib import Path
import json

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


class ImageNetValidation(Dataset):
    """Load the pinned Hugging Face validation split saved by download_data.py."""

    def __init__(self, root: Path, model):
        from datasets import load_from_disk

        self.dataset = load_from_disk(str(root))
        if len(self.dataset) != 50000:
            raise ValueError("Expected 50,000 ImageNet validation images")
        self.preprocessing = resolve_model_data_config(model)
        self.transform = create_transform(**self.preprocessing, is_training=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        return self.transform(row["image"].convert("RGB")), int(row["label"])


def subset_indices(n_total: int, n_images: int, seed: int = 0) -> list[int]:
    return np.random.default_rng(seed).choice(n_total, n_images, replace=False).tolist()


def experiment_indices(n_total: int, n_images: int, seed: int, saved: Path | None) -> list[int]:
    if saved is None:
        return subset_indices(n_total, n_images, seed)
    indices = json.loads(saved.read_text(encoding="utf-8"))
    if indices != subset_indices(n_total, 5000, seed):
        raise ValueError("Saved indices must match the fixed seed-0 5,000-image sample")
    if not 1 <= n_images <= len(indices):
        raise ValueError("Requested sample must fit within the fixed 5,000-image pool")
    return indices[:n_images]
