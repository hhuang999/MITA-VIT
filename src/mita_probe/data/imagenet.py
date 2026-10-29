"""ImageNet-1k validation split (pinned Hugging Face revision), the fixed 5,000-image subset and the shifted-crop noise floor."""

from pathlib import Path
import json

import numpy as np
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import Dataset

IMAGENET_REPO = "ILSVRC/imagenet-1k"
IMAGENET_REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"


class ShiftedCenterCrop:
    """torchvision CenterCrop with the window moved right by `dx` pixels (noise floor)."""

    def __init__(self, size: int, dx: int = 1):
        self.size, self.dx = size, dx

    def __call__(self, image):
        from torchvision.transforms import functional as TF

        width, height = image.size
        top = int(round((height - self.size) / 2.0))
        left = int(round((width - self.size) / 2.0)) + self.dx
        if left + self.size > width:
            raise ValueError("Shifted crop leaves the resized image")
        return TF.crop(image, top, left, self.size, self.size)


def shifted_transform(transform, dx: int = 1):
    """Replace the eval CenterCrop of a timm transform by a shifted one."""
    from torchvision.transforms import CenterCrop, Compose

    steps = list(transform.transforms)
    crops = [i for i, step in enumerate(steps) if isinstance(step, CenterCrop)]
    if len(crops) != 1:
        raise ValueError("Expected exactly one CenterCrop in the eval transform")
    size = steps[crops[0]].size
    steps[crops[0]] = ShiftedCenterCrop(size[0] if isinstance(size, (tuple, list)) else size, dx)
    return Compose(steps)


class ImageNetValidation(Dataset):
    """Load the pinned Hugging Face validation split saved by download_data.py."""

    def __init__(self, root: Path, model, *, preprocessing=None, shift: int = 0):
        from datasets import load_from_disk

        self.dataset = load_from_disk(str(root))
        if len(self.dataset) != 50000:
            raise ValueError("Expected 50,000 ImageNet validation images")
        self.preprocessing = preprocessing or resolve_model_data_config(model)
        self.transform = create_transform(**self.preprocessing, is_training=False)
        if shift:
            self.transform = shifted_transform(self.transform, shift)

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
