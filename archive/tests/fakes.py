"""Part masks used only by archived tests."""

import torch
from torch.utils.data import Dataset


class FakeParts(Dataset):
    """(image, object, valid, part fractions [196, 41], index): left half head (part 0), right half body (part 1)."""

    def __init__(self, n=3):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        fractions = torch.zeros(196, 41)
        columns = torch.arange(196) % 14
        fractions[columns < 5, 0] = 1
        fractions[(columns >= 5) & (columns < 10), 1] = 1
        fractions[columns >= 10, 40] = 1
        obj = columns < 10
        return torch.randn(3, 224, 224, generator=generator), obj, torch.ones(196, dtype=torch.bool), fractions, index
