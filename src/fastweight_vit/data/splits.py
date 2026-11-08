"""Pilot / formal split of a mask dataset.

The dataset's images are shuffled with a seeded generator; the first `pilot` images form the pilot set (the
register-neuron search runs on it) and the rest the formal set. Both lists are returned sorted so data loading
stays sequential.
"""

import numpy as np

SPLITS = ('pilot', 'formal', 'all')


def split_indices(n_total: int, split: str, pilot: int = 500, seed: int = 0) -> list[int]:
    if split not in SPLITS:
        raise ValueError(f'Unknown split {split}; choose from {SPLITS}')
    if split == 'all':
        return list(range(n_total))
    if not 0 < pilot < n_total:
        raise ValueError('The pilot set must be a proper, nonempty part of the dataset')
    order = np.random.default_rng(seed).permutation(n_total)
    chosen = order[:pilot] if split == 'pilot' else order[pilot:]
    return sorted(int(index) for index in chosen)
