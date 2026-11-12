"""Fixed, index-ordered examples from the original seeded ClickMe gallery."""

from pathlib import Path

import numpy as np

KEYS = ('image', 'human', 'map_R0', 'map_C3', 'map_C2')
SELECTED = {'figure_1b': [47, 287], 'figure_D1': [712, 1308, 3048, 4691]}


def load_gallery(folder):
    """Load all 24 samples, checking completeness before choosing any example."""
    paths = sorted(Path(folder).glob('gallery_*.npz'))
    if not paths:
        raise ValueError(f'Missing seeded ClickMe gallery: {folder}/gallery_*.npz')
    chunks = {key: [] for key in ('index', *KEYS)}
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for key in chunks:
                if key not in arrays:
                    raise ValueError(f'{path}: missing gallery array {key}')
                value = arrays[key]
                if len(value) != len(arrays['index']) or not np.isfinite(value).all():
                    raise ValueError(f'{path}: incompatible or non-finite {key}')
                chunks[key].append(value)
    data = {key: np.concatenate(values) for key, values in chunks.items()}
    indices = data['index']
    if len(indices) != 24 or len(np.unique(indices)) != 24:
        raise ValueError(f'Expected 24 unique seeded gallery indices, got {len(indices)} samples')
    order = np.argsort(indices, kind='stable')
    data = {key: value[order] for key, value in data.items()}
    if data['index'][:6].tolist() != SELECTED['figure_1b'] + SELECTED['figure_D1']:
        raise ValueError('Gallery indices differ from the fixed selection')
    for key in ('map_R0', 'map_C3', 'map_C2'):
        if data[key].shape != (24, 196):
            raise ValueError(f'Expected 24 x 196 patch scores: {key} {data[key].shape}')
    return data
