"""Model registry: load a studied model by name, with its checkpoint verified.

| name | model | ImageNet preprocessing |
| --- | --- | --- |
| deit | DeiT-S/16, supervised | timm default for the checkpoint (Resize 248, crop 0.9) |
| deit_b | DeiT-B/16, supervised (the larger model) | timm default for the checkpoint (Resize 248, crop 0.9) |
| dino | DINO ViT-S/16 + official linear head | DINO linear evaluation (Resize 256, crop 0.875) |
| mita | MiTA-DeiT-S (dense re-implementation) | MiTA evaluation (Resize 256, crop 0.875) |

Every model exposes `blocks`, `patch_embed` and `num_prefix_tokens`, so the probe treats them alike.
"""

import hashlib

from ..runtime import configure_runtime, project_path

configure_runtime()

from .deit import load_deit  # noqa: E402
from .dino import HEAD, HEAD_SHA256, load_dino_model  # noqa: E402
from .mita import CHECKPOINT, CHECKPOINT_SHA256, load_mita_model  # noqa: E402

NAMES = ('deit', 'deit_b', 'dino', 'mita')
# Resize(int(224 / 0.875)) bicubic, CenterCrop(224): DINO linear and MiTA evaluation.
CROP_0875 = dict(input_size=(3, 224, 224), interpolation='bicubic', crop_pct=0.875,
                 crop_mode='center', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
IMAGENET_PREPROCESSING = {'deit': None, 'deit_b': None, 'dino': CROP_0875, 'mita': CROP_0875}


def verified(path, sha256):
    path = project_path(str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f'{path} has SHA-256 {digest}, expected {sha256}')
    return path


def load_model(name, device='cpu'):
    if name == 'deit':
        return load_deit(device)
    if name == 'deit_b':
        return load_deit(device, base=True)
    if name == 'dino':
        return load_dino_model(verified(HEAD, HEAD_SHA256), device)
    if name == 'mita':
        return load_mita_model(verified(CHECKPOINT, CHECKPOINT_SHA256), device)
    raise ValueError(f'Unknown model {name}; choose from {NAMES}')


def data_reference(model):
    """The timm module whose pretrained data config defines the default preprocessing."""
    return getattr(model, 'backbone', model)
