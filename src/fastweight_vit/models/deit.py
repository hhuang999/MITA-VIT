"""Supervised DeiT-family ViTs trained with ImageNet-1k labels, at pinned timm revisions.

| key | timm model | used for |
| --- | --- | --- |
| small | deit_small_patch16_224.fb_in1k (DeiT-S) | every experiment |
| base | deit_base_patch16_224.fb_in1k (DeiT-B) | every experiment |
"""

import timm

CHECKPOINTS = {
    'small': ('deit_small_patch16_224.fb_in1k', '91327a9c99f98fe6b524cd4d397b7226b80e1365'),
    'base': ('deit_base_patch16_224.fb_in1k', 'b78cc5532a69df6bcad9c3a8d76653fd20b31ac6'),
}
NAME, REVISION = CHECKPOINTS['small']
REPO = f'timm/{NAME}'


def load_deit(device='cpu', pretrained=True, base=False):
    name, revision = CHECKPOINTS['base' if base else 'small']
    return timm.create_model(name, pretrained=pretrained,
                             pretrained_cfg_overlay={'hf_hub_id': f'timm/{name}@{revision}'}).eval().to(device)
