"""DeiT-S/16 trained with ImageNet-1k labels, at a pinned timm revision."""

import timm

NAME = 'deit_small_patch16_224.fb_in1k'
REPO = f'timm/{NAME}'
REVISION = '91327a9c99f98fe6b524cd4d397b7226b80e1365'


def load_deit(device='cpu', pretrained=True):
    return timm.create_model(NAME, pretrained=pretrained,
                             pretrained_cfg_overlay={'hf_hub_id': f'{REPO}@{REVISION}'}).eval().to(device)
