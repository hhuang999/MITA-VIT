"""DeiT-S/16 and DeiT-B/16 trained with ImageNet-1k labels, at pinned timm revisions."""

import timm

NAME = 'deit_small_patch16_224.fb_in1k'
REPO = f'timm/{NAME}'
REVISION = '91327a9c99f98fe6b524cd4d397b7226b80e1365'
BASE_NAME = 'deit_base_patch16_224.fb_in1k'
BASE_REVISION = 'b78cc5532a69df6bcad9c3a8d76653fd20b31ac6'


def load_deit(device='cpu', pretrained=True, base=False):
    name, revision = (BASE_NAME, BASE_REVISION) if base else (NAME, REVISION)
    return timm.create_model(name, pretrained=pretrained,
                             pretrained_cfg_overlay={'hf_hub_id': f'timm/{name}@{revision}'}).eval().to(device)
