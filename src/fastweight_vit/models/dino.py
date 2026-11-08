"""DINO ViT-S/16 with the official linear head (dino_deitsmall16_linearweights.pth).

Matches facebookresearch/dino eval_linear.py for ViT-S/16: concatenate the CLS token of
the last 4 blocks, each passed through the final norm, then one linear layer (76.98%).
The backbone is timm's converted checkpoint; every reading sees its blocks directly.
"""

from pathlib import Path

import timm
import torch
from torch import nn

DINO_NAME = 'vit_small_patch16_224.dino'
DINO_REVISION = '10e440b8a34dfd657a90f2fcaa988c6b6a2f0da4'
HEAD = Path('checkpoints/dino/dino_deitsmall16_linearweights.pth')
HEAD_URL = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth'
HEAD_SHA256 = '23250c67b1ae117ef8149c8afd9c212390bdb6f00e33f92be8c17cc306dc6c1b'
N_LAST_BLOCKS = 4


class DinoLinear(nn.Module):
    def __init__(self, backbone, linear):
        super().__init__()
        self.backbone, self.linear = backbone, linear

    @property
    def blocks(self):
        return self.backbone.blocks

    @property
    def patch_embed(self):
        return self.backbone.patch_embed

    @property
    def num_prefix_tokens(self):
        return self.backbone.num_prefix_tokens

    def forward(self, x):
        b = self.backbone
        x = b.norm_pre(b.patch_drop(b._pos_embed(b.patch_embed(x))))
        features = []
        for i, block in enumerate(b.blocks):
            x = block(x)
            if i >= len(b.blocks) - N_LAST_BLOCKS:
                features.append(b.norm(x)[:, 0])
        return self.linear(torch.cat(features, -1))


def load_dino_model(head_path: Path, device='cpu', pretrained=True):
    backbone = timm.create_model(DINO_NAME, pretrained=pretrained, num_classes=0,
                                 pretrained_cfg_overlay={'hf_hub_id': f'timm/{DINO_NAME}@{DINO_REVISION}'})
    linear = nn.Linear(N_LAST_BLOCKS * backbone.embed_dim, 1000)
    if head_path is not None:
        state = torch.load(head_path, map_location='cpu', weights_only=True)['state_dict']
        linear.load_state_dict({key.removeprefix('module.linear.'): value for key, value in state.items()})
    return DinoLinear(backbone, linear).eval().to(device)
