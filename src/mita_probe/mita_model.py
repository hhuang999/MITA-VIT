"""MiTA-DeiT-S with a dense re-implementation of the official MiTA_Attention.

Matches github.com/QishuaiWen/MiTA mita/mita_attention.py (commit 95460f7):
router = 2D average pool of the first 196 query tokens (CLS included, last patch
dropped), m = 25, k = 25, argmax routing over all queries, summary slots scored as
q.router/d, all slots in one softmax. Weights load into the timm DeiT-S skeleton.
"""

import argparse
from pathlib import Path

import timm
import torch
from timm.data import create_transform

from .anatomy import Plan, install

MITA_SIDE, MITA_K = 5, 25
CHECKPOINT = Path('checkpoints/mita/best_checkpoint.pth')
CHECKPOINT_SHA256 = 'd45434671d1096c6d7bfcba38407eb6a632049adf1b1137d8ea85fa638778ac2'
# DeiT/MiTA evaluation: Resize(int(224 / 0.875)) bicubic, CenterCrop(224).
PREPROCESSING = dict(input_size=(3, 224, 224), interpolation='bicubic', crop_pct=0.875,
                     crop_mode='center', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def native_plan():
    return Plan(summary_side=MITA_SIDE, landmark_mode='mita', native_k=MITA_K)


def load_mita_model(path: Path, device='cpu'):
    """Load the 'model' weights (not EMA) with a weights-only unpickler."""
    torch.serialization.add_safe_globals([argparse.Namespace])
    state = torch.load(path, map_location='cpu', weights_only=True)['model']
    model = timm.create_model('deit_small_patch16_224', pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for block in model.blocks:
        install(block.attn, native_plan(), model.patch_embed.grid_size, model.num_prefix_tokens)
    return model


def mita_transform():
    return create_transform(**PREPROCESSING, is_training=False)
