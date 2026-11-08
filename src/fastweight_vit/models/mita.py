"""MiTA-DeiT-S and MiTA-DeiT-T with a dense re-implementation of the official MiTA_Attention.

Matches github.com/QishuaiWen/MiTA mita/mita_attention.py (commit 95460f7):
router = 2D average pool of the first 196 query tokens (CLS included, last patch
dropped), m = pool_size^2 = 25, k = kv_topk^2 = 25, argmax routing over all queries, summary slots
scored as q.router/d, all slots in one softmax. Weights load into the timm DeiT skeleton named in the
checkpoint's training arguments; pool_size and kv_topk come from those arguments when saved (the
MiTA-DeiT-S checkpoint predates them and used the defaults 5 and 5).
"""

import argparse
from pathlib import Path

import timm
import torch

from ..execution.execute import Plan, install

MITA_SIDE, MITA_K = 5, 25
CHECKPOINT = Path('checkpoints/mita/best_checkpoint.pth')
CHECKPOINT_SHA256 = 'd45434671d1096c6d7bfcba38407eb6a632049adf1b1137d8ea85fa638778ac2'
TINY_CHECKPOINT = Path('checkpoints/mita_tiny/best_checkpoint.pth')
TINY_DRIVE_FOLDER = 'https://drive.google.com/drive/folders/1KnFqHpJLc_NddIjKnRJEb9McQjN4DXgk'
SKELETONS = ('deit_small_patch16_224', 'deit_tiny_patch16_224')


def native_plan(side=MITA_SIDE, k=MITA_K):
    return Plan(summary_side=side, landmark_mode='mita', native_k=k)


def checkpoint_settings(args):
    """(timm skeleton, m_side, k) from the saved training arguments."""
    if args.model not in SKELETONS:
        raise ValueError(f'Unexpected MiTA skeleton {args.model}')
    side, root = getattr(args, 'pool_size', None) or MITA_SIDE, getattr(args, 'kv_topk', None) or 5
    return args.model, side, root ** 2


def build_mita(state, args, device='cpu'):
    skeleton, side, k = checkpoint_settings(args)
    model = timm.create_model(skeleton, pretrained=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    model.mita_plan = native_plan(side, k)
    for block in model.blocks:
        install(block.attn, model.mita_plan, model.patch_embed.grid_size, model.num_prefix_tokens)
    return model


def load_mita_model(path: Path, device='cpu'):
    """Load the 'model' weights (not EMA) with a weights-only unpickler."""
    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    return build_mita(checkpoint['model'], checkpoint['args'], device)

