"""Metrics and reference maps: deletion faithfulness (SRG parts), same-ring enrichment, ClickMe alignment,
Spearman, high-norm tokens, GAE / centre / random, the top-k map."""

import numpy as np
import pandas as pd
import pytest
import timm
import torch

from fastweight_vit import metrics, references
from fastweight_vit.enrichment import auroc, geometry, measures as enrichment_measures
from fastweight_vit.execution import Plan, capture_input_norms, execute, resolve
from fastweight_vit.faithfulness import curve, keep_mask, measures, rank, run_kept
from fastweight_vit.metrics import kl_divergence, spearman
from fastweight_vit.models.dino import load_dino_model
from fastweight_vit.tokens import centre_rings, high_norm_tokens

MAIN = dict(m_side=4, k=16, routing='spatial', cls='own', scoring='query', summary_scale='mita')


@pytest.fixture(scope='module')
def deit():
    torch.manual_seed(0)
    return timm.create_model('deit_small_patch16_224', pretrained=False).eval()


def test_attention_mass_is_the_topk_map_with_every_token(deit):
    images = torch.randn(2, 3, 224, 224)
    layers = [9, 10, 11, 12]
    dense = references.compute('attention', deit, images, layers=layers)
    every_token = resolve({'mode': 'L3', 'k': 197}, MAIN)
    torch.testing.assert_close(dense, references.topk_map(deit, images, every_token, layers), rtol=1e-5, atol=1e-4)
    heads, tokens = deit.blocks[0].attn.num_heads, 197
    assert dense.shape == (2, 196) and (dense > 0).all() and (dense.sum(1) < len(layers) * heads * tokens).all()
    topk = references.topk_map(deit, images, resolve({'mode': 'L3'}, MAIN), layers)
    assert topk.shape == (2, 196) and (topk >= 0).all()
