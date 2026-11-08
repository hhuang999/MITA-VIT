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


@pytest.fixture(scope='module')
def dino():
    torch.manual_seed(0)
    return load_dino_model(None, pretrained=False)


def masked_forward(model, images, keep):
    """Reference: every layer's attention may read only the kept tokens (the probe's execution path)."""
    heads = model.blocks[0].attn.num_heads
    batch, tokens = keep.shape
    mask = keep[:, None, None, :].expand(batch, heads, tokens, tokens)
    with execute(model, {layer: Plan(token_mask=mask) for layer in range(1, len(model.blocks) + 1)}):
        return model(images)


@pytest.mark.parametrize('name', ['deit', 'dino'])
def test_masking_tokens_equals_dropping_them(name, request):
    model = request.getfixturevalue(name)
    images = torch.randn(2, 3, 224, 224)
    order = rank(torch.randn(2, 196), torch.Generator().manual_seed(0))
    keep = keep_mask(order, torch.tensor([40, 120]))  # unequal counts: rows are grouped by kept count
    assert keep.sum(1).tolist() == [41, 121] and keep[:, 0].all()
    with torch.inference_mode():
        torch.testing.assert_close(run_kept(model, images, keep), masked_forward(model, images, keep),
                                   atol=2e-5, rtol=0)
        full = keep_mask(order, 196)
        torch.testing.assert_close(run_kept(model, images, full), model(images), atol=2e-5, rtol=0)


def test_rank_breaks_ties_by_seed_and_conditions_are_complements(deit):
    scores = torch.tensor([[1., 1., 1., 0.] + [0.] * 192])
    first = rank(scores, torch.Generator().manual_seed(1))
    assert set(first[0, :3].tolist()) == {0, 1, 2}
    assert torch.equal(first, rank(scores, torch.Generator().manual_seed(1)))
    top = keep_mask(first, 3, top=True)
    drop_top = keep_mask(first, 196 - 3, top=False)
    assert top[0, 1:4].all() and not drop_top[0, 1:4].any() and (top[:, 1:] ^ drop_top[:, 1:]).all()


def test_curves_and_measures_are_consistent(deit):
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        original = deit(images)
        order = rank(references.centre(2), torch.Generator().manual_seed(0))
        bottom = curve(deit, images, original, order, [0], 'drop_bottom')
        drop = curve(deit, images, original, order, [0, 8], 'drop_top')
    assert bottom[0]['agree'].eq(1).all() and drop[0]['agree'].eq(1).all()
    assert torch.allclose(drop[0]['kl'], torch.zeros(2), atol=1e-6)
    m = measures(original, original)
    assert m['p_target'].equal(original.softmax(-1).amax(-1))


def test_kl_direction():
    original = torch.tensor([[2., 0.], [0., 2.], [0., 2.]])
    executed = torch.tensor([[1., 0.], [2., 0.], [0., 2.]])
    p, q = original.double().softmax(-1), executed.double().softmax(-1)
    torch.testing.assert_close(kl_divergence(original, executed).double(), (p * (p.log() - q.log())).sum(-1))
    torch.testing.assert_close(kl_divergence(original, original), torch.zeros(3))


ROWS, COLS = np.divmod(np.arange(196), 14)


def square(top, left, size):
    return torch.from_numpy((ROWS >= top) & (ROWS < top + size) & (COLS >= left) & (COLS < left + size))


def test_object_scores_are_perfect_and_centre_scores_follow_centre_objects():
    fg = torch.stack([square(4, 4, 6), square(0, 0, 5)])
    valid = torch.ones_like(fg)
    perfect = enrichment_measures(fg.float(), fg, valid, fg.flip(0))
    for key in ['mass', 'pointing', 'auroc', 'precision']:
        np.testing.assert_allclose(perfect[key], 1.0)
    assert (perfect['enrichment'] > 0).all()
    central = enrichment_measures(references.centre(2), fg, valid, fg.flip(0))
    assert central['auroc'][0] > 0.95 and central['auroc'][1] < 0.5  # centred object vs corner object
    assert central['enrichment'][0] < 0.05  # at equal distance from the centre the centre map has no edge


def test_auroc_counts_ties_as_half_and_skips_missing_classes():
    scores = torch.tensor([[1., 1., 0., 2.]])
    positive = torch.tensor([[True, False, False, False]])
    valid = torch.ones_like(positive)
    assert auroc(scores, positive, valid)[0] == (0.5 + 1 + 0) / 3
    assert np.isnan(auroc(scores, torch.zeros_like(positive), valid)[0])


def test_geometry_area_and_distance():
    fg = torch.stack([square(6, 6, 2), square(0, 0, 2)])
    area, distance = geometry(fg, torch.ones_like(fg))
    np.testing.assert_allclose(area, 4 / 196)
    assert distance[0] == 0 and distance[1] > 8


def test_alignment_is_one_for_the_human_map():
    coarse = torch.rand(3, 1, 7, 7, generator=torch.Generator().manual_seed(0))
    human = torch.nn.functional.interpolate(coarse, size=(224, 224), mode='bilinear', align_corners=False)[:, 0]
    values = metrics.alignment(metrics.per_patch(human), human)
    assert torch.allclose(values['spearman_token'], torch.ones(3, dtype=torch.float64))
    assert (values['spearman_pixel'] > 0.8).all()


def test_spearman_matches_pandas_with_ties():
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 4, (5, 196), generator=g).float()
    b = torch.rand(5, 196, generator=g)
    ours = spearman(a, b).numpy()
    ref = [np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1] for x, y in zip(a.numpy(), b.numpy())]
    assert np.allclose(ours, ref)


def test_high_norm_uses_the_midpoint_median():
    # Even count: the median is the midpoint of the two middle norms (2 and 4 -> 3), so only 10 > 9 is high.
    norms = torch.tensor([[1., 2., 4., 8.5, 10., 1.], [2., 2., 2., 6., 7., 2.]])
    assert high_norm_tokens(norms, 3).tolist() == [[False, False, False, False, True, False],
                                                   [False, False, False, False, True, False]]


def test_block_input_norms_exclude_prefixes_and_leave_output_unchanged():
    model = timm.create_model('vit_small_patch14_reg4_dinov2', pretrained=False, img_size=224, depth=2).eval()
    images = torch.randn(1, 3, 224, 224)
    before_norm = []
    handle = model.blocks[0].norm1.register_forward_pre_hook(lambda module, args: before_norm.append(args[0].clone()))
    with torch.inference_mode():
        original = model(images)
        with capture_input_norms(model) as norms:
            observed = model(images)
    handle.remove()
    torch.testing.assert_close(observed, original, atol=0, rtol=0)
    assert model.num_prefix_tokens == 5 and norms[1].shape == (1, 256)
    torch.testing.assert_close(norms[1], before_norm[-1][:, 5:].norm(dim=-1))
    assert all(not block._forward_pre_hooks for block in model.blocks)


def test_centre_rings_are_symmetric():
    rings = centre_rings()
    assert rings[0] == -1 and len(rings) == 197
    grid = rings[1:].reshape(14, 14)
    assert torch.equal(grid, grid.flip(0)) and torch.equal(grid, grid.T)
    assert set(grid[6:8, 6:8].flatten().tolist()) == {0}


@pytest.mark.parametrize('name', ['deit', 'dino'])
def test_gae_returns_finite_nonnegative_patch_maps(name, request):
    model = request.getfixturevalue(name)
    scores = references.gae(model, torch.randn(2, 3, 224, 224))
    assert scores.shape == (2, 196) and torch.isfinite(scores).all() and (scores >= 0).all()
    assert all(p.grad is None for p in model.parameters())


def test_reference_maps():
    centre = references.centre(3)
    assert centre.shape == (3, 196) and centre[0].argmax().item() in {90, 91, 104, 105}
    a = references.random_scores(2, torch.Generator().manual_seed(0))
    assert torch.equal(a, references.random_scores(2, torch.Generator().manual_seed(0)))


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
