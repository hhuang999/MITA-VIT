"""Explanation maps and the token-removal faithfulness tests built on them."""

import pytest
import torch

from fastweight_vit import explain
from fastweight_vit.faithfulness import curve, keep_mask, measures, rank, run_kept, saco
from fastweight_vit.models.deit import load_deit
from fastweight_vit.models.dino import load_dino_model
from fastweight_vit.execution import Plan, capture_qk, execute, resolve

SPEC = resolve(dict(mode='L3'), dict(m_side=4, k=16, routing='spatial', cls='own', scoring='query',
                                      summary_scale='mita'))


@pytest.fixture(scope='module')
def deit():
    torch.manual_seed(0)
    return load_deit(pretrained=False)


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
        order = rank(explain.centre(2), torch.Generator().manual_seed(0))
        keep = curve(deit, images, original, order, [196, 8], 'keep_top')
        drop = curve(deit, images, original, order, [0], 'drop_top')
    assert keep[196]['agree'].eq(1).all() and drop[0]['agree'].eq(1).all()
    assert torch.allclose(keep[196]['kl'], torch.zeros(2), atol=1e-6)
    m = measures(original, original)
    assert m['p_target'].equal(original.softmax(-1).amax(-1))


def test_saco_is_bounded_and_deterministic(deit):
    images = torch.randn(2, 3, 224, 224)
    scores = torch.randn(2, 196)
    with torch.inference_mode():
        original = deit(images)
        order = rank(scores, torch.Generator().manual_seed(0))
        a = saco(deit, images, original, scores, order, groups=4)
        b = saco(deit, images, original, scores, order, groups=4)
    assert a.shape == (2,) and (a.abs() <= 1 + 1e-6).all() and torch.equal(a, b)


@pytest.mark.parametrize('name', ['deit', 'dino'])
def test_cls_attention_and_rollout_match_direct_computation(name, request):
    model = request.getfixturevalue(name)
    images = torch.randn(2, 3, 224, 224)
    layers = explain.head_layers(model)
    assert layers == ((12,) if name == 'deit' else (9, 10, 11, 12))
    with torch.inference_mode(), capture_qk(model) as qk:
        model(images)
    scale = model.blocks[0].attn.scale
    probs = {layer: ((q * scale) @ k.transpose(-2, -1)).softmax(-1) for layer, (q, k) in qk.items()}
    expected = torch.stack([probs[layer][:, :, 0, 1:].mean(1) for layer in layers]).mean(0)
    torch.testing.assert_close(explain.cls_attention(model, images), expected, atol=1e-5, rtol=0)
    eye = torch.eye(197)
    relevance, rows = eye.expand(2, -1, -1), []
    for layer in range(1, 13):
        a = 0.5 * probs[layer].mean(1) + 0.5 * eye
        relevance = (a / a.sum(-1, keepdim=True)) @ relevance
        if layer in layers:
            rows.append(relevance[:, 0, 1:])
    torch.testing.assert_close(explain.rollout(model, images), torch.stack(rows).mean(0), atol=1e-5, rtol=0)


def test_recording_attention_leaves_the_output_and_forward_unchanged(deit):
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        original = deit(images)
        with explain.record_attention(deit) as probs:
            torch.testing.assert_close(deit(images), original, atol=1e-5, rtol=0)
        assert len(probs) == 12 and probs[3].shape == (2, 6, 197, 197)
    assert all('forward' not in block.attn.__dict__ for block in deit.blocks)


@pytest.mark.parametrize('name', ['deit', 'dino'])
def test_gradient_methods_return_finite_nonnegative_patch_maps(name, request):
    model = request.getfixturevalue(name)
    images = torch.randn(2, 3, 224, 224)
    for method in [explain.gae, explain.gradcam]:
        scores = method(model, images)
        assert scores.shape == (2, 196) and torch.isfinite(scores).all() and (scores >= 0).all()
    ig = explain.integrated_gradients(model, images, steps=3)
    assert ig.shape == (2, 196) and torch.isfinite(ig).all()
    assert all(p.grad is None for p in model.parameters())


def test_probe_scores_account_for_every_read(deit):
    images = torch.randn(2, 3, 224, 224)
    layers = [11, 12]
    count = explain.probe(deit, images, SPEC, layers, 'count')
    mass = explain.probe(deit, images, SPEC, layers, 'mass')
    value = explain.probe(deit, images, SPEC, layers, 'value')
    heads = 6
    # Without CLS: at most (16 regions + 1 CLS set) * heads * k reads per layer; mass at most heads * 197 rows.
    assert (count.sum(1) <= len(layers) * heads * 17 * 16).all() and (count >= 0).all()
    assert (mass.sum(1) <= len(layers) * heads * 197 + 1e-3).all() and (mass >= 0).all()
    assert value.shape == (2, 196) and torch.isfinite(value).all()
    assert ((mass > 0) == (count > 0)).all()


def test_reference_maps():
    centre = explain.centre(3)
    assert centre.shape == (3, 196) and centre[0].argmax().item() in {90, 91, 104, 105}
    a = explain.random_scores(2, torch.Generator().manual_seed(0))
    assert torch.equal(a, explain.random_scores(2, torch.Generator().manual_seed(0)))
