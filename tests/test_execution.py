"""Top-k reading sets (landmarks, shared sets, routes, masks) and execution against timm."""

from pathlib import Path

import numpy as np
import pytest
import timm
import torch

from fastweight_vit.execution import (Pattern, Plan, build_pattern, capture_qk, execute, expand_settings,
                                      grid_landmarks, landmarks, make_plan, mixed_attention, random_pattern, resolve,
                                      spatial_routes, token_mask)
from fastweight_vit.models.mita import install, native_plan

from fakes import config_for

FIXTURE = Path(__file__).parent / 'fixtures' / 'reading_pattern_reference.npz'
M_SIDES = {1: 1, 4: 2, 16: 4, 25: 5, 49: 7, 196: 14}


def grid_region_membership(grid=14, regions=4):
    """Exact adaptive-average-pooling windows (overlap for 14 -> 4)."""
    membership = torch.zeros(regions * regions, grid, grid, dtype=torch.bool)
    for y in range(regions):
        for x in range(regions):
            membership[y * regions + x, y * grid // regions:((y + 1) * grid + regions - 1) // regions,
                       x * grid // regions:((x + 1) * grid + regions - 1) // regions] = True
    return membership.flatten(1)


GRID = (14, 14)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return timm.create_model('deit_small_patch16_224', pretrained=False).eval()


def full_plans(model, plan_for):
    return {layer: plan_for(layer) for layer in range(1, len(model.blocks) + 1)}


def test_hand_worked_example_of_the_four_attention_modes():
    """Four tokens on a 1 x 4 grid, two regions (tokens 1-2 and 3-4), scale 1: region 1 reads tokens {3, 2},
    region 2 reads {1, 4}; the two landmarks are the mean queries of the regions. The expected outputs of queries 1
    and 2 were worked out by hand (three decimals): softmax over all keys (full), over the two summary slots only
    (summary; slot value = landmark's own softmax over all keys @ V), over the region's reading set only (shared),
    and over the reading set plus the summary slots (mixed)."""
    q = torch.tensor([[3., 1.], [-1., 1.], [1., -1.], [1., -1.]])[None, None]
    k = torch.tensor([[1., 0.], [0., 2.], [2., 2.], [0., -1.]])[None, None]
    v = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., -1.]])[None, None]
    lm = grid_landmarks(q, (1, 4), (1, 2))
    pattern = Pattern(torch.tensor([[[[2, 1], [0, 3]]]]), torch.tensor([[[0, 0, 1, 1]]]))
    mask = token_mask(pattern)
    full = mixed_attention(q, k, v, 1., torch.ones_like(mask))[0, 0]
    summary = mixed_attention(q, k, v, 1., None, lm)[0, 0]
    shared = mixed_attention(q, k, v, 1., mask)[0, 0]
    mixed = mixed_attention(q, k, v, 1., mask, lm)[0, 0]
    expected = {  # outputs of queries 1 and 2, computed by hand
        'full': ([0.997, 0.993], [0.150, 0.880]), 'summary': ([0.843, 0.805], [0.843, 0.805]),
        'shared': ([0.998, 1.000], [0.119, 1.000]), 'mixed': ([0.994, 0.996], [0.206, 0.977])}
    for name, out in [('full', full), ('summary', summary), ('shared', shared), ('mixed', mixed)]:
        torch.testing.assert_close(out[:2], torch.tensor(expected[name]), atol=2e-3, rtol=0)


@pytest.mark.parametrize('m', sorted(M_SIDES))
def test_spatial_routes_stay_inside_a_pooling_window(m):
    side = M_SIDES[m]
    routes = spatial_routes(GRID, side)
    assert routes.shape == (197,) and routes[0] == 0
    membership = grid_region_membership(14, side)
    assert membership[routes[1:], torch.arange(196)].all()
    assert set(routes[1:].tolist()) == set(range(m))


def test_shared_pattern_reproduces_the_reference_reading_with_dot_routes_and_routed_cls():
    """The fixture freezes an earlier implementation's reading sets (timm masks) on fixed Q/K."""
    saved = np.load(FIXTURE)
    q, k = torch.from_numpy(saved['q']), torch.from_numpy(saved['k'])
    prefix = ''
    new, scores = build_pattern(q, k, m_side=4, k_top=16, scale=8 ** -0.5, grid=GRID, routing='dot',
                                cls='routed', landmark_mode='clean')
    assert np.array_equal(new.readsets.numpy(), saved[prefix + 'readsets'])
    assert np.array_equal(new.routes.numpy(), saved[prefix + 'routes'])
    assert np.array_equal(token_mask(new).numpy(), saved[prefix + 'mask']) and scores.shape == (2, 3, 16, 197)


def test_own_cls_set_and_spatial_routes_in_mask():
    q, k = torch.randn(1, 2, 197, 8), torch.randn(1, 2, 197, 8)
    pattern, _ = build_pattern(q, k, m_side=4, k_top=5, scale=1., grid=GRID)
    mask = token_mask(pattern)
    assert (mask.sum(-1) == 5).all()
    expected_cls = (q[:, :, 0:1] @ k.transpose(-1, -2)).squeeze(2).topk(5, -1).indices
    assert torch.equal(mask[:, :, 0], torch.zeros(1, 2, 197, dtype=torch.bool).scatter_(-1, expected_cls, True))
    routes = spatial_routes(GRID, 4)
    for token in [1, 50, 196]:
        assert torch.equal(mask[:, :, token], mask[:, :, 1 + (routes[1:] == routes[token]).nonzero()[0, 0]])


def test_random_pattern_is_seeded_distinct_and_keeps_routes():
    q, k = torch.randn(2, 3, 197, 8), torch.randn(2, 3, 197, 8)
    pattern, _ = build_pattern(q, k, m_side=4, k_top=16, scale=1., grid=GRID)
    one = random_pattern(pattern, torch.Generator().manual_seed(3))
    two = random_pattern(pattern, torch.Generator().manual_seed(3))
    assert torch.equal(one.readsets, two.readsets) and torch.equal(one.cls_set, two.cls_set)
    assert one.routes is pattern.routes
    assert (token_mask(one).sum(-1) == 16).all()
    assert not torch.equal(one.readsets, pattern.readsets)


@pytest.mark.parametrize('fused', [False, True])
def test_dense_full_mask_matches_timm_and_restores_forward(model, fused):
    for block in model.blocks:
        block.attn.fused_attn = fused
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        original = model(images)
        with capture_qk(model) as qk:
            assert torch.equal(model(images), original)
        shared = full_plans(model, lambda layer: Plan(token_mask=token_mask(build_pattern(
            *qk[layer], m_side=4, k_top=197, scale=1., grid=GRID)[0])))
        with execute(model, shared):
            torch.testing.assert_close(model(images), original, atol=1e-5, rtol=0)
        assert all('forward' not in block.attn.__dict__ for block in model.blocks)
        assert torch.equal(model(images), original)


def test_single_layer_scope_drop_and_constant_summary(model):
    images = torch.randn(2, 3, 224, 224)
    block = model.blocks[4]
    captured = []
    handle = block.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    with torch.inference_mode():
        model(images)
        x = captured[-1]
        with execute(model, {5: Plan(drop=True)}):
            dropped = block(x)
            assert 'forward' in block.attn.__dict__ and 'forward' not in model.blocks[3].attn.__dict__
        torch.testing.assert_close(dropped, x + block.ls2(block.mlp(block.norm2(x))), atol=0, rtol=0)
        attention = block.attn
        out = []
        hook = attention.register_forward_hook(lambda module, args, output: out.append(output))
        with execute(model, {5: Plan(summary_side=1)}):
            block(x)
        hook.remove()
        handle.remove()
        torch.testing.assert_close(out[0], out[0][:, :1].expand_as(out[0]), atol=1e-6, rtol=0)


def test_mita_native_equals_frozen_execution_of_its_own_pattern(model):
    for block in model.blocks:
        install(block.attn, native_plan(), model.patch_embed.grid_size, 1)
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        native = model(images)
        with capture_qk(model) as qk:
            assert torch.equal(model(images), native)
        plans = full_plans(model, lambda layer: Plan(token_mask=token_mask(build_pattern(
            *qk[layer], m_side=5, k_top=25, scale=model.blocks[0].attn.scale, grid=GRID,
            routing='dot', cls='routed', landmark_mode='mita')[0]), summary_side=5, landmark_mode='mita'))
        with execute(model, plans):
            torch.testing.assert_close(model(images), native, atol=0, rtol=0)
        assert all('forward' in block.attn.__dict__ for block in model.blocks)
        assert torch.equal(model(images), native)
    q = qk[1][0]
    torch.testing.assert_close(landmarks(q, GRID, 5, mode='mita'), grid_landmarks(q[:, :, :196], GRID, (5, 5)))


def test_k_sweep_settings_and_plans():
    config = config_for('k_sweep')
    specs = expand_settings(config)
    assert len(specs) == 6 + 6 * 5 and {spec['mode'] for spec in specs} == {'L3', 'rand'}
    q, k = torch.randn(2, 6, 197, 64), torch.randn(2, 6, 197, 64)
    reading = make_plan(resolve(dict(mode='L3', k=8), config['main']), q, k, 0.125, GRID)
    assert (reading.token_mask.sum(-1) == 8).all() and reading.summary_side is None
    spec = resolve(dict(mode='rand', k=8, seed=0), config['main'])
    one = make_plan(spec, q, k, 0.125, GRID, layer=3, batch_start=64)
    assert torch.equal(one.token_mask, make_plan(spec, q, k, 0.125, GRID, layer=3, batch_start=64).token_mask)
    assert not torch.equal(one.token_mask, make_plan(spec, q, k, 0.125, GRID, layer=4, batch_start=64).token_mask)
    assert make_plan(resolve(dict(mode='drop'), config['main']), q, k, 0.125, GRID).drop
    with pytest.raises(ValueError):
        resolve(dict(mode='L4'), config['main'])
