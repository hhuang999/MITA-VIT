"""The probe: pattern reading (landmarks, top-k sets, routes, masks) and execution against timm."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from mita_probe.data import shifted_transform
from mita_probe.models.deit import load_deit
from mita_probe.models.mita import install, native_plan
from mita_probe.probe import (M_SIDES, Pattern, Plan, build_pattern, capture_qk, coverage, execute, grid_landmarks,
                              kmeans_landmarks, landmarks, mixed_attention, per_query_pattern, random_pattern,
                              spatial_routes, token_mask, union_fraction)
from mita_probe.probe.pattern import region_scores

FIXTURE = Path(__file__).parent / 'fixtures' / 'v1_reading_pattern.npz'


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
    return load_deit(pretrained=False)


def full_plans(model, plan_for):
    return {layer: plan_for(layer) for layer in range(1, len(model.blocks) + 1)}


def test_hand_example_of_design_section_4_3():
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
    expected = {  # q1 and q2 rows of the two tables in design section 4.3
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


@pytest.mark.parametrize('mode', ['clean', 'kmeans'])
def test_shared_pattern_reproduces_v1_reading_with_dot_routes_and_routed_cls(mode):
    """v1 (archive/v1) read these sets with timm masks; the fixture freezes its output on fixed Q/K."""
    saved = np.load(FIXTURE)
    q, k = torch.from_numpy(saved['q']), torch.from_numpy(saved['k'])
    prefix = '' if mode == 'clean' else 'kmeans_'
    new, scores = build_pattern(q, k, m_side=4, k_top=16, scale=8 ** -0.5, grid=GRID, routing='dot',
                                cls='routed', landmark_mode=mode)
    assert np.array_equal(new.readsets.numpy(), saved[prefix + 'readsets'])
    assert np.array_equal(new.routes.numpy(), saved[prefix + 'routes'])
    assert np.array_equal(token_mask(new).numpy(), saved[prefix + 'mask']) and scores.shape == (2, 3, 16, 197)


def test_kmeans_matches_independent_lloyd_loop_and_handles_empty_clusters():
    patches = torch.randn(2, 3, 12, 2)
    initial = patches[:, :, :4].clone()
    initial[:, :, 3] = initial[:, :, 2]  # Duplicate centers create an empty cluster.
    actual = kmeans_landmarks(patches, initial)
    expected = initial.clone()
    for batch in range(2):
        for head in range(3):
            points = patches[batch, head]
            centers = initial[batch, head].clone()
            for _ in range(10):
                assignments = (points[:, None] - centers).square().sum(-1).argmin(-1)
                for region in range(4):
                    members = points[assignments == region]
                    if len(members):
                        centers[region] = members.mean(0)
            expected[batch, head] = centers
    torch.testing.assert_close(actual, expected)
    identical = torch.ones(1, 1, 196, 8)
    centers = kmeans_landmarks(identical, grid_landmarks(identical, (14, 14)))
    assert torch.isfinite(centers).all() and centers.eq(1).all()


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


def test_m196_spatial_shared_sets_equal_per_query_top_k():
    q, k = torch.randn(2, 3, 197, 8), torch.randn(2, 3, 197, 8)
    shared, _ = build_pattern(q, k, m_side=14, k_top=7, scale=8 ** -0.5, grid=GRID)
    assert torch.equal(token_mask(shared), token_mask(per_query_pattern(q, k, 7, 8 ** -0.5)))


def test_attention_scoring_is_mean_region_attention():
    q, k = torch.randn(1, 2, 197, 8), torch.randn(1, 2, 197, 8)
    lm = landmarks(q, GRID, 4)
    scores = region_scores(q, k, lm, 1., scoring='attention', grid=GRID, m_side=4)
    probs = (q @ k.transpose(-1, -2)).softmax(-1)[:, :, 1:]
    region = grid_region_membership(14, 4)[5]
    torch.testing.assert_close(scores[:, :, 5], probs[:, :, region].mean(2))


def test_random_pattern_is_seeded_distinct_and_keeps_routes():
    q, k = torch.randn(2, 3, 197, 8), torch.randn(2, 3, 197, 8)
    pattern, _ = build_pattern(q, k, m_side=4, k_top=16, scale=1., grid=GRID)
    one = random_pattern(pattern, torch.Generator().manual_seed(3))
    two = random_pattern(pattern, torch.Generator().manual_seed(3))
    assert torch.equal(one.readsets, two.readsets) and torch.equal(one.cls_set, two.cls_set)
    assert one.routes is pattern.routes
    assert (token_mask(one).sum(-1) == 16).all()
    assert not torch.equal(one.readsets, pattern.readsets)


def test_coverage_and_union():
    q, k = torch.randn(2, 3, 197, 8), torch.randn(2, 3, 197, 8)
    probs = (q @ k.transpose(-1, -2)).softmax(-1)
    full, _ = build_pattern(q, k, m_side=4, k_top=197, scale=1., grid=GRID)
    patch, cls = coverage(probs, token_mask(full))
    torch.testing.assert_close(patch, torch.ones(2)); torch.testing.assert_close(cls, torch.ones(2))
    assert torch.equal(union_fraction(full), torch.ones(2))
    small, _ = build_pattern(q, k, m_side=4, k_top=8, scale=1., grid=GRID)
    patch, cls = coverage(probs, token_mask(small))
    assert ((patch > 0) & (patch < 1)).all() and (union_fraction(small) <= (17 * 8) / 197).all()


@pytest.mark.parametrize('fused', [False, True])
def test_dense_full_mask_matches_timm_and_restores_forward(model, fused):
    for block in model.blocks:
        block.attn.fused_attn = fused
    images = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        original = model(images)
        with capture_qk(model) as qk:
            assert torch.equal(model(images), original)
        plans = full_plans(model, lambda layer: Plan(token_mask=token_mask(
            per_query_pattern(*qk[layer], 197, model.blocks[0].attn.scale))))
        with execute(model, plans):
            executed = model(images)
        torch.testing.assert_close(executed, original, atol=1e-5, rtol=0)
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


def test_shifted_crop_moves_window_one_pixel():
    from timm.data import create_transform
    transform = create_transform(input_size=(3, 224, 224), crop_pct=0.9, interpolation='bicubic')
    image = Image.fromarray(torch.randint(0, 255, (300, 400, 3), dtype=torch.uint8).numpy())
    base, moved = transform(image), shifted_transform(transform, 1)(image)
    torch.testing.assert_close(moved[:, :, :-1], base[:, :, 1:], atol=0, rtol=0)
    assert not torch.equal(moved, base)


def test_shapley_split_sums_to_mixed_minus_random_per_image():
    from mita_probe.stats import anatomy_table, resample_indices, shapley
    import pandas as pd
    rng = np.random.default_rng(0)
    table = pd.DataFrame({name: rng.integers(0, 2, 50).astype(float) for name in
                          ['rand', 'randC', 'L3', 'L4', 'L2', 'L1_k32']})
    phi_c, phi_r = shapley(table['rand'], table['randC'], table['L3'], table['L4'])
    np.testing.assert_allclose(phi_c + phi_r, table['L4'] - table['rand'])
    summary = anatomy_table(table, resample_indices(50))
    assert summary.loc['phi_C', 'mean'] + summary.loc['phi_R', 'mean'] == pytest.approx(
        summary.loc['L4', 'mean'] - summary.loc['rand', 'mean'])
    bounded = summary.drop(index='sharing_index')  # random toy data can make its denominator 0
    assert (bounded['low'] <= bounded['mean'] + 1e-12).all() and (bounded['mean'] <= bounded['high'] + 1e-12).all()
    assert (resample_indices(50)[0] == np.random.default_rng(0).integers(50, size=50)).all()
