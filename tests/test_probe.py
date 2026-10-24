import math

import pytest
import torch

from mita_probe.baselines import random_readsets
from mita_probe.grouping import grid_landmarks, kmeans_landmarks
from mita_probe.probe import execution_mask, read_patterns


def test_hand_calculation():
    q = torch.tensor([[3., 1.], [-1., 1.], [1., -1.], [1., -1.]])[None, None]
    keys = torch.tensor([[1., 0.], [0., 2.], [2., 2.], [0., -1.]])[None, None]
    landmarks = grid_landmarks(q, (1, 4), (1, 2))
    torch.testing.assert_close(landmarks[0, 0], torch.tensor([[1., 1.], [1., -1.]]))
    pattern = read_patterns(q, keys, landmarks, k_top=2, scale=1)
    assert pattern.readsets[0, 0, 0].tolist() == [2, 1]
    assert set(pattern.readsets[0, 0, 1].tolist()) == {0, 3}
    assert pattern.routes[0, 0].tolist() == [0, 0, 1, 1]
    logits = q @ keys.transpose(-1, -2)
    weights = logits.masked_fill(~execution_mask(pattern), -torch.inf).softmax(-1)
    torch.testing.assert_close(
        weights[0, 0, 0], torch.tensor([0., 0.002472623, 0.997527377, 0.]),
    )
    torch.testing.assert_close(
        logits.softmax(-1)[0, 0, 0],
        torch.tensor([0.0067, 0.0025, 0.9907, 0.0001]), atol=1e-4, rtol=0,
    )


def test_adaptive_windows_and_average_scores():
    q = torch.randn(2, 3, 196, 8)
    keys = torch.randn(2, 3, 197, 8)
    landmarks = grid_landmarks(q, (14, 14))
    grid = q.reshape(2, 3, 14, 14, 8)
    for row in range(4):
        for col in range(4):
            region = grid[:, :, math.floor(row * 14 / 4):math.ceil((row + 1) * 14 / 4),
                          math.floor(col * 14 / 4):math.ceil((col + 1) * 14 / 4)]
            region = region.flatten(2, 3)
            mean_scores = ((region / math.sqrt(8)) @ keys.transpose(-1, -2)).mean(-2)
            landmark_scores = (landmarks[:, :, row * 4 + col] / math.sqrt(8)).unsqueeze(-2)
            landmark_scores = (landmark_scores @ keys.transpose(-1, -2)).squeeze(-2)
            torch.testing.assert_close(landmark_scores, mean_scores, atol=1e-4, rtol=0)
    assert math.ceil(14 / 4) > math.floor(14 / 4)  # Adjacent windows overlap.


@pytest.mark.parametrize("grouping", ["grid", "kmeans"])
def test_all_keys_is_exact_identity_and_cls_can_be_read(grouping):
    q, keys, values = (torch.randn(2, 3, 197, 8) for _ in range(3))
    landmarks = grid_landmarks(q[:, :, 1:], (14, 14))
    if grouping == "kmeans":
        landmarks = kmeans_landmarks(q[:, :, 1:], landmarks)
    pattern = read_patterns(q, keys, landmarks, 197)
    mask = execution_mask(pattern)
    assert mask.all()
    assert (pattern.readsets == 0).sum(-1).eq(1).all()
    assert pattern.routes.shape == (2, 3, 197)
    logits = (q * 8**-0.5) @ keys.transpose(-1, -2)
    actual = logits.masked_fill(~mask, -torch.inf).softmax(-1) @ values
    expected = logits.softmax(-1) @ values
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_every_valid_k_has_exactly_k_keys_and_finite_output():
    q, keys, values = (torch.randn(1, 2, 197, 8) for _ in range(3))
    landmarks = grid_landmarks(q[:, :, 1:], (14, 14))
    logits = (q * 8**-0.5) @ keys.transpose(-1, -2)
    for k_top in range(1, 198):
        mask = execution_mask(read_patterns(q, keys, landmarks, k_top))
        assert mask.sum(-1).eq(k_top).all()
        weights = logits.masked_fill(~mask, -torch.inf).softmax(-1)
        assert torch.isfinite(weights @ values).all()
        assert weights[~mask].eq(0).all()


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
    torch.testing.assert_close(kmeans_landmarks(patches, initial), actual, atol=0, rtol=0)
    identical = torch.ones(1, 1, 196, 8)
    centers = kmeans_landmarks(identical, grid_landmarks(identical, (14, 14)))
    assert torch.isfinite(centers).all()
    assert centers.eq(1).all()


def test_random_sets_are_seeded_without_replacement_and_preserve_routes():
    q, keys = (torch.randn(2, 3, 197, 8) for _ in range(2))
    pattern = read_patterns(q, keys, grid_landmarks(q[:, :, 1:], (14, 14)), 16)
    rng = torch.Generator().manual_seed(4)
    random = random_readsets(pattern, rng)
    repeat = random_readsets(pattern, torch.Generator().manual_seed(4))
    torch.testing.assert_close(random.readsets, repeat.readsets)
    assert random.routes is pattern.routes
    assert random.readsets.sort(-1).values.diff(dim=-1).gt(0).all()
    assert not torch.equal(random.readsets, random_readsets(pattern, rng).readsets)
    assert not torch.equal(random.readsets, random_readsets(pattern, torch.Generator().manual_seed(3)).readsets)
