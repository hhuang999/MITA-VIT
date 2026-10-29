import numpy as np
import pytest
import timm
import torch

from mita_probe.models import execute_attention, read_attention
from mita_probe.probe import ReadingPattern
from mita_probe.validation import (centre_rings, execute_online, layer_settings, replace_matched_patches,
                                  replace_random_patches, tie_metrics)


def test_ties_ignore_invalid_and_zero_mass():
    counts, first, expected = tie_metrics(
        np.array([[9, 9, 99], [0, 0, 0], [1, 2, 3]]),
        np.array([[False, True, True], [True, False, False], [False, True, True]]),
        np.array([[True, True, False], [True]*3, [False]*3]))
    assert counts[0] == 2 and first[0] == 0 and expected[0] == .5
    assert np.isnan(counts[1:]).all() and np.isnan(expected[1:]).all()


def test_patch_random_preserves_cls_dose_routes_and_never_reinserts():
    scores = torch.arange(12.).reshape(1, 1, 1, 12).expand(2, 2, 3, -1)
    reads = torch.tensor([0, 9, 10, 11]).expand(2, 2, 3, -1)
    pattern = ReadingPattern(reads, torch.zeros(2, 2, 12, dtype=torch.long))
    dose = torch.tensor([0, 2, 3]).reshape(1, 1, 3).expand(2, 2, 3)
    result = replace_random_patches(pattern, scores, dose, torch.Generator().manual_seed(0))
    overlap = (result.readsets[..., :, None] == reads[..., None, :]).any(-1).sum(-1)
    assert torch.equal(4 - overlap, dose)
    assert (result.readsets == 0).any(-1).all()
    assert (result.readsets.sort(-1).values.diff(dim=-1) > 0).all()
    assert result.routes is pattern.routes
    repeat = replace_random_patches(pattern, scores, dose, torch.Generator().manual_seed(0))
    assert torch.equal(result.readsets, repeat.readsets)
    with pytest.raises(ValueError):
        replace_random_patches(pattern, scores, dose + 2, torch.Generator())


def test_layer_paths_unique_and_exact():
    settings = layer_settings()
    assert len(settings) == len(set(layers for _, layers in settings)) == 19
    assert dict(settings)['prefix_3'] == (1, 2, 3)
    assert dict(settings)['suffix_3'] == (10, 11, 12)


def test_online_single_layer_identity_scope_and_restoration():
    model = timm.create_model('deit_small_patch16_224', pretrained=False, depth=2).eval()
    images = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        original = model(images)
        with read_attention(model, k_top=8, layers=(2,)) as frozen:
            observed = model(images)
        with execute_attention(model, frozen):
            expected = model(images)
        with execute_online(model, 8, layers=(2,)) as online:
            actual = model(images)
        assert set(online) == {2}
        assert torch.equal(online[2].readsets, frozen[2].readsets)
        assert torch.equal(online[2].routes, frozen[2].routes)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with execute_online(model, 197):
            identity = model(images)
        torch.testing.assert_close(identity, original, atol=1e-5, rtol=0)
        torch.testing.assert_close(observed, original, atol=0, rtol=0)
        torch.testing.assert_close(model(images), original, atol=0, rtol=0)
    assert all(not block.attn._forward_pre_hooks for block in model.blocks)
    with pytest.raises(RuntimeError):
        with execute_online(model, 8):
            raise RuntimeError('Test cleanup on exception')
    assert all(not block.attn._forward_pre_hooks for block in model.blocks)


def test_centre_rings_are_symmetric():
    rings = centre_rings()
    assert rings[0] == -1 and len(rings) == 197
    grid = rings[1:].reshape(14, 14)
    assert torch.equal(grid, grid.flip(0)) and torch.equal(grid, grid.T)
    assert set(grid[6:8, 6:8].flatten().tolist()) == {0}


def test_matched_removal_keeps_count_and_rings():
    torch.manual_seed(0)
    B, H, m, k, n = 3, 2, 4, 25, 197
    scores = torch.randn(B, H, m, n)
    reads = scores.topk(k, -1).indices
    pattern = ReadingPattern(reads, torch.zeros(B, H, n, dtype=torch.long))
    forbidden = torch.rand(B, n) < .5
    forbidden[:, 0] = False
    rings = centre_rings()
    result, removed = replace_matched_patches(pattern, scores, forbidden, rings, torch.Generator().manual_seed(1))
    blocked = forbidden[:, None, None, :].expand(B, H, m, n).gather(-1, reads)
    assert torch.equal(removed.sum(-1), blocked.sum(-1))
    for value in rings[1:].unique().tolist():
        inside = rings[reads] == value
        assert torch.equal((removed & inside).sum(-1), (blocked & inside).sum(-1))
    assert not (removed & (reads == 0)).any()
    kept = torch.where(removed, -1, reads)
    assert torch.equal(torch.where(removed, -1, result.readsets), kept)
    assert all(len(set(row.tolist())) == k for row in result.readsets.flatten(0, 2))


def test_matched_removal_equals_target_when_ring_is_all_forbidden():
    reads = torch.tensor([[[[0, 91, 92, 105, 106]]]])  # CLS plus the four central patches (ring 0)
    scores = torch.arange(197.).flip(0).expand(1, 1, 1, 197)
    pattern = ReadingPattern(reads, torch.zeros(1, 1, 197, dtype=torch.long))
    forbidden = torch.zeros(1, 197, dtype=torch.bool)
    forbidden[0, [91, 92, 105, 106]] = True
    _, removed = replace_matched_patches(pattern, scores, forbidden, centre_rings(), torch.Generator().manual_seed(0))
    assert removed.tolist() == [[[[False, True, True, True, True]]]]
