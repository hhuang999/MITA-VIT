"""Neuron-level interventions: category removal, random and centre-matched controls, high-norm tokens."""

import pytest
import timm
import torch

from fastweight_vit.interventions import (centre_rings, replace_category, replace_matched_patches,
                                      replace_random_patches, sink_tokens)
from fastweight_vit.execution import Pattern, capture_cls_attention, capture_input_norms


def test_sink_uses_midpoint_median_strict_threshold():
    norms = torch.tensor([[1., 2., 4., 9.], [1., 1., 1., 4.]])
    assert sink_tokens(norms).tolist() == [[False]*4, [False, False, False, True]]


def test_category_refills_without_leakage_duplicates_and_preserves_routes():
    scores = torch.tensor([[[[9., 8., 7., 6., 5., 4.]]]])
    pattern = Pattern(torch.tensor([[[[0, 1, 2]]]]), torch.zeros(1, 1, 6, dtype=torch.long))
    forbidden = torch.tensor([[False, True, True, False, False, True]])
    replaced, valid, dose = replace_category(pattern, scores, forbidden)
    assert replaced.readsets.tolist() == [[[[0, 3, 4]]]]
    assert valid.item() and dose.item() == 2
    assert replaced.routes is pattern.routes
    forbidden[:, 4] = True
    replaced, valid, _ = replace_category(pattern, scores, forbidden)
    assert not valid.item()
    assert torch.equal(replaced.readsets, pattern.readsets)


def test_patch_random_preserves_cls_dose_routes_and_never_reinserts():
    scores = torch.arange(12.).reshape(1, 1, 1, 12).expand(2, 2, 3, -1)
    reads = torch.tensor([0, 9, 10, 11]).expand(2, 2, 3, -1)
    pattern = Pattern(reads, torch.zeros(2, 2, 12, dtype=torch.long))
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


def test_block_input_norms_exclude_prefixes_and_leave_output_unchanged():
    model = timm.create_model('vit_small_patch14_reg4_dinov2', pretrained=False, img_size=224, depth=2).eval()
    images = torch.randn(1, 3, 224, 224)
    before_norm = []
    handle = model.blocks[0].norm1.register_forward_pre_hook(lambda module, args: before_norm.append(args[0].clone()))
    with torch.inference_mode():
        original = model(images)
        with capture_input_norms(model) as norms, capture_cls_attention(model) as cls:
            observed = model(images)
    handle.remove()
    torch.testing.assert_close(observed, original, atol=0, rtol=0)
    assert model.num_prefix_tokens == 5
    assert norms[1].shape == (1, 256) and cls['heatmap'].shape == (1, 256)
    torch.testing.assert_close(norms[1], before_norm[-1][:, 5:].norm(dim=-1))
    assert all(not block._forward_pre_hooks for block in model.blocks)


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
    pattern = Pattern(reads, torch.zeros(B, H, n, dtype=torch.long))
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
    pattern = Pattern(reads, torch.zeros(1, 1, 197, dtype=torch.long))
    forbidden = torch.zeros(1, 197, dtype=torch.bool)
    forbidden[0, [91, 92, 105, 106]] = True
    _, removed = replace_matched_patches(pattern, scores, forbidden, centre_rings(), torch.Generator().manual_seed(0))
    assert removed.tolist() == [[[[False, True, True, True, True]]]]
