import torch

from mita_probe.metrics import foreground_ratio, heatmap_metrics
from mita_probe.pet import background_regions, grid_region_membership, token_foreground


def test_trimap_boundaries_and_half_foreground_threshold():
    mask = torch.full((32, 32), 3)
    mask[:8, :16] = 1
    mask[8:16, :16] = 2
    mask[16:, :16] = 2
    mask[16:, 16:] = 1
    foreground, valid = token_foreground(mask)
    assert foreground.tolist() == [True, False, False, True]
    assert valid.tolist() == [True, False, True, True]


def test_foreground_ratio_excludes_cls_and_invalid_and_keeps_repeats():
    fg = torch.tensor([[True, False, True]])
    valid = torch.tensor([[True, True, False]])
    reads = torch.tensor([[[[0, 1, 2], [1, 2, 3]]]])
    assert foreground_ratio(reads, fg, valid).item() == .5
    only_first = torch.tensor([[True, False]])
    assert foreground_ratio(reads, fg, valid, only_first).item() == .5
    empty = torch.zeros_like(only_first)
    assert foreground_ratio(reads, fg, valid, empty).isnan().all()


def test_heatmap_mass_and_pointing_ignore_invalid_tokens():
    fg = torch.tensor([[True, False, False]])
    valid = torch.tensor([[True, True, False]])
    mass, hit = heatmap_metrics(torch.tensor([[3., 1., 100.]]), fg, valid)
    assert mass.item() == .75 and hit.item() == 1
    mass, hit = heatmap_metrics(torch.zeros(1, 3), fg, valid)
    assert mass.isnan().all() and hit.isnan().all()


def test_grid_background_windows_match_adaptive_pooling():
    membership = grid_region_membership()
    x = torch.arange(196).float().reshape(1, 1, 14, 14)
    expected = torch.nn.functional.adaptive_avg_pool2d(x, (4, 4)).flatten()
    actual = (membership * x.flatten()).sum(1) / membership.sum(1)
    torch.testing.assert_close(actual, expected)
    fg = torch.zeros(1, 196, dtype=torch.bool)
    valid = torch.ones_like(fg)
    assert background_regions(fg, valid).all()
