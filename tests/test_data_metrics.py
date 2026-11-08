"""Data contracts (subset, Pet trimap tokens, shifted crop) and per-image metrics."""

import torch

from fastweight_vit.data import subset_indices, token_foreground
from fastweight_vit.metrics import agreement, heatmap_metrics, kl_divergence


def test_agreement_compares_predictions_not_labels_and_kl_direction():
    original = torch.tensor([[2., 0.], [0., 2.], [0., 2.]])
    executed = torch.tensor([[1., 0.], [2., 0.], [0., 2.]])
    assert agreement(original, executed).tolist() == [True, False, True]
    p, q = original.double().softmax(-1), executed.double().softmax(-1)
    expected = (p * (p.log() - q.log())).sum(-1)
    torch.testing.assert_close(kl_divergence(original, executed).double(), expected)
    torch.testing.assert_close(kl_divergence(original, original), torch.zeros(3))


def test_subset_sampling_is_shared_reproducible_and_without_replacement():
    indices = subset_indices(50000, 5000)
    assert indices == subset_indices(50000, 5000, seed=0)
    assert len(set(indices)) == 5000
    assert indices != subset_indices(50000, 5000, seed=1)


def test_trimap_boundaries_and_half_foreground_threshold():
    mask = torch.full((32, 32), 3)
    mask[:8, :16] = 1
    mask[8:16, :16] = 2
    mask[16:, :16] = 2
    mask[16:, 16:] = 1
    foreground, valid = token_foreground(mask)
    assert foreground.tolist() == [True, False, False, True]
    assert valid.tolist() == [True, False, True, True]


def test_heatmap_mass_and_pointing_ignore_invalid_tokens():
    fg = torch.tensor([[True, False, False]])
    valid = torch.tensor([[True, True, False]])
    mass, hit = heatmap_metrics(torch.tensor([[3., 1., 100.]]), fg, valid)
    assert mass.item() == .75 and hit.item() == 1
    mass, hit = heatmap_metrics(torch.zeros(1, 3), fg, valid)
    assert mass.isnan().all() and hit.isnan().all()
