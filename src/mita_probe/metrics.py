"""Prediction fidelity and paired image-level bootstrap intervals."""

import numpy as np
import torch
from torch.nn import functional as F


def agreement(original: torch.Tensor, executed: torch.Tensor) -> torch.Tensor:
    return original.argmax(-1).eq(executed.argmax(-1))


def kl_divergence(original: torch.Tensor, executed: torch.Tensor) -> torch.Tensor:
    """Return KL(p_original || p_executed) for each image."""
    return F.kl_div(
        executed.log_softmax(-1), original.log_softmax(-1),
        reduction="none", log_target=True,
    ).sum(-1)


def paired_bootstrap_difference(
    ours: np.ndarray,
    random: np.ndarray,
    *,
    seed: int = 0,
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """ours: [images]; random: [seeds, images]; resample shared image indices.

    Seeds are averaged before bootstrapping. This interval measures image sampling
    uncertainty, not uncertainty over random seeds.
    """
    ours, random = np.asarray(ours), np.asarray(random)
    if ours.ndim != 1 or random.ndim != 2 or random.shape[1] != len(ours) or not len(ours):
        raise ValueError("Expected ours [images] and random [seeds, same images]")
    difference = ours.astype(float) - random.mean(axis=0)
    rng = np.random.default_rng(seed)
    means = np.array([
        difference[rng.integers(len(difference), size=len(difference))].mean()
        for _ in range(n_resamples)
    ])
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(difference.mean()), float(low), float(high)


def foreground_ratio(readsets, foreground, valid, regions=None):
    """Per-image selected foreground fraction; exclude CLS and invalid tokens.

    Count occurrences across heads/regions, not the union of selected tokens.
    An optional [batch, regions] mask selects only background source regions.
    """
    batch = readsets.shape[0]
    zero = torch.zeros(batch, 1, dtype=torch.bool, device=foreground.device)
    fg = torch.cat([zero, foreground & valid], dim=1)
    available = torch.cat([zero, valid], dim=1)
    selected_fg = fg.gather(1, readsets.flatten(1)).reshape_as(readsets)
    selected_valid = available.gather(1, readsets.flatten(1)).reshape_as(readsets)
    if regions is not None:
        region_mask = regions[:, None, :, None]
        selected_fg = selected_fg & region_mask
        selected_valid = selected_valid & region_mask
    numerator = selected_fg.flatten(1).sum(1)
    denominator = selected_valid.flatten(1).sum(1)
    return torch.where(denominator > 0, numerator / denominator, torch.nan)


def heatmap_metrics(heatmap, foreground, valid):
    """Foreground mass fraction and pointing hit on valid spatial tokens."""
    if (heatmap < 0).any() or not torch.isfinite(heatmap).all():
        raise ValueError("Heatmap values must be finite and nonnegative")
    mass = (heatmap * valid).sum(-1)
    ratio = (heatmap * foreground * valid).sum(-1) / mass
    # torch.argmax resolves equal maxima by the first flattened spatial index.
    maximum = heatmap.masked_fill(~valid, -torch.inf).argmax(-1)
    hit = foreground.gather(1, maximum[:, None]).squeeze(1).float()
    usable = valid.any(-1) & mass.gt(0)
    return torch.where(usable, ratio, torch.nan), torch.where(usable, hit, torch.nan)
