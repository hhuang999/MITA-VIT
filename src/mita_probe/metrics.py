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
