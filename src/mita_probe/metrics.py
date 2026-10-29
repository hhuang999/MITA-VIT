"""Per-image measures: prediction agreement, KL to the original output, heatmap localisation."""

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
