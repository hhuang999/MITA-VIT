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


def average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Row-wise ranks (0-based) with ties given their average rank, as in scipy.stats.rankdata."""
    rows, n = values.shape
    order = values.argsort(dim=-1, stable=True)
    ordered = values.gather(-1, order)
    starts = torch.ones_like(ordered, dtype=torch.bool)
    starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    group = starts.cumsum(-1) - 1 + torch.arange(rows, device=values.device)[:, None] * n
    position = torch.arange(n, device=values.device, dtype=torch.float64).expand(rows, n).flatten()
    total = torch.zeros(rows * n, dtype=torch.float64, device=values.device).index_add_(0, group.flatten(), position)
    count = torch.zeros(rows * n, dtype=torch.float64, device=values.device).index_add_(
        0, group.flatten(), torch.ones_like(position))
    mean_rank = (total / count.clamp_min(1))[group.flatten()].view(rows, n)
    return torch.empty_like(mean_rank).scatter_(-1, order, mean_rank)


def spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise Spearman correlation of [rows, n] tensors (tie-aware); NaN when a row is constant."""
    ra, rb = average_ranks(a.flatten(1)), average_ranks(b.flatten(1))
    ra, rb = ra - ra.mean(-1, keepdim=True), rb - rb.mean(-1, keepdim=True)
    return (ra * rb).sum(-1) / (ra.norm(dim=-1) * rb.norm(dim=-1))
