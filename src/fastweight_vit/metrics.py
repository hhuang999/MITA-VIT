"""Per-image measures: KL to the original output, tie-aware Spearman, and agreement with ClickMe human maps.

ClickMe (Fel et al. 2022 protocol): per image and map,
- spearman_pixel: Spearman between the map upsampled (bilinear) to 224 x 224 and the human map;
- spearman_token: Spearman between the 196 patch scores and the human map averaged per patch;
- *_shuffled: the same against another image's human map (the previous image in the batch: roll(1)). A fixed prior such as
  the centre scores the same on both, so own - shuffled (spearman_pixel) is the image-specific part reported in
  the paper.
"""

import torch
from torch.nn import functional as F


def kl_divergence(original: torch.Tensor, executed: torch.Tensor) -> torch.Tensor:
    """Return KL(p_original || p_executed) for each image."""
    return F.kl_div(
        executed.log_softmax(-1), original.log_softmax(-1),
        reduction="none", log_target=True,
    ).sum(-1)


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


ALIGNMENT = ['spearman_token', 'spearman_token_shuffled', 'spearman_pixel', 'spearman_pixel_shuffled']


def per_patch(maps):
    """[B, 224, 224] -> [B, 196] patch means."""
    return F.avg_pool2d(maps[:, None], 16).flatten(1)


def upsample(scores):
    """[B, 196] -> [B, 224, 224], bilinear."""
    return F.interpolate(scores.view(-1, 1, 14, 14), size=(224, 224), mode='bilinear', align_corners=False)[:, 0]


def alignment(scores, human):
    """Dict of [B] tensors; the shuffled control pairs each image with the previous one's human map (roll(1))."""
    other = human.roll(1, 0)
    pixel = upsample(scores)
    out = dict(spearman_token=spearman(scores, per_patch(human)), spearman_pixel=spearman(pixel, human),
               spearman_token_shuffled=spearman(scores, per_patch(other)),
               spearman_pixel_shuffled=spearman(pixel, other))
    if len(scores) == 1:
        for key in ('spearman_token_shuffled', 'spearman_pixel_shuffled'):
            out[key] = torch.full_like(out[key], float('nan'))
    return out
