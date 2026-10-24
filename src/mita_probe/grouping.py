"""Per-image, per-head landmarks from patch queries (CLS excluded)."""

import torch
from torch.nn import functional as F


def grid_landmarks(
    patches: torch.Tensor,
    grid_size: tuple[int, int],
    output_size: tuple[int, int] = (4, 4),
) -> torch.Tensor:
    batch, heads, _, dim = patches.shape
    grid = patches.reshape(batch * heads, *grid_size, dim).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, output_size)
    return pooled.flatten(2).transpose(1, 2).reshape(batch, heads, -1, dim)


def kmeans_landmarks(
    patches: torch.Tensor,
    initial: torch.Tensor,
    iterations: int = 10,
) -> torch.Tensor:
    """Lloyd's Euclidean k-means; retain a previous center if its cluster is empty."""
    centers = initial.clone()
    for _ in range(iterations):
        distances = (
            patches.square().sum(-1, keepdim=True)
            + centers.square().sum(-1).unsqueeze(-2)
            - 2 * patches @ centers.transpose(-2, -1)
        )
        assignments = distances.argmin(-1)
        membership = F.one_hot(assignments, centers.shape[-2]).to(patches.dtype)
        counts = membership.sum(-2).unsqueeze(-1)
        means = membership.transpose(-2, -1) @ patches / counts.clamp_min(1)
        centers = torch.where(counts > 0, means, centers)
    return centers
