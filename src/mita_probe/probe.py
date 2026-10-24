"""Read a summary from original Q/K and build its strict execution mask."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ReadingPattern:
    readsets: torch.Tensor  # [batch, heads, regions, k], zero-based token indices
    routes: torch.Tensor  # [batch, heads, tokens], zero-based region indices


def read_patterns(
    queries: torch.Tensor,
    keys: torch.Tensor,
    landmarks: torch.Tensor,
    k_top: int,
    scale: float | None = None,
) -> ReadingPattern:
    scale = queries.shape[-1] ** -0.5 if scale is None else scale
    scores = (landmarks * scale) @ keys.transpose(-2, -1)
    readsets = scores.topk(k_top, dim=-1).indices
    routes = (queries @ landmarks.transpose(-2, -1)).argmax(-1)
    return ReadingPattern(readsets, routes)


def execution_mask(pattern: ReadingPattern) -> torch.Tensor:
    """True means a key is allowed, matching timm / PyTorch SDPA semantics."""
    batch, heads, regions, _ = pattern.readsets.shape
    n_tokens = pattern.routes.shape[-1]
    allowed = torch.zeros(
        batch, heads, regions, n_tokens,
        dtype=torch.bool, device=pattern.readsets.device,
    )
    allowed.scatter_(-1, pattern.readsets, True)
    return allowed.gather(2, pattern.routes.unsqueeze(-1).expand(-1, -1, -1, n_tokens))
