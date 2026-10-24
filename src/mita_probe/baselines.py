"""Random reading sets with unchanged query-to-region assignments."""

import torch

from .probe import ReadingPattern


def random_readsets(pattern: ReadingPattern, generator: torch.Generator) -> ReadingPattern:
    """Draw uniform subsets without replacement; the caller owns the seeded RNG."""
    batch, heads, regions, k_top = pattern.readsets.shape
    n_tokens = pattern.routes.shape[-1]
    weights = torch.ones(batch * heads * regions, n_tokens, device=pattern.readsets.device)
    sampled = torch.multinomial(weights, k_top, replacement=False, generator=generator)
    return ReadingPattern(sampled.reshape(batch, heads, regions, k_top), pattern.routes)
