"""Neuron-level interventions on a probe pattern: remove chosen tokens from reading sets and refill.

All functions take a frozen pattern (read from the original forward) and the scores that ranked
its tokens, and return a new pattern of the same budget. Removing a category (foreground,
background, high-norm) refills each set with its best-scoring outside tokens of other
categories; the random and centre-matched controls remove the same number of tokens per set and
refill the same way, so the conditions differ only in which tokens leave.
"""

import numpy as np
import torch

from .execution import Pattern


def sink_tokens(norms, threshold=3.):
    """Use the midpoint median, including for an even number of patches."""
    return norms > threshold * torch.quantile(norms, .5, dim=-1, keepdim=True)


def replace_category(pattern, scores, forbidden):
    """Keep surviving reads; fill from original scores. Return feasibility per image.

    Infeasible images retain their original pattern for safe batched inference,
    but callers MUST exclude their outputs and record the reason.
    """
    k = pattern.readsets.shape[-1]
    blocked = forbidden[:, None, None, :].expand_as(scores)
    selected_blocked = blocked.gather(-1, pattern.readsets)
    eligible = (~forbidden).sum(-1).ge(k)
    original = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, pattern.readsets, True)
    order = scores.masked_fill(blocked | original, -torch.inf).argsort(
        dim=-1, descending=True, stable=True)
    rank = (selected_blocked.long().cumsum(-1) - 1).clamp_min(0)
    proposed = torch.where(selected_blocked, order.gather(-1, rank), pattern.readsets)
    proposed = torch.where(eligible[:, None, None, None], proposed, pattern.readsets)
    return Pattern(proposed, pattern.routes), eligible, selected_blocked.sum(-1)


def replace_random_patches(pattern, scores, number, generator, prefix=1):
    """Match each set's dose, preserve selected prefixes, refill outside old set."""
    reads = pattern.readsets
    k = reads.shape[-1]
    available = reads.ge(prefix)
    if (number < 0).any() or (number > available.sum(-1)).any():
        raise ValueError('Dose exceeds selected patch count')
    if scores.shape[-1] - k < int(number.max()):
        raise ValueError('Insufficient outside-set replacement candidates')
    order = torch.rand(reads.shape, device=reads.device, generator=generator)
    order = order.masked_fill(~available, torch.inf).argsort(dim=-1, stable=True)
    remove = torch.zeros_like(reads, dtype=torch.bool)
    remove.scatter_(-1, order, torch.arange(k, device=reads.device) < number[..., None])
    original = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, reads, True)
    ranked = scores.masked_fill(original, -torch.inf).argsort(
        dim=-1, descending=True, stable=True)
    rank = (remove.long().cumsum(-1) - 1).clamp_min(0)
    changed = torch.where(remove, ranked.gather(-1, rank), reads)
    return Pattern(changed, pattern.routes)


def centre_rings(grid=14, prefix=1):
    """Ring id of every token by distance to the image centre; prefix tokens get -1."""
    rows, cols = np.divmod(np.arange(grid * grid), grid)
    distance = (rows - (grid - 1) / 2) ** 2 + (cols - (grid - 1) / 2) ** 2
    rings = np.unique(distance, return_inverse=True)[1]
    return torch.cat([torch.full((prefix,), -1), torch.from_numpy(rings)])


def replace_matched_patches(pattern, scores, forbidden, rings, generator):
    """Remove, per set, as many members as it holds forbidden ones, drawn at random from the same rings.

    The refill is the same as replace_random_patches, so only which members leave differs.
    """
    reads = pattern.readsets
    k = reads.shape[-1]
    blocked = forbidden[:, None, None, :].expand(*reads.shape[:3], -1).gather(-1, reads)
    ring = rings.to(reads.device)[reads]
    key = torch.rand(reads.shape, device=reads.device, generator=generator)
    remove = torch.zeros_like(reads, dtype=torch.bool)
    for value in ring.unique().tolist():
        if value < 0:
            continue
        inside = ring == value
        need = (blocked & inside).sum(-1, keepdim=True)
        order = key.masked_fill(~inside, torch.inf).argsort(dim=-1, stable=True)
        rank = torch.empty_like(order).scatter_(-1, order, torch.arange(k, device=reads.device).expand_as(order))
        remove |= inside & (rank < need)
    original = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, reads, True)
    ranked = scores.masked_fill(original, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    position = (remove.long().cumsum(-1) - 1).clamp_min(0)
    changed = torch.where(remove, ranked.gather(-1, position), reads)
    return Pattern(changed, pattern.routes), remove
