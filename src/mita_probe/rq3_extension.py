"""Preregistered RQ3 diagnostics and constant-budget interventions."""

from contextlib import contextmanager

import numpy as np
import torch

from .probe import ReadingPattern


@contextmanager
def capture_input_norms(model):
    """Capture block inputs, before norm1; exclude every prefix token."""
    norms, handles = {}, []

    def hook(layer):
        def capture(module, inputs):
            norms[layer] = inputs[0][:, model.num_prefix_tokens:].float().norm(dim=-1)
        return capture

    try:
        for layer, block in enumerate(model.blocks, 1):
            handles.append(block.register_forward_pre_hook(hook(layer)))
        yield norms
    finally:
        for handle in handles:
            handle.remove()


def sink_tokens(norms, threshold=3.):
    """Use the midpoint median, including for an even number of patches."""
    return norms > threshold * torch.quantile(norms, .5, dim=-1, keepdim=True)


def token_counts(readsets, n_tokens):
    counts = torch.zeros(readsets.shape[0], n_tokens, device=readsets.device)
    flat = readsets.flatten(1)
    return counts.scatter_add_(1, flat, torch.ones_like(flat, dtype=counts.dtype))


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
    return ReadingPattern(proposed, pattern.routes), eligible, selected_blocked.sum(-1)


def replace_random(pattern, scores, number, generator):
    """Remove exactly number original members and replace from outside that set."""
    k = pattern.readsets.shape[-1]
    random_order = torch.rand(pattern.readsets.shape, device=scores.device,
                              generator=generator).argsort(-1)
    remove = torch.zeros_like(pattern.readsets, dtype=torch.bool)
    remove.scatter_(-1, random_order, torch.arange(k, device=scores.device) < number[..., None])
    original_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, pattern.readsets, True)
    replacements = scores.masked_fill(original_mask, -torch.inf).argsort(
        dim=-1, descending=True, stable=True)[..., :k]
    replacement_rank = (remove.long().cumsum(-1) - 1).clamp_min(0)
    reads = torch.where(remove, replacements.gather(-1, replacement_rank), pattern.readsets)
    return ReadingPattern(reads, pattern.routes)


def numpy_heatmap_metrics(heatmap, foreground, valid):
    """Independent CPU implementation, with NaN for zero usable mass."""
    heatmap = np.asarray(heatmap, dtype=np.float64)
    mass = (heatmap * valid).sum(-1)
    usable = valid.any(-1) & (mass > 0)
    ratio = np.divide((heatmap * foreground * valid).sum(-1), mass,
                      out=np.full_like(mass, np.nan), where=mass > 0)
    peak = np.where(valid, heatmap, -np.inf).argmax(-1)
    hit = np.take_along_axis(foreground, peak[..., None], -1)[..., 0].astype(float)
    hit[~usable] = np.nan
    return ratio, hit, peak


def bootstrap(values, resamples=1000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(mean=np.nan, low=np.nan, high=np.nan, n=0)
    rng = np.random.default_rng(0)
    means = np.array([values[rng.integers(len(values), size=len(values))].mean()
                      for _ in range(resamples)])
    low, high = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), low=float(low), high=float(high), n=len(values))
