"""Approved supplementary controls; original experiment definitions stay intact."""

from contextlib import contextmanager

import numpy as np
import torch

from .grouping import grid_landmarks
from .models import _selected_layers
from .probe import ReadingPattern, execution_mask, read_patterns


def tie_metrics(heatmap, foreground, valid):
    heatmap = np.asarray(heatmap, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    usable = valid.any(-1) & ((heatmap * valid).sum(-1) > 0)
    masked = np.where(valid, heatmap, -np.inf)
    ties = valid & (masked == masked.max(-1, keepdims=True))
    count = ties.sum(-1).astype(float)
    expected = np.divide((ties & foreground).sum(-1), count,
                         out=np.full(len(count), np.nan), where=count > 0)
    first = np.take_along_axis(foreground, masked.argmax(-1)[:, None], -1)[:, 0].astype(float)
    for array in [count, expected, first]:
        array[~usable] = np.nan
    return count, first, expected


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
    return ReadingPattern(changed, pattern.routes)


def layer_settings():
    settings = [(f'layer_{layer:02}', (layer,)) for layer in range(1, 13)]
    for depth in [3, 6, 9]:
        settings += [(f'prefix_{depth}', tuple(range(1, depth + 1))),
                     (f'suffix_{depth}', tuple(range(13 - depth, 13)))]
    return settings + [('all12', tuple(range(1, 13)))]


@contextmanager
def execute_online(model, k_top, *, layers=None):
    """Recompute both read sets and routes from the current attention input."""
    selected = _selected_layers(model, layers)
    if not 1 <= k_top <= model.patch_embed.num_patches + model.num_prefix_tokens:
        raise ValueError('Invalid reading budget')
    patterns, handles = {}, []

    def make_hook(layer):
        @torch.no_grad()
        def restrict(attention, args, kwargs):
            if kwargs.get('attn_mask') is not None or kwargs.get('is_causal', False):
                raise ValueError('Expected unmasked, noncausal attention')
            x = args[0]
            batch, tokens, _ = x.shape
            qkv = attention.qkv(x).reshape(batch, tokens, 3, attention.num_heads,
                                           attention.head_dim).permute(2, 0, 3, 1, 4)
            q, key, _ = qkv.unbind(0)
            q, key = attention.q_norm(q), attention.k_norm(key)
            landmarks = grid_landmarks(q[:, :, model.num_prefix_tokens:],
                                       model.patch_embed.grid_size, (4, 4))
            pattern = read_patterns(q, key, landmarks, k_top, attention.scale)
            patterns[layer] = pattern
            return args, {**kwargs, 'attn_mask': execution_mask(pattern)}
        return restrict

    try:
        for layer in selected:
            handles.append(model.blocks[layer - 1].attn.register_forward_pre_hook(
                make_hook(layer), with_kwargs=True))
        yield patterns
    finally:
        for handle in handles:
            handle.remove()
