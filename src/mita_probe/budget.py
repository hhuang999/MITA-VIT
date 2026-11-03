"""Attention budget (decisions B27-B30): where each query's softmax weight goes, and edits of what is read.

Every query's attention weights sum to 1: its budget. `instrument` runs selected layers with an explicit
softmax and records, per layer and head, how much budget every token receives (from all queries, and from
the CLS query alone). For MiTA models the weight a query puts on a summary slot is passed back to the
tokens the slot averages: A = P_token + P_slot S, where S[s, j] = softmax_j(landmark_s . key_j). S's rows
sum to 1, so A's rows still sum to 1 and A V equals MiTA's output.

The same forward can edit what a dense layer reads (`Edits`):
- block: tokens no query may read (logit -inf before the softmax; the budget moves to the other tokens);
- zero: tokens whose value is set to 0 (the budget is still spent on them but brings nothing back);
- mean: tokens whose value is replaced by a fixed per-head vector (only a constant is brought back).

`categories` and `shares` split the budget into CLS, register, high-norm, object, background and unknown
(boundary) tokens. `ring_enrichment` scores a chosen set of patches against the same-distance-ring base rate.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from .interventions import centre_rings
from .probe.execute import selected_layers, split_qkv
from .probe.pattern import build_pattern, landmarks, token_mask

CATEGORIES = ('cls', 'register', 'high', 'object', 'background', 'unknown')
CLS, REGISTER, HIGH, OBJECT, BACKGROUND, UNKNOWN = range(len(CATEGORIES))


@dataclass
class Edits:
    """Per one-based layer: [batch, tokens] bool masks over all tokens (prefix included)."""
    block: dict = field(default_factory=dict)
    zero: dict = field(default_factory=dict)
    mean: dict = field(default_factory=dict)  # layer -> (mask [batch, tokens], values [heads, head_dim])

    def touches(self, layer):
        return layer in self.block or layer in self.zero or layer in self.mean


def effective_rank(matrix):
    """exp(entropy of the normalised singular values) of [..., rows, cols]."""
    values = torch.linalg.svdvals(matrix.double())
    p = values / values.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.exp(-(p * p.clamp_min(1e-30).log()).sum(-1)).float()


def mita_attention(attention, q, k, plan, grid, prefix):
    """MiTA's attention as one [batch, heads, tokens, tokens] matrix (summary slots passed back)."""
    pattern, _ = build_pattern(q, k, m_side=plan.summary_side, k_top=plan.native_k, scale=attention.scale,
                               grid=grid, routing='dot', cls='routed', prefix=prefix,
                               landmark_mode=plan.landmark_mode)
    mask = token_mask(pattern)
    lm = landmarks(q, grid, plan.summary_side, prefix, plan.landmark_mode)
    logits = ((q * attention.scale) @ k.transpose(-2, -1)).masked_fill(~mask, -torch.inf)
    key = lm * attention.scale if plan.summary_scale == 'mita' else lm
    p = torch.cat([logits, (q * attention.scale) @ key.transpose(-2, -1)], -1).softmax(-1)
    n_tokens = k.shape[2]
    spread = ((lm * attention.scale) @ k.transpose(-2, -1)).softmax(-1)
    return p[..., :n_tokens] + p[..., n_tokens:] @ spread, mask


@contextmanager
def instrument(model, layers=None, *, edits=None, values=False, rank=False, selected=False):
    """Record each selected layer's budget and optionally edit what dense layers read.

    Yields {layer: dict}: 'received' [B, H, N] (budget each token gets from all queries), 'cls' [B, H, N]
    (from the CLS query); with values=True 'value_norm' [B, H, N]; with rank=True 'rank' [B, H] (effective
    rank of each head's output over all queries); with selected=True 'selected' [B, N] (tokens read by any
    query of any head: MiTA's top-k sets, or each query's top-k for dense layers with k = model.mita_k or 25).
    """
    edits = edits or Edits()
    plan = getattr(model, 'mita_plan', None)
    if plan is not None and any(edits.touches(layer) for layer in range(1, len(model.blocks) + 1)):
        raise ValueError('Edits are implemented for dense attention only')
    layers = selected_layers(model, layers)
    grid, prefix = model.patch_embed.grid_size, model.num_prefix_tokens
    store, saved = {}, {}

    def make_forward(layer, attention):
        def forward(x, attn_mask=None, is_causal=False):
            if attn_mask is not None or is_causal:
                raise ValueError('Expected unmasked, noncausal self-attention')
            batch, tokens, _ = x.shape
            q, k, v = split_qkv(attention, x)
            if plan is not None:
                a, read = mita_attention(attention, q, k, plan, grid, prefix)
            else:
                logits = (q * attention.scale) @ k.transpose(-2, -1)
                if layer in edits.block:
                    logits = logits.masked_fill(edits.block[layer][:, None, None, :], -torch.inf)
                a = logits.softmax(-1)
                read = None
                if layer in edits.zero:
                    v = v.masked_fill(edits.zero[layer][:, None, :, None], 0.)
                if layer in edits.mean:
                    where, fill = edits.mean[layer]
                    v = torch.where(where[:, None, :, None], fill[None, :, None, :].to(v.dtype), v)
            out = a @ v
            record = dict(received=a.sum(-2), cls=a[:, :, 0])
            if values:
                record['value_norm'] = v.norm(dim=-1)
            if rank:
                record['rank'] = effective_rank(out)
            if selected:
                if read is None:
                    top = a.topk(min(getattr(model, 'mita_k', 25), tokens), dim=-1).indices
                    read = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, top, True)
                record['selected'] = read.any(2).any(1)
            store[layer] = record
            out = out.transpose(1, 2).reshape(batch, tokens, -1)
            return attention.proj_drop(attention.proj(attention.norm(out)))
        return forward

    try:
        for layer in layers:
            attention = model.blocks[layer - 1].attn
            saved[layer] = attention.__dict__.get('forward')
            attention.forward = make_forward(layer, attention)
        yield store
    finally:
        for layer, forward in saved.items():
            attention = model.blocks[layer - 1].attn
            if forward is None:
                del attention.forward
            else:
                attention.forward = forward


def categories(foreground, valid, high, prefix=1):
    """[B, prefix + patches] category codes: CLS, register (prefix tokens after CLS), high, object, ..."""
    patches = torch.full(foreground.shape, BACKGROUND, dtype=torch.long, device=foreground.device)
    patches[foreground & valid] = OBJECT
    patches[~valid] = UNKNOWN
    patches[high] = HIGH
    head = torch.full((foreground.shape[0], prefix), REGISTER, dtype=torch.long, device=foreground.device)
    head[:, 0] = CLS
    return torch.cat([head, patches], 1)


def shares(received, codes):
    """[B, categories] share of the budget per category; received [B, H, N] from any number of queries."""
    per_token = received.float().mean(1)
    per_token = per_token / per_token.sum(1, keepdim=True)
    return torch.zeros(codes.shape[0], len(CATEGORIES), device=codes.device).scatter_add_(1, codes, per_token)


def ring_enrichment(chosen, foreground, valid, grid=14):
    """Same-distance-ring enrichment of a chosen patch set: object rate among chosen minus the ring's rate.

    chosen, foreground, valid: [B, patches] bool. Rings weighted by how many patches were chosen in them;
    NaN when nothing valid was chosen. The scoring rule of plausibility.measures, for a given set.
    """
    rings = centre_rings(grid)[1:].to(chosen.device)
    fg = (foreground & valid).double()
    total = torch.zeros(chosen.shape[0], dtype=torch.float64, device=chosen.device)
    weights = torch.zeros_like(total)
    for ring in rings.unique().tolist():
        members = (rings == ring) & valid
        picked = members & chosen
        count = picked.sum(1).double()
        rate = (fg * picked).sum(1) / count.clamp_min(1) - (fg * members).sum(1) / members.sum(1).double().clamp_min(1)
        total += torch.where(count > 0, rate * count, torch.zeros_like(rate))
        weights += count
    return torch.where(weights > 0, total / weights.clamp_min(1), torch.full_like(total, float('nan')))
