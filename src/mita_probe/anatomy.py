"""V2 anatomy probes: per-query, summary-only, shared top-k and mixed (MiTA) attention.

Discrete choices (reading sets, query routes, CLS sets) are read from the original
forward and frozen. Summary slots are rebuilt from the current Q/K/V of the
executed layer: their region membership is fixed by the grid, so no choice is made.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .grouping import grid_landmarks
from .models import _selected_layers

M_SIDES = {1: 1, 4: 2, 16: 4, 25: 5, 49: 7, 196: 14}


@dataclass(frozen=True)
class Pattern:
    readsets: torch.Tensor  # [batch, heads, groups, k]; groups = regions or tokens (per query)
    routes: torch.Tensor  # [batch, heads, tokens] group read by each query
    cls_set: torch.Tensor | None = None  # [batch, heads, k]; CLS reads its own set when given


@dataclass(frozen=True)
class Plan:
    """What one executed layer may read. None token_mask means no token slots."""
    token_mask: torch.Tensor | None = None  # [batch, heads, tokens, tokens], True = allowed
    summary_side: int | None = None  # m_side of summary slots
    summary_scale: str = 'mita'  # 'mita': q.lm/d as in the official code; 'standard': q.lm/sqrt(d)
    landmark_mode: str = 'clean'  # 'clean': patches only; 'mita': official grid (CLS in, last patch out)
    native_k: int | None = None  # recompute MiTA reading sets/routes from the current Q/K
    drop: bool = False  # attention branch outputs zero


def spatial_routes(grid, m_side, prefix=1, device=None):
    """Assign each patch to one region whose adaptive-pooling window contains it."""
    rows = torch.arange(grid[0], device=device) * m_side // grid[0]
    cols = torch.arange(grid[1], device=device) * m_side // grid[1]
    regions = (rows[:, None] * m_side + cols[None, :]).flatten()
    return torch.cat([torch.zeros(prefix, dtype=torch.long, device=device), regions])


def landmarks(q, grid, m_side, prefix=1, mode='clean'):
    if mode == 'clean':
        return grid_landmarks(q[:, :, prefix:], grid, (m_side, m_side))
    if mode == 'mita':
        return grid_landmarks(q[:, :, :grid[0] * grid[1]], grid, (m_side, m_side))
    raise ValueError(f'Unknown landmark mode {mode}')


def region_scores(q, k, lm, scale, *, scoring='query', grid=None, m_side=None, prefix=1):
    """'query': landmark logits (mean of region logits); 'attention': mean region attention."""
    if scoring == 'query':
        return (lm * scale) @ k.transpose(-2, -1)
    if scoring == 'attention':
        probs = ((q * scale) @ k.transpose(-2, -1)).softmax(-1)
        return grid_landmarks(probs[:, :, prefix:], grid, (m_side, m_side))
    raise ValueError(f'Unknown scoring {scoring}')


def build_pattern(q, k, *, m_side, k_top, scale, grid, routing='spatial', cls='own',
                  scoring='query', prefix=1, landmark_mode='clean'):
    """Return the shared top-k pattern and the region scores used to rank tokens."""
    lm = landmarks(q, grid, m_side, prefix, landmark_mode)
    scores = region_scores(q, k, lm, scale, scoring=scoring, grid=grid, m_side=m_side, prefix=prefix)
    readsets = scores.topk(k_top, dim=-1).indices
    if routing == 'spatial':
        routes = spatial_routes(grid, m_side, prefix, q.device).expand(*q.shape[:3])
    elif routing == 'dot':
        routes = (q @ lm.transpose(-2, -1)).argmax(-1)
    else:
        raise ValueError(f'Unknown routing {routing}')
    cls_set = None
    if cls == 'own':
        cls_set = ((q[:, :, 0] * scale).unsqueeze(-2) @ k.transpose(-2, -1)).squeeze(-2).topk(k_top, -1).indices
    elif cls != 'routed':
        raise ValueError(f'Unknown CLS mode {cls}')
    return Pattern(readsets, routes, cls_set), scores


def per_query_pattern(q, k, k_top, scale):
    readsets = ((q * scale) @ k.transpose(-2, -1)).topk(k_top, dim=-1).indices
    routes = torch.arange(q.shape[2], device=q.device).expand(*q.shape[:3])
    return Pattern(readsets, routes)


def random_pattern(pattern, generator):
    """Uniform sets without replacement; routes unchanged; CLS keeps a set of its own if it had one."""
    def draw(shape):
        weights = torch.ones(shape[:-1].numel(), n_tokens, device=pattern.readsets.device)
        return torch.multinomial(weights, shape[-1], replacement=False, generator=generator).reshape(shape)
    n_tokens = pattern.routes.shape[-1]
    cls_set = None if pattern.cls_set is None else draw(pattern.cls_set.shape)
    return Pattern(draw(pattern.readsets.shape), pattern.routes, cls_set)


def token_mask(pattern):
    batch, heads, groups, _ = pattern.readsets.shape
    n_tokens = pattern.routes.shape[-1]
    allowed = torch.zeros(batch, heads, groups, n_tokens, dtype=torch.bool, device=pattern.readsets.device)
    allowed.scatter_(-1, pattern.readsets, True)
    mask = allowed.gather(2, pattern.routes.unsqueeze(-1).expand(-1, -1, -1, n_tokens))
    cls_set = getattr(pattern, 'cls_set', None)  # v1 ReadingPattern has none
    if cls_set is not None:
        cls_row = torch.zeros(batch, heads, n_tokens, dtype=torch.bool, device=mask.device)
        mask[:, :, 0] = cls_row.scatter_(-1, cls_set, True)
    return mask


def coverage(probs, mask):
    """Original attention mass inside the allowed set: patch queries and CLS, per image."""
    covered = (probs * mask).sum(-1)
    return covered[:, :, 1:].mean((1, 2)), covered[:, :, 0].mean(1)


def union_fraction(pattern):
    """Fraction of all tokens read by any region set or the CLS set, averaged over heads."""
    batch, heads, _, _ = pattern.readsets.shape
    n_tokens = pattern.routes.shape[-1]
    used = torch.zeros(batch, heads, n_tokens, dtype=torch.bool, device=pattern.readsets.device)
    used.scatter_(-1, pattern.readsets.flatten(2), True)
    if pattern.cls_set is not None:
        used.scatter_(-1, pattern.cls_set, True)
    return used.float().mean(-1).mean(-1)


def split_qkv(attention, x):
    batch, tokens, _ = x.shape
    qkv = attention.qkv(x).reshape(batch, tokens, 3, attention.num_heads,
                                   attention.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    return attention.q_norm(q), attention.k_norm(k), v


@contextmanager
def capture_qk(model, layers=None):
    """Observe each selected layer's normalized Q/K from the latest forward."""
    selected = _selected_layers(model, layers)
    captured, handles = {}, []

    def make_hook(layer, attention):
        @torch.no_grad()
        def hook(module, inputs, output):
            batch, tokens, _ = output.shape
            q, k, _ = output.reshape(batch, tokens, 3, attention.num_heads,
                                     attention.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            captured[layer] = (attention.q_norm(q), attention.k_norm(k))
        return hook

    try:
        for layer in selected:
            attention = model.blocks[layer - 1].attn
            handles.append(attention.qkv.register_forward_hook(make_hook(layer, attention)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def mixed_attention(q, k, v, scale, mask=None, lm=None, summary_scale='mita'):
    """One softmax over allowed token slots and, given landmarks, their summary slots."""
    logits = (q * scale) @ k.transpose(-2, -1)
    logits = logits.masked_fill(~mask, -torch.inf) if mask is not None else torch.full_like(logits, -torch.inf)
    values = v
    if lm is not None:
        summary = ((lm * scale) @ k.transpose(-2, -1)).softmax(-1) @ v
        key = lm * scale if summary_scale == 'mita' else lm
        logits = torch.cat([logits, (q * scale) @ key.transpose(-2, -1)], -1)
        values = torch.cat([v, summary], 2)
    return logits.softmax(-1) @ values


def mixed_forward(attention, x, plan, grid, prefix=1):
    batch, tokens, _ = x.shape
    if plan.drop:
        return x.new_zeros(batch, tokens, attention.proj.out_features)
    q, k, v = split_qkv(attention, x)
    mask = plan.token_mask
    if plan.native_k:
        pattern, _ = build_pattern(q, k, m_side=plan.summary_side, k_top=plan.native_k,
                                   scale=attention.scale, grid=grid, routing='dot', cls='routed',
                                   prefix=prefix, landmark_mode=plan.landmark_mode)
        mask = token_mask(pattern)
    lm = landmarks(q, grid, plan.summary_side, prefix, plan.landmark_mode) if plan.summary_side else None
    out = mixed_attention(q, k, v, attention.scale, mask, lm, plan.summary_scale)
    out = out.transpose(1, 2).reshape(batch, tokens, -1)
    return attention.proj_drop(attention.proj(attention.norm(out)))


def install(attention, plan, grid, prefix=1):
    def forward(x, attn_mask=None, is_causal=False):
        if attn_mask is not None or is_causal:
            raise ValueError('The anatomy probe expects unmasked, noncausal self-attention')
        return mixed_forward(attention, x, plan, grid, prefix)
    attention.forward = forward


@contextmanager
def execute(model, plans):
    """Temporarily replace each selected layer's attention forward; restore on exit."""
    selected = _selected_layers(model, plans)
    saved = {}
    try:
        for layer in selected:
            attention = model.blocks[layer - 1].attn
            saved[layer] = attention.__dict__.get('forward')
            install(attention, plans[layer], model.patch_embed.grid_size, model.num_prefix_tokens)
        yield
    finally:
        for layer, forward in saved.items():
            attention = model.blocks[layer - 1].attn
            if forward is None:
                del attention.forward
            else:
                attention.forward = forward
