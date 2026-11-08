"""Run a model with selected attention layers restricted to a plan, and observe layers.

`execute` swaps each selected layer's attention forward for a dense re-implementation that
reads only the allowed token slots (and optional region summary slots); `capture_*` hooks
observe Q/K or block-input norms without changing the output. MiTA-DeiT-S runs through the same
forward with its native plan (summary slots, routing recomputed from the current Q/K).
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .pattern import build_pattern, landmarks, token_mask


@dataclass(frozen=True)
class Plan:
    """What one executed layer may read. None token_mask means no token slots."""
    token_mask: torch.Tensor | None = None  # [batch, heads, tokens, tokens], True = allowed
    summary_side: int | None = None  # m_side of summary slots
    summary_scale: str = 'mita'  # 'mita': q.lm/d as in the official code; 'standard': q.lm/sqrt(d)
    landmark_mode: str = 'clean'  # 'clean': patches only; 'mita': official grid (CLS in, last patch out)
    native_k: int | None = None  # recompute MiTA reading sets/routes from the current Q/K
    drop: bool = False  # attention branch outputs zero


def selected_layers(model, layers):
    if model.training:
        raise ValueError('Reading attention requires model.eval()')
    selected = tuple(range(1, len(model.blocks) + 1)) if layers is None else tuple(layers)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('Select a nonempty set of distinct layers')
    if any(layer < 1 or layer > len(model.blocks) for layer in selected):
        raise ValueError('Layer numbers are one-based')
    return selected


def split_qkv(attention, x):
    batch, tokens, _ = x.shape
    qkv = attention.qkv(x).reshape(batch, tokens, 3, attention.num_heads,
                                   attention.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    return attention.q_norm(q), attention.k_norm(k), v


@contextmanager
def capture_qk(model, layers=None):
    """Observe each selected layer's normalized Q/K from the latest forward."""
    selected = selected_layers(model, layers)
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


@contextmanager
def capture_input_norms(model, prefix=None):
    """Capture each block's input norm per spatial token (before norm1; prefix tokens excluded).

    prefix: number of leading non-patch tokens (default model.num_prefix_tokens; 2 with a test-time register).
    """
    norms, handles = {}, []
    prefix = model.num_prefix_tokens if prefix is None else prefix

    def hook(layer):
        def capture(module, inputs):
            norms[layer] = inputs[0][:, prefix:].float().norm(dim=-1)
        return capture

    try:
        for layer, block in enumerate(model.blocks, 1):
            handles.append(block.register_forward_pre_hook(hook(layer)))
        yield norms
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
            raise ValueError('Expected unmasked, noncausal self-attention')
        return mixed_forward(attention, x, plan, grid, prefix)
    attention.forward = forward


@contextmanager
def execute(model, plans):
    """Temporarily replace each selected layer's attention forward; restore on exit."""
    selected = selected_layers(model, plans)
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
