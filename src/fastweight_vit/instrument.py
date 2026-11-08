"""Attention received per token, for dense ViTs and for MiTA (summary slots passed back to tokens).

`instrument` runs selected layers with an explicit softmax and records, per layer and head, how much attention
every token receives from all queries ('received' [B, H, N]); with values=True also the value norm ||v_hj||. It
gives R0 for the model variants of Table 1 (the test-time register and MiTA-DeiT-S) and the high-norm counts of
the register-neuron search.

For MiTA models the weight a query puts on a summary slot is passed back to the tokens the slot averages:
A = P_token + P_slot S, where S[s, j] = softmax_j(landmark_s . key_j). S's rows sum to 1, so A's rows still sum
to 1 and A V equals MiTA's output.
"""

from contextlib import contextmanager

import torch

from .execution.execute import selected_layers, split_qkv
from .execution.pattern import build_pattern, landmarks, token_mask


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
def instrument(model, layers=None, *, values=False):
    """Yields {layer: dict(received=[B, H, N])} (and 'value_norm' [B, H, N] with values=True), filled by a forward."""
    plan = getattr(model, 'mita_plan', None)
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
                a, _ = mita_attention(attention, q, k, plan, grid, prefix)
            else:
                a = ((q * attention.scale) @ k.transpose(-2, -1)).softmax(-1)
            out = a @ v
            record = dict(received=a.sum(-2))
            if values:
                record['value_norm'] = v.norm(dim=-1)
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
