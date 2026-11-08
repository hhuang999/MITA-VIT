"""Reference maps and the top-k map, all as one score per patch token [batch, 196] (larger = more important).

| method | idea |
| --- | --- |
| gae | Chefer et al. ICCV 2021: rollout of head-averaged positive (grad * attention), for the model's top-1 class |
| centre | isotropic Gaussian on the patch grid (sigma = grid / 4) |
| random | seeded uniform scores |
| attention | attention each patch receives from every query and head (R0 up to a positive factor) |
| topk | attention each patch receives when every layer reads only the shared top-k reading sets (`execution`) |

`topk` and `attention` are compared in the top-k map similarity check (appendix B).
"""

from contextlib import contextmanager

import torch

from .execution import base_pattern, capture_qk, selected_layers, token_mask
from .execution.execute import split_qkv

METHODS = ('gae', 'centre', 'random', 'attention', 'topk')


@contextmanager
def record_attention(model, layers=None):
    """Run selected layers with an explicit softmax and keep each layer's probabilities [B, H, N, N].

    The probabilities keep their autograd graph when the forward runs with gradients enabled (GAE).
    """
    selected = selected_layers(model, layers)
    probs, saved = {}, {}

    def make_forward(layer, attention):
        def forward(x, attn_mask=None, is_causal=False):
            if attn_mask is not None or is_causal:
                raise ValueError('Expected unmasked, noncausal self-attention')
            batch, tokens, _ = x.shape
            q, k, v = split_qkv(attention, x)
            p = ((q * attention.scale) @ k.transpose(-2, -1)).softmax(-1)
            probs[layer] = p
            out = (p @ v).transpose(1, 2).reshape(batch, tokens, -1)
            return attention.proj_drop(attention.proj(attention.norm(out)))
        return forward

    try:
        for layer in selected:
            attention = model.blocks[layer - 1].attn
            saved[layer] = attention.__dict__.get('forward')
            attention.forward = make_forward(layer, attention)
        yield probs
    finally:
        for layer, forward in saved.items():
            attention = model.blocks[layer - 1].attn
            if forward is None:
                del attention.forward
            else:
                attention.forward = forward


def _target_logits(model, images, target):
    logits = model(images)
    target = logits.argmax(-1) if target is None else target
    return logits.gather(1, target[:, None]).sum()


def attention_mass(model, images, layers):
    """Attention each patch receives from all queries and heads, unrestricted, summed over the layers."""
    with torch.no_grad(), record_attention(model, layers) as probs:
        model(images)
    return sum(probs[layer].sum((1, 2)) for layer in layers)[:, model.num_prefix_tokens:]


def gae(model, images, target=None):
    """Chefer et al. 2021 (generic attention-model explainability): R += mean_h[(dA * A)+] R, layer by layer."""
    with torch.enable_grad(), record_attention(model) as probs:
        images = images.detach().requires_grad_(True)
        score = _target_logits(model, images, target)
        layers = sorted(probs)
        grads = torch.autograd.grad(score, [probs[layer] for layer in layers])
    first = probs[layers[0]]
    relevance = torch.eye(first.shape[-1], device=first.device).expand(first.shape[0], -1, -1)
    for layer, grad in zip(layers, grads):
        cam = (grad * probs[layer].detach()).clamp_min(0).mean(1)
        relevance = relevance + cam @ relevance
    return relevance[:, 0, model.num_prefix_tokens:]


def topk_map(model, images, spec, layers):
    """Attention each patch receives (all heads and queries) when attention is restricted to the reading sets of
    `spec` (execution.resolve), summed over the given layers."""
    scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
    with torch.no_grad(), capture_qk(model, layers) as qk:
        model(images)
    total = 0
    for layer in layers:
        q, k = qk[layer]
        pattern, _ = base_pattern(spec, q, k, scale, grid)
        probs = ((q * scale) @ k.transpose(-2, -1)).masked_fill(~token_mask(pattern), -torch.inf).softmax(-1)
        total = total + probs.sum((1, 2))
    return total[:, model.num_prefix_tokens:]


def centre(batch, grid=14, sigma=None, device=None):
    """Isotropic Gaussian centred on the patch grid (sigma = grid / 4 patches by default)."""
    sigma = grid / 4 if sigma is None else sigma
    position = torch.arange(grid, device=device, dtype=torch.float32) - (grid - 1) / 2
    gauss = torch.exp(-(position[:, None] ** 2 + position[None, :] ** 2) / (2 * sigma ** 2))
    return gauss.flatten().expand(batch, -1)


def random_scores(batch, generator, n_patches=196, device=None):
    return torch.rand(batch, n_patches, generator=generator, device=device)


def compute(method, model, images, *, layers=None, generator=None, target=None):
    """Dispatch by name (gae, centre, random, attention). target: class indices for GAE (default: top-1)."""
    if method == 'gae':
        return gae(model, images, target)
    if method == 'centre':
        return centre(len(images), device=images.device)
    if method == 'random':
        return random_scores(len(images), generator, device=images.device)
    if method == 'attention':
        return attention_mass(model, images, layers)
    raise ValueError(f'Unknown method {method}; choose from {METHODS}')
