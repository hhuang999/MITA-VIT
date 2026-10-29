"""Patch-importance maps for RQ3/RQ4: our probe and the baselines, all as one score per patch token.

Every method returns a float tensor [batch, 196] (larger = more important) for the original model's
top-1 class, so all of them can be ranked, thresholded and tested the same way:

| method | needs gradients | idea |
| --- | --- | --- |
| probe | no | attention each patch-neuron receives when the model reads only through the probe's sets |
| cls_attention | no | CLS-to-patch attention of the layers the classifier reads, averaged over heads |
| rollout | no | Attention Rollout (Abnar & Zuidema 2020): 0.5 A + 0.5 I multiplied through the layers |
| gae | yes | Chefer et al. ICCV 2021: rollout of head-averaged positive (grad * attention) |
| gradcam | yes | Grad-CAM on the last block's normalised tokens (as in pytorch-grad-cam for ViTs) |
| integrated_gradients | yes | Integrated Gradients from a zero (mean) image, summed |attribution| per patch |
| centre / random / mask | no | references: centre Gaussian, seeded random scores, the ground-truth object mask |

For DINO the classifier reads the CLS token of the last four blocks, so the CLS-based maps average over
those blocks; gradient methods follow the head automatically.
"""

from contextlib import contextmanager

import torch

from .models.dino import N_LAST_BLOCKS, DinoLinear
from .probe import base_pattern, capture_qk, selected_layers, token_mask
from .probe.execute import split_qkv

METHODS = ('probe', 'cls_attention', 'rollout', 'gae', 'gradcam', 'integrated_gradients', 'centre', 'random',
           'mask')


def head_layers(model):
    """One-based layers whose CLS token the classifier reads."""
    depth = len(model.blocks)
    return tuple(range(depth - N_LAST_BLOCKS + 1, depth + 1)) if isinstance(model, DinoLinear) else (depth,)


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


def cls_attention(model, images, layers=None):
    layers = layers or head_layers(model)
    with torch.no_grad(), record_attention(model, layers) as probs:
        model(images)
    prefix = model.num_prefix_tokens
    return torch.stack([probs[layer][:, :, 0, prefix:].mean(1) for layer in layers]).mean(0)


def _rollout_rows(probs, prefix, rows_at):
    """Roll 0.5 A + 0.5 I through the layers; return the CLS row at each requested layer, averaged."""
    first = probs[min(probs)]
    eye = torch.eye(first.shape[-1], device=first.device, dtype=first.dtype)
    relevance, rows = eye.expand(first.shape[0], -1, -1), []
    for layer in sorted(probs):
        a = 0.5 * probs[layer].mean(1) + 0.5 * eye
        relevance = (a / a.sum(-1, keepdim=True)) @ relevance
        if layer in rows_at:
            rows.append(relevance[:, 0, prefix:])
    return torch.stack(rows).mean(0)


def rollout(model, images):
    with torch.no_grad(), record_attention(model) as probs:
        model(images)
    return _rollout_rows(probs, model.num_prefix_tokens, head_layers(model))


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


def gradcam(model, images, target=None):
    store = {}

    def hook(module, inputs, output):
        store['tokens'] = output

    handle = model.blocks[-1].norm1.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            images = images.detach().requires_grad_(True)
            grad, = torch.autograd.grad(_target_logits(model, images, target), store['tokens'])
    finally:
        handle.remove()
    prefix = model.num_prefix_tokens
    tokens, grads = store['tokens'][:, prefix:].detach(), grad[:, prefix:]
    return (tokens * grads.mean(1, keepdim=True)).sum(-1).clamp_min(0)


def integrated_gradients(model, images, target=None, steps=20, patch=16):
    """|attribution| summed per patch; the baseline is the all-zero normalised image (the dataset mean)."""
    with torch.no_grad():
        target = model(images).argmax(-1) if target is None else target
    total = torch.zeros_like(images)
    for step in range(1, steps + 1):
        with torch.enable_grad():
            point = (images * step / steps).detach().requires_grad_(True)
            grad, = torch.autograd.grad(_target_logits(model, point, target), point)
        total += grad
    attribution = (images * total / steps).abs().sum(1)
    batch, height, width = attribution.shape
    return attribution.reshape(batch, height // patch, patch, width // patch, patch).sum((2, 4)).flatten(1)


def probe(model, images, spec, layers, weight='mass'):
    """Importance of each patch-neuron under the probe's reading sets, summed over the given layers.

    weight='mass': attention it receives (all heads and queries) when attention is restricted to the sets;
    weight='count': how many sets (heads x regions, plus CLS sets) contain it;
    weight='value': 'mass' times the norm of its value vector in each head (what it writes, not only what
    it is read with).
    """
    scale, grid = model.blocks[0].attn.scale, model.patch_embed.grid_size
    values = {}
    if weight == 'value':
        hooks = [model.blocks[layer - 1].attn.qkv.register_forward_hook(
            lambda module, inputs, output, layer=layer: values.__setitem__(layer, output)) for layer in layers]
    try:
        with torch.no_grad(), capture_qk(model, layers) as qk:
            model(images)
    finally:
        if weight == 'value':
            for hook in hooks:
                hook.remove()
    total = 0
    for layer in layers:
        q, k = qk[layer]
        pattern, _ = base_pattern(spec, q, k, scale, grid)
        if weight == 'count':
            counts = torch.zeros(q.shape[0], q.shape[2], device=q.device)
            sets = pattern.readsets.flatten(1)
            if pattern.cls_set is not None:
                sets = torch.cat([sets, pattern.cls_set.flatten(1)], 1)
            total = total + counts.scatter_add_(1, sets, torch.ones_like(sets, dtype=counts.dtype))
            continue
        probs = ((q * scale) @ k.transpose(-2, -1)).masked_fill(~token_mask(pattern), -torch.inf).softmax(-1)
        if weight == 'value':
            attention = model.blocks[layer - 1].attn
            output = values[layer]
            v = output.reshape(*output.shape[:2], 3, attention.num_heads, attention.head_dim)[:, :, 2]
            probs = probs * v.norm(dim=-1).transpose(1, 2)[:, :, None, :]
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


def compute(method, model, images, *, spec=None, layers=None, weight='mass', foreground=None, generator=None,
            target=None):
    """Dispatch by name; see the table in the module docstring. target: class indices for the gradient
    methods (default: the model's own top-1)."""
    if method == 'probe':
        return probe(model, images, spec, layers, weight)
    if method == 'cls_attention':
        return cls_attention(model, images, layers)
    if method == 'rollout':
        return rollout(model, images)
    if method == 'gae':
        return gae(model, images, target)
    if method == 'gradcam':
        return gradcam(model, images, target)
    if method == 'integrated_gradients':
        return integrated_gradients(model, images, target)
    if method == 'centre':
        return centre(len(images), device=images.device)
    if method == 'random':
        return random_scores(len(images), generator, device=images.device)
    if method == 'mask':
        return foreground.float()
    raise ValueError(f'Unknown method {method}; choose from {METHODS}')
