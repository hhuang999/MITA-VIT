"""Faithfulness of patch-importance maps by removing patch tokens (RQ1 sufficiency/necessity, RQ4).

Removing a patch = no token may read it, in any layer. This equals dropping it from the sequence after the
positional embedding (token dropping; Jain et al. ICLR 2022), which is how it is computed; the test suite
checks the equality against masking the patch out of every layer's attention, the operation the probe uses.
The probe and every baseline are tested alike. CLS and other prefix tokens are always kept.

Measures per image, against the original model's prediction: top-1 agreement, probability of the original
top-1 class, and KL(original || reduced).
"""

import torch

from .metrics import kl_divergence
from .models.dino import N_LAST_BLOCKS, DinoLinear

CONDITIONS = ('keep_top', 'drop_top', 'drop_bottom')  # insertion (MoRF), deletion MoRF, deletion LeRF


def rank(scores, generator):
    """Patch order by descending score; equal scores are ordered by a seeded random permutation."""
    noise = torch.rand(scores.shape, generator=generator, device=scores.device)
    shuffled = noise.argsort(-1)
    order = scores.gather(-1, shuffled).argsort(dim=-1, descending=True, stable=True)
    return shuffled.gather(-1, order)


def keep_mask(order, counts, prefix=1, top=True):
    """[batch, prefix + patches] bool: prefix tokens plus the first (top=True) or last `counts` ranked patches."""
    batch, patches = order.shape
    counts = torch.as_tensor(counts, device=order.device).expand(batch)
    position = torch.arange(patches, device=order.device).expand(batch, -1)
    chosen = position < counts[:, None] if top else position >= patches - counts[:, None]
    keep = torch.zeros(batch, patches, dtype=torch.bool, device=order.device).scatter_(1, order, chosen)
    return torch.cat([torch.ones(batch, prefix, dtype=torch.bool, device=order.device), keep], 1)


def run_kept(model, images, keep):
    """Model output when every layer may read only the kept tokens.

    Computed by dropping the other tokens after the positional embedding, which equals masking them out of
    every layer's attention (tests/test_explain.py) and runs the fused attention on a shorter sequence.
    Rows are grouped by how many tokens they keep.
    """
    counts = keep.sum(1)
    groups = [(counts == count).nonzero()[:, 0] for count in counts.unique().tolist()]
    logits = torch.cat([_dropped_forward(model, images[rows], keep[rows]) for rows in groups])
    return logits[torch.cat(groups).argsort()]


def _dropped_forward(model, images, keep):
    backbone = getattr(model, 'backbone', model)
    x = backbone.norm_pre(backbone.patch_drop(backbone._pos_embed(backbone.patch_embed(images))))
    x = x[keep].reshape(len(images), -1, x.shape[-1])
    if not isinstance(model, DinoLinear):
        return backbone.forward_head(backbone.norm(backbone.blocks(x)))
    features = []
    for i, block in enumerate(backbone.blocks):
        x = block(x)
        if i >= len(backbone.blocks) - N_LAST_BLOCKS:
            features.append(backbone.norm(x)[:, 0])
    return model.linear(torch.cat(features, -1))


def measures(original, reduced):
    target = original.argmax(-1)
    return dict(agree=reduced.argmax(-1).eq(target).float(),
                p_target=reduced.softmax(-1).gather(1, target[:, None])[:, 0],
                kl=kl_divergence(original.double(), reduced.double()).float())


def curve(model, images, original, order, budgets, condition, prefix=None):
    """{budget: measures} for one condition; budgets count patches kept (keep_top) or removed (drop_*).

    prefix: tokens always kept before the patches (default model.num_prefix_tokens; 2 with a test-time register,
    which is inserted at norm_pre, i.e. before the tokens are dropped).
    """
    prefix = model.num_prefix_tokens if prefix is None else prefix
    patches = order.shape[1]
    out = {}
    for budget in budgets:
        if condition == 'keep_top':
            keep = keep_mask(order, budget, prefix, top=True)
        elif condition == 'drop_top':
            keep = keep_mask(order, patches - budget, prefix, top=False)
        elif condition == 'drop_bottom':
            keep = keep_mask(order, patches - budget, prefix, top=True)
        else:
            raise ValueError(f'Unknown condition {condition}')
        out[budget] = measures(original, run_kept(model, images, keep))
    return out


def saco(model, images, original, scores, order, groups=10):
    """Salience-guided faithfulness coefficient (Wu et al., CVPR 2024), one value per image in [-1, 1].

    Patches are split into `groups` equal rank groups; removing group i alone changes p(original top-1) by
    d_i. For every pair i < j (group i ranked higher, s_i = summed score >= s_j), the pair adds +(s_i - s_j)
    when d_i >= d_j and -(s_i - s_j) otherwise; the total is divided by the summed weights.
    """
    prefix, patches = model.num_prefix_tokens, order.shape[1]
    target = original.argmax(-1)
    p_full = original.softmax(-1).gather(1, target[:, None])[:, 0]
    bounds = torch.linspace(0, patches, groups + 1).round().long().tolist()
    drops, sums = [], []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        members = order[:, start:stop]
        keep = torch.ones(order.shape[0], patches, dtype=torch.bool, device=order.device)
        keep.scatter_(1, members, False)
        keep = torch.cat([torch.ones(order.shape[0], prefix, dtype=torch.bool, device=order.device), keep], 1)
        reduced = run_kept(model, images, keep)
        drops.append(p_full - reduced.softmax(-1).gather(1, target[:, None])[:, 0])
        sums.append(scores.gather(1, members).sum(1))
    drops, sums = torch.stack(drops, 1), torch.stack(sums, 1)
    total = weights = torch.zeros(order.shape[0], device=order.device)
    for i in range(groups):
        for j in range(i + 1, groups):
            weight = sums[:, i] - sums[:, j]
            total = total + torch.where(drops[:, i] >= drops[:, j], weight, -weight)
            weights = weights + weight.abs()
    return torch.where(weights > 0, total / weights.clamp_min(1e-12), torch.zeros_like(total))
