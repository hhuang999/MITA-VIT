"""Faithfulness of patch-importance maps by removing patch tokens (RQ1 sufficiency/necessity, RQ4).

Removing a patch = no token may read it, in any layer. Removed tokens are still computed as queries but are
never read, so the CLS token and every kept token compute exactly what they would if the removed tokens
had been dropped from the sequence after the positional embedding (token dropping; Jain et al. ICLR 2022).
This is the same operation the probe uses, so the probe and every baseline are tested alike. CLS and
other prefix tokens are always kept.

Measures per image, against the original model's prediction: top-1 agreement, probability of the original
top-1 class, and KL(original || reduced).
"""

import torch

from .metrics import kl_divergence
from .probe import Plan, execute

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
    """Model output when every layer may read only the kept tokens."""
    heads = model.blocks[0].attn.num_heads
    batch, tokens = keep.shape
    mask = keep[:, None, None, :].expand(batch, heads, tokens, tokens)
    with execute(model, {layer: Plan(token_mask=mask) for layer in range(1, len(model.blocks) + 1)}):
        return model(images)


def measures(original, reduced):
    target = original.argmax(-1)
    return dict(agree=reduced.argmax(-1).eq(target).float(),
                p_target=reduced.softmax(-1).gather(1, target[:, None])[:, 0],
                kl=kl_divergence(original.double(), reduced.double()).float())


def curve(model, images, original, order, budgets, condition):
    """{budget: measures} for one condition; budgets count patches kept (keep_top) or removed (drop_*)."""
    prefix, patches = model.num_prefix_tokens, order.shape[1]
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
