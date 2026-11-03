"""Which patches a "was the budget used?" test removes (decision B29), per layer and image.

All functions take and return [B, patches] bool masks (no prefix tokens) for one layer.
- C1 `top_ordinary`: as many ordinary (not high-norm) patches as there are high-norm ones, those receiving
  the most budget at that layer: the strictest same-count control.
- C2 `ring_random`: the same number per same-distance ring, drawn uniformly from that ring's ordinary
  patches (from all ordinary patches when a ring has too few; counted in `short`).
- `equal_mass`: the fewest top ordinary patches whose budget reaches the high-norm patches' budget;
  `feasible` is False when all ordinary patches together receive less.
"""

import torch

from .interventions import centre_rings


def top_ordinary(received, high):
    """received: [B, P] budget per patch at this layer; high: [B, P] bool."""
    count = high.sum(1)
    order = received.masked_fill(high, -torch.inf).argsort(dim=1, descending=True, stable=True)
    ranks = torch.arange(received.shape[1], device=received.device).expand_as(order)
    return torch.zeros_like(high).scatter_(1, order, ranks < count[:, None])


def ring_random(high, generator, grid=14):
    """Returns (mask, short): short[b] = how many draws fell back to other rings. Drawn on the CPU."""
    device, high = high.device, high.cpu()
    rings = centre_rings(grid)[1:]
    out = torch.zeros_like(high)
    short = torch.zeros(high.shape[0], dtype=torch.long, device=high.device)
    for b in range(high.shape[0]):
        need_elsewhere = 0
        for ring in rings.unique().tolist():
            members = (rings == ring)
            need = int((high[b] & members).sum())
            if not need:
                continue
            pool = (members & ~high[b] & ~out[b]).nonzero().flatten()
            take = min(need, len(pool))
            if take:
                pick = pool[torch.randperm(len(pool), generator=generator)[:take]]
                out[b, pick] = True
            need_elsewhere += need - take
        if need_elsewhere:
            pool = (~high[b] & ~out[b]).nonzero().flatten()
            pick = pool[torch.randperm(len(pool), generator=generator)[:need_elsewhere]]
            out[b, pick] = True
            short[b] = need_elsewhere
    return out.to(device), short.to(device)


def equal_mass(received, high):
    """Returns (mask, feasible)."""
    target = (received * high).sum(1)
    ordinary = received.masked_fill(high, -torch.inf)
    values, order = ordinary.sort(dim=1, descending=True, stable=True)
    values = values.clamp_min(0).nan_to_num(neginf=0.)
    before = values.cumsum(1) - values  # budget of the patches ranked above
    take = (before < target[:, None]) & (target[:, None] > 0) & torch.isfinite(ordinary.gather(1, order))
    feasible = (values.sum(1) >= target) | (target == 0)
    return torch.zeros_like(high).scatter_(1, order, take), feasible
