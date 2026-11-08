"""Same-distance-ring enrichment: does a patch map point at the object beyond what the image centre explains?

Per image, given scores [196], the object mask [196] and valid patches [196] (boundary-only patches are
excluded everywhere):

| measure | meaning |
| --- | --- |
| mass | share of the (non-negative) score mass on object patches |
| pointing | the single highest-scoring patch is on the object |
| auroc | probability that a random object patch outscores a random background patch |
| auroc_shuffled | the same scores against another image's mask; a pure centre prior scores equally on both |
| precision | share of the top-|object| patches that are object (|object| = number of object patches) |
| enrichment | object rate among the top-|object| patches minus the object rate of their distance ring, averaged
               over rings with weight = selected patches (comparison at equal distance from the centre) |

The paper reports `enrichment`; the other measures are written next to it (same pass, same images). Image
geometry: object area share and object-centroid distance from the grid centre.
"""

import numpy as np
import torch

from .tokens import centre_rings

FIELDS = ['mass', 'pointing', 'auroc', 'auroc_shuffled', 'precision', 'enrichment', 'object_area', 'object_distance']


def auroc(scores, positive, valid):
    """Mann-Whitney AUROC per image over valid patches; NaN when either class is empty."""
    out = torch.full((scores.shape[0],), float('nan'), dtype=torch.float64)
    for n in range(scores.shape[0]):
        s = scores[n][valid[n]].double()
        p = positive[n][valid[n]]
        pos, neg = s[p], s[~p]
        if len(pos) and len(neg):
            greater = (pos[:, None] > neg[None, :]).double().mean()
            ties = (pos[:, None] == neg[None, :]).double().mean()
            out[n] = greater + 0.5 * ties
    return out


def geometry(foreground, valid, grid=14):
    """Object area share among valid patches and object-centroid distance (in patches) from the centre."""
    rows, cols = np.divmod(np.arange(grid * grid), grid)
    fg = (foreground & valid).cpu().numpy()
    area = fg.sum(1) / np.maximum(valid.cpu().numpy().sum(1), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (fg * rows).sum(1) / fg.sum(1)
        c = (fg * cols).sum(1) / fg.sum(1)
    return area, np.hypot(r - (grid - 1) / 2, c - (grid - 1) / 2)


def measures(scores, foreground, valid, other_foreground, generator=None):
    """Dict of per-image numpy arrays; other_foreground is another image's mask (for auroc_shuffled)."""
    scores, foreground, valid = scores.detach().float().cpu(), foreground.cpu(), valid.cpu()
    other_foreground = other_foreground.cpu()
    fg = foreground & valid
    positive = scores.clamp_min(0) * valid
    mass = (positive * fg).sum(1) / positive.sum(1)
    peak = scores.masked_fill(~valid, -torch.inf).argmax(1)
    pointing = fg.gather(1, peak[:, None])[:, 0].float()
    n_fg = fg.sum(1)
    noise = torch.rand(scores.shape, generator=generator) * 1e-6 * (scores.abs().amax(1, keepdim=True) + 1e-12)
    order = (scores + noise).masked_fill(~valid, -torch.inf).argsort(1, descending=True)
    selected = torch.zeros_like(fg)
    for n in range(scores.shape[0]):
        selected[n, order[n, :int(n_fg[n])]] = True
    precision = (selected & fg).sum(1) / n_fg.clamp_min(1)
    rings = centre_rings()[1:]
    enrichment = torch.zeros(scores.shape[0], dtype=torch.float64)
    for n in range(scores.shape[0]):
        weights, total = 0, 0.
        for ring in rings.unique().tolist():
            members = (rings == ring) & valid[n]
            chosen = members & selected[n]
            if chosen.any():
                rate = fg[n][chosen].double().mean() - fg[n][members].double().mean()
                total += float(rate) * int(chosen.sum())
                weights += int(chosen.sum())
        enrichment[n] = total / weights if weights else float('nan')
    usable = (n_fg > 0) & ((valid & ~fg).sum(1) > 0)
    nan = torch.tensor(float('nan'))
    area, distance = geometry(foreground, valid)
    return dict(
        mass=torch.where(usable & (positive.sum(1) > 0), mass, nan).numpy(),
        pointing=torch.where(usable, pointing, nan).numpy(),
        auroc=auroc(scores, fg, valid).numpy(),
        auroc_shuffled=auroc(scores, other_foreground & valid, valid).numpy(),
        precision=torch.where(usable, precision, nan).numpy(),
        enrichment=enrichment.numpy(),
        object_area=area, object_distance=distance)
