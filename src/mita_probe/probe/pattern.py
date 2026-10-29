"""Read a probe pattern from captured Q/K: which tokens (neurons) each query may read.

Viewed as a fast-weight MLP, one attention head on one image has a hidden neuron per token
(key = input weight, value = output weight). A pattern keeps, for each image region, the
top-k neurons its landmark (mean region query) scores highest, and routes every query to one
region. Discrete choices are read from the original forward and then frozen.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

M_SIDES = {1: 1, 4: 2, 16: 4, 25: 5, 49: 7, 196: 14}


@dataclass(frozen=True)
class Pattern:
    readsets: torch.Tensor  # [batch, heads, groups, k]; groups = regions or tokens (per query)
    routes: torch.Tensor  # [batch, heads, tokens] group read by each query
    cls_set: torch.Tensor | None = None  # [batch, heads, k]; CLS reads its own set when given


def grid_landmarks(patches, grid_size, output_size=(4, 4)):
    """Adaptive 2D average pooling of per-token vectors [batch, heads, tokens, dim] on the patch grid."""
    batch, heads, _, dim = patches.shape
    grid = patches.reshape(batch * heads, *grid_size, dim).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(grid, output_size)
    return pooled.flatten(2).transpose(1, 2).reshape(batch, heads, -1, dim)


def kmeans_landmarks(patches, initial, iterations=10):
    """Lloyd's Euclidean k-means from grid centres (Clustered Attention grouping); keep empty clusters' centres."""
    centers = initial.clone()
    for _ in range(iterations):
        distances = (patches.square().sum(-1, keepdim=True) + centers.square().sum(-1).unsqueeze(-2)
                     - 2 * patches @ centers.transpose(-2, -1))
        membership = F.one_hot(distances.argmin(-1), centers.shape[-2]).to(patches.dtype)
        counts = membership.sum(-2).unsqueeze(-1)
        means = membership.transpose(-2, -1) @ patches / counts.clamp_min(1)
        centers = torch.where(counts > 0, means, centers)
    return centers


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
    if mode == 'kmeans':
        patches = q[:, :, prefix:]
        return kmeans_landmarks(patches, grid_landmarks(patches, grid, (m_side, m_side)))
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
    cls_set = pattern.cls_set
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
