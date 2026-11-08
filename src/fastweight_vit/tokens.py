"""Token geometry shared by the readings and metrics: high-norm patches and distance-to-centre rings."""

import numpy as np
import torch


def high_norm_tokens(norms, ratio=3.):
    """[B, patches] bool: block-input norm above `ratio` times the image's median patch norm.

    The midpoint median is used, also for an even number of patches.
    """
    return norms > ratio * torch.quantile(norms, .5, dim=-1, keepdim=True)


def centre_rings(grid=14, prefix=1):
    """Ring id of every token by distance to the image centre; prefix tokens get -1."""
    rows, cols = np.divmod(np.arange(grid * grid), grid)
    distance = (rows - (grid - 1) / 2) ** 2 + (cols - (grid - 1) / 2) ** 2
    rings = np.unique(distance, return_inverse=True)[1]
    return torch.cat([torch.full((prefix,), -1), torch.from_numpy(rings)])
