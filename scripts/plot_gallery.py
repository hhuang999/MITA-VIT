"""Qualitative figures from gallery_<dataset>.npz (decision B16: random images, shown in index order).

methods: one row per image (the first --rows images), columns = input, ground truth, then each method's map.
layers: one row per image (the first --rows images), columns = input, then the probe at layers 1-12.
A map is shown as a spotlight: each patch's score is replaced by its rank within the image (so methods with
different scales are comparable), cubed (so the top quarter of patches stands out), upsampled bilinearly, and
the image is darkened where the result is low.

Usage: python scripts/plot_gallery.py methods RUN/gallery_imagenet_s.npz --output figures/figure12_gallery
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from fastweight_vit.plotting import METHOD_LABELS, save

METHODS = ['probe_mass', 'cls_attention', 'rollout', 'gae', 'gradcam', 'integrated_gradients']
PART_COLORS = np.array([[42, 120, 214], [138, 137, 134], [235, 104, 52], [27, 175, 122], [237, 161, 0]]) / 255


def spotlight(rgb, scores):
    """rgb uint8 [224, 224, 3]; scores [196] -> float image."""
    ranks = (scores.argsort().argsort().astype(np.float32) / (len(scores) - 1)) ** 3
    weight = F.interpolate(torch.from_numpy(ranks).view(1, 1, 14, 14), size=(224, 224), mode='bilinear',
                           align_corners=False)[0, 0].numpy()
    return rgb / 255 * (0.08 + 0.92 * weight[..., None])


def truth_image(rgb, truth):
    """Object patches [196], a human map [224, 224] or part ids [224, 224] drawn over a pale image."""
    pale = 0.35 + 0.65 * rgb / 255 * 0.5
    if truth.shape == (196,):
        mask = np.kron(truth.reshape(14, 14), np.ones((16, 16)))
        return pale * (1 - 0.6 * mask[..., None]) + 0.6 * mask[..., None] * PART_COLORS[0]
    if truth.dtype.kind == 'f':
        weight = (truth - truth.min()) / max(truth.max() - truth.min(), 1e-12)
        return rgb / 255 * (0.15 + 0.85 * weight[..., None])
    out = pale.copy()
    for rank, part in enumerate(sorted(set(np.unique(truth)) - {40})):
        out[truth == part] = PART_COLORS[rank % len(PART_COLORS)]
    return out


def methods_figure(data, rows, output):
    columns = ['input', 'truth'] + [m for m in METHODS if m in data]
    fig, axes = plt.subplots(rows, len(columns), figsize=(1.45 * len(columns), 1.5 * rows), squeeze=False)
    for r in range(rows):
        rgb = data['rgb'][r]
        for c, column in enumerate(columns):
            ax = axes[r, c]
            image = rgb / 255 if column == 'input' else truth_image(rgb, data['truth'][r]) if column == 'truth' \
                else spotlight(rgb, data[column][r].astype(np.float32))
            ax.imshow(np.clip(image, 0, 1))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title({'input': 'Input', 'truth': 'Ground truth'}.get(column, METHOD_LABELS.get(column, column)),
                             fontsize=7)
    fig.tight_layout(pad=0.3)
    save(fig, output, dpi=110)  # photo grids: smaller PNG; the PDF keeps full resolution


def layers_figure(data, rows, output):
    fig, axes = plt.subplots(rows, 13, figsize=(13 * 1.1, 1.15 * rows), squeeze=False)
    for r in range(rows):
        rgb = data['rgb'][r]
        axes[r, 0].imshow(rgb)
        for layer in range(12):
            axes[r, layer + 1].imshow(np.clip(spotlight(rgb, data['probe_layers'][r, layer].astype(np.float32)), 0, 1))
        for c in range(13):
            axes[r, c].set_xticks([]), axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title('Input' if c == 0 else f'Layer {c}', fontsize=7)
    fig.tight_layout(pad=0.3)
    save(fig, output, dpi=110)  # photo grids: smaller PNG; the PDF keeps full resolution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', choices=['methods', 'layers'])
    parser.add_argument('gallery')
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data = dict(np.load(args.gallery))
    rows = min(args.rows, len(data['index']))
    (methods_figure if args.figure == 'methods' else layers_figure)(data, rows, args.output)


if __name__ == '__main__':
    main()
