"""Held-out ImageNet validation images for the v5 lens (decision B36): disjoint from both evaluation sets.

The mean maps (C1) and position probes (C2) are estimated on images that are evaluated nowhere. Both
evaluation sets come from the ImageNet validation split: ImageNet-S919 validation (12,419 images, matched by
file name) and the ClickMe validation set (17,408 images, stored without file names). Candidates are drawn
in a seeded order from the ImageNet validation images that are not ImageNet-S images; a candidate is
dropped when it is a near-duplicate of a ClickMe image of the same class: the Pearson correlation of 32 x 32
grey thumbnails, under any of the candidate geometries below, reaches `threshold`.

| geometry | how the ImageNet image is turned into a square before the thumbnail |
| --- | --- |
| squash | the whole image resized to a square |
| crop | resize the short side, centre crop a square (ClickMe's crop_pct is unknown; the full short side) |
"""

import numpy as np
from PIL import Image

GEOMETRIES = ('squash', 'crop')


def thumbnail(image, geometry='squash', size=32):
    """RGB PIL image -> z-normalised grey [size * size] float32 vector."""
    image = image.convert('L')
    if geometry == 'crop':
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    elif geometry != 'squash':
        raise ValueError(f'Unknown geometry {geometry}')
    x = np.asarray(image.resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    return (x - x.mean()) / (x.std() + 1e-6)


def best_match(candidate, references):
    """candidate: {geometry: [d]}; references: [n, d] -> (best correlation, its geometry)."""
    if len(references) == 0:
        return -1., None
    best, where = -1., None
    for geometry, vector in candidate.items():
        value = float((references @ vector).max() / len(vector))
        if value > best:
            best, where = value, geometry
    return best, where


def select(order, names, labels, excluded_names, clickme_thumbnails, image_of, n, threshold=.9):
    """Walk `order` (row indices) and keep the first n rows that are neither excluded nor ClickMe duplicates.

    labels: row -> class; clickme_thumbnails: {class: [m, d]}; image_of(row) -> PIL image.
    Returns (kept rows, dict of counts and the correlations seen).
    """
    kept, duplicates, correlations = [], [], []
    skipped = 0
    for row in order:
        if names[row] in excluded_names:
            skipped += 1
            continue
        image = image_of(row)
        candidate = {geometry: thumbnail(image, geometry) for geometry in GEOMETRIES}
        value, geometry = best_match(candidate, clickme_thumbnails.get(labels[row], np.zeros((0, 1))))
        correlations.append(value)
        if value >= threshold:
            duplicates.append(dict(row=int(row), name=names[row], correlation=value, geometry=geometry))
            continue
        kept.append(int(row))
        if len(kept) == n:
            break
    if len(kept) < n:
        raise ValueError(f'Only {len(kept)} held-out images available, {n} requested')
    return kept, dict(examined=len(correlations), skipped_imagenet_s=skipped, clickme_duplicates=duplicates,
                      correlation_quantiles={q: float(np.quantile(correlations, q))
                                             for q in (.5, .9, .99, 1.)} if correlations else {})


def square(image, geometry):
    """The candidate geometry applied to a PIL image (no resizing)."""
    if geometry == 'squash':
        return image
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def clickme_like(image, geometry, size=256):
    """An ImageNet image brought to the ClickMe storage format: the matched geometry, then size x size RGB."""
    return square(image.convert('RGB'), geometry).resize((size, size), Image.BILINEAR)
