"""Image-level statistics: paired bootstrap over images (1,000 resamples, seed 0, 95% percentile intervals)."""

import numpy as np


def resample_indices(n, resamples=1000, seed=0):
    """Bootstrap draws: rng.integers(n, size=n) repeated `resamples` times."""
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(n, size=n) for _ in range(resamples)])


def interval(values, draws, confidence=.95):
    values = np.asarray(values, dtype=float)
    means = values[draws].mean(1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(values.mean()), float(low), float(high)


def ratio_interval(numerator, denominator, draws, confidence=.95):
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    ratios = numerator[draws].mean(1) / denominator[draws].mean(1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(ratios, [tail, 1 - tail])
    return float(numerator.mean() / denominator.mean()), float(low), float(high)


def trapezoid(x, y):
    """Area under each row of y [rows, points] over the shared x [points]."""
    y = np.asarray(y, float)
    return ((y[:, 1:] + y[:, :-1]) / 2 * np.diff(np.asarray(x, float))).sum(1)


def paired(table, base, others, draws, **labels):
    """Rows of mean (base - other) with bootstrap intervals; table: one row per image, one column per method."""
    rows = []
    for other in others:
        diff = (table[base] - table[other]).to_numpy()
        mean, low, high = interval(diff, draws)
        rows.append(dict(labels, base=base, other=other, difference=mean, low=low, high=high,
                         verdict='higher' if low > 0 else 'lower' if high < 0 else 'uncertain'))
    return rows

