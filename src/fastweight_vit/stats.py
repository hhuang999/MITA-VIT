"""Image-level statistics for V2: Shapley split, sharing index and paired bootstrap."""

import numpy as np
import pandas as pd


def resample_indices(n, resamples=1000, seed=0):
    """Same draws as the v1 loop: rng.integers(n, size=n) repeated `resamples` times."""
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


def shapley(rand, rand_c, shared, mixed):
    """Two-player Shapley values of compression (C) and routing (R); sum = mixed - rand."""
    phi_c = .5 * ((rand_c - rand) + (mixed - shared))
    phi_r = .5 * ((shared - rand) + (mixed - rand_c))
    return phi_c, phi_r


def anatomy_table(table, draws, upper='L1_k32'):
    """table: columns rand, randC, L3, L4, L2, upper, ... (one row per image)."""
    phi_c, phi_r = shapley(table['rand'], table['randC'], table['L3'], table['L4'])
    out = {name: interval(table[name], draws) for name in table.columns}
    out['phi_C'] = interval(phi_c, draws)
    out['phi_R'] = interval(phi_r, draws)
    out['query_specific'] = interval(table[upper] - table['L4'], draws)
    out['sharing_index'] = ratio_interval(table['L4'] - table['rand'], table[upper] - table['rand'], draws)
    return pd.DataFrame(out, index=['mean', 'low', 'high']).T


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


def summarise(table, draws, **labels):
    """Rows of mean with bootstrap interval for each column of a per-image table (NaN-free columns)."""
    return [dict(labels, quantity=name, **dict(zip(('mean', 'low', 'high'), interval(table[name].to_numpy(), draws))))
            for name in table.columns]
