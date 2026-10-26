"""RQ1 settings and image-paired result aggregation."""

import numpy as np
import pandas as pd

from .metrics import paired_bootstrap_difference


def settings(config):
    for k in config["k"]:
        for method in config["methods"]:
            seeds = config["random_seeds"] if method == "random" else [0]
            for seed in seeds:
                yield method, k, seed


def mean_interval(values, config):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(config["subset_seed"])
    means = [values[rng.integers(len(values), size=len(values))].mean()
             for _ in range(config["bootstrap_resamples"])]
    tail = (1 - config["confidence_level"]) / 2
    return np.quantile(means, [tail, 1 - tail]).tolist()


def validate_completed_settings(predictions, results, identity, config):
    """Accept only whole, ordered, image-paired settings for a continuation run."""
    keys = ['method', 'k', 'seed']
    if predictions.empty or predictions.duplicated(keys + ['index']).any():
        raise ValueError('Resume requires nonempty, unique image-setting rows')
    expected = set(settings(config))
    completed = set()
    for (method, k, seed), frame in predictions.groupby(keys, sort=False):
        key = (method, int(k), int(seed))
        if key not in expected or len(frame) != config['n_images']:
            raise ValueError('Resume requires complete authorized settings')
        for field in ['index', 'target', 'original']:
            if not np.array_equal(frame[field], identity[field]):
                raise ValueError('Resume images, order, labels and original predictions must match')
        if not np.isfinite(frame.kl).all() or not np.array_equal(
                frame.agree, (frame.original == frame.executed).astype(int)):
            raise ValueError('Invalid saved prediction metrics')
        completed.add(key)
    saved = set()
    for row in results:
        method = 'random' if row['readset'] == 'random' else row['grouping']
        key = (method, int(row['k']), int(row['seed']))
        if key in saved:
            raise ValueError('Duplicate saved setting summary')
        saved.add(key)
    if saved != completed:
        raise ValueError('Saved summaries and predictions contain different settings')
    return completed


def aggregate_predictions(predictions: pd.DataFrame, config: dict):
    """Align by image ID; never treat random seeds as extra independent images."""
    summaries, contrasts = [], []
    for k in config["k"]:
        paired = {}
        for method in config["methods"]:
            frame = predictions.query("k == @k and method == @method")
            if frame.duplicated(["index", "seed"]).any():
                raise ValueError("Duplicate image/seed predictions")
            matrix = frame.pivot(index="index", columns="seed", values="agree").sort_index()
            expected = config["random_seeds"] if method == "random" else [0]
            if (set(matrix.columns) != set(expected) or matrix.isna().any().any()
                    or len(matrix) != config["n_images"]):
                raise ValueError("Missing images or seeds in predictions")
            paired[method] = matrix
            per_image = matrix.mean(axis=1).to_numpy()
            low, high = mean_interval(per_image, config)
            summaries.append({
                "method": method, "k": k, "n_images": len(matrix),
                "n_seeds": len(expected), "agree": per_image.mean(),
                "seed_std": matrix.mean(axis=0).std(ddof=1) if len(expected) > 1 else 0.,
                "agree_ci_low": low, "agree_ci_high": high,
                "kl": frame["kl"].mean(),
                "acc_original": frame["original_correct"].mean(),
                "acc_executed": frame["executed_correct"].mean(),
            })
        grid, random = paired["grid"], paired["random"]
        if not all(matrix.index.equals(grid.index) for matrix in paired.values()):
            raise ValueError("Methods must use the same image IDs")
        delta, low, high = paired_bootstrap_difference(
            grid[0].to_numpy(), random.to_numpy().T,
            seed=config["subset_seed"], n_resamples=config["bootstrap_resamples"],
            confidence=config["confidence_level"],
        )
        contrasts.append({"k": k, "contrast": "grid_minus_random", "difference": delta,
                          "ci_low": low, "ci_high": high})
    return summaries, contrasts
