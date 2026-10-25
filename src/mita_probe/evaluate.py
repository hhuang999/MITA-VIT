"""Image-paired comparisons of saved execution results."""

import pandas as pd

from .metrics import paired_bootstrap_difference


def compare_runs(left: pd.DataFrame, right: pd.DataFrame, *, metric="agree",
                 seed=0, n_resamples=1000, confidence=.95):
    """Compare two single-setting runs after validating exact image pairing."""
    if left.empty or right.empty:
        raise ValueError("Cannot compare empty runs")
    if left["index"].duplicated().any() or right["index"].duplicated().any():
        raise ValueError("Select exactly one setting and seed per run")
    left, right = left.set_index("index").sort_index(), right.set_index("index").sort_index()
    if not left.index.equals(right.index):
        raise ValueError("Runs must contain the same image IDs")
    for field in ["target", "original"]:
        if field in left and field in right and not left[field].equals(right[field]):
            raise ValueError(f"Paired runs disagree on {field}")
    if left[metric].isna().any() or right[metric].isna().any():
        raise ValueError("Missing paired measurements")
    mean, low, high = paired_bootstrap_difference(
        left[metric].to_numpy(), right[metric].to_numpy()[None],
        seed=seed, n_resamples=n_resamples, confidence=confidence,
    )
    return {"difference": mean, "ci_low": low, "ci_high": high, "n_images": len(left)}
