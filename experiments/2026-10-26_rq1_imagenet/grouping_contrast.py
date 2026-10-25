"""Exploratory paired image bootstrap: K-means minus grid, no model rerun."""
import sys
from pathlib import Path

import pandas as pd
import yaml

from mita_probe.metrics import paired_bootstrap_difference

run = Path(sys.argv[1])
config = yaml.safe_load((run / "config.yaml").read_text())
data = pd.read_csv(run / "predictions.csv")
rows = []
for k in config["k"]:
    grid = data[(data.k == k) & (data.method == "grid")].set_index("index").sort_index()
    km = data[(data.k == k) & (data.method == "kmeans")].set_index("index").sort_index()
    if not grid.index.equals(km.index):
        raise ValueError("Unpaired images")
    delta, low, high = paired_bootstrap_difference(
        km.agree.to_numpy(), grid.agree.to_numpy()[None, :],
        seed=config["subset_seed"], n_resamples=config["bootstrap_resamples"],
        confidence=config["confidence_level"],
    )
    rows.append({"k": k, "contrast": "kmeans_minus_grid", "difference": delta,
                 "ci_low": low, "ci_high": high, "analysis": "exploratory; no multiplicity adjustment"})
frame = pd.DataFrame(rows)
frame.to_csv(run / "kmeans_vs_grid.csv", index=False)
print(frame.to_string(index=False))
