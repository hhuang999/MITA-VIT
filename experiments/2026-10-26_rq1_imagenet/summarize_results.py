import json
import sys
from pathlib import Path

import pandas as pd

run = Path(sys.argv[1])
summary = pd.read_csv(run / "summary.csv")
results = pd.read_csv(run / "results.csv")
contrasts = pd.read_csv(run / "contrasts.csv")
print("| k | m×k | Grid | Random mean ± SD | K-means |")
print("| --- | ---: | ---: | ---: | ---: |")
for k, frame in summary.groupby("k"):
    frame = frame.set_index("method")
    g, r, km = (frame.loc[name] for name in ["grid", "random", "kmeans"])
    print(f"| {k} | {16*k} | {100*g.agree:.2f}% | {100*r.agree:.2f}% ± {100*r.seed_std:.2f} | {100*km.agree:.2f}% |")
print("\nIntervals and auxiliary metrics:")
print(summary.to_string(index=False))
print("\nPaired contrasts:")
print(contrasts.to_string(index=False))
print("\nRandom seed ranges:")
print(results[results.readset == "random"].groupby("k").agree.agg(["min", "max"]).to_string())
print("\nChecks:")
print((run / "checks.json").read_text())
print("\nFirst tested k meeting the 90% reference:")
for method, frame in summary.groupby("method"):
    selected = frame[frame.agree >= .9]
    print(method, int(selected.k.min()) if len(selected) else "none")
