import json
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[2]
local = root / "experiments/2026-10-25_hpc_local_smoke"
remote = root / "experiments/2026-10-26_hpc_first/smoke"
key = ["method", "k", "seed", "index"]
a = pd.read_csv(local / "predictions.csv").sort_values(key).reset_index(drop=True)
b = pd.read_csv(remote / "predictions.csv").sort_values(key).reset_index(drop=True)
report = {
    "same_settings_and_indices": a[key].equals(b[key]),
    "rows": len(b),
    "same_targets": a.target.equals(b.target),
    "same_original_predictions": a.original.equals(b.original),
    "same_executed_predictions": a.executed.equals(b.executed),
    "executed_prediction_differences": int((a.executed != b.executed).sum()),
    "max_abs_kl_difference": float((a.kl - b.kl).abs().max()),
    "by_method": {
        method: {
            "rows": int((a.method == method).sum()),
            "executed_prediction_differences": int(((a.executed != b.executed) & (a.method == method)).sum()),
            "max_abs_kl_difference": float((a.kl[a.method == method] - b.kl[b.method == method]).abs().max()),
        }
        for method in ["grid", "random", "kmeans"]
    },
}
(remote.parent / "local_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
