"""Audit downloaded RQ1 tables against image-level outputs and fixed inputs."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

from mita_probe.data import subset_indices
from mita_probe.rq1 import aggregate_predictions, settings

run = Path(sys.argv[1]).resolve()
config = yaml.safe_load((run / "config.yaml").read_text())
metadata = json.loads((run / "metadata.json").read_text())
checks = json.loads((run / "checks.json").read_text())
indices = json.loads((run / "indices.json").read_text())
predictions = pd.read_csv(run / "predictions.csv")
identity = pd.read_csv(run / "identity_predictions.csv")
results = pd.read_csv(run / "results.csv")
expected = set(settings(config))
audit = {
    "formal_size": config["n_images"] == 5000,
    "fixed_indices": indices == subset_indices(50000, 5000, 0),
    "indices_file_hash": hashlib.sha256((run / "indices.json").read_bytes()).hexdigest() == metadata["indices_sha256"],
    "identity_rows": len(identity) == 5000 and identity["index"].tolist() == indices,
    "identity_predictions": bool((identity.original == identity.identity).all()),
    "identity_record": checks["identity_agreement"] == 1 and checks["max_read_logit_error"] == 0 and checks["max_identity_logit_error"] < 1e-5,
    "all_settings": set(predictions.groupby(["method", "k", "seed"]).groups) == expected,
    "full_row_count": len(predictions) == 5000 * 35,
    "no_duplicate_rows": not predictions.duplicated(["method", "k", "seed", "index"]).any(),
    "finite_numeric_values": bool(np.isfinite(predictions.select_dtypes("number").to_numpy()).all()),
    "agreement_matches_predictions": bool((predictions.agree == (predictions.original == predictions.executed)).all()),
    "accuracy_matches_labels": bool(((predictions.original_correct == (predictions.original == predictions.target)) & (predictions.executed_correct == (predictions.executed == predictions.target))).all()),
    "source_commit_consistent": results.git_commit.nunique() == 1 and results.git_commit.iloc[0] == metadata["git_commit"],
    "declared_complete": checks["settings_completed"] == 35,
}
reference = identity.set_index("index")
for (method, k, seed), frame in predictions.groupby(["method", "k", "seed"]):
    label = f"{method}_{k}_{seed}"
    audit[f"indices_{label}"] = frame["index"].tolist() == indices
    aligned = reference.loc[frame["index"]]
    audit[f"original_{label}"] = bool(np.array_equal(frame.original.to_numpy(), aligned.original.to_numpy()) and np.array_equal(frame.target.to_numpy(), aligned.target.to_numpy()))
    result = results[(results.k == k) & (results.seed == seed) & (results.grouping == ("kmeans" if method == "kmeans" else "grid")) & (results.readset == ("random" if method == "random" else "ours"))]
    audit[f"table_{label}"] = len(result) == 1 and bool(np.isclose(result.iloc[0].agree, frame.agree.mean()) and np.isclose(result.iloc[0].kl, frame.kl.mean()))
summary, contrasts = aggregate_predictions(predictions, config)
for filename, recomputed in [("summary.csv", summary), ("contrasts.csv", contrasts)]:
    actual = pd.read_csv(run / filename)
    expected_frame = pd.DataFrame(recomputed)
    try:
        pd.testing.assert_frame_equal(actual, expected_frame, check_exact=False, rtol=1e-10, atol=1e-12)
        audit[f"recomputed_{filename}"] = True
    except AssertionError:
        audit[f"recomputed_{filename}"] = False
report = {
    "passed": all(audit.values()), "checks": audit,
    "prediction_rows": len(predictions), "settings": len(expected),
    "source_commit": metadata["git_commit"],
    "hash_scope": "Downloaded artifact bytes before Git line-ending normalization",
    "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.iterdir()) if p.is_file() and p.name != "audit.json"},
}
(run / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"passed": report["passed"], "checks": len(audit), "failed": [k for k, v in audit.items() if not v], "prediction_rows": len(predictions)}, indent=2))
if not report["passed"]:
    raise SystemExit(1)
