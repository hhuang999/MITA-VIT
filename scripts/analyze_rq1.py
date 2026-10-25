"""Merge the approved extension and recompute paired RQ1 statistics."""

import argparse
from pathlib import Path

from mita_probe.runtime import configure_runtime

ROOT = configure_runtime()

import numpy as np
import pandas as pd
import yaml

from mita_probe.evaluate import compare_runs
from mita_probe.io import load_config, write_csv, write_json
from mita_probe.rq1 import aggregate_predictions, mean_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("extension", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = load_config(ROOT / "configs/rq1.yaml") | {"k": [4, 8, 16, 32, 64, 128]}
    (args.output / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    predictions = pd.concat([pd.read_csv(path / "predictions.csv")
                             for path in [args.original, args.extension]], ignore_index=True)
    identity = pd.read_csv(args.original / "identity_predictions.csv").sort_values("index")
    extra_identity = pd.read_csv(args.extension / "identity_predictions.csv").sort_values("index")
    pd.testing.assert_frame_equal(identity.reset_index(drop=True), extra_identity.reset_index(drop=True))
    summary, seed_mean_contrasts = aggregate_predictions(predictions, config)
    contrasts = []
    for k in config["k"]:
        grid = predictions.query("method == 'grid' and k == @k and seed == 0")
        for other in ["kmeans", "random"]:
            control = predictions.query("method == @other and k == @k and seed == 0")
            contrasts.append({"k": k, "contrast": f"grid_minus_{other}_seed0",
                              **compare_runs(grid, control)})
    for row in summary:
        part = predictions[
            (predictions.method == row["method"]) & (predictions.k == row["k"])]
        kl = part.groupby("index").kl.mean().to_numpy()
        row["kl_ci_low"], row["kl_ci_high"] = mean_interval(kl, config)
    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "paired_contrasts.csv", contrasts)
    write_csv(args.output / "seed_mean_contrasts.csv", seed_mean_contrasts)
    pd.concat([pd.read_csv(p / "results.csv") for p in [args.original, args.extension]]).to_csv(
        args.output / "results.csv", index=False)
    histogram = pd.read_csv(args.extension / "group_query_counts.csv")
    counts = []
    for (layer, kind), frame in histogram.groupby(["layer", "kind"]):
        values = np.repeat(frame.query_count.to_numpy(), frame.frequency.to_numpy())
        counts.append({"layer": layer, "kind": kind, "minimum": values.min(),
                       "median": float(np.median(values)), "maximum": values.max(),
                       "mean": values.mean(), "empty_fraction": float((values == 0).mean()),
                       "n_groups": len(values)})
    write_csv(args.output / "group_size_summary.csv", counts)
    write_json(args.output / "provenance.json", {
        "sources": [str(args.original), str(args.extension)],
        "n_predictions": len(predictions), "n_images": len(identity),
        "same_identity_predictions": True, "identity_agreement": float(identity.agree.mean()),
        "acc_original": float((identity.original == identity.target).mean()),
        "paired_comparison": "mita_probe.evaluate.compare_runs; grid minus control seed 0",
        "group_counts": "from original-model Q, independent of k; observed in k=128 pass",
        "minimum_k_at_90_percent": {
            method: next((int(row["k"]) for row in summary
                          if row["method"] == method and row["agree"] >= .9), None)
            for method in ["grid", "kmeans"]},
    })
    (args.output / "log.txt").write_text(
        "Combined 35 original and 7 approved extension settings; exact image/label/original-prediction pairing passed.\n"
        "Confidence intervals: 1000 paired image bootstrap resamples, seed=0.\n", encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
