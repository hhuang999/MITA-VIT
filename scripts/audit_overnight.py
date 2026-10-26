"""Check saved overnight results against fixed samples and recomputed statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mita_probe.io import load_config, write_json
from mita_probe.rq1 import mean_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rq", choices=["rq1", "rq2", "rq3"])
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    run = args.run
    config = load_config(run / "config.yaml")
    checks = []

    def check(name, condition):
        checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    if args.rq in ["rq1", "rq2"]:
        identity = pd.read_csv(run / "identity_predictions.csv").sort_values("index").reset_index(drop=True)
        reference = pd.read_csv(args.reference / "identity_predictions.csv").sort_values("index").reset_index(drop=True)
        check("same fixed image IDs, labels and original predictions", identity.equals(reference))
        check("full-image identity is exact", identity.agree.eq(1).all())
        frame = pd.read_csv(run / "predictions.csv")
        settings = ["k", "method", "seed"] if args.rq == "rq1" else ["k", "layer"]
        expected = 7 if args.rq == "rq1" else 24
        check("expected settings", frame.groupby(settings).ngroups == expected)
        check("no duplicate image-setting rows", not frame.duplicated(settings+["index"]).any())
        for key, values in frame.groupby(settings):
            check(f"{key}: complete sample", set(values["index"]) == set(reference["index"]))
            check(f"{key}: agreement from predictions", np.array_equal(values.agree, (values.original == values.executed).astype(int)))
            check(f"{key}: finite KL", np.isfinite(values.kl).all() and values.kl.min() > -1e-10)
            left = values.sort_values("index").reset_index(drop=True)
            check(f"{key}: fixed labels/original", left[["target", "original"]].equals(reference[["target", "original"]]))
        if args.rq == "rq2":
            results = pd.read_csv(run / "results.csv")
            for row in results.itertuples():
                values = frame[(frame.k == row.k) & (frame.layer == row.layer)]
                low, high = mean_interval(values.agree.to_numpy(), config)
                check(f"k={row.k},l={row.layer}: recomputed agreement and CI", np.allclose(
                    [values.agree.mean(), low, high], [row.agree, row.agree_ci_low, row.agree_ci_high], atol=1e-12))
                check(f"k={row.k},l={row.layer}: index mapping", values.layer_index.eq(row.layer-1).all())
        else:
            hist = pd.read_csv(run / "group_query_counts.csv")
            for (layer, kind), values in hist.groupby(["layer", "kind"]):
                total = 196 if kind == "euclidean_patch_clusters" else 197
                check(f"layer {layer} {kind}: histogram count", values.frequency.sum() == 5000*6*16)
                check(f"layer {layer} {kind}: query conservation", (values.query_count*values.frequency).sum() == 5000*6*total)
    else:
        layers = pd.read_csv(run / "layer_predictions.csv")
        maps = pd.read_csv(run / "heatmap_predictions.csv")
        observed = pd.read_csv(run / "observe_predictions.csv")
        backgrounds = pd.read_csv(run / "background_predictions.csv")
        n = 3669
        check("all Pet test IDs", set(layers["index"]) == set(range(n)))
        check("two k times twelve layers", len(layers) == n*24)
        check("three maps for both k", len(maps) == n*6)
        check("four background layers for both k", len(backgrounds) == n*8)
        check("no duplicate layer rows", not layers.duplicated(["index", "k", "layer"]).any())
        check("no duplicate heatmap rows", not maps.duplicated(["index", "k", "method"]).any())
        check("observe logits and predictions unchanged", observed.agree.eq(1).all() and observed.batch_max_logit_error.eq(0).all())
        check("all observed records", len(observed) == 2*n)
        check("foreground ratios bounded", layers.fg_ratio.dropna().between(0, 1).all())
        check("heatmap mass bounded and finite", maps.fg_mass.between(0, 1+1e-6).all())
        check("pointing binary", maps.pointing.isin([0, 1]).all())
        check("same foreground area for each image", layers.groupby("index").fg_area_ratio.nunique().eq(1).all())
        for method in ["cls_attention", "center_gaussian"]:
            left = maps[(maps.method == method) & (maps.k == 16)].sort_values("index")
            right = maps[(maps.method == method) & (maps.k == 32)].sort_values("index")
            check(f"{method}: identical across k", np.array_equal(left[["fg_mass", "pointing"]].to_numpy(), right[["fg_mass", "pointing"]].to_numpy()))
        summary = pd.read_csv(run / "results.csv")
        for row in summary.itertuples():
            values = layers[(layers.k == row.k) & (layers.layer == row.layer)].fg_ratio.dropna().to_numpy()
            low, high = mean_interval(values, config)
            check(f"k={row.k},l={row.layer}: recomputed foreground CI", np.allclose(
                [values.mean(), low, high], [row.fg_ratio, row.fg_ci_low, row.fg_ci_high], atol=1e-12))
    write_json(run / "audit.json", {"n_checks": len(checks), "all_passed": True, "checks": checks})
    print(f"{args.rq}: {len(checks)} checks passed")


if __name__ == "__main__":
    main()
