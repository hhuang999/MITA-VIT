"""Pair each RQ2 setting with the same-k all-layer RQ1 intervention."""

import argparse
from pathlib import Path

import pandas as pd

from mita_probe.evaluate import compare_runs
from mita_probe.io import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("rq1", type=Path)
    args = parser.parse_args()
    single = pd.read_csv(args.run / "predictions.csv")
    all_layers = pd.read_csv(args.rq1 / "predictions.csv")
    results = []
    for (k, layer), frame in single.groupby(["k", "layer"]):
        reference = all_layers[(all_layers.k == k) & (all_layers.method == "grid")]
        results.append({"k": int(k), "layer": int(layer),
                        "contrast": "single_layer_minus_all_layers", **compare_runs(frame, reference)})
    write_csv(args.run / "all_layer_comparison.csv", results)
    print("24 paired single-layer versus all-layer comparisons written")


if __name__ == "__main__":
    main()
