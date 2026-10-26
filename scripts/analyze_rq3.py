"""Image-paired foreground/pointing contrasts for the requested RQ3 maps."""

import argparse
from pathlib import Path

import pandas as pd

from mita_probe.evaluate import compare_runs
from mita_probe.io import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.run / "heatmap_predictions.csv")
    results = []
    for k in [16, 32]:
        ours = frame[(frame.k == k) & (frame.method == "selection")]
        for method in ["cls_attention", "center_gaussian"]:
            reference = frame[(frame.k == k) & (frame.method == method)]
            for metric in ["fg_mass", "pointing"]:
                results.append({"k": k, "contrast": f"selection_minus_{method}", "metric": metric,
                                **compare_runs(ours, reference, metric=metric)})
    write_csv(args.run / "heatmap_contrasts.csv", results)
    print("Eight image-paired RQ3 map contrasts written")


if __name__ == "__main__":
    main()
