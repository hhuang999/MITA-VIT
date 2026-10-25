"""Create Figure 3 from saved single-layer results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.run / "results.csv")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8, 4.7), layout="constrained")
    for k, color, marker in [(8, "#0072B2", "o"), (32, "#D55E00", "s")]:
        values = frame[frame.k == k].sort_values("layer")
        ax.plot(values.layer, values.agree*100, marker=marker, color=color, label=f"k = {k}")
        ax.fill_between(values.layer, values.agree_ci_low*100, values.agree_ci_high*100,
                        color=color, alpha=.18)
    ax.set(xticks=range(1, 13), xlabel="Replaced layer (one layer at a time)",
           ylabel="Prediction agreement (%)", ylim=(0, 102))
    ax.set_title(f"RQ2 | ImageNet validation, {frame.n_images.iloc[0]:,} images", loc="left")
    ax.grid(axis="y", color="#dddddd")
    ax.legend(frameon=False)
    fig.supxlabel("Grid summaries | m = 16 | Shading: 95% image bootstrap CI", fontsize=9)
    for extension in ["png", "pdf"]:
        fig.savefig(args.run / f"rq2_layers.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
