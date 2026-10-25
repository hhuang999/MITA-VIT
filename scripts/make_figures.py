"""Plot RQ1 result CSVs without rerunning or modifying the experiment."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--title", default="RQ1 pilot | Imagenette, 500 images")
    args = parser.parse_args()
    summary = pd.read_csv(args.run / "summary.csv")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    styles = [("grid", "Grid summaries", "#0072B2", "o"),
              ("random", "Random read sets (5 seeds)", "#777777", "s"),
              ("kmeans", "K-means summaries", "#D55E00", "^")]
    for metric, ylabel, filename in [
        ("agree", "Prediction agreement (%)", "rq1_agreement"),
        ("kl", "KL(original || executed)", "rq1_kl"),
    ]:
        fig, ax = plt.subplots(figsize=(7.6, 4.8), layout="constrained")
        for method, label, color, marker in styles:
            values = summary[summary.method == method].sort_values("k")
            scale = 100 if metric == "agree" else 1
            ax.plot(values.k, values[metric] * scale, label=label, color=color,
                    marker=marker, linewidth=2, markersize=6)
            if metric == "agree" and method == "random":
                ax.errorbar(values.k, values.agree * 100, yerr=values.seed_std * 100,
                            color=color, fmt="none", capsize=4)
        ax.set_xscale("log", base=2)
        ticks = sorted(summary.k.unique())
        ax.set_xticks(ticks, labels=ticks)
        ax.set_xlabel("Tokens per reading set, k (log scale)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", linewidth=.7)
        ax.set_axisbelow(True)
        if metric == "agree":
            ax.axhline(90, color="#999999", linestyle="--", linewidth=1)
            ax.text(max(ticks), 91.5, "90% reference", ha="right", color="#666666", fontsize=9)
            ax.set_ylim(0, 103)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(args.title, loc="left", fontsize=13, pad=14)
        ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1,
                  loc="upper left" if metric == "agree" else "upper right", fontsize=10)
        fig.get_layout_engine().set(rect=(0, .055, 1, .945))
        footer = ("All 12 layers | m = 16 | Frozen original-model summaries"
                  + (" | Error bars: ±1 seed SD" if metric == "agree" else ""))
        fig.text(.5, .015, footer, ha="center", fontsize=8, color="#555555")
        for extension in ["png", "pdf"]:
            fig.savefig(args.run / f"{filename}.{extension}", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
