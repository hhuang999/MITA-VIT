"""Figure 4: foreground curves and three predetermined qualitative examples."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    results = pd.read_csv(args.run / "results.csv")
    examples = pd.read_csv(args.run / "example_maps.csv")
    ids = json.loads((args.run / "metadata.json").read_text())["example_indices"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(15, 8), layout="constrained")
    outer = fig.add_gridspec(1, 2, width_ratios=[1, 2])
    ax = fig.add_subplot(outer[0])
    for k, color in [(16, "#0072B2"), (32, "#D55E00")]:
        values = results[results.k == k].sort_values("layer")
        ax.plot(values.layer, values.fg_ratio*100, marker="o", color=color, label=f"k = {k}")
        ax.fill_between(values.layer, values.fg_ci_low*100, values.fg_ci_high*100, color=color, alpha=.18)
    baseline = results.iloc[0]
    ax.axhline(baseline.fg_area_ratio*100, color="#777777", ls="--", label="Foreground token share")
    ax.axhspan(baseline.area_ci_low*100, baseline.area_ci_high*100, color="#777777", alpha=.1)
    ax.set(xticks=range(1, 13), xlabel="Layer", ylabel="Selected foreground tokens (%)", ylim=(0, 100))
    ax.grid(axis="y", color="#dddddd")
    ax.set_title("Foreground reading across layers", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower left")
    right = outer[1].subgridspec(3, 4, wspace=.02, hspace=.05)
    for row, index in enumerate(ids):
        frame = examples[examples["index"] == index].sort_values("token")
        rgb = Image.open(args.run / f"example_{index}.png")
        for column, method in enumerate([None, "selection", "cls_attention", "background_reads"]):
            panel = fig.add_subplot(right[row, column])
            panel.imshow(rgb)
            if method:
                values = frame[method].to_numpy().reshape(14, 14)
                normalized = values / max(values.max(), 1e-12)
                rgba = plt.get_cmap("magma")(normalized)
                rgba[..., 3] = .6 * normalized
                panel.imshow(rgba, extent=(-.5, 223.5, 223.5, -.5), interpolation="nearest")
            if method == "background_reads":
                region = int(frame.source_region.iloc[0])
                if region >= 0:
                    y, x = divmod(region, 4)
                    y0, x0 = y*14//4, x*14//4
                    y1, x1 = ((y+1)*14+3)//4, ((x+1)*14+3)//4
                    panel.add_patch(Rectangle((x0*16-.5, y0*16-.5), (x1-x0)*16, (y1-y0)*16,
                                              fill=False, edgecolor="#00FFFF", linewidth=2))
            panel.set_xticks([])
            panel.set_yticks([])
            if row == 0:
                panel.set_title(["Image", "Selected counts", "CLS attention", "Background reads"][column], fontsize=10)
            if column == 0:
                panel.set_ylabel(f"Test index {index}", fontsize=9)
    fig.suptitle("RQ3 | Oxford-IIIT Pet test | Observe mode, m = 16", x=.01, ha="left", fontsize=15)
    fig.supxlabel("95% image bootstrap CI | Examples: k = 16; counts averaged over layers 9–12 | Cyan: source window in layer 12", fontsize=10)
    for extension in ["png", "pdf"]:
        fig.savefig(args.run / f"rq3_foreground.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
