"""Plot paired continuous deletion contrasts from the v13 exploratory CSV.

Run from any directory with the project Python environment. The PDF is exactly
5.5 inches wide; all plotted marks and text remain vector/editable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SKILL_SCRIPTS = Path(os.environ.get("NATURE_FIGURE_SCRIPTS", Path.home() / ".codex/skills/nature-figure/scripts"))


INPUT = HERE / "budget_differences.csv"
OUTPUT = HERE / "deletion_curves.pdf"
BUDGETS = (1, 2, 4, 8, 16, 32, 64, 98, 147)
MODELS = {
    "deit": ("DeiT-S", "#0072B2", "o", -0.055),
    "deit_b": ("DeiT-B", "#D55E00", "s", 0.000),
    "dino": ("DINO", "#3D6557", "^", 0.055),
}


def plot() -> None:
    data = pd.read_csv(INPUT)
    data = data[(data.a == "C3") & (data.b == "stand_in_no_high")
                & data.metric.isin(("p_target", "kl"))].copy()
    if len(data) != 2 * len(MODELS) * len(BUDGETS):
        raise ValueError("Expected two metrics, three models, and all nine budgets")
    if data.duplicated(["metric", "model", "budget"]).any() or not np.isfinite(data[["mean", "low", "high"]]).all().all():
        raise ValueError("Duplicate or non-finite plotted data")
    for (metric, model), group in data.groupby(["metric", "model"]):
        if tuple(sorted(group.budget)) != BUDGETS or not (group.low <= group["mean"]).all() or not (group["mean"] <= group.high).all():
            raise ValueError(f"Incomplete or invalid interval: {metric}, {model}")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 7,
        "axes.titlesize": 7.4,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55), dpi=300)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.205, top=0.765, wspace=0.35)
    specs = (
        ("p_target", "Original top-1 probability", "螖 probability", (-0.007, 0.034)),
        ("kl", "KL from original output", "螖 KL (nats)", (-0.058, 0.185)),
    )
    for ax, (metric, title, ylabel, ylim), letter in zip(axes, specs, "ab"):
        ax.axhline(0, color="#777777", lw=0.65, zorder=0)
        for model, (label, color, marker, offset) in MODELS.items():
            part = data[(data.metric == metric) & (data.model == model)].sort_values("budget")
            x = np.log2(part.budget.to_numpy() + 1) + offset
            y = part["mean"].to_numpy()
            low = part.low.to_numpy()
            high = part.high.to_numpy()
            # Offsets expose overlapping pointwise intervals at small budgets.
            # The first, zero-budget point is exactly shared by every method.
            ax.plot(np.r_[0, x], np.r_[0, y], color=color, lw=1.05, alpha=0.93,
                    zorder=2, label=label)
            ax.errorbar(x, y, yerr=np.vstack((y - low, high - y)), fmt=marker,
                        color=color, ecolor=color, ms=3.0, lw=0.7, capsize=1.35,
                        capthick=0.7, markeredgecolor="white", markeredgewidth=0.32,
                        zorder=3)
        ax.plot(0, 0, marker="o", ms=2.3, color="#333333", markerfacecolor="white", zorder=4)
        ax.set_xlim(-0.20, np.log2(148) + 0.19)
        ax.set_ylim(*ylim)
        shown = (0, 1, 4, 16, 64, 147)
        ax.set_xticks(np.log2(np.array(shown) + 1), labels=[str(value) for value in shown])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=6)
        ax.text(-0.21, 1.08, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
        ax.tick_params(axis="both", direction="out", length=2.4, width=0.6, pad=2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.96),
               ncol=3, handlelength=1.5, columnspacing=1.9)
    fig.text(0.545, 0.065, "Removed patches", ha="center", va="center", fontsize=7)
    if (SKILL_SCRIPTS / "audit_panel_alignment.py").is_file():
        sys.path.insert(0, str(SKILL_SCRIPTS))
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            fig, json_out=HERE / "deletion_curves.alignment.json",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=True, strict=True,
        )
    fig.savefig(OUTPUT, metadata={"Title": "C3 versus no-high control: deletion-budget contrasts",
                                  "Creator": "research/v13/endpoints/plot_curves.py"})
    fig.savefig(HERE / "deletion_curves.svg")
    fig.savefig(HERE / "deletion_curves.png", dpi=300)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    plot()
