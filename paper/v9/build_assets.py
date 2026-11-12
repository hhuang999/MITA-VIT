"""Build v9 presentation assets from the unchanged, supplied result CSVs.

Run with Python 3.12: python paper/v9/build_assets.py
No inference, fitting, resampling or new statistical decisions are performed.
"""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = ("deit", "deit_b", "dino")
METRICS = ("enrichment", "clickme", "srg")
CONTROLS = {
    "C1": "stand_in_centre",
    "C2": "stand_in_blur",
    "C3": "stand_in_no_high",
}


def read_csv(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def unique_index(rows, fields):
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in indexed:
            raise ValueError(f"Duplicate result key: {key}")
        indexed[key] = row
    return indexed


def number(value):
    return f"{float(value):.3f}".replace("-", "$-$")


def make_table(values, tests, variants):
    marks = []

    def row(label, method, tested=True):
        cells = []
        for model in MODELS:
            for metric in METRICS:
                text = number(values[model, metric, method]["mean"])
                verdict = tests[model, metric, method, "R0"]["verdict"] if tested else "comparable"
                control = CONTROLS.get(method)
                above_control = control is not None and tests[model, metric, method, control]["verdict"] == "higher"
                if above_control:
                    marks.append({"model": model, "metric": metric, "method": method, "control": control})
                if verdict == "higher" and above_control:
                    text = r"\textbf{" + text + "}"
                symbols = {"higher": r"\uparrow", "lower": r"\downarrow", "comparable": ""}[verdict]
                if above_control:
                    symbols += r"\diamond"
                if symbols:
                    text += "$^{" + symbols + "}$"
                cells.append(text)
        return label + " & " + " & ".join(cells) + r" \\"

    def variant_row(label, variant):
        cells = []
        for metric in METRICS:
            value = variants.get((variant, metric, "value"))
            if value is None:
                cells.append("--")
                continue
            verdict = variants[variant, metric, "variant - DeiT-S R0"]["verdict"]
            mark = {"higher": r"\up", "lower": r"\dn", "comparable": ""}[verdict]
            cells.append(number(value["mean"]) + mark)
        return label + " & " + " & ".join(cells + ["--"] * 6) + r" \\"

    lines = [
        "% v9 presentation; all values and verdicts are read unchanged from results/.",
        r"\begin{tabular}{@{}l@{\hspace{4pt}}rrr@{\hspace{7pt}}rrr@{\hspace{7pt}}rrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{DeiT-S} & \multicolumn{3}{c}{DeiT-B} & \multicolumn{3}{c}{DINO ViT-S}\\",
        r"\cmidrule(r{6pt}){2-4}\cmidrule(r{6pt}){5-7}\cmidrule{8-10}",
        "Reading" + " & Enr. & Click. & Del." * 3 + r"\\",
        r"\midrule",
        row("Received attention", "R0", tested=False),
        row("Contribution weighting", "C3"),
        row(r"\quad zero high-norm tokens", "stand_in_no_high"),
        row("Mean-map subtraction", "C1"),
        row(r"\quad subtract centre Gaussian", "stand_in_centre"),
        row("Probe relocation", "C2"),
        row(r"\quad matched blur", "stand_in_blur"),
        row("All three corrections", "R_all"),
        r"\midrule",
        variant_row("+ test-time register", "register"),
        variant_row("Top-$k$ training (MiTA)", "mita"),
        row("GAE (reference)", "gae", tested=False),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (HERE / "tables/main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return marks


def make_effect_plot(tests):
    fig, axes = plt.subplots(1, 2, figsize=(3.42, 1.67), sharex=True, sharey=True)
    comparisons = [("R0", "#2566a5", "o", 0.12), ("stand_in_no_high", "#ba6c25", "s", -0.12)]
    plotted = []
    for ax, metric, title in zip(axes, METRICS[:2], ("Enrichment", "ClickMe")):
        ax.axvline(0, color="0.65", lw=0.7, zorder=0)
        ax.axvline(0.02, color="0.35", lw=0.8, ls=(0, (3, 2)), zorder=0)
        for index, model in enumerate(MODELS):
            for reference, color, marker, offset in comparisons:
                result = tests[model, metric, "C3", reference]
                mean, low, high = (float(result[key]) for key in ("mean", "low", "high"))
                ax.errorbar(mean, 2 - index + offset, xerr=[[mean - low], [high - mean]],
                            color=color, marker=marker, markersize=4, capsize=2, lw=0.9)
                plotted.append({key: result[key] for key in ("model", "metric", "a", "b", "mean", "low", "high", "verdict")})
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlim(-0.009, 0.082)
        ax.set_xticks([0, 0.02, 0.06], ["0", ".02", ".06"])
        ax.set_ylim(-0.4, 2.4)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks([2, 1, 0], ["DeiT-S", "DeiT-B", "DINO"])
    handles = [Line2D([], [], color=color, marker=marker, lw=0.8, markersize=4)
               for _, color, marker, _ in comparisons]
    fig.legend(handles, ["C3 - R0", "C3 - control"], frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.54, -0.015), columnspacing=1.0,
               handlelength=1.1, handletextpad=0.4)
    fig.text(0, 0.98, "(b) Norm-weighting effects", fontsize=8.5, weight="bold", va="top")
    fig.subplots_adjust(left=0.15, right=0.99, top=0.77, bottom=0.26, wspace=0.2)
    fig.savefig(HERE / "figures/contribution_effects.pdf")
    fig.savefig(HERE / "figures/contribution_effects.png", dpi=220)
    plt.close(fig)
    return plotted


def make_schematic():
    fig, ax = plt.subplots(figsize=(1.93, 1.52))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    for y in (0.24, 0.47, 0.70):
        ax.plot([0.10, 0.48, 0.90], [0.47, y, 0.47], color="#6d8295", lw=0.8, zorder=0)
        ax.add_patch(Circle((0.48, y), 0.071, facecolor="#e4edf5", edgecolor="#2566a5", lw=0.9))
    for x in (0.10, 0.90):
        ax.add_patch(Circle((x, 0.47), 0.075, facecolor="white", edgecolor="0.25", lw=0.8))
    ax.text(0.10, 0.47, r"$q_i^h$", ha="center", va="center", fontsize=9)
    ax.text(0.90, 0.47, r"$o_i^h$", ha="center", va="center", fontsize=9)
    ax.text(0.48, 0.47, r"$a_{ij}^h$", ha="center", va="center", fontsize=8)
    ax.text(0.27, 0.83, r"keys $k_j^h$", ha="center", fontsize=8)
    ax.text(0.76, 0.83, r"$W_O^h v_j^h$", ha="center", fontsize=8)
    ax.text(0.48, 0.10, "hidden units = tokens", ha="center", fontsize=8)
    ax.text(0.48, 0.01, "softmax-coupled activations", ha="center", fontsize=7.5)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(HERE / "figures/fast_weight_lens.pdf")
    plt.close(fig)


def make_sanity_table():
    rows = read_csv("paper/v9/evidence/sanity.csv")
    rows = unique_index([row for row in rows if row["level"] == "embed"],
                        ("model", "method"))
    methods = [("Received attention", "R0"), ("Mean-map subtraction", "C1"),
               ("Probe relocation", "C2"), ("Contribution weighting", "C3"),
               ("All three corrections", "R_all"), ("Centre Gaussian", "centre")]
    lines = ["% Historical sanity results, copied without new inference or statistics.",
             r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Reading & DeiT-S & DeiT-B & DINO ViT-S\\", r"\midrule"]
    for label, method in methods:
        if method == "centre":
            lines.append(r"\midrule")
        cells = [number(rows[model, method]["median"]) for model in MODELS]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (HERE / "tables/sanity_check.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main():
    for name in ("figures", "tables"):
        (HERE / name).mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.linewidth": 0.6, "xtick.major.size": 2,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    values = unique_index(read_csv("results/table1_long.csv"), ("model", "metric", "method"))
    comparisons = [row for row in read_csv("results/correct_tests.csv")
                   if row["test"] in ("correction - R0", "correction - stand-in", "stand-in - R0")]
    tests = unique_index(comparisons, ("model", "metric", "a", "b"))
    variants = unique_index(read_csv("results/variants.csv"), ("variant", "metric", "test"))
    marks = make_table(values, tests, variants)
    plotted = make_effect_plot(tests)
    make_schematic()
    make_sanity_table()
    paths = ("results/table1_long.csv", "results/correct_tests.csv", "results/variants.csv",
             "paper/v9/evidence/sanity.csv")
    record = {"input_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
              "control_markers": marks, "effect_plot_rows": plotted}
    (HERE / "asset_sources.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Built table and figures from existing results; {len(marks)} matched-control markers.")


if __name__ == "__main__":
    main()
