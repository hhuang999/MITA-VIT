"""Render candidate figures from saved evidence, without new statistical analysis.

Use --audit-scripts to locate the installed scientific-figure QA helpers.
The final-size PDF, editable SVG and PNG use the same Matplotlib canvas.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("deit", "deit_b", "dino")
NAMES = {"deit": "DeiT-S", "deit_b": "DeiT-B", "dino": "DINO ViT-S"}
COLORS = {"deit": "#2F6685", "deit_b": "#BB7D32", "dino": "#7B6A90"}
STYLES = {"deit": "-", "deit_b": "--", "dino": ":"}
CONTROLS = {"C1": "stand_in_centre", "C2": "stand_in_blur", "C3": "stand_in_no_high"}
Y = {(model, metric): value for model, positions in zip(MODELS, ((7, 6), (4, 3), (1, 0)))
     for metric, value in zip(("enrichment", "clickme"), positions)}
ROW_TICKS = [7, 6, 4, 3, 1, 0]
ROW_LABELS = ["DeiT-S  Enr.", "ClickMe", "DeiT-B  Enr.", "ClickMe", "DINO  Enr.", "ClickMe"]


def read_rows(relative):
    with (ROOT / relative).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def contrast_index(rows):
    result = {}
    for row in rows:
        if row["test"] not in ("correction - R0", "correction - stand-in"):
            continue
        key = (row["model"], row["metric"], row["a"], row["b"])
        if key in result:
            raise ValueError(f"Ambiguous stored contrast: {key}")
        result[key] = row
    return result


def panel_header(ax, letter, title, subtitle=None, pad=14):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(-10, pad),
                textcoords="offset points", fontsize=8.5, weight="bold",
                ha="left", va="bottom", annotation_clip=False)
    ax.set_title(title, loc="left", fontsize=7.5, weight="bold", pad=pad)
    if subtitle:
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left", color="#525252", clip_on=False)


def interval_point(ax, row, y, marker="o", size=3.5):
    mean, low, high = (float(row[field]) for field in ("mean", "low", "high"))
    if not low <= mean <= high:
        raise ValueError(f"Stored interval does not contain its mean: {row}")
    color = COLORS[row["model"]]
    face = color if row["verdict"] in ("higher", "lower") else "white"
    ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]],
                marker=marker, ms=size, mfc=face, mec=color, mew=0.85,
                color=color, lw=0.75, elinewidth=0.75, capsize=1.8, zorder=3)


def effect_axes(ax, limit, ticks, margin, labels=False):
    ax.axvspan(-margin, margin, color="#F0F0EE", lw=0, zorder=0)
    ax.axvline(0, color="#929292", lw=0.6, zorder=1)
    ax.set_xlim(*limit)
    ax.set_xticks(ticks)
    ax.set_ylim(-0.8, 7.8)
    ax.set_yticks(ROW_TICKS, ROW_LABELS if labels else [""] * 6)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Paired difference", fontsize=7, labelpad=3)


def write_source_data(path, rows):
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(fig, stem, out, require_matplotlib_panel_alignment, axes, row_groups, column_groups=()):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=[ax.get_gid() for ax in axes],
              row_groups=row_groups, column_groups=list(column_groups),
              json_out=out / f"{stem}.alignment.json",
              overlay_svg=out / f"{stem}.alignment.svg",
              tolerance_pt=1.5, gutter_tolerance_pt=1.5,
              require_panel_labels=True, strict=True)
    # Preserve the exact physical canvas: no tight crop and no manuscript rescaling.
    fig.savefig(out / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{stem}.svg")
    fig.savefig(out / f"{stem}.png", dpi=300)
    plt.close(fig)


def make_repair_figure(tests, out, alignment):
    fig = plt.figure(figsize=(5.5, 1.85))
    grid = fig.add_gridspec(1, 3, left=0.17, right=0.985, bottom=0.30,
                           top=0.77, wspace=0.32)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]
    selected = []
    for index, (ax, title, reference) in enumerate(zip(
            axes[:2], ("Alignment gain", "Beyond token removal"),
            ("R0", "stand_in_no_high"))):
        ax.set_gid(chr(97 + index))
        effect_axes(ax, (-0.025, 0.085), [0, 0.04, 0.08], 0.02, labels=index == 0)
        panel_header(ax, chr(97 + index), title,
                     "C3 - received attention" if index == 0 else "C3 - zero high-norm tokens")
        for model in MODELS:
            for metric in ("enrichment", "clickme"):
                row = tests[model, metric, "C3", reference]
                interval_point(ax, row, Y[model, metric])
                selected.append({"panel": chr(97 + index), **row})
    ax = axes[2]
    ax.set_gid("c")
    effect_axes(ax, (-0.06, 0.06), [-0.05, 0, 0.05], 0.05)
    panel_header(ax, "c", "Deletion boundary", "C3 - received attention")
    for model, y in zip(MODELS, (6.5, 3.5, 0.5)):
        row = tests[model, "srg", "C3", "R0"]
        interval_point(ax, row, y, size=4)
        selected.append({"panel": "c", **row})
    fig.text(0.17, 0.045, "Filled: higher/lower; open: comparable. Grey: practical-margin band.",
             fontsize=7, ha="left", va="bottom", color="#525252")
    export(fig, "01_repair_benefits_limits", out, alignment, axes, [["a", "b", "c"]])
    write_source_data(out / "01_repair_benefits_limits.source.csv", selected)
    return {"source_rows": len(selected), "width_in": 5.5, "height_in": 1.85}


def make_bridge_figure(diagnostics, tests, out, alignment):
    fig = plt.figure(figsize=(5.5, 2.85))
    upper = fig.add_gridspec(1, 3, left=0.17, right=0.985, bottom=0.62,
                            top=0.85, wspace=0.43)
    lower = fig.add_gridspec(1, 3, left=0.17, right=0.985, bottom=0.17,
                            top=0.43, wspace=0.43)
    top_axes = [fig.add_subplot(upper[0, col]) for col in range(3)]
    low_axes = [fig.add_subplot(lower[0, col]) for col in range(3)]
    panels = [("M1", "R0", "Specificity", "Mean-map\nshare (%)", (0, 25), [0, 10, 20]),
              ("M2", "probe", "Location", "Index-probe\naccuracy (%)", (0, 105), [0, 50, 100]),
              # Leave a little room below zero so DINO's zero series is visible above the frame.
              ("M3", "R0", "Contribution", "High-norm\nshare (%)", (-2, 48), [0, 20, 40])]
    diagnostic_data = []
    for index, (ax, spec) in enumerate(zip(top_axes, panels)):
        panel, variant, title, ylabel, limits, ticks = spec
        letter = chr(97 + index)
        ax.set_gid(letter)
        panel_header(ax, letter, title)
        ax.axvspan(8.5, 12.5, color="#F0F0EE", lw=0, zorder=0)
        for model in MODELS:
            rows = sorted([r for r in diagnostics if r["model"] == model
                           and r["panel"] == panel and r["variant"] == variant],
                          key=lambda r: int(r["layer"]))
            if len(rows) != 12:
                raise ValueError(f"Expected 12 diagnostic layers: {model}/{panel}")
            x = [int(row["layer"]) for row in rows]
            mean, low, high = [np.array([float(row[key]) for row in rows]) * 100
                               for key in ("mean", "low", "high")]
            ax.fill_between(x, low, high, color=COLORS[model], alpha=0.13, lw=0)
            ax.plot(x, mean, color=COLORS[model], ls=STYLES[model], lw=1.05)
            diagnostic_data.extend({"figure_panel": letter, **row} for row in rows)
        ax.set_xlim(0.5, 12.5)
        ax.set_xticks([1, 4, 8, 12])
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        ax.tick_params(axis="both", labelsize=6.5)
        ax.set_ylabel(ylabel, fontsize=6.5, labelpad=3)
        ax.set_xlabel("Layer", fontsize=6.5, labelpad=2)
    paired_data = []
    for index, (ax, method, title) in enumerate(zip(
            low_axes, ("C1", "C2", "C3"),
            ("Mean subtraction", "Probe relocation", "Norm weighting"))):
        letter = chr(100 + index)
        ax.set_gid(letter)
        panel_header(ax, letter, title, pad=6)
        effect_axes(ax, (-0.14, 0.09), [-0.12, -0.06, 0, 0.06],
                    0.02, labels=index == 0)
        ax.tick_params(axis="both", labelsize=6.5)
        ax.set_xlabel("Paired difference", fontsize=6.5, labelpad=3)
        for model in MODELS:
            for metric in ("enrichment", "clickme"):
                for reference, marker, offset in (("R0", "o", 0.20),
                                                   (CONTROLS[method], "D", -0.20)):
                    row = tests[model, metric, method, reference]
                    interval_point(ax, row, Y[model, metric] + offset, marker=marker, size=2.8)
                    paired_data.append({"figure_panel": letter, **row})
    model_handles = [Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=1.1,
                            label=NAMES[m]) for m in MODELS]
    model_handles.append(Patch(facecolor="#F0F0EE", label="Layers 9-12"))
    fig.legend(handles=model_handles, loc="upper center", bbox_to_anchor=(0.55, 1.015),
               frameon=False, ncol=4, fontsize=6.4, handlelength=1.5,
               handletextpad=0.4, columnspacing=1.2)
    ref_handles = [Line2D([], [], color="#505050", marker=marker, mfc="white", ms=3,
                          lw=0, label=label) for marker, label in
                   (("o", "Vs received attention"), ("D", "Vs matched control"))]
    fig.legend(handles=ref_handles, loc="lower center", bbox_to_anchor=(0.55, 0.005),
               frameon=False, ncol=2, fontsize=6.4, handletextpad=0.4,
               columnspacing=1.8)
    axes = top_axes + low_axes
    export(fig, "02_diagnosis_correction_bridge", out, alignment, axes,
           [["a", "b", "c"], ["d", "e", "f"]],
           [["a", "d"], ["b", "e"], ["c", "f"]])
    write_source_data(out / "02_diagnosis_correction_bridge.diagnostics.csv", diagnostic_data)
    write_source_data(out / "02_diagnosis_correction_bridge.effects.csv", paired_data)
    return {"diagnostic_rows": len(diagnostic_data), "paired_rows": len(paired_data),
            "width_in": 5.5, "height_in": 2.85}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-scripts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "output/figure_candidates")
    args = parser.parse_args()
    sys.path.insert(0, str(args.audit_scripts.resolve()))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    args.out.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.5, "xtick.major.width": 0.6,
        "ytick.major.size": 2.5, "ytick.major.width": 0.6,
        "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    diagnostic_rows = read_rows("results/diagnose.csv")
    test_rows = read_rows("results/correct_tests.csv")
    tests = contrast_index(test_rows)
    manifest = {
        "status": "Candidates awaiting author selection; v9 unchanged",
        "inputs": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                   for path in ("results/diagnose.csv", "results/correct_tests.csv")},
        "source_counts": {"diagnostics": len(diagnostic_rows), "contrasts": len(test_rows)},
        "figure_1": make_repair_figure(tests, args.out, require_matplotlib_panel_alignment),
        "figure_2": make_bridge_figure(diagnostic_rows, tests, args.out,
                                       require_matplotlib_panel_alignment),
        "statistics": "Original paired image-bootstrap means, 95% intervals and registered verdicts; no recalculation",
    }
    (args.out / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
