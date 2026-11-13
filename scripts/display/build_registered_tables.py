"""Rebuild the manuscript's registered-result tables from unchanged CSV summaries."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "paper" / "tables"
MODELS = ("deit", "deit_b", "dino")
MODEL_NAMES = {"deit": "DeiT-S", "deit_b": "DeiT-B", "dino": "DINO ViT-S"}
METRICS = ("enrichment", "clickme", "srg")
CONTROLS = {"C1": "stand_in_centre", "C2": "stand_in_blur", "C3": "stand_in_no_high"}


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def unique(rows, fields):
    index = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in index:
            raise ValueError(f"Duplicate result key: {key}")
        index[key] = row
    return index


def write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def number(value):
    rounded = round(float(value), 3)
    return ("$-$" if rounded < 0 else "") + f"{abs(rounded):.3f}"


def main_results():
    values = unique(read_csv("table1_long.csv"), ("model", "metric", "method"))
    tests = unique((r for r in read_csv("correct_tests.csv") if r["test"] in
                    ("correction - R0", "stand-in - R0", "correction - stand-in")),
                   ("model", "metric", "a", "b"))
    variants = unique(read_csv("variants.csv"), ("variant", "metric", "test"))

    def row(label, method, tested=True):
        cells = []
        for model in MODELS:
            for metric in METRICS:
                value = number(values[model, metric, method]["mean"])
                verdict = tests[model, metric, method, "R0"]["verdict"] if tested else "comparable"
                control = CONTROLS.get(method)
                above_control = control is not None and tests[model, metric, method, control]["verdict"] == "higher"
                if verdict == "higher" and above_control:
                    value = r"\textbf{" + value + "}"
                symbol = {"higher": r"\uparrow", "lower": r"\downarrow", "comparable": ""}[verdict]
                if above_control:
                    symbol += r"\diamond"
                if symbol:
                    value += "$^{" + symbol + "}$"
                cells.append(value)
        return label + " & " + " & ".join(cells) + r" \\"

    def variant_row(label, variant):
        cells = []
        for metric in METRICS:
            result = variants.get((variant, metric, "value"))
            if result is None:
                cells.append("--")
            else:
                verdict = variants[variant, metric, "variant - DeiT-S R0"]["verdict"]
                mark = {"higher": r"\up", "lower": r"\dn", "comparable": ""}[verdict]
                cells.append(number(result["mean"]) + mark)
        return label + " & " + " & ".join(cells + ["--"] * 6) + r" \\"

    lines = [
        "% Generated from results/table1_long.csv, correct_tests.csv and variants.csv.",
        r"\begin{tabular}{@{}l@{\hspace{4pt}}rrr@{\hspace{7pt}}rrr@{\hspace{7pt}}rrr@{}}",
        r"\toprule",
        r" & \multicolumn{3}{c}{DeiT-S} & \multicolumn{3}{c}{DeiT-B} & \multicolumn{3}{c}{DINO ViT-S}\\",
        r"\cmidrule(r{6pt}){2-4}\cmidrule(r{6pt}){5-7}\cmidrule{8-10}",
        "Reading" + " & Enr. & Click. & Del." * 3 + r"\\",
        r"\midrule",
        row("Received attention", "R0", False),
        row("Value weighting", "C3"),
        row(r"\quad zero high-norm scores", "stand_in_no_high"),
        row("Mean-map subtraction", "C1"),
        row(r"\quad subtract centre Gaussian", "stand_in_centre"),
        row("Probe relocation", "C2"),
        row(r"\quad matched blur", "stand_in_blur"),
        row("All three corrections", "R_all"),
        r"\midrule",
        variant_row("+ test-time register", "register"),
        variant_row("Top-$k$ training (MiTA)", "mita"),
        row("GAE (reference)", "gae", False),
        r"\bottomrule", r"\end{tabular}",
    ]
    write("registered_main.tex", lines)


def paired_differences():
    rows = unique((r for r in read_csv("correct_tests.csv") if r["test"] in
                   ("correction - R0", "correction - stand-in")),
                  ("model", "metric", "a", "b"))
    blocks = (
        ("Value weighting $-$ received attention", "C3", "R0"),
        ("Mean-map subtraction $-$ received attention", "C1", "R0"),
        ("Probe relocation $-$ received attention", "C2", "R0"),
        ("All three corrections $-$ received attention", "R_all", "R0"),
        ("Value weighting $-$ zero high-norm scores (its control)", "C3", "stand_in_no_high"),
        ("Mean-map subtraction $-$ centre-Gaussian subtraction (its control)", "C1", "stand_in_centre"),
        ("Probe relocation $-$ matched blur (its control)", "C2", "stand_in_blur"),
    )

    def cell(row):
        verdict = {"higher": "h", "comparable": "c", "lower": "l"}[row["verdict"]]
        interval = f'[{number(row["low"])}, {number(row["high"])}]'
        return f'{number(row["mean"])} {{\\fontsize{{8}}{{9}}\\selectfont {interval}}}\\,\\textit{{{verdict}}}'

    lines = ["% Generated from results/correct_tests.csv; no statistical recalculation.",
             r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Model & Same-ring enrichment & ClickMe alignment & Deletion score \\", r"\midrule"]
    for block_index, (title, method, comparator) in enumerate(blocks):
        if block_index == 4:
            lines.append(r"\midrule")
        elif block_index:
            lines.append(r"\addlinespace[2pt]")
        lines.append(r"\multicolumn{4}{l}{\textit{" + title + r"}} \\")
        for model in MODELS:
            values = [cell(rows[model, metric, method, comparator]) for metric in METRICS]
            lines.append(MODEL_NAMES[model] + " & " + " & ".join(values) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    write("registered_paired.tex", lines)


def sanity_check():
    rows = unique((r for r in read_csv("sanity.csv") if r["level"] == "embed"), ("model", "method"))
    methods = (("Received attention", "R0"), ("Mean-map subtraction", "C1"),
               ("Probe relocation", "C2"), ("Value weighting", "C3"),
               ("All three corrections", "R_all"), ("Centre Gaussian", "centre"))
    lines = ["% Historical registered randomisation results from results/sanity.csv.",
             r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
             r"Reading & DeiT-S & DeiT-B & DINO ViT-S\\", r"\midrule"]
    for label, method in methods:
        if method == "centre":
            lines.append(r"\midrule")
        lines.append(label + " & " + " & ".join(number(rows[model, method]["median"]) for model in MODELS) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    write("registered_sanity.tex", lines)


def diagnostics_per_layer():
    rows = unique(read_csv("diagnose.csv"), ("model", "panel", "variant", "layer"))
    panels = (("M1", "R0", 100.0, "{:.1f}"), ("M2", "probe", 100.0, "{:.1f}"),
              ("M3", "R0", 100.0, "{:.1f}"), ("quality", "R0", 1.0, "{:.3f}"))
    lines = ["% Generated from results/diagnose.csv.", r"\footnotesize", r"\setlength{\tabcolsep}{2.6pt}",
             r"\begin{tabular}{r" + r" >{\centering\arraybackslash}p{0.8cm}" * 9 +
             r" >{\centering\arraybackslash}p{1.05cm}" * 3 + "}", r"\toprule",
             " & \\multicolumn{3}{c}{\\makecell{Specificity:\\\\mean-map share (\\%)}}"
             " & \\multicolumn{3}{c}{\\makecell{Location:\\\\probe accuracy (\\%)}}"
             " & \\multicolumn{3}{c}{\\makecell{Output magnitude:\\\\high-norm share (\\%)}}"
             " & \\multicolumn{3}{c}{\\makecell{Map quality:\\\\ClickMe alignment}} \\\\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
             "Layer" + " & S & B & D" * 4 + r" \\", r"\midrule"]
    for layer in range(1, 13):
        cells = [str(layer)]
        for panel, variant, scale, fmt in panels:
            for model in MODELS:
                value = float(rows[model, panel, variant, str(layer)]["mean"]) * scale
                rendered = fmt.format(value)
                if float(rendered) == 0:
                    rendered = rendered.lstrip("-")
                else:
                    rendered = rendered.replace("-", "$-$")
                cells.append(rendered)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    write("diagnostics_per_layer.tex", lines)


def main():
    main_results()
    paired_differences()
    sanity_check()
    diagnostics_per_layer()
    print("Rebuilt registered_main, registered_paired, registered_sanity and diagnostics_per_layer")


if __name__ == "__main__":
    main()
