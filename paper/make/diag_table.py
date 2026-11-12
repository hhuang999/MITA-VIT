"""Build a per-layer diagnostics table (LaTeX) from results/diagnose.csv.

Usage: python -I diag_table.py <path/to/diagnose.csv> <out.tex>
Columns: mean-map share (%), position-probe accuracy (%), share of received attention on
high-norm tokens (%), image-specific ClickMe alignment of single-layer received attention.
"""
import csv
import sys

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, newline="")))
models = ["deit", "deit_b", "dino"]
want = [("M1", "R0", 100.0, "{:.1f}"), ("M2", "probe", 100.0, "{:.1f}"),
        ("M3", "R0", 100.0, "{:.1f}"), ("quality", "R0", 1.0, "{:.3f}")]
val = {}
for r in rows:
    val[(r["model"], r["panel"], r["variant"], int(r["layer"]))] = float(r["mean"])


def fmt(x, scale, f):
    s = f.format(x * scale)
    if s.startswith("-"):
        # avoid "-0.0" for tiny negatives
        if float(s) == 0.0:
            s = s[1:]
        else:
            s = "$-$" + s[1:]
    return s


lines = []
lines.append("% Per-layer diagnostics, generated from results/diagnose.csv by diag_table.py.")
lines.append("\\footnotesize")
lines.append("\\setlength{\\tabcolsep}{2.6pt}")
lines.append("\\begin{tabular}{r" + " >{\\centering\\arraybackslash}p{0.8cm}" * 9 + " >{\\centering\\arraybackslash}p{1.05cm}" * 3 + "}")
lines.append("\\toprule")
lines.append(" & \\multicolumn{3}{c}{\\makecell{Specificity:\\\\mean-map share (\\%)}}"
             " & \\multicolumn{3}{c}{\\makecell{Location:\\\\probe accuracy (\\%)}}"
             " & \\multicolumn{3}{c}{\\makecell{Contribution:\\\\high-norm share (\\%)}}"
             " & \\multicolumn{3}{c}{\\makecell{Map quality:\\\\ClickMe alignment}} \\\\")
lines.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}")
lines.append("Layer" + " & S & B & D" * 4 + " \\\\")
lines.append("\\midrule")
for layer in range(1, 13):
    cells = [str(layer)]
    for panel, variant, scale, f in want:
        for m in models:
            cells.append(fmt(val[(m, panel, variant, layer)], scale, f))
    lines.append(" & ".join(cells) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
open(out, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("\n".join(lines))
