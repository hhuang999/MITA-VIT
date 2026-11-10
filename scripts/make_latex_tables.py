"""Build the unchanged paired-difference source used by the paper's Table B.2."""

import argparse
import math
import os
import re
import textwrap
from pathlib import Path

import pandas as pd

MODELS = ["deit", "deit_b", "dino"]


MODEL_NAME = {"deit": "DeiT-S", "deit_b": "DeiT-B", "dino": "DINO ViT-S"}


NAME = {
    "R0": "received attention",
    "C3": "contribution weighting",
    "C2": "probe relocation",
    "C1": "mean-map subtraction",
    "stand_in_no_high": "zero high-norm tokens",
    "stand_in_blur": "matched blur",
    "stand_in_centre": "centre-Gaussian subtraction",
    "centre": "centre Gaussian",
    "mean_map": "mean map",
    "random": "random",
    "gae": "GAE",
    "register": "+ test-time register",
    "mita": "top-$k$ training (MiTA)",
    "R_all": "all three corrections (order A)",
}


METRICS = ["srg", "enrichment", "clickme"]


METRIC_LONG = {
    "srg": "deletion faithfulness (SRG)",
    "enrichment": "same-ring object enrichment",
    "clickme": "image-specific ClickMe alignment",
}


METRIC_LINES = {
    "srg": ["Deletion faithfulness", "(SRG)"],
    "enrichment": ["Same-ring object", "enrichment"],
    "clickme": ["Image-specific", "ClickMe alignment"],
}


LETTER = {"higher": "h", "comparable": "c", "lower": "l"}


INTERVALS = ("95% paired bootstrap intervals over images (1,000 resamples, seed 0; 2.5th and "
             "97.5th percentiles).")


LEGEND = ("Verdict letters: h = higher, c = comparable, l = lower (higher or lower only if the "
          "interval excludes 0 and |difference| exceeds the margin, 0.05 for deletion "
          "faithfulness (SRG) and 0.02 otherwise).")


ORDERS = ("Order A = contribution weighting, then probe relocation, then mean-map subtraction "
          "(pre-registered); order B = contribution weighting, then mean-map subtraction, then "
          "probe relocation.")


def cap(s):
    return s[:1].upper() + s[1:] if s[:1].isalpha() else s


def num(x, d=3):
    """Number with d decimals and a LaTeX minus sign; '---' for missing values."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    s = f"{float(x):.{d}f}"
    if s.startswith("-") and float(s) == 0.0:   # rounds to zero: print without a sign
        return s[1:]
    return "$-$" + s[1:] if s.startswith("-") else s


def ci(m, lo, hi, d=3):
    """Mean with its interval in small type."""
    return f"{num(m, d)} {{\\scriptsize [{num(lo, d)}, {num(hi, d)}]}}"


def letter(v):
    return r"\textit{%s}" % LETTER.get(str(v), "?")


def n_fmt(n):
    return f"{int(n):,}"


def n_text(pairs):
    """pairs: list of (model, n). '12,150 (DeiT-S, DeiT-B) and 12,144 (DINO ViT-S)' or '17,408'."""
    groups = {}
    for m, n in pairs:
        groups.setdefault(int(n), []).append(MODEL_NAME[m])
    if len(groups) == 1:
        return n_fmt(next(iter(groups)))
    parts = [f"{n_fmt(n)} ({', '.join(ms)})" for n, ms in groups.items()]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def mc(lines, align="l"):
    """One cell on several lines (makecell); a single line is returned as is."""
    if isinstance(lines, str):
        return lines
    if len(lines) == 1:
        return lines[0]
    return r"\makecell[%s]{%s}" % (align, r" \\ ".join(lines))


def hc(lines):
    """Centred column header on several lines."""
    return mc(lines, "c")


def metric_header(me):
    return hc(METRIC_LINES[me])


def group(title, ncol):
    return r"\multicolumn{%d}{l}{\textit{%s}} \\" % (ncol, title)


def tabular(spec, header_lines, rows, eol=r"\\"):
    """header_lines: list of LaTeX lines (with \\\\ or rules); rows: list of lists, the string
    'MID' for a \\midrule, or any other string (written as is)."""
    out = [r"\begin{tabular}{%s}" % spec, r"\toprule"]
    out += header_lines
    out.append(r"\midrule")
    for i, r in enumerate(rows):
        if isinstance(r, str):
            out.append(r"\midrule" if r == "MID" else r)
        else:
            out.append(" & ".join(r) + " " + eol)
            nxt = rows[i + 1] if i + 1 < len(rows) else None
            if isinstance(nxt, list) and any(r"\makecell[l]" in c for c in r + nxt):
                out.append(r"\addlinespace[2pt]")   # air between rows with two-line cells
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def emit(comments, body, size=r"\footnotesize", colsep="4pt"):
    """comments: list of (key, text); written as wrapped % lines above the tabular(s)."""
    out = []
    for key, text in comments:
        lead = f"% {key}: " if key else "% "
        out += textwrap.wrap(text, width=100, initial_indent=lead, subsequent_indent="%   ",
                             break_on_hyphens=False, break_long_words=False)
    if size:
        out.append(size)
    if colsep:
        out.append(r"\setlength{\tabcolsep}{%s}" % colsep)
    out.append(body)
    return "\n".join(out) + "\n"


def std_comments(title, data, sources, caption, intervals=INTERVALS, legend=False, extra=(),
                 makecell=True):
    c = [("", title)]
    if data:
        c.append(("Data", data))
    if intervals:
        c.append(("Intervals", intervals))
    if legend:
        c.append(("Legend", LEGEND))
    c += list(extra)
    pk = r"\usepackage{booktabs}" + (r" and \usepackage{makecell}" if makecell else "")
    c.append(("Source", "Generated by scripts/make_latex_tables.py from "
              + ", ".join("results/" + s for s in sources) + f". Needs {pk}. Caption and "
              "label are added in the paper; use the file inside a table environment."))
    c.append(("Suggested caption", caption))
    return c


class Results:
    def __init__(self, root):
        self.root = root
        self._cache = {}

    def __call__(self, rel):
        if rel not in self._cache:
            self._cache[rel] = pd.read_csv(os.path.join(self.root, rel))
        return self._cache[rel]


def one(df, **kw):
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    if len(sel) != 1:
        raise KeyError(f"expected one row for {kw}, found {len(sel)}")
    return sel.iloc[0]


DATA_WHERE = {"srg": "ImageNet-S images", "enrichment": "ImageNet-S images",
              "clickme": "ClickMe images"}


def data_text(df, metrics, **kw):
    """'Deletion faithfulness (SRG): 2,000 ImageNet-S images; ...' with n from df."""
    parts = [f"{METRIC_LONG[me]}: {ns_of(df, me, **kw)} {DATA_WHERE[me]}" for me in metrics]
    return cap("; ".join(parts)) + "."


def ns_of(df, metric, **kw):
    """'n' text of one metric over the three models (first matching row per model)."""
    pairs = []
    for m in MODELS:
        sel = df[(df["model"] == m) & (df["metric"] == metric)]
        for k, v in kw.items():
            sel = sel[sel[k] == v]
        if len(sel):
            pairs.append((m, sel["n"].iloc[0]))
    return n_text(pairs)


DIFF_ROWS = [
    ("C1", "R0", "correction - R0", False),
    ("C2", "R0", "correction - R0", False),
    ("C3", "R0", "correction - R0", False),
    ("R_all", "R0", "correction - R0", True),
    ("C1", "stand_in_centre", "correction - stand-in", True),
    ("C2", "stand_in_blur", "correction - stand-in", True),
    ("C3", "stand_in_no_high", "correction - stand-in", True),
    ("stand_in_centre", "R0", "stand-in - R0", False),
    ("stand_in_blur", "R0", "stand-in - R0", False),
    ("stand_in_no_high", "R0", "stand-in - R0", False),
]


def t_paired_differences(R):
    ct = R("correct_tests.csv")
    rows, prev_test = [], None
    for a, b, test, primary in DIFF_ROWS:
        if prev_test is not None:
            rows.append("MID" if test != prev_test else r"\addlinespace[3pt]")
        prev_test = test
        title = cap(NAME[a]) + " $-$ " + NAME[b] + (r"$^\dagger$" if primary else "")
        rows.append(group(title, 4))
        for m in MODELS:
            cells = [MODEL_NAME[m]]
            for me in METRICS:
                r = one(ct, a=a, b=b, model=m, metric=me, test=test)
                cells.append(ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows.append(cells)
    header = ["Model & " + " & ".join(metric_header(me) for me in METRICS) + r" \\"]
    com = std_comments(
        "Paired differences, layers 9-12, all three models: each correction and each stand-in "
        "against received attention, and each correction against its stand-in; mean with "
        "interval and verdict letter. Dagger = pre-registered primary comparison. " + ORDERS,
        data_text(ct, METRICS, a="C3", b="R0", test="correction - R0"),
        ["correct_tests.csv"],
        r"Paired differences, layers 9--12: mean with 95\% paired-bootstrap interval and verdict "
        r"(\textit{h} = higher, \textit{c} = comparable, \textit{l} = lower; higher or lower only "
        r"if the interval excludes 0 and $|$difference$|$ exceeds the margin, 0.05 for deletion "
        r"faithfulness and 0.02 otherwise). $^\dagger$ = pre-registered primary comparison. "
        r"Order A: contribution weighting, then probe relocation, then mean-map subtraction. "
        r"Deletion faithfulness: 2,000 ImageNet-S images; enrichment: ImageNet-S; ClickMe: "
        r"17,408 images.",
        legend=True)
    return emit(com, tabular("lccc", header, rows), size=r"\footnotesize", colsep="4pt")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=root / 'results')
    parser.add_argument('--out', type=Path, default=root / 'build/rebuilt/paper/tables')
    args = parser.parse_args()
    target = args.out / 'appendix' / 'paired_differences.tex'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(t_paired_differences(Results(str(args.results))), encoding='utf-8')
    print('written to', target)


if __name__ == '__main__':
    main()
