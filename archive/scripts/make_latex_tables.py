"""LaTeX tables of the paper, built from the summary tables in results/.

  python scripts/make_latex_tables.py [--results results] [--out paper/tables]

Writes <out>/table1.tex (main results: enrichment / ClickMe, layers 9-12), <out>/table1_full.tex (three metrics with
95% intervals, all references, both orders of the three corrections) and one file per appendix table in
<out>/appendix/. Each file holds only the tabular(s), with the font size and column spacing it needs to fit the
NeurIPS 2025 text width (5.5 in), and starts with comment lines that say what the table shows, the data, the
intervals, the verdict-letter legend, the source CSV files and a suggested caption. Caption, label and the table
float are added in the paper. LaTeX needs booktabs, makecell, amsmath and amssymb.

Number format: 3 decimals for scores and differences, agreement rates in % with 1 decimal, $-$ for minus signs.
Verdict words and "holds" flags are taken from the CSV files as stored (they are not recomputed here). No internal code
(R0, C1-C3, M1-M3, P1-P7, D2, ...) is printed in the table text; the names follow paper/naming.md.
"""

import argparse
import math
import os
import re
import textwrap
from pathlib import Path

import pandas as pd

MODELS = ["deit", "deit_b", "dino"]
MODEL_NAME = {"deit": "DeiT-S", "deit_b": "DeiT-B", "dino": "DINO ViT-S"}

# Paper names of the readings (lower case; cap() for the start of a cell).
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
    "R_all_alt": "all three corrections (order B)",
    "D2": "probability discount",
    "D2_shuffle": "shuffled probability discount",
    "C3D2": "contribution weighting + probability discount",
    "C2MLP": "probe relocation (MLP probe)",
    "C2conf": "probe relocation (confident tokens only)",
    "C1C3": "contribution weighting + mean-map subtraction",
    "C2C3": "contribution weighting + probe relocation",
    "C1C2": "probe relocation + mean-map subtraction",
    "C3v": "value-norm weighting",
    "C3shuffle": "shuffled contribution weights",
    "C3headmean": "head-mean contribution weights",
}

METRICS = ["srg", "enrichment", "clickme"]
METRIC_LONG = {
    "srg": "deletion faithfulness (SRG)",
    "enrichment": "same-ring object enrichment",
    "clickme": "image-specific ClickMe alignment",
}
# Two-line forms for table cells and column headers.
METRIC_LINES = {
    "srg": ["Deletion faithfulness", "(SRG)"],
    "enrichment": ["Same-ring object", "enrichment"],
    "clickme": ["Image-specific", "ClickMe alignment"],
}
METRIC_SHORT = {"srg": "SRG", "enrichment": "Enrichment", "clickme": "ClickMe"}  # narrow headers
LETTER = {"higher": "h", "comparable": "c", "lower": "l"}

# Comment-header texts.
INTERVALS = ("95% paired bootstrap intervals over images (1,000 resamples, seed 0; 2.5th and "
             "97.5th percentiles).")
INTERVALS_PLAIN = ("95% bootstrap intervals over images (1,000 resamples, seed 0; paired over "
                   "images for differences).")
LEGEND = ("Verdict letters: h = higher, c = comparable, l = lower (higher or lower only if the "
          "interval excludes 0 and |difference| exceeds the margin, 0.05 for deletion "
          "faithfulness (SRG) and 0.02 otherwise).")
ORDERS = ("Order A = contribution weighting, then probe relocation, then mean-map subtraction "
          "(pre-registered); order B = contribution weighting, then mean-map subtraction, then "
          "probe relocation.")
DATA_3 = ("Deletion faithfulness (SRG): 2,000 ImageNet-S images; same-ring object enrichment: "
          "ImageNet-S; image-specific ClickMe alignment: 17,408 ClickMe images.")
DATA_2 = ("Same-ring object enrichment: ImageNet-S; image-specific ClickMe alignment: 17,408 "
          "ClickMe images.")


# --------------------------------------------------------------------------- formatting

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
    """Mean with its interval in small type (as in table1_full.tex)."""
    return f"{num(m, d)} {{\\scriptsize [{num(lo, d)}, {num(hi, d)}]}}"


def pct(x):
    return num(100.0 * x, 1)


def letter(v):
    return r"\textit{%s}" % LETTER.get(str(v), "?")


def ceil_to(x, d=3):
    f = 10 ** d
    return math.ceil(x * f - 1e-9) / f


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


def metric_cell(me):
    return mc(METRIC_LINES[me])


def metric_header(me):
    return hc(METRIC_LINES[me])


def diff_lines(a, b):
    """'A $-$ B' with the paper names, on two lines."""
    return [cap(NAME[a]), "$-$ " + NAME[b]]


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


def sub(title, first=False):
    """Small italic heading line above a sub-tabular."""
    return ("" if first else r"\par\medskip") + r"\textit{%s}\par\smallskip" % title


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


# --------------------------------------------------------------------------- data access

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


def maybe(df, **kw):
    try:
        return one(df, **kw)
    except KeyError:
        return None


def max_half(rows):
    return max((r["high"] - r["low"]) / 2 for r in rows)


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


# --------------------------------------------------------------------------- RQ1 extensions

def t_single_layer(R):
    sl, ks = R("single_layer.csv"), R("k_sweep.csv")
    sl = sl.copy()
    sl["layer"] = sl["layer"].astype(int)
    rows, used = [], []
    for layer in range(1, 13):
        cells = [str(layer)]
        for m in MODELS:
            for test in ("restricted to reading sets", "attention off"):
                r = one(sl, model=m, layer=layer, test=test)
                used.append(r)
                cells.append(pct(r["mean"]))
        rows.append(cells)
    rows.append("MID")
    cells = ["all 12"]
    for m in MODELS:
        r = one(ks, model=m, sets="reading", k=16)       # execution test, every layer at once
        off = one(sl, model=m, layer=0, test="attention off")  # layer 0 = all 12 layers off
        used += [r, off]
        cells += [pct(r["mean"]), pct(off["mean"])]
    rows.append(cells)
    n = int(sl["n"].iloc[0])
    hw = ceil_to(100 * max_half(used), 1)
    header = [
        r" & \multicolumn{2}{c}{DeiT-S} & \multicolumn{2}{c}{DeiT-B} & "
        r"\multicolumn{2}{c}{DINO ViT-S} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Layer & restricted & removed & restricted & removed & restricted & removed \\",
    ]
    com = std_comments(
        "One layer at a time: top-1 agreement (%) with the unmodified model when only that "
        "layer's queries are restricted to their region's reading sets (k = 16; 'restricted') or "
        "only that layer's attention branch is removed ('removed'). Row 'all 12': every layer at "
        "once (restricted = the k = 16 execution test; removed = no attention in any layer).",
        f"{n_fmt(n)} ImageNet validation images. Removal runs used resize 256 / crop 0.875 for "
        f"every model.",
        ["single_layer.csv", "k_sweep.csv"],
        f"One layer at a time: top-1 agreement (\\%) with the unmodified model on {n_fmt(n)} "
        f"ImageNet validation images when only that layer's queries are restricted to their "
        f"region's reading sets ($k = 16$) or only that layer's attention branch is removed. Row "
        f"``all 12'': every layer at once (restricted = the $k = 16$ execution test; removed = "
        f"no attention in any layer). 95\\% bootstrap intervals have half-width $\\le$ "
        f"{hw:.1f} points.",
        intervals=f"95% bootstrap intervals over images (1,000 resamples) have half-width <= "
                  f"{hw:.1f} points (not shown).",
        makecell=False)
    return emit(com, tabular("lcccccc", header, rows), size=r"\small", colsep="4pt")


def t_head_vs_body(R):
    pr, ms = R("part_reliance.csv"), R("map_similarity.csv")
    rows_a = []
    for m in MODELS:
        r = one(pr, model=m)
        rows_a.append([MODEL_NAME[m], n_fmt(r["n"]), num(r["head"]), num(r["body"]),
                       ci(r["mean"], r["low"], r["high"]), r["verdict"],
                       num(r["head"] / r["body"], 2)])
    a = tabular("lrccccc", [
        r"Model & $n$ & Head drop & Body drop & Head $-$ body & Verdict & Head / body \\"], rows_a)
    rows_b = []
    for m in MODELS:
        r = one(ms, model=m)
        rows_b.append([MODEL_NAME[m], n_fmt(r["n"]), num(r["median"]), num(r["q25"]),
                       num(r["q75"]), "yes" if r["median"] > 0.5 else "no"])
    b = tabular("lrcccc", [
        r"Model & $n$ & Median & 25th pct. & 75th pct. & Median $> 0.5$ \\"], rows_b)
    com = std_comments(
        "Extensions of the execution question. (a) Head vs body patches: drop in log-probability "
        "(nats) of the original top-1 class after removing 4 head or 4 body patches from all "
        "layers (5 seeds), the paired difference head - body with its interval, the verdict by "
        "the earlier study's rule (higher if the lower bound > 0), and the ratio of the two "
        "means. (b) Per-image Spearman correlation between the top-k reading-set map (k = 16) and "
        "the full received-attention map (layers 9-12): median and quartiles (no interval).",
        "(a) PartImageNet (ImageNet training images with both a removable animal head and "
        "body), n per model in the table; (b) ImageNet-S, n in the table.",
        ["part_reliance.csv", "map_similarity.csv"],
        r"Extensions of the execution question. (a) Head vs body patches (PartImageNet, images "
        r"with both a removable animal head and body): drop in log-probability (nats) of the "
        r"original top-1 class after removing 4 head or 4 body patches from all layers (5 seeds), "
        r"paired difference with 95\% paired-bootstrap interval, verdict by the earlier study's "
        r"rule (higher if the lower bound $> 0$), and the ratio of the two means. These are "
        r"ImageNet training images. (b) Per-image Spearman correlation between the top-$k$ "
        r"reading-set map ($k = 16$) and the full received-attention map (layers 9--12, "
        r"ImageNet-S): median and quartiles (no interval).",
        intervals=INTERVALS + " Only for the head - body difference in (a).",
        makecell=False)
    body = "\n".join([sub("(a) Head vs body patches", first=True), a,
                      sub("(b) Top-$k$ reading-set map vs full received-attention map"), b])
    return emit(com, body, size=r"\small", colsep="4pt")


# --------------------------------------------------------------------------- pre-registration

PRED_NAME = {
    "P1": "Reading sets beat random sets (execution)",
    "P2": "Premises fail with depth (diagnosis)",
    "P3": "All three corrections beat received attention (main prediction)",
    "P4a": "Negative controls: contribution weighting on DINO ViT-S",
    "P4b": "Negative controls: probe relocation at layers 1--4",
    "P5": "Last-layer recovery (descriptive)",
    "P6": "Model-side variants not higher",
    "P7": "Primary comparisons (no direction)",
}


def _models_text(s):
    s = str(s)
    if s == "deit+register, mita":
        return r"DeiT-S + test-time register; top-$k$ training (MiTA)"
    parts = [p.strip() for p in s.split(",")]
    return ", ".join(MODEL_NAME.get(p, p) for p in parts)


_DETAIL_WORDS = {"srg": METRIC_LONG["srg"], "enrichment": METRIC_LONG["enrichment"],
                 "clickme": METRIC_LONG["clickme"]}
_STAND_IN = {"C1": "stand_in_centre", "C2": "stand_in_blur", "C3": "stand_in_no_high"}


def _detail_tex(s):
    s = str(s)
    # premise measures and corrections by their paper names
    s = s.replace("M2 deepens: accuracy L12 < L1",
                  "position-probe accuracy falls with depth: accuracy at layer 12 < layer 1")
    s = s.replace("M3: DINO high-norm share upper bound",
                  "share of received attention on high-norm tokens: upper bound")
    s = s.replace("M3: largest high-norm share in",
                  "share of received attention on high-norm tokens: largest in")
    s = re.sub(r"\b(C[123]) vs stand-in:",
               lambda m: f"{NAME[m.group(1)]} vs {NAME[_STAND_IN[m.group(1)]]}:", s)
    s = re.sub(r"\b(srg|enrichment|clickme)\b", lambda m: _DETAIL_WORDS[m.group(1)], s)
    s = s.replace("<=", r"$\le$").replace(" < ", " $<$ ")
    s = re.sub(r"(?<=\s)-(\d)", r"$-$\1", s)      # minus sign
    s = re.sub(r"(\d)-(\d)", r"\1--\2", s)         # number range
    return cap(s)


def t_predictions(R):
    p = R("predictions.csv")
    rows, prev = [], None
    for _, r in p.iterrows():
        code = str(r["prediction"]).split(" (")[0]
        if code != prev:
            if prev is not None:
                rows.append(r"\addlinespace[3pt]")
            rows.append(r"\multicolumn{3}{l}{\textit{%s}} \tabularnewline" % PRED_NAME[code])
        h = r["holds"]
        if isinstance(h, str):
            h = {"True": True, "False": False}.get(h, h)
        holds = "---" if (h is None or (isinstance(h, float) and math.isnan(h))) else (
            "yes" if bool(h) else "no")
        rows.append([r"\raggedright " + _models_text(r["model"]), holds,
                     r"\raggedright " + _detail_tex(r["detail"])])
        prev = code
    header = [r"Model(s) & Holds & Detail \tabularnewline"]
    com = std_comments(
        "Pre-registered predictions and their outcomes, one row per registered test as stored. "
        "'Primary comparisons (no direction)' lists each correction against its stand-in on the "
        "three metrics; no direction was predicted (holds = ---). 'All three corrections beat "
        "received attention' has 1 of 4 cells higher, which its pre-agreed reading scores as "
        "'partly holds'. 'Last-layer recovery (descriptive)' has no holds/fails row; it is "
        "reported in last_layer_recovery.tex.",
        "As registered for each prediction (decision rules in the appendix on pre-registration).",
        ["predictions.csv"],
        r"Pre-registered predictions and their outcomes (decision rules in Appendix~C). The "
        r"primary comparisons (each correction against its stand-in, three metrics) have no "
        r"predicted direction (holds = ---). ``All three corrections beat received attention'' "
        r"has 1 of 4 cells higher, which its pre-agreed reading scores as ``partly holds''. "
        r"Last-layer recovery is descriptive and is reported in its own table.",
        intervals=None, makecell=False)
    return emit(com, tabular(r"p{2.5cm}cp{9.6cm}", header, rows, eol=r"\tabularnewline"),
                size=r"\footnotesize", colsep="4pt")


def t_negative_controls(R):
    nc, rc = R("negative_controls.csv"), R("history/reproduction_check.csv")
    groups = [("P4a", [("dino", m) for m in METRICS],
               "Contribution weighting $-$ received attention, layers 9--12"),
              ("P4b", [(mo, me) for mo in MODELS for me in ("enrichment", "clickme")],
               "Probe relocation $-$ received attention, layers 1--4")]
    rows = []
    for gi, (test, cells_list, title) in enumerate(groups):
        if gi:
            rows.append(r"\addlinespace[3pt]")
        rows.append(group(title, 5))
        for mo, me in cells_list:
            r = maybe(nc, test=test, model=mo, metric=me)
            if r is None:
                continue
            rows.append([MODEL_NAME[mo], metric_cell(me), n_fmt(r["n"]),
                         ci(r["mean"], r["low"], r["high"], 4), r["verdict"]])
    a = tabular("llrcc", [r"Model & Metric & $n$ & Difference & Verdict \\"], rows)
    rows_b = []
    for mo in MODELS:
        for me in ("enrichment", "srg", "clickme"):
            r = maybe(rc, model=mo, metric=me)
            if r is None:
                continue
            rows_b.append([MODEL_NAME[mo] if me == "enrichment" else "", metric_cell(me),
                           n_fmt(r["n"]), num(r["new"], 4), num(r["old"], 4),
                           num(1000 * r["difference"], 3),
                           "yes" if str(r["holds"]) == "True" else "no"])
    b = tabular("llrcccc", [
        r"Model & Metric & $n$ & \makecell{Received\\attention} & "
        r"\makecell{Earlier full-\\attention run} & \makecell{Difference\\($\times 10^{-3}$)} & "
        r"\makecell{$|$Difference$|$\\$< 0.01$} \\"], rows_b)
    com = std_comments(
        "Checks. (a) Pre-registered negative controls: contribution weighting - received "
        "attention on DINO ViT-S (layers 9-12) and probe relocation - received attention at "
        "layers 1-4; differences (4 decimals) with intervals and verdicts (words). Deletion "
        "faithfulness (SRG) is not used for the second control by design. (b) Reproduction "
        "check: mean received attention (layers 9-12) of the new runs vs the earlier "
        "full-attention runs on the same images; registered criterion |difference| < 0.01.",
        "Deletion faithfulness (SRG): 2,000 ImageNet-S images; same-ring object enrichment: "
        "ImageNet-S; image-specific ClickMe alignment: 17,408 ClickMe images; n per row in the "
        "table.",
        ["negative_controls.csv", "history/reproduction_check.csv"],
        r"Checks. (a) Pre-registered negative controls: contribution weighting $-$ received "
        r"attention on DINO ViT-S (layers 9--12) and probe relocation $-$ received attention at "
        r"layers 1--4; differences (4 decimals) with 95\% paired-bootstrap intervals and verdicts "
        r"(margin 0.05 for deletion faithfulness, 0.02 otherwise); deletion faithfulness is not "
        r"used for the second control by design. (b) Reproduction check: mean received attention "
        r"(layers 9--12) of the new runs vs the earlier full-attention runs on the same images; "
        r"registered criterion $|$difference$|$ $< 0.01$.",
        intervals=INTERVALS + " Only in (a).",
        extra=[("Verdicts", "(a) uses the words higher / comparable / lower: interval excludes 0 "
                "and |difference| exceeds the margin (0.05 for SRG, 0.02 otherwise) for higher or "
                "lower.")])
    body = "\n".join([sub("(a) Negative controls", first=True), a,
                      sub("(b) Reproduction check"), b])
    return emit(com, body, size=r"\footnotesize", colsep="4pt")


def t_sanity(R):
    s = R("sanity.csv")
    s = s[s["level"] == "embed"]
    names = ["R0", "C1", "C2", "C3", "R_all", "centre"]
    rows = []
    for key in names:
        label = "Centre Gaussian (reference)" if key == "centre" else cap(NAME[key])
        cells = [label]
        for m in MODELS:
            r = one(s, model=m, method=key)
            ok = str(r["passes"]) == "True"
            cells.append(f"{num(r['median'])} & {'passes' if ok else 'fails'}")
        rows.append(cells)
        if key == "R_all":
            rows.append("MID")
    header = [
        r" & \multicolumn{2}{c}{DeiT-S} & \multicolumn{2}{c}{DeiT-B} & "
        r"\multicolumn{2}{c}{DINO ViT-S} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Reading & median & & median & & median & \\",
    ]
    com = std_comments(
        "Model-randomisation sanity check: median per-image Spearman correlation between the map "
        "of the fully randomised model (all stages re-initialised down to the patch embedding) "
        "and the map of the trained model (layers 9-12). A reading passes if the median is below "
        "0.5. The centre Gaussian does not depend on the model and is the designed failing "
        "example. " + ORDERS,
        "1,000 ImageNet-S images.",
        ["sanity.csv (level = embed)"],
        r"Model-randomisation sanity check on 1,000 ImageNet-S images: median per-image Spearman "
        r"correlation between the map of the fully randomised model (all stages re-initialised "
        r"down to the patch embedding) and the map of the trained model (layers 9--12). A reading "
        r"passes if the median is below 0.5. Order A: contribution weighting, then probe "
        r"relocation, then mean-map subtraction. The centre Gaussian does not depend on the model "
        r"and is the designed failing example.",
        intervals="None (medians).", makecell=False)
    return emit(com, tabular("lcccccc", header, rows), size=r"\small", colsep="4pt")


# --------------------------------------------------------------------------- full results

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


def t_model_side_variants(R):
    va, ct = R("variants.csv"), R("correct_tests.csv")
    rows = []
    for key, title in (("register", "DeiT-S + test-time register"),
                       ("mita", "DeiT-S with top-$k$ training (MiTA)")):
        rows.append(group(title, 6))
        for me in METRICS:
            v = maybe(va, variant=key, metric=me, test="value")
            d = maybe(va, variant=key, metric=me, test="variant - DeiT-S R0")
            if d is None:
                rows.append([metric_cell(me), "---", "n/a", "n/a", "", ""])
                continue
            p6 = d["holds_p6"]
            p6 = "not higher" if str(p6) == "True" else ("---" if pd.isna(p6) else "higher")
            rows.append([metric_cell(me), n_fmt(d["n"]), num(v["mean"]),
                         ci(d["mean"], d["low"], d["high"]), d["verdict"], p6])
        rows.append("MID")
    rows.append(group("For comparison: contribution weighting $-$ received attention (DeiT-S)",
                      6))
    for me in METRICS:
        r = one(ct, a="C3", b="R0", model="deit", metric=me, test="correction - R0")
        rows.append([metric_cell(me), n_fmt(r["n"]), "",
                     ci(r["mean"], r["low"], r["high"]), r["verdict"], ""])
    header = [r"Metric & $n$ & \makecell{Received\\attention} & "
              r"\makecell{Difference vs\\received attention\\of DeiT-S} & Verdict & "
              r"\makecell{Pre-registered\\check} \\"]
    com = std_comments(
        "Changing the model (DeiT-S, layers 9-12): received attention of DeiT-S with a test-time "
        "register and of DeiT-S with top-k training (MiTA), paired with received attention of "
        "DeiT-S by image index; mean difference with interval, verdict (words), and the "
        "pre-registered check 'Model-side variants not higher', part (2): not higher on "
        "enrichment and ClickMe. Top-k training has no deletion score (no token removal for "
        "MiTA) and uses its own crop. Last rows: contribution weighting - received attention on "
        "DeiT-S for comparison.",
        "Deletion faithfulness (SRG): 2,000 ImageNet-S images; same-ring object enrichment: "
        "ImageNet-S; image-specific ClickMe alignment: 17,408 ClickMe images; n per row in the "
        "table (top-k training: own crop 0.875, hence a different enrichment n).",
        ["variants.csv", "correct_tests.csv"],
        r"Changing the model (DeiT-S, layers 9--12): received attention of DeiT-S with a "
        r"test-time register and with top-$k$ training (MiTA), paired with received attention of "
        r"DeiT-S by image index; mean difference with 95\% paired-bootstrap interval, verdict, and "
        r"the pre-registered check that model-side variants are not higher on enrichment and "
        r"ClickMe. Top-$k$ training has no deletion score and uses its own crop. Last rows: "
        r"contribution weighting $-$ received attention on DeiT-S, for comparison.",
        extra=[("Verdicts", "Words higher / comparable / lower: higher or lower only if the "
                "interval excludes 0 and |difference| exceeds the margin (0.05 for SRG, 0.02 "
                "otherwise).")])
    return emit(com, tabular("lrcccc", header, rows), size=r"\footnotesize", colsep="4pt")


def t_combination_orders(R):
    od = R("order.csv")
    rows, same = [], 0
    for mi, me in enumerate(METRICS):
        if mi:
            rows.append(r"\addlinespace[3pt]")
        rows.append(group(cap(METRIC_LONG[me]), 6))
        for m in MODELS:
            v = {k: one(od, model=m, metric=me, method=k, test="value")
                 for k in ("R0", "R_all", "R_all_alt")}
            d1 = one(od, model=m, metric=me, test="order", a="R_all", b="R0")
            d2 = one(od, model=m, metric=me, test="order", a="R_all_alt", b="R_all")
            d3 = one(od, model=m, metric=me, test="order", a="R_all_alt", b="R0")
            same += int(d1["verdict"] == d3["verdict"])
            rows.append([MODEL_NAME[m], num(v["R0"]["mean"]), num(v["R_all"]["mean"]),
                         num(v["R_all_alt"]["mean"]),
                         ci(d1["mean"], d1["low"], d1["high"]) + r"\," + letter(d1["verdict"]),
                         ci(d2["mean"], d2["low"], d2["high"]) + r"\," + letter(d2["verdict"])])
    total = 3 * len(METRICS)
    vals = od[od["test"] == "value"]
    hw = ceil_to(max_half([r for _, r in vals.iterrows()]))
    header = [r"Model & \makecell{Received\\attention} & \makecell{All three\\corrections\\"
              r"(order A)} & \makecell{All three\\corrections\\(order B)} & "
              r"\makecell{All three corrections\\(order A) $-$\\received attention} & "
              r"\makecell{All three corrections\\(order B) $-$ all three\\corrections (order A)} \\"]
    com = std_comments(
        "Combination orders (layers 9-12): values of received attention and of all three "
        "corrections in order A (pre-registered) and order B (descriptive check), and paired "
        "differences with intervals and verdict letters. " + ORDERS + f" The two orders get the "
        f"same verdict against received attention in {same} of {total} cells ('robust' required "
        f"all {total}).",
        data_text(od, METRICS, method="R0", test="value"),
        ["order.csv"],
        r"Combination orders (layers 9--12). Order A (pre-registered): contribution weighting, "
        r"then probe relocation, then mean-map subtraction; order B (descriptive check): "
        r"contribution weighting, then mean-map subtraction, then probe relocation. Values (95\% "
        r"bootstrap intervals $\le \pm$" + f"{hw:.3f}" + r") and paired differences with 95\% "
        r"paired-bootstrap intervals and verdicts (\textit{h}/\textit{c}/\textit{l} = higher / "
        r"comparable / lower). The two orders get the same verdict against received attention in "
        + f"{same} of {total} cells (``robust'' required all {total}).",
        intervals=INTERVALS + f" Intervals of the values (not shown) are <= +-{hw:.3f}.",
        legend=True)
    return emit(com, tabular("lccccc", header, rows), size=r"\footnotesize", colsep="3.5pt")


def t_last_layer_recovery(R):
    ll = R("last_layer.csv")
    readings = ["C3", "C1", "C2", "R_all"]
    rows = []
    for mi, m in enumerate(("deit", "deit_b")):
        if mi:
            rows.append("MID")
        rows.append(group(MODEL_NAME[m], 4))
        best = [mc(["Best received-attention", "layer (8--11)"])]
        drop = ["Drop to layer 12"]
        for me in METRICS:
            r = one(ll, model=m, metric=me, reading="C3")
            best.append(str(r["best_layer"]).replace("R0_l", "layer "))
            drop.append(num(r["drop"]))
        rows += [best, drop]
        for key in readings:
            lines = {"R_all": ["Recovered by all three", "corrections (order A)"]}.get(
                key, ["Recovered by", NAME[key]])
            cells = [mc(lines)]
            for me in METRICS:
                r = one(ll, model=m, metric=me, reading=key)
                cells.append(ci(r["recovered"], r["low"], r["high"]))
            rows.append(cells)
    header = [" & " + " & ".join(metric_header(me) for me in METRICS) + r" \\"]
    com = std_comments(
        "Last-layer recovery (pre-registered as descriptive; supervised models). Drop = received "
        "attention at the best of layers 8-11 minus received attention at layer 12. Share "
        "recovered = (corrected layer-12 map - received attention at layer 12) / drop: 1 = full "
        "recovery, negative = worse than received attention at layer 12. " + ORDERS,
        DATA_3,
        ["last_layer.csv"],
        r"Last-layer recovery (descriptive; supervised models). Drop = received attention at the "
        r"best of layers 8--11 minus received attention at layer 12. Share recovered = (corrected "
        r"layer-12 map $-$ received attention at layer 12) / drop: 1 = full recovery, negative = "
        r"worse than received attention at layer 12. 95\% paired-bootstrap intervals of the ratio "
        r"of means. Order A: contribution weighting, then probe relocation, then mean-map "
        r"subtraction. Deletion faithfulness: 2,000 ImageNet-S images; enrichment: ImageNet-S; "
        r"ClickMe: 17,408 images.",
        intervals=INTERVALS + " Intervals of the ratio of means.")
    return emit(com, tabular("lccc", header, rows), size=r"\footnotesize", colsep="4pt")


def t_contribution_checks(R):
    mech, dose, gap = (R("supplementary/c3_mechanism.csv"), R("supplementary/dose.csv"),
                       R("supplementary/gae_gap.csv"))
    rows = []
    for b in ("C3shuffle", "C3headmean", "C3v"):
        if b != "C3shuffle":
            rows.append(r"\addlinespace[3pt]")
        rows.append(group(cap(NAME["C3"]) + " $-$ " + NAME[b], 3))
        for m in MODELS:
            cells = [MODEL_NAME[m]]
            for me in ("enrichment", "clickme"):
                r = one(mech, a="C3", b=b, model=m, metric=me)
                cells.append(ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows.append(cells)
    a = tabular("lcc", [r"Model & " + metric_header("enrichment") + " & "
                         + metric_header("clickme") + r" \\"], rows)
    rows_b = []
    for m in ("deit", "deit_b"):
        for me in ("enrichment", "clickme"):
            for imp, bb in (("C3 - R0", "R0"), ("C3 - stand_in_no_high", "stand_in_no_high")):
                r = one(dose, model=m, metric=me, improvement=imp)
                rows_b.append([MODEL_NAME[m], metric_cell(me), mc(diff_lines("C3", bb)),
                               n_fmt(r["n"]), num(r["mean_dose"]),
                               ci(r["spearman"], r["low"], r["high"])])
    b = tabular("lllrcc", [
        r"Model & Metric & Improvement & $n$ & \makecell{Mean share on\\high-norm tokens} & "
        r"\makecell{Spearman(share,\\improvement)} \\"], rows_b)
    rows_c = []
    for m in ("deit", "deit_b"):
        for me in METRICS:
            r = one(gap, model=m, metric=me)
            rows_c.append([MODEL_NAME[m], metric_cell(me), n_fmt(r["n"]), num(r["c3_gain"]),
                           num(r["gae_gap"]), ci(r["share_closed"], r["low"], r["high"])])
    c = tabular("llrccc", [
        r"Model & Metric & $n$ & \makecell{Contribution weighting\\$-$ received attention} & "
        r"\makecell{GAE $-$\\received attention} & Share closed \\"], rows_c)
    com = std_comments(
        "Exploratory checks of contribution weighting (layers 9-12). (a) Mechanism: contribution "
        "weighting against versions whose weights ||W_O v_j|| are shuffled among the patch tokens "
        "of each head (shuffled contribution weights, 5 seeds), replaced by the head mean "
        "(head-mean contribution weights), or replaced by ||v_j|| (value-norm weighting). "
        "(b) Dose-response: across images, Spearman correlation between the share of received "
        "attention on high-norm tokens and the image's improvement. (c) Share of the gap between "
        "received attention and GAE closed by contribution weighting, (contribution weighting - "
        "received attention) / (GAE - received attention).",
        DATA_3 + " n per row in (b) and (c).",
        ["supplementary/c3_mechanism.csv", "supplementary/dose.csv",
         "supplementary/gae_gap.csv"],
        r"Exploratory checks of contribution weighting (layers 9--12). (a) Mechanism: contribution "
        r"weighting against its weights $\lVert W_O v_j\rVert$ shuffled among the patch tokens of "
        r"each head (5 seeds), replaced by the head mean, or replaced by $\lVert v_j\rVert$ "
        r"(value-norm weighting); verdicts \textit{h}/\textit{c}/\textit{l} = higher / comparable "
        r"/ lower. (b) Dose--response: across images, Spearman correlation between the share of "
        r"received attention on high-norm tokens and the image's improvement. (c) Share of the "
        r"gap between received attention and GAE closed by contribution weighting. All intervals "
        r"are 95\% (paired) bootstrap intervals over images.",
        intervals=INTERVALS_PLAIN, legend=True)
    body = "\n".join([sub("(a) Mechanism", first=True), a, sub("(b) Dose--response"), b,
                      sub("(c) Share of the GAE gap closed"), c])
    return emit(com, body, size=r"\footnotesize", colsep="3pt")


def t_per_layer_gains(R):
    dl = R("supplementary/dose_layers.csv")
    dl = dl.copy()
    dl["layer"] = dl["layer"].astype(str)
    layers = ["8", "9", "10", "11", "12", "9-12"]
    rows, used = [], []
    for mi, m in enumerate(("deit", "deit_b")):
        if mi:
            rows.append("MID")
        for L in layers:
            cells = [MODEL_NAME[m] if L == layers[0] else "",
                     "9--12 summed" if L == "9-12" else L]
            for me in ("enrichment", "clickme"):
                for test in ("C3 - R0", "stand_in_no_high - R0"):
                    r = one(dl, model=m, layer=L, metric=me, test=test)
                    used.append(r)
                    cells.append(num(r["mean"]) + r"\," + letter(r["verdict"]))
            rows.append(cells)
    hw = ceil_to(max_half(used))
    h1 = hc(["Contribution", "weighting $-$", "received", "attention"])
    h2 = hc(["Zero high-norm", "tokens $-$", "received", "attention"])
    header = [
        r" & & \multicolumn{2}{c}{Same-ring object enrichment} & "
        r"\multicolumn{2}{c}{Image-specific ClickMe alignment} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        f"Model & Layer & {h1} & {h2} & {h1} & {h2} \\\\",
    ]
    com = std_comments(
        "Per-layer gains (exploratory, supervised models): gain over received attention of "
        "contribution weighting and of zeroing high-norm tokens when only that layer is read, "
        "and for layers 9-12 summed; mean difference and verdict letter.",
        data_text(dl, ["enrichment", "clickme"], test="C3 - R0"),
        ["supplementary/dose_layers.csv"],
        r"Per-layer gains (exploratory): gain over received attention of contribution weighting "
        r"and of zeroing high-norm tokens when only that layer is read, and for layers 9--12 "
        r"summed. Enrichment: ImageNet-S; ClickMe: 17,408 images. 95\% paired-bootstrap intervals "
        r"have half-width $\le$ " + f"{hw:.3f}" + r". Verdicts \textit{h}/\textit{c}/\textit{l} "
        r"= higher / comparable / lower.",
        intervals=INTERVALS + f" Not shown; half-width <= {hw:.3f}.",
        legend=True)
    return emit(com, tabular("llcccc", header, rows), size=r"\footnotesize", colsep="5pt")


def t_pairwise_combinations(R):
    cb = R("supplementary/combinations.csv")
    comps = [("C1C3", "R0"), ("C2C3", "R0"), ("C1C2", "R0"), ("C1C3", "C3"), ("C2C3", "C3")]
    rows = []
    for ci_, (a, b) in enumerate(comps):
        if ci_:
            rows.append(r"\addlinespace[3pt]")
        rows.append(group(cap(NAME[a]) + " $-$ " + NAME[b], 4))
        for m in MODELS:
            cells = [MODEL_NAME[m]]
            for me in METRICS:
                r = maybe(cb, a=a, b=b, model=m, metric=me)
                cells.append("---" if r is None else
                             ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows.append(cells)
    header = ["Model & " + " & ".join(metric_header(me) for me in METRICS) + r" \\"]
    com = std_comments(
        "Pairwise combinations (exploratory), layers 9-12, applied in the order contribution "
        "weighting, then probe relocation, then mean-map subtraction, with the absent correction "
        "skipped; paired differences with intervals and verdict letters. Deletion faithfulness "
        "(SRG) was computed for contribution weighting + mean-map subtraction only (--- "
        "otherwise).",
        data_text(cb, METRICS, a="C1C3", b="R0"),
        ["supplementary/combinations.csv"],
        r"Pairwise combinations (exploratory), layers 9--12, applied in the order contribution "
        r"weighting, then probe relocation, then mean-map subtraction, with the absent correction "
        r"skipped. Paired differences with 95\% paired-bootstrap intervals and verdicts "
        r"(\textit{h}/\textit{c}/\textit{l} = higher / comparable / lower). Deletion faithfulness "
        r"was computed for contribution weighting + mean-map subtraction only (2,000 ImageNet-S "
        r"images); enrichment: ImageNet-S; ClickMe: 17,408 images.",
        legend=True)
    return emit(com, tabular("lccc", header, rows), size=r"\footnotesize", colsep="4pt")


# --------------------------------------------------------------------------- probe relocation

def t_probe_errors(R):
    er, mp = R("c2_analysis/errors.csv"), R("c2_analysis/mean_prediction.csv")
    er = er.copy()
    er["layer"] = er["layer"].astype(int)
    rows, used = [], []
    for mi, m in enumerate(MODELS):
        if mi:
            rows.append("MID")
        for L in (9, 10, 11, 12):
            def g(src, q):
                r = one(er, model=m, layer=L, source=src, metric=q)
                if src == "probe":
                    used.append(r)
                return r["mean"]
            rows.append([MODEL_NAME[m] if L == 9 else "", str(L), num(g("probe", "exact")),
                         num(g("probe", "top5")),
                         f"{num(g('probe', 'near3x3'))} / {num(g('stand_in_blur', 'near3x3'))}",
                         f"{num(g('probe', 'far_share'))} / {num(g('stand_in_blur', 'far_share'))}",
                         f"{num(g('probe', 'distance'))} / {num(g('stand_in_blur', 'distance'))}"])
    hw = ceil_to(max_half(used))
    a = tabular("llccccc", [
        r"Model & Layer & Exact & Top-5 & $P$(3$\times$3) & \makecell{Far share\\of errors} & "
        r"\makecell{Expected\\distance} \\"], rows)
    mp = mp.copy()
    mp["layer"] = mp["layer"].astype(int)
    rows_b = []
    for mi, m in enumerate(MODELS):
        if mi:
            rows_b.append("MID")
        for L in (9, 10, 11, 12):
            mx = one(mp, model=m, layer=L, quantity="max_ratio")
            pc = one(mp, model=m, layer=L, quantity="pearson_mean_map")
            rows_b.append([MODEL_NAME[m] if L == 9 else "", str(L),
                           ci(mx["mean"], mx["low"], mx["high"]),
                           ci(pc["mean"], pc["low"], pc["high"])])
    b = tabular("llcc", [
        r"Model & Layer & Peak / uniform & Pearson with mean map \\"], rows_b)
    n = int(er["n"].iloc[0])
    com = std_comments(
        "Position-probe errors, layers 9-12 (exploratory). (a) Exact and top-5 accuracy; "
        "probability within the 3x3 neighbourhood of the true patch; far share of errors = "
        "probability more than 3 patches away / probability off the true patch; expected "
        "distance (patches). Pairs give probe / matched blur (exact and top-5 of the blur are 1 "
        "by construction). (b) The probe's mean prediction over all tokens: its peak divided by "
        "the uniform value 1/196, and its Pearson correlation with the mean map of received "
        "attention of that layer.",
        f"All patch tokens of the {n_fmt(n)} held-out test images (of the probe).",
        ["c2_analysis/errors.csv", "c2_analysis/mean_prediction.csv"],
        r"Position-probe errors, layers 9--12 (exploratory; all patch tokens of the "
        f"{n_fmt(n)} held-out test images). (a) Exact and top-5 accuracy; probability within "
        r"the 3$\times$3 neighbourhood of the true patch; far share of errors = probability more "
        r"than 3 patches away / probability off the true patch; expected distance (patches). "
        r"Pairs give probe / matched blur (exact and top-5 of the blur are 1 by construction). "
        r"95\% bootstrap intervals of the probe values have half-width $\le$ " + f"{hw:.3f}"
        + r". (b) The probe's mean prediction over all tokens: its peak divided by the uniform "
        r"value 1/196, and its Pearson correlation with the received-attention mean map of that "
        r"layer, with 95\% bootstrap intervals (images resampled; the point estimate is not a "
        r"mean over images and can lie outside its interval).",
        intervals=(f"95% bootstrap intervals over images (1,000 resamples). (a) not shown, "
                   f"half-width of the probe values <= {hw:.3f}. (b) shown; images resampled, "
                   f"the point estimate is not a mean over images and can lie outside its "
                   f"interval."))
    body = "\n".join([sub("(a) Error geometry", first=True), a, sub("(b) Mean prediction"), b])
    return emit(com, body, size=r"\small", colsep="4pt")


def t_score_shift(R):
    sh, ct = R("c2_analysis/shift.csv"), R("correct_tests.csv")
    q = [("object share R0", "Object share, received attention"),
         ("object share after", "Object share after the change"),
         ("object share moved to background", "Moved to background"),
         ("pearson_mean of (moved - R0)", "Pearson(change, mean map)"),
         ("spearman_mean of (moved - R0)", "Spearman(change, mean map)"),
         ("pearson_centre of (moved - R0)", "Pearson(change, closeness)"),
         ("spearman_centre of (moved - R0)", "Spearman(change, closeness)")]
    rows, used = [], []
    for key, label in q:
        cells = [label]
        for m in MODELS:
            for src in ("C2", "stand_in_blur"):
                r = one(sh, model=m, source=src, quantity=key)
                used.append(r)
                cells.append(num(r["mean"]))
        rows.append(cells)
    rows.append("MID")
    cells = [mc(["Change in same-ring object", r"enrichment vs received attention$^\ast$"])]
    for m in MODELS:
        for src in ("C2", "stand_in_blur"):
            r = one(ct, a=src, b="R0", model=m, metric="enrichment")
            cells.append(num(r["mean"]))
    rows.append(cells)
    hw = ceil_to(max_half(used))
    n = int(sh["n"].iloc[0])
    pr, bl = hc(["Probe", "relocation"]), hc(["Matched", "blur"])
    header = [
        r" & \multicolumn{2}{c}{DeiT-S} & \multicolumn{2}{c}{DeiT-B} & "
        r"\multicolumn{2}{c}{DINO ViT-S} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        f"Quantity & {pr} & {bl} & {pr} & {bl} & {pr} & {bl} \\\\",
    ]
    com = std_comments(
        "Where probe relocation and the matched blur move the score (exploratory, layers 9-12). "
        "Object share = score on object patches / score on all valid patches; moved to "
        "background = object share of received attention - object share after the change. "
        "Correlations: per image, (map - received attention) against the layer 9-12 mean map or "
        "against closeness to the centre, averaged over images. Asterisk: mean paired difference "
        "in same-ring object enrichment, from paired_differences.tex.",
        f"{n_fmt(n)} ImageNet-S images (last row: ImageNet-S, n as in paired_differences.tex).",
        ["c2_analysis/shift.csv", "correct_tests.csv"],
        r"Where probe relocation and the matched blur move the score (exploratory; "
        + n_fmt(n) + r" ImageNet-S images, layers 9--12). Object share = score on object patches "
        r"/ score on all valid patches; moved to background = object share of received attention "
        r"$-$ object share after the change. Correlations: per image, (map $-$ received attention) "
        r"against the layer 9--12 mean map or against closeness to the centre, averaged over "
        r"images. 95\% bootstrap intervals have half-width $\le$ " + f"{hw:.3f}" + r". "
        r"$^\ast$Mean paired difference in same-ring object enrichment (paired-differences table).",
        intervals=INTERVALS_PLAIN + f" Not shown; half-width <= {hw:.3f}.")
    return emit(com, tabular("lcccccc", header, rows), size=r"\footnotesize", colsep="4pt")


def t_globalisation(R):
    gl = R("supplementary/globalisation.csv")
    gl = gl.copy()
    gl["layer"] = gl["layer"].astype(int)
    rows = []
    for mi, m in enumerate(MODELS):
        if mi:
            rows.append("MID")
        for L in (10, 11, 12):
            cells = [MODEL_NAME[m] if L == 10 else "", str(L)]
            for qn in ("rho_exact", "rho_entropy", "rho_move"):
                r = one(gl, model=m, layer=L, quantity=qn)
                cells.append(ci(r["mean"], r["low"], r["high"]))
            rows.append(cells)
    n = int(gl["n"].iloc[0])
    header = [r"Model & Layer & $\rho$(group, accuracy) & $\rho$(group, entropy) & "
              r"\makecell{$\rho$(group, probe-\\relocation move)} \\"]
    com = std_comments(
        "Within-layer globalisation (exploratory). Within each image, tokens are split into 5 "
        "groups by their cosine similarity with the image's mean token (5 = most global); per "
        "image, Spearman correlation (rho) of group number with the probe's exact accuracy, the "
        "entropy of its prediction and the expected move of probe relocation, averaged over "
        "images. The registered rule (higher similarity -> lower accuracy and higher entropy in "
        "at least 2 models at each of layers 10-12) is not met.",
        f"{n_fmt(n)} held-out test images of the probe, high-norm tokens excluded.",
        ["supplementary/globalisation.csv"],
        r"Within-layer globalisation (exploratory; " + n_fmt(n) + r" held-out test images, "
        r"high-norm tokens excluded). Within each image, tokens are split into 5 groups by their "
        r"cosine similarity with the image's mean token (5 = most global); per image, Spearman "
        r"correlation of group number with the probe's exact accuracy, the entropy of its "
        r"prediction and the expected move of probe relocation, averaged over images, with 95\% "
        r"bootstrap intervals. The registered rule (higher similarity $\to$ lower accuracy and "
        r"higher entropy in at least 2 models at each of layers 10--12) is not met.",
        intervals="95% bootstrap intervals over images (1,000 resamples).")
    return emit(com, tabular("llccc", header, rows), size=r"\footnotesize", colsep="4pt")


def t_mlp_probe(R):
    mp, cv = R("supplementary/mlp_probe.csv"), R("supplementary/c2_variants.csv")
    mp = mp.copy()
    mp["layer"] = mp["layer"].astype(int)
    rows = []
    for mi, m in enumerate(MODELS):
        if mi:
            rows.append("MID")
        for L in (9, 10, 11, 12):
            li = one(mp, model=m, layer=L, probe="linear", quantity="exact")
            ml = one(mp, model=m, layer=L, probe="mlp", quantity="exact")
            rows.append([MODEL_NAME[m] if L == 9 else "", str(L),
                         ci(li["mean"], li["low"], li["high"]),
                         ci(ml["mean"], ml["low"], ml["high"]), num(ml["mean"] - li["mean"])])
    a = tabular("llccc", [r"Model & Layer & Linear & MLP & MLP $-$ linear \\"], rows)
    rows_b = []
    for key in ("C2MLP", "C2conf"):
        if key != "C2MLP":
            rows_b.append(r"\addlinespace[3pt]")
        rows_b.append(group(cap(NAME[key]) + " $-$ " + NAME["R0"], 3))
        for m in MODELS:
            cells = [MODEL_NAME[m]]
            for me in ("enrichment", "clickme"):
                r = one(cv, a=key, b="R0", model=m, metric=me)
                cells.append(ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows_b.append(cells)
    b = tabular("lcc", [r"Model & " + metric_header("enrichment") + " & "
                         + metric_header("clickme") + r" \\"], rows_b)
    n = int(mp["n"].iloc[0])
    com = std_comments(
        "Probe capacity and probe-relocation variants (exploratory). (a) Exact accuracy of the "
        "linear and the two-layer MLP position probe, with intervals, and their difference. "
        "(b) Probe relocation with the MLP probe, and probe relocation of only the tokens whose "
        "top probe probability is >= 0.5 (confident tokens only), against received attention "
        "(layers 9-12): paired differences with intervals and verdict letters.",
        f"(a) the {n_fmt(n)} held-out test images of the probe; (b) "
        + data_text(cv, ["enrichment", "clickme"], a="C2MLP"),
        ["supplementary/mlp_probe.csv", "supplementary/c2_variants.csv"],
        r"Probe capacity and probe-relocation variants (exploratory). (a) Exact accuracy of the "
        r"linear and the two-layer MLP position probe on the " + n_fmt(n) + r" held-out test "
        r"images, with 95\% bootstrap intervals. (b) Probe relocation with the MLP probe, and "
        r"probe relocation of only the tokens whose top probe probability is $\ge 0.5$, against "
        r"received attention (layers 9--12): paired differences with 95\% paired-bootstrap "
        r"intervals and verdicts (\textit{h}/\textit{c}/\textit{l} = higher / comparable / "
        r"lower). Enrichment: ImageNet-S; ClickMe: 17,408 images.",
        intervals=INTERVALS_PLAIN, legend=True)
    body = "\n".join([sub("(a) Exact accuracy, linear vs MLP probe", first=True), a,
                      sub("(b) Probe-relocation variants"), b])
    return emit(com, body, size=r"\footnotesize", colsep="4pt")


def t_probability_discount(R):
    d2 = R("d2/tests.csv")
    vals = d2[d2["test"] == "value"]
    meths = [("R0", ["Received", "attention"]), ("C2", ["Probe", "relocation"]),
             ("C3", ["Contribution", "weighting"]), ("D2", ["Probability", "discount"]),
             ("D2_shuffle", ["Shuffled", "probability", "discount"]),
             ("C3D2", ["Contribution", "weighting +", "probability", "discount"])]
    rows, used = [], []
    for mi, me in enumerate(("enrichment", "clickme")):
        if mi:
            rows.append(r"\addlinespace[3pt]")
        rows.append(group(cap(METRIC_LONG[me]), 7))
        for m in MODELS:
            cells = [MODEL_NAME[m]]
            for key, _ in meths:
                r = one(vals, model=m, metric=me, method=key)
                used.append(r)
                cells.append(num(r["mean"]))
            rows.append(cells)
    hw = ceil_to(max_half(used))
    a = tabular("lcccccc", ["Model & " + " & ".join(hc(l) for _, l in meths) + r" \\"], rows)
    tests = [("D2 - R0", "D2", "R0"), ("D2 - D2_shuffle", "D2", "D2_shuffle"),
             ("D2 - C2", "D2", "C2"), ("C3D2 - C3", "C3D2", "C3")]
    rows_b = []
    for ti, (key, x, y) in enumerate(tests):
        if ti:
            rows_b.append(r"\addlinespace[3pt]")
        rows_b.append(group(cap(NAME[x]) + " $-$ " + NAME[y], 3))
        for m in MODELS:
            cells = [MODEL_NAME[m]]
            for me in ("enrichment", "clickme"):
                r = one(d2, model=m, metric=me, test=key)
                cells.append(ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows_b.append(cells)
    b = tabular("lcc", [r"Model & " + metric_header("enrichment") + " & "
                         + metric_header("clickme") + r" \\"], rows_b)
    com = std_comments(
        "Discount without relocation (exploratory, layers 9-12). Probability discount keeps each "
        "unit's received-attention score at its patch and multiplies it by the probe's "
        "probability that the unit is still there; shuffled probability discount uses the same "
        "probabilities permuted among the tokens of the image and layer (5 seeds); contribution "
        "weighting + probability discount applies contribution weighting first. (a) Means. "
        "(b) Paired differences with intervals and verdict letters. Contribution weighting + "
        "probability discount was registered for report only under the branch that did not "
        "occur and is shown for completeness.",
        data_text(d2, ["enrichment", "clickme"], method="R0", test="value"),
        ["d2/tests.csv"],
        r"Discount without relocation (exploratory, layers 9--12). The probability discount "
        r"keeps each unit's received-attention score at its patch and multiplies it by the "
        r"probe's probability that the unit is still there; the shuffled probability discount "
        r"uses the same probabilities permuted among the tokens of the image and layer (5 seeds). "
        r"(a) Means (95\% bootstrap intervals $\le \pm$" + f"{hw:.3f}" + r"). (b) Paired "
        r"differences with 95\% paired-bootstrap intervals and verdicts (\textit{h}/\textit{c}/"
        r"\textit{l} = higher / comparable / lower). Enrichment: ImageNet-S; ClickMe: 17,408 "
        r"images. Contribution weighting + probability discount was registered for report only "
        r"under the branch that did not occur and is shown for completeness.",
        intervals=INTERVALS_PLAIN + f" (a) not shown, <= +-{hw:.3f}.", legend=True)
    body = "\n".join([sub("(a) Values", first=True), a, sub("(b) Paired differences"), b])
    return emit(com, body, size=r"\footnotesize", colsep="4pt")


def t_smoothing(R):
    ct, od = R("correct_tests.csv"), R("order.csv")
    changes = [("blur", "stand_in_blur"), ("C2", "C2"), ("alt", "R_all_alt")]

    def get(kind, m, me):
        if kind == "blur":
            return one(ct, a="stand_in_blur", b="R0", model=m, metric=me)
        if kind == "C2":
            return one(ct, a="C2", b="R0", model=m, metric=me, test="correction - R0")
        return one(od, a="R_all_alt", b="R0", model=m, metric=me, test="order")

    rows = []
    for ki, (kind, key) in enumerate(changes):
        if ki:
            rows.append(r"\addlinespace[3pt]")
        rows.append(group(cap(NAME[key]) + " $-$ " + NAME["R0"], 4))
        for m in MODELS:
            c, e = get(kind, m, "clickme"), get(kind, m, "enrichment")
            rows.append([MODEL_NAME[m],
                         num(c["mean"]) + r"\," + letter(c["verdict"]),
                         num(e["mean"]) + r"\," + letter(e["verdict"]),
                         num(c["mean"] - e["mean"])])
    header = [r"Model & ClickMe & Enrichment & ClickMe $-$ enrichment \\"]
    com = std_comments(
        "Does image-specific ClickMe alignment reward smoothing? (exploratory compilation, no new "
        "runs; layers 9-12.) Mean paired change against received attention on ClickMe and on "
        "same-ring object enrichment, with verdict letters, from paired_differences.tex and "
        "combination_orders.tex, and their difference (a difference of means over different "
        "image sets, so no interval). ClickMe human maps are themselves blurred. " + ORDERS,
        data_text(ct, ["enrichment", "clickme"], a="stand_in_blur", b="R0"),
        ["correct_tests.csv", "order.csv"],
        r"Does ClickMe alignment reward smoothing? (exploratory compilation, no new runs; layers "
        r"9--12.) Mean paired change against received attention on image-specific ClickMe "
        r"alignment (17,408 images) and on same-ring object enrichment (ImageNet-S), from the "
        r"paired-differences and combination-order tables, and their difference (a difference of "
        r"means over different image sets, so no interval). Verdicts \textit{h}/\textit{c}/"
        r"\textit{l} = higher / comparable / lower. Order B: contribution weighting, then "
        r"mean-map subtraction, then probe relocation. ClickMe human maps are themselves blurred.",
        intervals=INTERVALS + " Used only for the verdicts; none for the last column.",
        legend=True, makecell=False)
    return emit(com, tabular("lccc", header, rows), size=r"\footnotesize", colsep="6pt")


# --------------------------------------------------------------------------- main

# (output name, builder, name of the old table in paper/review/tables)
TABLES = [
    ("single_layer", t_single_layer, "single_layer"),
    ("head_vs_body", t_head_vs_body, "parts"),
    ("predictions", t_predictions, "predictions"),
    ("negative_controls", t_negative_controls, "controls"),
    ("sanity_check", t_sanity, "sanity"),
    ("paired_differences", t_paired_differences, "diffs"),
    ("model_side_variants", t_model_side_variants, "variants"),
    ("combination_orders", t_combination_orders, "order"),
    ("last_layer_recovery", t_last_layer_recovery, "last_layer"),
    ("contribution_checks", t_contribution_checks, "c3_checks"),
    ("per_layer_gains", t_per_layer_gains, "dose_layers"),
    ("pairwise_combinations", t_pairwise_combinations, "pairs"),
    ("probe_errors", t_probe_errors, "probe_errors"),
    ("score_shift", t_score_shift, "shift"),
    ("globalisation", t_globalisation, "global"),
    ("mlp_probe", t_mlp_probe, "mlp"),
    ("probability_discount", t_probability_discount, "d2"),
    ("smoothing", t_smoothing, "smoothing"),
]


# --- main results table and full table ----------------------------------------------------------------------
ROWS_MAIN = [('centre', 'Centre Gaussian', 0), ('R0', 'Received attention', 0),
             ('C3', 'Contribution weighting', 0), ('stand_in_no_high', 'Zero high-norm tokens', 1),
             ('C2', 'Probe relocation', 0), ('stand_in_blur', 'Matched blur', 1),
             ('C1', 'Mean-map subtraction', 0), ('stand_in_centre', 'Centre-Gaussian subtraction', 1),
             ('register', '+ test-time register', 0), ('mita', 'Top-$k$ training (MiTA)', 0), ('gae', 'GAE (reference)', 0)]


def fmt(x, digits=3):
    text = f'{x:.{digits}f}'
    if text.startswith('-'):
        text = '$-$' + text[1:] if float(text) != 0 else text[1:]
    return text


def value_table(results):
    """mean, low, high per (model, metric, method) for every reading, reference, order and model-side variant."""
    long = pd.read_csv(results / 'table1_long.csv')[['model', 'metric', 'method', 'mean', 'low', 'high']]
    order = pd.read_csv(results / 'order.csv')
    order = order[(order.test == 'value') & (order.method == 'R_all_alt')][['model', 'metric', 'method', 'mean', 'low', 'high']]
    variants = pd.read_csv(results / 'variants.csv')
    variants = variants[variants.test == 'value'].rename(columns={'variant': 'method'}).assign(model='deit')
    variants = variants[['model', 'metric', 'method', 'mean', 'low', 'high']]
    return pd.concat([long, order, variants]).set_index(['model', 'metric', 'method'])


def table1(results, out):
    values = value_table(results)
    lines = ['% Table 1 (main text): object enrichment / image-specific ClickMe alignment, layers 9--12.',
             '% Generated by scripts/make_latex_tables.py from results/table1_long.csv and results/variants.csv.',
             '% Needs \\usepackage{booktabs}. Caption and label are added in the paper.',
             '\\begin{tabular}{lccc}', '\\toprule',
             'Reading & DeiT-S & DeiT-B & DINO ViT-S \\\\', '\\midrule']
    for method, label, indent in ROWS_MAIN:
        cells = []
        for model in MODELS:
            key_e, key_c = (model, 'enrichment', method), (model, 'clickme', method)
            if key_e in values.index:
                cells.append(f"{fmt(values.loc[key_e, 'mean'])} / {fmt(values.loc[key_c, 'mean'])}")
            else:
                cells.append('---')
        lines.append(('\\quad ' if indent else '') + label + ' & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    (out / 'table1.tex').write_text('\n'.join(lines))
    print('table1.tex written')


ROWS_FULL = ROWS_MAIN[1:8] + [('R_all', 'All three corrections (order A)', 0),
                              ('R_all_alt', 'All three corrections (order B)', 0)] + ROWS_MAIN[8:10] + [
    ('random', 'Random', 0), ('centre', 'Centre Gaussian', 0), ('mean_map', 'Mean map', 0), ('gae', 'GAE', 0)]
FULL_METRICS = [('srg', 'Deletion faithfulness (SRG; 2,000 ImageNet-S images)'),
           ('enrichment', 'Same-ring object enrichment (ImageNet-S)'),
           ('clickme', 'Image-specific ClickMe alignment (17,408 images)')]


def table1_full(results, out):
    values = value_table(results)
    lines = ['% Full Table 1 (appendix): three metrics, layers 9--12, mean [95% paired-bootstrap interval over images].',
             '% Order A = contribution weighting, then probe relocation, then mean-map subtraction (pre-registered);',
             '% order B = contribution weighting, then mean-map subtraction, then probe relocation.',
             '% Top-k training (MiTA) has no deletion score (no token removal for MiTA).',
             '% Generated by scripts/make_latex_tables.py from results/table1_long.csv, results/order.csv and',
             '% results/variants.csv. Needs \\usepackage{booktabs}. Caption and label are added in the paper.',
             '% Font size and column spacing below fit the 5.5 in text width; keep them inside the table environment.',
             '\\footnotesize', '\\setlength{\\tabcolsep}{4pt}',
             '\\begin{tabular}{lccc}', '\\toprule',
             'Reading & DeiT-S & DeiT-B & DINO ViT-S \\\\']
    for metric, title in FULL_METRICS:
        lines += ['\\midrule', f'\\multicolumn{{4}}{{l}}{{\\emph{{{title}}}}} \\\\', '\\addlinespace[1pt]']
        for method, label, indent in ROWS_FULL:
            if method in ('random',):
                lines.append('\\addlinespace[2pt]')
            cells = []
            for model in MODELS:
                key = (model, metric, method)
                if key in values.index:
                    row = values.loc[key]
                    cells.append(f"{fmt(row['mean'])} {{\\scriptsize [{fmt(row['low'])}, {fmt(row['high'])}]}}")
                elif method == 'mita' and model == 'deit':
                    cells.append('n/a')
                else:
                    cells.append('---')
            lines.append(('\\quad ' if indent else '') + label + ' & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '']
    (out / 'table1_full.tex').write_text('\n'.join(lines))
    print('table1_full.tex written')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=root / 'results')
    parser.add_argument('--out', type=Path, default=root / 'paper' / 'tables')
    args = parser.parse_args()
    (args.out / 'appendix').mkdir(parents=True, exist_ok=True)
    table1(args.results, args.out)
    table1_full(args.results, args.out)
    R = Results(str(args.results))
    for name, fn, _old in TABLES:
        (args.out / 'appendix' / f'{name}.tex').write_text(fn(R), encoding='utf-8')
    print(f'wrote table1.tex, table1_full.tex and {len(TABLES)} appendix tables to {args.out}')


if __name__ == "__main__":
    main()
