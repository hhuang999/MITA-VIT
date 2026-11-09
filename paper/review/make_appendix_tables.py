#!/usr/bin/env python3
"""Build the appendix tables (App. B-E) of the review paper from the result CSV files.

Usage:
    python make_appendix_tables.py <results dir> <out dir>

Reads only the CSV files under <results dir> (the layout of mita-vit/results) and writes one
.tex file per table into <out dir>. Each file is a complete table float (caption above the
tabular, booktabs rules) for NeurIPS 2025 (text width 397 pt). Requires pandas only; the
LaTeX side needs booktabs, amsmath and amssymb.

Number format: 3 decimals for scores and differences, agreement rates in % with 1 decimal,
$-$ for minus signs. Verdict words and the "holds" flags are taken from the CSV files as stored
(they are not recomputed here).
"""

import math
import os
import re
import sys

import pandas as pd

MODELS = ["deit", "deit_b", "dino"]
MODEL_NAME = {"deit": "DeiT-S", "deit_b": "DeiT-B", "dino": "DINO", "mita": "MiTA-DeiT-S"}
METRICS = ["srg", "enrichment", "clickme"]
METRIC_NAME = {"srg": "SRG", "enrichment": "Enrichment", "clickme": "ClickMe"}
METRIC_LONG = {
    "srg": "deletion faithfulness (SRG)",
    "enrichment": "same-ring object enrichment",
    "clickme": "image-specific ClickMe alignment",
}
LETTER = {"higher": "h", "comparable": "c", "lower": "l"}
RALL = r"$R_{\mathrm{all}}$"
ALT = "Other order"
LEGEND = (r"Verdicts: \textit{h} = higher, \textit{c} = comparable, \textit{l} = lower (interval excludes 0 and "
          r"$|$difference$|$ exceeds the margin, 0.05 for SRG and 0.02 otherwise).")


# --------------------------------------------------------------------------- formatting

def num(x, d=3):
    """Number with d decimals and a LaTeX minus sign; '---' for missing values."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "---"
    s = f"{float(x):.{d}f}"
    if s.startswith("-") and float(s) == 0.0:   # rounds to zero: print without a sign
        return s[1:]
    return "$-$" + s[1:] if s.startswith("-") else s


def ci(m, lo, hi, d=3):
    return f"{num(m, d)} [{num(lo, d)}, {num(hi, d)}]"


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
    """pairs: list of (model, n). '12,150 (DeiT-S, DeiT-B) and 12,144 (DINO)' or '17,408'."""
    groups = {}
    for m, n in pairs:
        groups.setdefault(int(n), []).append(MODEL_NAME[m])
    if len(groups) == 1:
        return n_fmt(next(iter(groups)))
    parts = [f"{n_fmt(n)} ({', '.join(ms)})" for n, ms in groups.items()]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def table(label, caption, body, size=r"\small", colsep="4pt"):
    """Wrap one or more tabulars (body) in a table float."""
    out = [r"\begin{table}[ht]", r"\centering", size]
    if colsep:
        out.append(r"\setlength{\tabcolsep}{%s}" % colsep)
    out += [r"\caption{%s}" % caption, r"\label{tab:%s}" % label, body, r"\end{table}", ""]
    return "\n".join(out)


def tabular(spec, header_lines, rows, eol=r"\\"):
    """header_lines: list of LaTeX lines (with \\\\ or rules); rows: list of lists or the
    string 'MID' for a \\midrule."""
    out = [r"\begin{tabular}{%s}" % spec, r"\toprule"]
    out += header_lines
    out.append(r"\midrule")
    for r in rows:
        if isinstance(r, str):
            out.append(r"\midrule" if r == "MID" else r)
        else:
            out.append(" & ".join(r) + " " + eol)
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def sub(title):
    """Small run-in title above a sub-tabular inside one table float."""
    return r"\par\medskip\textit{%s}\par\smallskip" % title


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


# --------------------------------------------------------------------------- App. B

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
        r" & \multicolumn{2}{c}{DeiT-S} & \multicolumn{2}{c}{DeiT-B} & \multicolumn{2}{c}{DINO} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Layer & restricted & removed & restricted & removed & restricted & removed \\",
    ]
    cap = (f"One layer at a time: top-1 agreement (\\%) with the unmodified model on {n_fmt(n)} "
           f"ImageNet validation images when only that layer's queries are restricted to their "
           f"region's reading sets ($k = 16$) or only that layer's attention branch is removed. "
           f"Row ``all 12'': every layer at once (restricted = the $k = 16$ execution test; "
           f"removed = no attention in any layer). 95\\% bootstrap intervals have half-width "
           f"$\\le$ {hw:.1f} points. Removal runs used Resize 256 / crop 0.875 for every model "
           f"(App.~A).")
    return table("single_layer", cap, tabular("lcccccc", header, rows))


def t_parts(R):
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
    cap = (r"Extensions of RQ1. (a) Head vs body patches (PartImageNet, images with both a "
           r"removable animal head and body): drop in log-probability (nats) of the original top-1 "
           r"class after removing 4 head or 4 body patches from all layers (5 seeds), paired "
           r"difference with 95\% paired-bootstrap interval, verdict by the earlier study's rule "
           r"(higher if the lower bound $> 0$), and the ratio of the two means. These are ImageNet "
           r"training images. (b) Per-image Spearman correlation between the top-$k$ reading-set "
           r"map ($k = 16$) and the full map R0 (layers 9--12, ImageNet-S): median and quartiles "
           r"(no interval).")
    body = sub("(a) Head vs body patches") + "\n" + a + "\n" + sub(
        "(b) Top-$k$ map vs full map") + "\n" + b
    return table("parts", cap, body)


def _models_text(s):
    s = str(s)
    if s == "deit+register, mita":
        return "+register, MiTA-DeiT-S"
    parts = [p.strip() for p in s.split(",")]
    return ", ".join(MODEL_NAME.get(p, p) for p in parts)


def _detail_tex(s):
    s = str(s)
    s = s.replace("<=", r"$\le$").replace(" < ", " $<$ ")
    s = re.sub(r"(?<=\s)-(\d)", r"$-$\1", s)      # minus sign
    s = re.sub(r"(\d)-(\d)", r"\1--\2", s)         # number range
    s = re.sub(r"\bsrg\b", "SRG", s)
    s = re.sub(r"\bclickme\b", "ClickMe", s)
    s = s.replace("stand-in", "stand-in")
    return s


def t_predictions(R):
    p = R("predictions.csv")
    rows, prev = [], None
    for _, r in p.iterrows():
        name = str(r["prediction"]).split(" (")[0]
        if prev is not None and name != prev:
            rows.append(r"\addlinespace[2pt]")
        h = r["holds"]
        if isinstance(h, str):
            h = {"True": True, "False": False}.get(h, h)
        holds = "---" if (h is None or (isinstance(h, float) and math.isnan(h))) else (
            "yes" if bool(h) else "no")
        rows.append([name if name != prev else "", _models_text(r["model"]), holds,
                     r"\raggedright " + _detail_tex(r["detail"])])
        prev = name
    header = [r"Prediction & Model(s) & Holds & Detail \\"]
    cap = (r"Registered predictions and their outcomes (decision rules in App.~C). P7 lists the "
           r"primary comparisons (correction $-$ stand-in, three metrics), for which no direction "
           r"was predicted (holds = ---). P3 has 1 of 4 cells higher, which its pre-agreed reading "
           r"scores as ``partly holds''. P5 (last-layer recovery) is descriptive and is reported in "
           r"Table~\ref{tab:last_layer}.")
    return table("predictions", cap,
                 tabular(r"lllp{7.3cm}", header, rows, eol=r"\tabularnewline"), size=r"\small")


def t_controls(R):
    nc, rc = R("negative_controls.csv"), R("history/reproduction_check.csv")
    order = [("P4a", "dino", m) for m in METRICS] + [
        ("P4b", mo, me) for mo in MODELS for me in ("enrichment", "clickme")]
    rows = []
    for test, mo, me in order:
        r = maybe(nc, test=test, model=mo, metric=me)
        if r is None:
            continue
        rows.append([test, MODEL_NAME[mo], METRIC_NAME[me], n_fmt(r["n"]),
                     ci(r["mean"], r["low"], r["high"], 4), r["verdict"]])
    a = tabular("lllrcc", [
        r"Control & Model & Metric & $n$ & Difference & Verdict \\"], rows)
    rows_b = []
    for mo in MODELS:
        for me in ("enrichment", "srg", "clickme"):
            r = maybe(rc, model=mo, metric=me)
            if r is None:
                continue
            rows_b.append([MODEL_NAME[mo], METRIC_NAME[me], n_fmt(r["n"]), num(r["new"], 4),
                           num(r["old"], 4), num(1000 * r["difference"], 3),
                           "yes" if str(r["holds"]) == "True" else "no"])
    b = tabular("llrcccc", [
        r"Model & Metric & $n$ & New R0 & Earlier & Difference ($\times 10^{-3}$) & $< 0.01$ \\"],
        rows_b)
    cap = (r"Checks. (a) Pre-registered negative controls (P4): P4a = C3 $-$ R0 on DINO "
           r"(layers 9--12), P4b = C2 $-$ R0 at layers 1--4; differences (4 decimals) with "
           r"95\% paired-bootstrap intervals and verdicts (margin 0.05 for SRG, 0.02 otherwise); "
           r"SRG is not used for P4b by design. Data: ImageNet-S (SRG, enrichment) and ClickMe. "
           r"(b) Reproduction check: mean of the new R0 (layers 9--12) vs the earlier "
           r"full-attention runs on the same images; registered criterion $|$difference$|$ "
           r"$< 0.01$.")
    body = sub("(a) Negative controls") + "\n" + a + "\n" + sub("(b) Reproduction check") + "\n" + b
    return table("controls", cap, body, size=r"\small")


def t_sanity(R):
    s = R("sanity.csv")
    s = s[s["level"] == "embed"]
    names = [("R0", "R0"), ("C1", "C1"), ("C2", "C2"), ("C3", "C3"),
             ("R_all", RALL + r" (C3$\to$C2$\to$C1)"), ("centre", "Centre Gaussian (reference)")]
    rows = []
    for key, label in names:
        cells = [label]
        for m in MODELS:
            r = one(s, model=m, method=key)
            ok = str(r["passes"]) == "True"
            cells.append(f"{num(r['median'])} & {'passes' if ok else 'fails'}")
        rows.append(cells)
        if key == "R_all":
            rows.append("MID")
    header = [
        r" & \multicolumn{2}{c}{DeiT-S} & \multicolumn{2}{c}{DeiT-B} & \multicolumn{2}{c}{DINO} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Reading & median & & median & & median & \\",
    ]
    cap = (r"Model-randomisation sanity check on 1,000 ImageNet-S images: median per-image "
           r"Spearman correlation between the map of the fully randomised model (all stages "
           r"re-initialised down to the patch embedding) and the map of the trained model "
           r"(layers 9--12). A reading passes if the median is below 0.5. The centre Gaussian does "
           r"not depend on the model and is the designed failing example.")
    return table("sanity", cap, tabular("lcccccc", header, rows))


# --------------------------------------------------------------------------- App. D

FULL_ROWS = [
    ("R0", "R0 (attention received)"),
    ("C3", "C3 contribution weighting"),
    ("stand_in_no_high", r"\hspace{0.6em}$\llcorner$ zero high-norm tokens"),
    ("C2", "C2 relocation by probe"),
    ("stand_in_blur", r"\hspace{0.6em}$\llcorner$ matched blur"),
    ("C1", "C1 subtract mean map"),
    ("stand_in_centre", r"\hspace{0.6em}$\llcorner$ subtract centre Gaussian"),
    ("R_all", RALL + r" (C3$\to$C2$\to$C1)"),
    ("R_all_alt", ALT + r" (C3$\to$C1$\to$C2)"),
    ("MID", None),
    ("GROUP", "References"),
    ("random", "Random"),
    ("centre", "Centre Gaussian"),
    ("mean_map", "Mean map"),
    ("gae", "GAE"),
    ("MID", None),
    ("GROUP", "Change the model (DeiT-S)"),
    ("register", "+ test-time register"),
    ("mita", "top-$k$ training (MiTA)"),
]


def _full_value(R, metric, model, key):
    t1, od, va = R("table1_long.csv"), R("order.csv"), R("variants.csv")
    if key in ("register", "mita"):
        if model != "deit":
            return None
        return maybe(va, variant=key, metric=metric, test="value")
    r = maybe(t1, model=model, metric=metric, method=key)
    if r is None and key == "R_all_alt":
        r = maybe(od, model=model, metric=metric, method=key, test="value")
    return r


def t_full(R, metric):
    rows, ns = [], {}
    for key, label in FULL_ROWS:
        if key == "MID":
            rows.append("MID")
            continue
        if key == "GROUP":
            rows.append([r"\multicolumn{4}{l}{\textit{%s}}" % label])
            continue
        cells = [label]
        for m in MODELS:
            r = _full_value(R, metric, m, key)
            if r is None:
                if key == "mita" and m == "deit" and metric == "srg":
                    cells.append("n/a")
                else:
                    cells.append("---")
                continue
            cells.append(ci(r["mean"], r["low"], r["high"]))
            if key == "R0":
                ns[m] = int(r["n"])
        rows.append(cells)
    header = [r"Reading & DeiT-S & DeiT-B & DINO \\"]
    data = {"srg": "2,000 ImageNet-S images", "enrichment": "ImageNet-S", "clickme": "ClickMe"}[metric]
    extra = ""
    if metric == "srg":
        extra = (" SRG is not defined for MiTA, whose summary slots pool fixed regions of the full "
                 "grid (App.~A).")
    if metric == "enrichment":
        mi = _full_value(R, metric, "deit", "mita")
        if mi is not None and int(mi["n"]) != ns["deit"]:
            extra = f" MiTA: $n$ = {n_fmt(mi['n'])} (own crop 0.875)."
    cap = (f"Full Table~1, {METRIC_LONG[metric]} (layers 9--12; higher is better), {data}, "
           f"$n$ = {n_text(list(ns.items()))} per model. Mean over images with 95\\% bootstrap "
           f"interval (1,000 resamples over images). Each correction is followed by its stand-in "
           f"($\\llcorner$). GAE uses gradients and the class and is a reference, not a competitor."
           f"{extra}")
    return table(f"full_{metric}", cap, tabular("lccc", header, rows), size=r"\small",
                 colsep="3pt")


DIFF_ROWS = [
    ("C1", "R0", "correction - R0", "C1 $-$ R0"),
    ("C2", "R0", "correction - R0", "C2 $-$ R0"),
    ("C3", "R0", "correction - R0", "C3 $-$ R0"),
    ("R_all", "R0", "correction - R0", RALL + r" $-$ R0$^\dagger$"),
    ("C1", "stand_in_centre", "correction - stand-in", r"C1 $-$ centre sub.$^\dagger$"),
    ("C2", "stand_in_blur", "correction - stand-in", r"C2 $-$ blur$^\dagger$"),
    ("C3", "stand_in_no_high", "correction - stand-in", r"C3 $-$ zero h.n.$^\dagger$"),
    ("stand_in_centre", "R0", "stand-in - R0", r"centre sub. $-$ R0"),
    ("stand_in_blur", "R0", "stand-in - R0", r"blur $-$ R0"),
    ("stand_in_no_high", "R0", "stand-in - R0", r"zero h.n. $-$ R0"),
]


def t_diffs(R):
    ct = R("correct_tests.csv")
    rows = []
    for mi, m in enumerate(MODELS):
        if mi:
            rows.append("MID")
        rows.append([r"\multicolumn{4}{l}{\textit{%s}}" % MODEL_NAME[m]])
        for a, b, test, label in DIFF_ROWS:
            cells = [label]
            for me in METRICS:
                r = one(ct, a=a, b=b, model=m, metric=me, test=test)
                cells.append(ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows.append(cells)
    header = [r"Comparison & SRG & Enrichment & ClickMe \\"]
    cap = (r"Paired differences, layers 9--12: mean with 95\% paired-bootstrap interval and "
           r"verdict. " + LEGEND + r" $^\dagger$ = pre-registered primary comparison. "
           r"centre sub.\ = subtract centre Gaussian; blur = matched blur; zero h.n.\ = zero "
           r"high-norm tokens. SRG: 2,000 ImageNet-S images; enrichment: ImageNet-S; ClickMe: "
           r"17,408 images.")
    return table("diffs", cap, tabular("lccc", header, rows), size=r"\footnotesize",
                 colsep="3pt")


def t_variants(R):
    va, ct = R("variants.csv"), R("correct_tests.csv")
    rows = []
    for key, label in (("register", "+ test-time register"), ("mita", "top-$k$ training (MiTA)")):
        for me in METRICS:
            v = maybe(va, variant=key, metric=me, test="value")
            d = maybe(va, variant=key, metric=me, test="variant - DeiT-S R0")
            if d is None:
                rows.append([label, METRIC_NAME[me], "---", "n/a", "n/a", "", ""])
                continue
            p6 = d["holds_p6"]
            p6 = "not higher" if str(p6) == "True" else ("---" if pd.isna(p6) else "higher")
            rows.append([label, METRIC_NAME[me], n_fmt(d["n"]), num(v["mean"]),
                         ci(d["mean"], d["low"], d["high"]), d["verdict"], p6])
        rows.append("MID")
    for me in METRICS:
        r = one(ct, a="C3", b="R0", model="deit", metric=me, test="correction - R0")
        rows.append(["For comparison: C3", METRIC_NAME[me], n_fmt(r["n"]), "",
                     ci(r["mean"], r["low"], r["high"]), r["verdict"], ""])
    header = [r"Row & Metric & $n$ & Value & Difference vs DeiT-S R0 & Verdict & P6 (2) \\"]
    cap = (r"Changing the model (DeiT-S, layers 9--12): R0 of DeiT-S with a test-time register "
           r"and of MiTA-DeiT-S (top-$k$ training), paired with DeiT-S R0 by image index; mean "
           r"difference with 95\% paired-bootstrap interval, verdict, and the registered check P6 "
           r"(2) (not higher on enrichment and ClickMe). MiTA has no SRG (App.~A) and uses its own "
           r"crop. Last rows: C3 $-$ R0 on DeiT-S for comparison.")
    return table("variants", cap, tabular("llrcccc", header, rows), size=r"\small", colsep="3.5pt")


def t_order(R):
    od = R("order.csv")
    rows, same = [], 0
    for mi, m in enumerate(MODELS):
        if mi:
            rows.append("MID")
        rows.append([r"\multicolumn{6}{l}{\textit{%s}}" % MODEL_NAME[m]])
        for me in METRICS:
            v = {k: one(od, model=m, metric=me, method=k, test="value")
                 for k in ("R0", "R_all", "R_all_alt")}
            d1 = one(od, model=m, metric=me, test="order", a="R_all", b="R0")
            d2 = one(od, model=m, metric=me, test="order", a="R_all_alt", b="R_all")
            d3 = one(od, model=m, metric=me, test="order", a="R_all_alt", b="R0")
            same += int(d1["verdict"] == d3["verdict"])
            rows.append([METRIC_NAME[me], num(v["R0"]["mean"]), num(v["R_all"]["mean"]),
                         num(v["R_all_alt"]["mean"]),
                         ci(d1["mean"], d1["low"], d1["high"]) + r"\," + letter(d1["verdict"]),
                         ci(d2["mean"], d2["low"], d2["high"]) + r"\," + letter(d2["verdict"])])
    total = 3 * len(METRICS)
    vals = od[od["test"] == "value"]
    hw = ceil_to(max_half([r for _, r in vals.iterrows()]))
    header = [r"Metric & R0 & " + RALL + r" & " + ALT + r" & " + RALL + r" $-$ R0 & " + ALT
              + r" $-$ " + RALL + r" \\"]
    cap = (r"Combination orders (layers 9--12): " + RALL + r" = C3$\to$C2$\to$C1 (registered), "
           + ALT + r" = C3$\to$C1$\to$C2 (descriptive check). Values (95\% bootstrap intervals "
           r"$\le \pm$" + f"{hw:.3f}" + r") and paired differences with 95\% paired-bootstrap "
           r"intervals and verdicts. " + LEGEND + f" The two orders get the same verdict against "
           f"R0 in {same} of {total} cells (``robust'' required all {total}).")
    return table("order", cap, tabular("lccccc", header, rows), size=r"\footnotesize",
                 colsep="3.5pt")


def t_last_layer(R):
    ll = R("last_layer.csv")
    readings = [("C3", "C3"), ("C1", "C1"), ("C2", "C2"), ("R_all", RALL)]
    rows = []
    for mi, m in enumerate(("deit", "deit_b")):
        if mi:
            rows.append("MID")
        rows.append([r"\multicolumn{4}{l}{\textit{%s}}" % MODEL_NAME[m]])
        best, drop = ["Best R0 layer (8--11)"], ["Drop to layer 12"]
        for me in METRICS:
            r = one(ll, model=m, metric=me, reading="C3")
            best.append(str(r["best_layer"]).replace("R0_l", "layer "))
            drop.append(num(r["drop"]))
        rows += [best, drop]
        for key, label in readings:
            cells = [label + " recovers"]
            for me in METRICS:
                r = one(ll, model=m, metric=me, reading=key)
                cells.append(ci(r["recovered"], r["low"], r["high"]))
            rows.append(cells)
    header = [r" & SRG & Enrichment & ClickMe \\"]
    cap = (r"Last-layer recovery (P5, descriptive; supervised models). Drop = R0 at the best of "
           r"layers 8--11 minus R0 at layer 12. Share recovered = (corrected layer-12 map $-$ R0 "
           r"layer 12) / drop: 1 = full recovery, negative = worse than R0 at layer 12. 95\% "
           r"paired-bootstrap intervals of the ratio of means. SRG: 2,000 ImageNet-S images; "
           r"enrichment: ImageNet-S; ClickMe: 17,408 images.")
    return table("last_layer", cap, tabular("lccc", header, rows), size=r"\small")


def t_c3_checks(R):
    mech, dose, gap = (R("supplementary/c3_mechanism.csv"), R("supplementary/dose.csv"),
                       R("supplementary/gae_gap.csv"))
    rows = []
    for b, label in (("C3shuffle", "shuffled weights"), ("C3headmean", "head-mean weights"),
                     ("C3v", r"$\lVert v_j\rVert$ weights")):
        for m in MODELS:
            cells = [r"C3 $-$ C3 with " + label if m == "deit" else "", MODEL_NAME[m]]
            for me in ("enrichment", "clickme"):
                r = one(mech, a="C3", b=b, model=m, metric=me)
                cells.append(ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows.append(cells)
    a = tabular("llcc", [r"Comparison & Model & Enrichment & ClickMe \\"], rows)
    rows_b = []
    for m in ("deit", "deit_b"):
        for me in ("enrichment", "clickme"):
            for imp, label in (("C3 - R0", "C3 $-$ R0"),
                               ("C3 - stand_in_no_high", "C3 $-$ zero high-norm")):
                r = one(dose, model=m, metric=me, improvement=imp)
                rows_b.append([MODEL_NAME[m], METRIC_NAME[me], label, n_fmt(r["n"]),
                               num(r["mean_dose"]), ci(r["spearman"], r["low"], r["high"])])
    b = tabular("lllrcc", [
        r"Model & Metric & Improvement & $n$ & Mean dose & Spearman(dose, improvement) \\"], rows_b)
    rows_c = []
    for m in ("deit", "deit_b"):
        for me in METRICS:
            r = one(gap, model=m, metric=me)
            rows_c.append([MODEL_NAME[m], METRIC_NAME[me], n_fmt(r["n"]), num(r["c3_gain"]),
                           num(r["gae_gap"]), ci(r["share_closed"], r["low"], r["high"])])
    c = tabular("llrccc", [r"Model & Metric & $n$ & C3 $-$ R0 & GAE $-$ R0 & Share closed \\"],
                rows_c)
    cap = (r"Exploratory checks of C3 (layers 9--12). (a) Mechanism: C3 against C3 with its "
           r"weights $\lVert W_O v_j\rVert$ shuffled among the patch tokens of each head (5 "
           r"seeds), replaced by the head mean, or replaced by $\lVert v_j\rVert$. " + LEGEND +
           r" (b) Dose--response: across images, Spearman correlation between the share of R0 on "
           r"high-norm tokens (dose) and the image's improvement. (c) Share of the gap between R0 "
           r"and GAE closed by C3, (C3 $-$ R0) / (GAE $-$ R0). All intervals are 95\% "
           r"(paired) bootstrap intervals over images. Enrichment: ImageNet-S; ClickMe: 17,408 "
           r"images; SRG: 2,000 ImageNet-S images.")
    body = "\n".join([sub("(a) Mechanism"), a, sub("(b) Dose--response"), b,
                      sub("(c) Share of the GAE gap closed"), c])
    return table("c3_checks", cap, body, size=r"\footnotesize", colsep="4pt")


def t_dose_layers(R):
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
    header = [
        r" & & \multicolumn{2}{c}{Enrichment} & \multicolumn{2}{c}{ClickMe} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"Model & Layer & C3 $-$ R0 & zero h.n. $-$ R0 & C3 $-$ R0 & zero h.n. $-$ R0 \\",
    ]
    cap = (r"Per-layer gains (exploratory): gain over R0 of C3 and of zeroing high-norm tokens "
           r"(zero h.n.) when only that layer is read, and for layers 9--12 summed. Enrichment: "
           r"ImageNet-S; ClickMe: 17,408 images. 95\% paired-bootstrap intervals have half-width "
           r"$\le$ " + f"{hw:.3f}" + r". " + LEGEND)
    return table("dose_layers", cap, tabular("llcccc", header, rows), size=r"\small")


def t_pairs(R):
    cb = R("supplementary/combinations.csv")
    comps = [("C1C3", "R0", "C1+C3 $-$ R0"), ("C2C3", "R0", "C2+C3 $-$ R0"),
             ("C1C2", "R0", "C1+C2 $-$ R0"), ("C1C3", "C3", "C1+C3 $-$ C3"),
             ("C2C3", "C3", "C2+C3 $-$ C3")]
    rows = []
    for mi, m in enumerate(MODELS):
        if mi:
            rows.append("MID")
        rows.append([r"\multicolumn{4}{l}{\textit{%s}}" % MODEL_NAME[m]])
        for a, b, label in comps:
            cells = [label]
            for me in METRICS:
                r = maybe(cb, a=a, b=b, model=m, metric=me)
                cells.append("---" if r is None else
                             ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows.append(cells)
    header = [r"Comparison & SRG & Enrichment & ClickMe \\"]
    cap = (r"Pairwise combinations (exploratory), layers 9--12, applied in the order "
           r"C3$\to$C2$\to$C1 with the absent correction skipped. Paired differences with 95\% "
           r"paired-bootstrap intervals and verdicts. " + LEGEND + r" SRG was computed for C1+C3 "
           r"only (2,000 ImageNet-S images); enrichment: ImageNet-S; ClickMe: 17,408 images.")
    return table("pairs", cap, tabular("lccc", header, rows), size=r"\footnotesize", colsep="3pt")


# --------------------------------------------------------------------------- App. E

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
        r"Model & Layer & Exact & Top-5 & $P$(3$\times$3) & Far share of errors & "
        r"Expected distance \\"], rows)
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
    cap = (r"Position-probe errors, layers 9--12 (exploratory; all patch tokens of the "
           f"{n_fmt(n)} held-out test images). (a) Exact and top-5 accuracy; probability within "
           r"the 3$\times$3 neighbourhood of the true patch; far share of errors = probability "
           r"more than 3 patches away / probability off the true patch; expected distance "
           r"(patches). Pairs give probe / matched blur (exact and top-5 of the blur are 1 by "
           r"construction). 95\% bootstrap intervals of the probe values have half-width $\le$ "
           f"{hw:.3f}" + r". (b) The probe's mean prediction over all tokens: its peak divided by "
           r"the uniform value 1/196, and its Pearson correlation with the model's R0 mean map of "
           r"that layer, with 95\% bootstrap intervals (images resampled; the point estimate is "
           r"not a mean over images and can lie outside its interval).")
    body = "\n".join([sub("(a) Error geometry"), a, sub("(b) Mean prediction"), b])
    return table("probe_errors", cap, body, size=r"\small")


def t_shift(R):
    sh, ct = R("c2_analysis/shift.csv"), R("correct_tests.csv")
    q = [("object share R0", "Object share, R0"),
         ("object share after", "Object share, after"),
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
    cells = [r"Enrichment change vs R0$^\ast$"]
    for m in MODELS:
        for src in ("C2", "stand_in_blur"):
            r = one(ct, a=src, b="R0", model=m, metric="enrichment")
            cells.append(num(r["mean"]))
    rows.append(cells)
    hw = ceil_to(max_half(used))
    n = int(sh["n"].iloc[0])
    header = [
        r" & \multicolumn{2}{c}{DeiT-S} & \multicolumn{2}{c}{DeiT-B} & \multicolumn{2}{c}{DINO} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"Quantity & C2 & blur & C2 & blur & C2 & blur \\",
    ]
    cap = (f"Where C2 and the matched blur move the score (exploratory; {n_fmt(n)} ImageNet-S "
           r"images, layers 9--12). Object share = score on object patches / score on all valid "
           r"patches; moved to background = object share of R0 $-$ object share after the change. "
           r"Correlations: per image, (map $-$ R0) against the layer 9--12 mean map or against "
           r"closeness to the centre, averaged over images. 95\% bootstrap intervals have "
           r"half-width $\le$ " + f"{hw:.3f}" + r". $^\ast$Mean paired difference in same-ring "
           r"enrichment (Table~\ref{tab:diffs}).")
    return table("shift", cap, tabular("lcccccc", header, rows), size=r"\small")


def t_global(R):
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
              r"$\rho$(group, C2 move) \\"]
    cap = (r"Within-layer globalisation (exploratory; " + n_fmt(n) + r" held-out test images, "
           r"high-norm tokens excluded). Within each image, tokens are split into 5 groups by their "
           r"cosine similarity with the image's mean token (5 = most global); per image, Spearman "
           r"correlation of group number with the probe's exact accuracy, the entropy of its "
           r"prediction and the expected C2 move, averaged over images, with 95\% bootstrap "
           r"intervals. The registered rule (higher similarity $\to$ lower accuracy and higher "
           r"entropy in at least 2 models at each of layers 10--12) is not met.")
    return table("global", cap, tabular("llccc", header, rows), size=r"\footnotesize",
                 colsep="4pt")


def t_mlp(R):
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
    for key, label in (("C2MLP", "C2 with MLP probe $-$ R0"),
                       ("C2conf", "C2 confident tokens only $-$ R0")):
        for m in MODELS:
            cells = [label if m == "deit" else "", MODEL_NAME[m]]
            for me in ("enrichment", "clickme"):
                r = one(cv, a=key, b="R0", model=m, metric=me)
                cells.append(ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows_b.append(cells)
    b = tabular("llcc", [r"Comparison & Model & Enrichment & ClickMe \\"], rows_b)
    n = int(mp["n"].iloc[0])
    cap = (r"Probe capacity and C2 variants (exploratory). (a) Exact accuracy of the linear and "
           r"the two-layer MLP position probe on the " + n_fmt(n) + r" held-out test images, with "
           r"95\% bootstrap intervals. (b) C2 with the MLP probe, and C2 relocating only tokens whose "
           r"top probe probability is $\ge 0.5$, against R0 (layers 9--12): paired differences "
           r"with 95\% paired-bootstrap intervals. " + LEGEND + r" Enrichment: ImageNet-S; "
           r"ClickMe: 17,408 images.")
    body = "\n".join([sub("(a) Exact accuracy, linear vs MLP probe"), a,
                      sub("(b) C2 variants"), b])
    return table("mlp", cap, body, size=r"\small")


def t_d2(R):
    d2 = R("d2/tests.csv")
    vals = d2[d2["test"] == "value"]
    meths = [("R0", "R0"), ("C2", "C2"), ("C3", "C3"), ("D2", "D2"),
             ("D2_shuffle", "D2 shuffled"), ("C3D2", "C3D2")]
    rows, used = [], []
    for m in MODELS:
        for me in ("enrichment", "clickme"):
            cells = [MODEL_NAME[m] if me == "enrichment" else "", METRIC_NAME[me]]
            for key, _ in meths:
                r = one(vals, model=m, metric=me, method=key)
                used.append(r)
                cells.append(num(r["mean"]))
            rows.append(cells)
    hw = ceil_to(max_half(used))
    a = tabular("llcccccc", ["Model & Metric & " + " & ".join(l for _, l in meths) + r" \\"], rows)
    tests = [("D2 - R0", "D2 $-$ R0"), ("D2 - D2_shuffle", "D2 $-$ D2 shuffled"),
             ("D2 - C2", "D2 $-$ C2"), ("C3D2 - C3", "C3D2 $-$ C3")]
    rows_b = []
    for key, label in tests:
        for m in MODELS:
            cells = [label if m == "deit" else "", MODEL_NAME[m]]
            for me in ("enrichment", "clickme"):
                r = one(d2, model=m, metric=me, test=key)
                cells.append(ci(r["mean"], r["low"], r["high"]) + r"\," + letter(r["verdict"]))
            rows_b.append(cells)
    b = tabular("llcc", [r"Comparison & Model & Enrichment & ClickMe \\"], rows_b)
    cap = (r"Discount without relocation (exploratory, layers 9--12). D2 keeps each unit's R0 "
           r"score at its patch and multiplies it by the probe's probability that the unit is "
           r"still there; D2 shuffled uses the same probabilities permuted among the tokens of the "
           r"image and layer (5 seeds); C3D2 applies C3 first. (a) Means (95\% bootstrap intervals "
           r"$\le \pm$" + f"{hw:.3f}" + r"). (b) Paired differences with 95\% paired-bootstrap "
           r"intervals. " + LEGEND + r" Enrichment: ImageNet-S; ClickMe: 17,408 images. C3D2 was "
           r"registered for report only under the branch that did not occur and is shown for "
           r"completeness.")
    body = "\n".join([sub("(a) Values"), a, sub("(b) Paired differences"), b])
    return table("d2", cap, body, size=r"\small")


def t_smoothing(R):
    ct, od = R("correct_tests.csv"), R("order.csv")
    changes = [("blur", "Matched blur $-$ R0"), ("C2", "C2 $-$ R0"),
               ("alt", ALT + " $-$ R0")]

    def get(kind, m, me):
        if kind == "blur":
            return one(ct, a="stand_in_blur", b="R0", model=m, metric=me)
        if kind == "C2":
            return one(ct, a="C2", b="R0", model=m, metric=me, test="correction - R0")
        return one(od, a="R_all_alt", b="R0", model=m, metric=me, test="order")

    rows = []
    for ki, (kind, label) in enumerate(changes):
        if ki:
            rows.append("MID")
        for m in MODELS:
            c, e = get(kind, m, "clickme"), get(kind, m, "enrichment")
            rows.append([label if m == "deit" else "", MODEL_NAME[m],
                         num(c["mean"]) + r"\," + letter(c["verdict"]),
                         num(e["mean"]) + r"\," + letter(e["verdict"]),
                         num(c["mean"] - e["mean"])])
    header = [r"Change & Model & ClickMe & Enrichment & ClickMe $-$ enrichment \\"]
    cap = (r"Does ClickMe reward smoothing? (exploratory compilation, no new runs; layers 9--12.) "
           r"Mean paired change against R0 on ClickMe (17,408 images) and on same-ring enrichment "
           r"(ImageNet-S), from Tables~\ref{tab:diffs} and~\ref{tab:order}, and their difference "
           r"(a difference of means over different image sets, so no interval). " + LEGEND +
           r" ClickMe human maps are themselves blurred (App.~A).")
    return table("smoothing", cap, tabular("llccc", header, rows), size=r"\small")


# --------------------------------------------------------------------------- main

TABLES = [
    ("single_layer", t_single_layer),
    ("parts", t_parts),
    ("predictions", t_predictions),
    ("controls", t_controls),
    ("sanity", t_sanity),
    ("full_srg", lambda R: t_full(R, "srg")),
    ("full_enrichment", lambda R: t_full(R, "enrichment")),
    ("full_clickme", lambda R: t_full(R, "clickme")),
    ("diffs", t_diffs),
    ("variants", t_variants),
    ("order", t_order),
    ("last_layer", t_last_layer),
    ("c3_checks", t_c3_checks),
    ("dose_layers", t_dose_layers),
    ("pairs", t_pairs),
    ("probe_errors", t_probe_errors),
    ("shift", t_shift),
    ("global", t_global),
    ("mlp", t_mlp),
    ("d2", t_d2),
    ("smoothing", t_smoothing),
]


def main(argv):
    if len(argv) != 3:
        sys.exit(__doc__)
    results, out = argv[1], argv[2]
    os.makedirs(out, exist_ok=True)
    R = Results(results)
    header = ("% Generated by make_appendix_tables.py from the CSV files in the results "
              "directory; do not edit by hand.\n")
    for name, fn in TABLES:
        tex = fn(R)
        with open(os.path.join(out, f"{name}.tex"), "w", encoding="utf-8") as f:
            f.write(header + tex)
        print(f"wrote {name}.tex")


if __name__ == "__main__":
    main(sys.argv)
