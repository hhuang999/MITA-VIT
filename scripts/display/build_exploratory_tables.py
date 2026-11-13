"""Make manuscript tables from the preserved exploratory summaries."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments"
OUT = ROOT / "paper" / "tables"
MODELS = {"deit": "DeiT-S", "deit_b": "DeiT-B", "dino": "DINO"}
def rows(filename):
    with (DATA / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

def write_table(name, cols, header, body):
    content = [r"\begin{tabular}{" + cols + "}", r"\toprule", header + r" \\", r"\midrule"]
    content.extend(" & ".join(row) + r" \\" for row in body)
    content += [r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(content) + "\n", encoding="utf-8", newline="\n")

def ci(mean, low, high):
    return f"{float(mean):.3f} [{float(low):.3f}, {float(high):.3f}]".replace("-", "$-$")

def spatial_table():
    study=json.loads((DATA/"spatial/summary.json").read_text(encoding="utf-8"))
    labels={"R0":"Received attention", "C3":"Value weighting (own lengths)",
            "donor_rank":"Donor-image positions", "ring_shuffle":"Within-ring permutation",
            "norm_only":"Own lengths, mean attention", "C3v":"Unprojected value norm",
            "net_output":"Norm after summing heads"}
    placement = {"R0": "Received attention", "C3": "Value weighting (own lengths)",
                 "donor_rank": "Same lengths in donor rank order",
                 "ring_shuffle": "Same lengths shuffled within radius",
                 "norm_only": "Own lengths, mean attention"}
    body=[]
    full=[]
    for method,label in labels.items():
        m=study["methods"][method]
        v=m["enrichment"]
        roc=m["auroc"]
        delta=ci(roc["delta_C3"],*roc["delta_C3_ci95"]) if method!="C3" else "---"
        row=[label,f"{roc['mean']:.3f}",delta,f"{v['mean']:.3f}"]
        full.append(row)
        if method in placement:
            delta = ci(v["delta_C3"], *v["delta_C3_ci95"]) if method != "C3" else "---"
            body.append([placement[method], f"{v['mean']:.3f}", delta, f"{roc['mean']:.3f}"])
    write_table("spatial.tex","@{}lrrr@{}",r"Readout & Enrichment & $\Delta$Enrichment [95\% CI] & AUROC",body)
    write_table("spatial_full.tex","@{}lrrr@{}",r"Reading & AUROC & $\Delta$AUROC [95\% CI] & Enrichment",full)


def appendix_tables():
    areas=rows("endpoints/deletion_areas.csv")
    click=rows("endpoints/clickme_bootstrap.csv")
    body=[]
    for model,name in MODELS.items():
        for control,label in (("R0","R0"),("stand_in_no_high","Zero high-norm")):
            row=[name,label]
            for metric in ("p_target","kl"):
                x=next(t for t in areas if (t["model"],t["a"],t["b"],t["metric"])==(model,"C3",control,metric))
                row.append(ci(x["mean"],x["low"],x["high"]))
            x=next(t for t in click if (t["model"],t["a"],t["b"])==(model,"C3",control))
            row.append(ci(x["mean"],x["batch_low"],x["batch_high"]))
            body.append(row)
    write_table("endpoint.tex","@{}llrrr@{}","Model & Comparator & Probability area & KL area & ClickMe",body)

    body = []
    for metric,filename in (("Enrichment","strata.csv"),("AUROC","auroc_strata.csv")):
        data = rows("geometry/" + filename)
        for model,name in MODELS.items():
            cell = {r["quartile"]:float(r["mean"]) for r in data
                    if r["model"] == model and r["control"] == "R0" and r["geometry"] == "object_area"}
            body.append([metric,name,*[f"{cell[f'Q{i}']:.3f}".replace("-0.000", "0.000").replace("-", "$-$") for i in range(1,5)]])
    write_table("geometry.tex","@{}llrrrr@{}","Metric & Model & Q1 & Q2 & Q3 & Q4",body)


def cls_table():
    methods = {"all_c3":"Value weighting, all queries", "cls_sum_head":"Value weighting, CLS only",
               "cls_net":"CLS, net vector"}
    metrics=("margin_spearman","auroc","ring_enrichment")
    body, full, effects = [], [], []
    for model,source in (("DeiT-S","cls"),("DeiT-B","receiver_review/deit_b")):
        records = rows(source + "/three_endpoint_summary.csv")
        means = {(r["method"],r["endpoint"]):float(r["mean"]) for r in records}
        for method,label in methods.items():
            if method == "cls_net":
                continue  # Full cancellation comparison is retained in the appendix table.
            body.append([model if method == "all_c3" else "", label,
                         *[f"{means[method,metric]:.3f}" for metric in metrics]])
        contrasts=[r for r in rows(source + "/three_endpoint_contrasts.csv")
                   if (r["a"],r["b"]) == ("cls_sum_head","all_c3")]
        values={r["endpoint"]:float(r["mean"]) for r in contrasts}
        body.append(["",r"\textbf{CLS $-$ all queries}",
                     *[rf"{{\boldmath${values[m]:+.3f}$}}" for m in metrics]])
        for method,label in {"all_r0":"All queries, attention",**methods,
                             "cls_attention":"CLS, attention"}.items():
            full.append([model,label,*[f"{means[method,metric]:.3f}" for metric in metrics]])
        for metric in metrics:
            r=next(r for r in contrasts if r["endpoint"] == metric)
            effects.append([model,{"margin_spearman":"Intervention $\\rho$","auroc":"Mask AUROC",
                                   "ring_enrichment":"Enrichment"}[metric],ci(r["mean"],r["low"],r["high"])])
    write_table("cls.tex","@{}llrrr@{}",r"Model & Readout & Intervention $\rho$ & Mask AUROC & Enrichment",body)
    write_table("cls_full.tex","@{}llrrr@{}",r"Model & Readout & Intervention $\rho$ & Mask AUROC & Enrichment",full)
    write_table("cls_effects.tex","@{}llr@{}",r"Model & Endpoint & CLS $-$ all queries [95\% CI]",effects)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-only", action="store_true", help="Build Tables 1 and 2 only.")
    args = parser.parse_args()
    spatial_table()
    cls_table()
    if not args.main_only:
        appendix_tables()
