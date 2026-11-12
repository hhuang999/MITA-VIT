"""Make the manuscript tables from the v13 exploratory results. Run from any directory."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "research" / "v13"
MODELS = {"deit": "DeiT-S", "deit_b": "DeiT-B", "dino": "DINO"}
def rows(filename):
    with (DATA / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

def write_table(name, cols, header, body):
    content = [r"\begin{tabular}{" + cols + "}", r"\toprule", header + r" \\", r"\midrule"]
    content.extend(" & ".join(row) + r" \\" for row in body)
    content += [r"\bottomrule", r"\end{tabular}"]
    (HERE / name).write_text("\n".join(content) + "\n", encoding="utf-8", newline="\n")

def ci(mean, low, high):
    return f"{float(mean):.3f} [{float(low):.3f}, {float(high):.3f}]".replace("-", "$-$")

def main():
    study=json.loads((DATA/"spatial/summary.json").read_text(encoding="utf-8"))
    labels={"R0":"Received attention (R0)", "C3":"Projected value norm (C3)", "C3v":"Unprojected value norm",
            "net_output":"Norm after summing heads", "norm_only":"Value norm only", "donor_rank":"Donor-image positions",
            "ring_shuffle":"Within-ring permutation"}
    body=[]
    for method,label in labels.items():
        m=study["methods"][method]
        v=m["enrichment"]
        delta=ci(v["delta_C3"],*v["delta_C3_ci95"]) if method!="C3" else "---"
        body.append([label,f"{v['mean']:.3f}",delta,f"{m['auroc']['mean']:.3f}"])
    write_table("spatial_table.tex","@{}lrrr@{}",r"Reading & Enrichment & Difference from C3 [95\% CI] & AUROC",body)

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
    write_table("endpoint_table.tex","@{}llrrr@{}","Model & Comparator & Probability area & KL area & ClickMe",body)

    body = []
    for metric,filename in (("Enrichment","strata.csv"),("AUROC","auroc_strata.csv")):
        data = rows("geometry/" + filename)
        for model,name in MODELS.items():
            cell = {r["quartile"]:float(r["mean"]) for r in data
                    if r["model"] == model and r["control"] == "R0" and r["geometry"] == "object_area"}
            body.append([metric,name,*[f"{cell[f'Q{i}']:.3f}".replace("-0.000", "0.000").replace("-", "$-$") for i in range(1,5)]])
    write_table("geometry_table.tex","@{}llrrrr@{}","Metric & Model & Q1 & Q2 & Q3 & Q4",body)

    records = rows("cls/three_endpoint_summary.csv")
    means = {(r["method"],r["endpoint"]):float(r["mean"]) for r in records}
    methods = {"all_r0":"All queries, attention", "all_c3":"All queries, sum of magnitudes",
               "cls_attention":"CLS, attention", "cls_sum_head":"CLS, sum of magnitudes",
               "cls_net":"CLS, net magnitude"}
    body = [[label,*[f"{means[method,metric]:.3f}" for metric in ("margin_spearman","ring_enrichment","auroc")]]
            for method,label in methods.items()]
    write_table("cls_table.tex","@{}lrrr@{}","Layer-12 readout & Margin Spearman & Enrichment & AUROC",body)

if __name__ == "__main__":
    main()
