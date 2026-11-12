"""Render the ImageNet-S subgroup summary at the physical size used on page one."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
source = root / "research/v13/geometry"
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.62), sharey=False)
for ax, filename, title in zip(axes, ["strata.csv", "auroc_strata.csv"], ["Object enrichment", "Patch AUROC"]):
    with (source / filename).open(newline="") as stream:
        data = list(csv.DictReader(stream))
    for model,label,colour,marker in [("deit","DeiT-S","#26668a","o"),("deit_b","DeiT-B","#a9510b","s")]:
        rows=[r for r in data if r["model"]==model and r["control"]=="R0" and
              r["geometry"]=="object_area" and r["quartile"].startswith("Q")]
        rows.sort(key=lambda r:r["quartile"])
        values=[float(r["mean"]) for r in rows]
        ax.errorbar(range(1,5),values,yerr=[[v-float(r["ci_low"]) for v,r in zip(values,rows)],
                    [float(r["ci_high"])-v for v,r in zip(values,rows)]],label=label, color=colour,
                    marker=marker,linestyle="-" if model=="deit" else "--",lw=1,ms=3,capsize=1.5)
    ax.axhline(0,color=".55",lw=.5)
    ax.set(xticks=[1,2,3,4], xlim=(.8,4.2),title=title,ylim=(-.009,.18),yticks=[0,.05,.1,.15])
    if title == "Patch AUROC":
        ax.set(ylim=(-.003,.06),yticks=[0,.02,.04,.06])
    ax.tick_params(length=2,pad=2)
axes[0].set_ylabel("C3 minus R0")
fig.text(.55,.21,"Foreground area quartile",ha="center",size=7)
fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False,fontsize=7,bbox_to_anchor=(.54,-.035))
fig.subplots_adjust(left=.15,right=.985,top=.84,bottom=.41,wspace=.4)
fig.savefig(root/"paper/v13/geometry_effects.pdf")
