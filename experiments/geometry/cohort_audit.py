"""Audit historical ImageNet-S enrichment denominators without changing saved results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.geometry.analyze import HERE, MODELS, ROOT


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    archived = pd.read_csv(ROOT / "results/correct_tests.csv")
    rows = []
    for model in MODELS:
        path = ROOT / "outputs/table1/imagenet_s" / model / "plausibility.csv.gz"
        data = pd.read_csv(path, usecols=["index", "method", "enrichment", "auroc",
                                          "object_area", "object_distance"])
        data = data[data.method.isin(("C3", "R0", "stand_in_no_high"))]
        if data.duplicated(["index", "method"]).any():
            raise ValueError(f"{model}: duplicate main rows")
        geom = data[data.method == "R0"].set_index("index")["object_area"].sort_index()
        enrichment = data.pivot(index="index", columns="method", values="enrichment").sort_index()
        auroc = data.pivot(index="index", columns="method", values="auroc").sort_index()
        if not geom.index.equals(enrichment.index) or not geom.index.equals(auroc.index):
            raise ValueError(f"{model}: image IDs differ across outcomes")
        area_zero = (geom == 0)
        area_one = (geom == 1)
        both = (geom > 0) & (geom < 1)
        for control in ("R0", "stand_in_no_high"):
            finite = np.isfinite(enrichment.C3) & np.isfinite(enrichment[control])
            delta = enrichment.C3[finite] - enrichment[control][finite]
            two_class_delta = (enrichment.C3[finite & both]
                               - enrichment[control][finite & both])
            archived_row = archived[(archived.model == model) & (archived.metric == "enrichment")
                                    & (archived.a == "C3") & (archived.b == control)].iloc[0]
            if int(archived_row.n) != len(delta) or not np.isclose(float(archived_row["mean"]),
                                                                    delta.mean(), atol=1e-12, rtol=0):
                raise ValueError(f"{model}/{control}: saved historical summary disagrees with per-image export")
            finite_auroc = np.isfinite(auroc.C3) & np.isfinite(auroc[control])
            rows.append({"model": model, "comparison": f"C3 - {control}",
                         "total_image_ids": len(geom),
                         "historical_finite_enrichment_n": int(finite.sum()),
                         "area_zero_n": int(area_zero.sum()), "area_one_n": int(area_one.sum()),
                         "two_classes_n": int(both.sum()),
                         "historical_two_classes_n": int((finite & both).sum()),
                         "finite_auroc_pair_n": int(finite_auroc.sum()),
                         "historical_mean": float(archived_row["mean"]),
                         "two_classes_mean": two_class_delta.mean(),
                         "two_classes_minus_historical": two_class_delta.mean() - float(archived_row["mean"])})
    pd.DataFrame(rows).to_csv(HERE / "cohort_audit.csv", index=False, float_format="%.12f")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
