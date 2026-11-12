"""Post-result endpoint sensitivity using saved patch AUROC, with frozen geometry cuts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analyze import GEOMETRY, HERE, MODELS, N_BOOT, ROOT, bootstrap_ci, read_model


def main() -> None:
    metadata = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
    rows, contrasts = [], []
    for mi, model in enumerate(MODELS):
        base, _, _ = read_model(model)
        path = ROOT / "outputs/supplementary_eval/imagenet_s" / model / "plausibility.csv.gz"
        data = pd.read_csv(path, usecols=["index", "method", "auroc"])
        data = data[data.method.isin(("C3", "R0", "stand_in_no_high"))]
        if data.duplicated(["index", "method"]).any():
            raise ValueError(f"{model}: duplicate AUROC rows")
        table = data.pivot(index="index", columns="method", values="auroc").sort_index()
        if not base.index.equals(table.index):
            raise ValueError(f"{model}: AUROC images differ from enrichment images")
        table = table.join(base[list(GEOMETRY)], validate="one_to_one")
        for ci, control in enumerate(("R0", "stand_in_no_high")):
            paired = table[["C3", control, "R0", *GEOMETRY]].copy()
            paired = paired.loc[:, ~paired.columns.duplicated()].dropna(subset=["C3", control, *GEOMETRY])
            paired["difference"] = paired["C3"] - paired[control]
            for vi, variable in enumerate(GEOMETRY):
                paired["quartile"] = np.searchsorted(metadata["cuts"][model][variable],
                                                      paired[variable], side="left") + 1
                groups = {0: paired}
                groups.update({q: paired[paired.quartile == q] for q in range(1, 5)})
                for q, group in groups.items():
                    values = group.difference.to_numpy()
                    if len(values) == 0:
                        raise ValueError(f"empty AUROC stratum: {model}/{control}/{variable}/Q{q}")
                    seed = 20261114 + 1000 * mi + 100 * ci + 10 * vi + q
                    low, high = bootstrap_ci(values, seed)
                    rows.append({"model": model, "control": control, "geometry": variable,
                                 "quartile": "all" if q == 0 else f"Q{q}",
                                 "n": len(values), "mean": values.mean(),
                                 "ci_low": low, "ci_high": high,
                                 "r0_median": group.R0.median()})
                lo, hi = (groups[q].difference.to_numpy() for q in (1, 4))
                rng = np.random.default_rng(20261114 + 1000 * mi + 100 * ci + 10 * vi + 9)
                boot = np.empty(N_BOOT)
                for start in range(0, N_BOOT, 100):
                    end = min(start+100, N_BOOT)
                    ld = rng.integers(0, len(lo), size=(end-start, len(lo)))
                    hd = rng.integers(0, len(hi), size=(end-start, len(hi)))
                    boot[start:end] = hi[hd].mean(axis=1) - lo[ld].mean(axis=1)
                contrasts.append({"model": model, "control": control, "geometry": variable,
                                  "contrast": "Q4-Q1", "n_q1": len(lo), "n_q4": len(hi),
                                  "mean": hi.mean() - lo.mean(),
                                  "ci_low": np.quantile(boot, 0.025),
                                  "ci_high": np.quantile(boot, 0.975)})
    pd.DataFrame(rows).to_csv(HERE / "auroc_strata.csv", index=False, float_format="%.9f")
    pd.DataFrame(contrasts).to_csv(HERE / "auroc_extreme_contrasts.csv", index=False, float_format="%.9f")
    print(f"Wrote {len(rows)} AUROC stratum rows and {len(contrasts)} contrasts")


if __name__ == "__main__":
    main()
