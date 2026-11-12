"""Exploratory ImageNet-S object-geometry moderation of the saved C3 reading.

Run from anywhere with: python research/v13/geometry/analyze.py
This reads immutable per-image exports and writes only beside this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODELS = ("deit", "deit_b", "dino")
CONTROLS = ("R0", "stand_in_no_high", "C3headmean", "C3shuffle_mean")
SEEDS = tuple(f"C3shuffle_s{i}" for i in range(5))
GEOMETRY = ("object_area", "object_distance")
USECOLS = ("index", "method", "enrichment", *GEOMETRY)
N_BOOT = 2000
RNG_SEED = 20261112


def read_model(model: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    main_path = ROOT / "outputs/table1/imagenet_s" / model / "plausibility.csv.gz"
    supp_path = ROOT / "outputs/supplementary_eval/imagenet_s" / model / "plausibility.csv.gz"
    main = pd.read_csv(main_path, usecols=USECOLS)
    supp = pd.read_csv(supp_path, usecols=USECOLS)
    main = main[main.method.isin(("C3", "R0"))].copy()
    supp = supp[supp.method.isin(("C3", "R0", "stand_in_no_high", "C3headmean", *SEEDS))].copy()
    for source_name, source in (("main", main), ("supplementary", supp)):
        if source.duplicated(["index", "method"]).any():
            raise ValueError(f"{model} {source_name}: duplicate image-method rows")
        expected = source.groupby("method").size()
        if expected.nunique() != 1:
            raise ValueError(f"{model} {source_name}: unequal method sample sizes: {expected}")
    audit = []
    for method in ("C3", "R0"):
        left = main[main.method == method].set_index("index").sort_index()
        right = supp[supp.method == method].set_index("index").sort_index()
        if not left.index.equals(right.index):
            raise ValueError(f"{model}: main/supplementary index mismatch ({method})")
        for column in ("enrichment", *GEOMETRY):
            bad = ~np.isclose(left[column], right[column], atol=1e-12, rtol=0, equal_nan=True)
            if column in GEOMETRY and bad.any():
                raise ValueError(f"{model}: main/supplementary {column} mismatch ({method})")
            if bad.any():
                audit.extend({"model": model, "method": method, "index": int(index),
                              "main": float(left.loc[index, column]),
                              "supplementary": float(right.loc[index, column])}
                             for index in left.index[bad])
    if len(audit) > 2 or any(row["model"] != "deit" or row["method"] != "R0"
                             or abs(row["main"] - row["supplementary"]) > 0.006757
                             for row in audit):
        raise ValueError(f"Unexpected main/supplementary disagreement: {audit}")
    base = main[main.method == "R0"].set_index("index").sort_index()
    for method, part in supp.groupby("method"):
        part = part.set_index("index").sort_index()
        if not base.index.equals(part.index):
            raise ValueError(f"{model}: index mismatch for {method}")
        for column in GEOMETRY:
            if not np.allclose(base[column], part[column], atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError(f"{model}: object geometry mismatch for {method}")
    return base, supp, audit


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 100):
        stop = min(start + 100, N_BOOT)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def data_by_model(model: str) -> tuple[pd.DataFrame, list[dict]]:
    base, supp, audit = read_model(model)
    values = supp.pivot(index="index", columns="method", values="enrichment").sort_index()
    values["C3shuffle_mean"] = values[list(SEEDS)].mean(axis=1, skipna=False)
    values = values.join(base[list(GEOMETRY)], validate="one_to_one")
    return values, audit


def main() -> None:
    metadata = {"analysis": "post-registration exploratory", "n_geometry_cohort": {},
                "quantile_rule": "empirical quartiles; exact ties assigned to lower bin",
                "cuts": {}, "n_boot": N_BOOT, "rng_seed": RNG_SEED,
                "sources": ["outputs/table1/imagenet_s/*/plausibility.csv.gz",
                            "outputs/supplementary_eval/imagenet_s/*/plausibility.csv.gz"]}
    rows = []
    contrasts = []
    audits = []
    base_indices = None
    for model_idx, model in enumerate(MODELS):
        table, audit = data_by_model(model)
        if base_indices is None:
            base_indices = table.index
        elif not base_indices.equals(table.index):
            raise ValueError(f"{model}: image IDs differ from DeiT-S")
        cohort = table[list(GEOMETRY)].dropna()
        cuts = {column: [float(v) for v in cohort[column].quantile([0.25, 0.5, 0.75])]
                for column in GEOMETRY}
        metadata["n_geometry_cohort"][model] = len(cohort)
        metadata["cuts"][model] = cuts
        audits.extend(audit)
        for control_idx, control in enumerate(CONTROLS):
            paired = table[["C3", control, *GEOMETRY, "R0"]].copy()
            # R0 appears twice when it is the comparison control; select columns once.
            paired = paired.loc[:, ~paired.columns.duplicated()].dropna(subset=["C3", control, *GEOMETRY])
            paired["difference"] = paired["C3"] - paired[control]
            for variable_idx, variable in enumerate(GEOMETRY):
                cut = cuts[variable]
                paired["quartile"] = np.searchsorted(cut, paired[variable].to_numpy(), side="left") + 1
                groups = {0: paired}
                groups.update({q: paired.loc[paired.quartile == q] for q in range(1, 5)})
                for q, group in groups.items():
                    values = group["difference"].to_numpy()
                    if len(values) == 0:
                        raise ValueError(f"empty stratum: {model}/{control}/{variable}/Q{q}")
                    seed = RNG_SEED + 1000 * model_idx + 100 * control_idx + 10 * variable_idx + q
                    low, high = bootstrap_ci(values, seed)
                    rows.append({"model": model, "control": control, "geometry": variable,
                                 "quartile": "all" if q == 0 else f"Q{q}",
                                 "n": len(values), "mean": values.mean(), "ci_low": low,
                                 "ci_high": high, "r0_median": group["R0"].median(),
                                 "geometry_min": group[variable].min(), "geometry_max": group[variable].max()})
                lo = groups[1]["difference"].to_numpy()
                hi = groups[4]["difference"].to_numpy()
                rng = np.random.default_rng(RNG_SEED + 1000 * model_idx + 100 * control_idx + 10 * variable_idx + 9)
                boot = np.empty(N_BOOT)
                for start in range(0, N_BOOT, 100):
                    stop = min(start + 100, N_BOOT)
                    lo_draw = rng.integers(0, len(lo), size=(stop - start, len(lo)))
                    hi_draw = rng.integers(0, len(hi), size=(stop - start, len(hi)))
                    boot[start:stop] = hi[hi_draw].mean(axis=1) - lo[lo_draw].mean(axis=1)
                contrasts.append({"model": model, "control": control, "geometry": variable,
                                  "contrast": "Q4-Q1", "n_q1": len(lo), "n_q4": len(hi),
                                  "mean": hi.mean() - lo.mean(),
                                  "ci_low": np.quantile(boot, 0.025),
                                  "ci_high": np.quantile(boot, 0.975)})
    HERE.mkdir(parents=True, exist_ok=True)
    metadata["main_supplementary_discrepancies"] = audits
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(HERE / "strata.csv", index=False, float_format="%.9f")
    pd.DataFrame(contrasts).to_csv(HERE / "extreme_contrasts.csv", index=False, float_format="%.9f")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {len(rows)} stratum rows and {len(contrasts)} contrasts")


if __name__ == "__main__":
    main()
