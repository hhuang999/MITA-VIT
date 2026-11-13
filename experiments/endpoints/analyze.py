"""Exploratory continuous deletion endpoints and ClickMe batch sensitivity.

Run from the repository root with the project Python environment. Inputs are
archived per-image CSV files. New outputs go to outputs/rerun/endpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("FASTWEIGHT_ENDPOINT_OUTPUT_DIR",
                          ROOT / "outputs/rerun/endpoints")).resolve()
ARCHIVE = Path(__file__).resolve().parent
MODELS = ("deit", "deit_b", "dino")
METHODS = ("R0", "C3", "stand_in_no_high")
CONTRASTS = (("C3", "R0"), ("C3", "stand_in_no_high"),
             ("stand_in_no_high", "R0"))
BUDGETS = (1, 2, 4, 8, 16, 32, 64, 98, 147)
X = np.array((0,) + BUDGETS, dtype=float) / 196
RESAMPLES = 1000
SEED = 0


def interval(values: np.ndarray, *, clusters: np.ndarray | None = None) -> tuple[float, float, float]:
    """Use the original image bootstrap RNG or resample equal-size batch means."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("A paired contrast has non-finite observations")
    if clusters is not None:
        group = pd.DataFrame({"value": values, "cluster": clusters}).groupby("cluster")
        if group.size().nunique() != 1:
            raise ValueError("Cluster sizes are unequal; weighted resampling is required")
        values = group.value.mean().to_numpy()
    rng = np.random.default_rng(SEED)
    sampled = np.array([values[rng.integers(len(values), size=len(values))].mean()
                        for _ in range(RESAMPLES)])
    lo, hi = np.quantile(sampled, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def area(separation: np.ndarray) -> np.ndarray:
    """Trapezoidal area per image, with exactly zero condition difference at b=0."""
    y = np.column_stack((np.zeros(len(separation)), separation))
    return (((y[:, 1:] + y[:, :-1]) / 2) * np.diff(X)).sum(axis=1)


def named(values: np.ndarray) -> dict[str, float]:
    mean, low, high = interval(values)
    return {"n": len(values), "mean": mean, "low": low, "high": high}


def validate_faith(frame: pd.DataFrame, model: str) -> np.ndarray:
    keys = ["index", "method", "condition", "budget"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{model}: duplicate faithfulness keys")
    if set(frame.budget) != set(BUDGETS) or set(frame.condition) != {"drop_bottom", "drop_top"}:
        raise ValueError(f"{model}: unexpected budget or deletion order")
    if frame.model.nunique() != 1 or frame.model.iloc[0] != model:
        raise ValueError(f"{model}: model mismatch")
    if not np.isfinite(frame[["agree", "p_target", "kl"]].to_numpy()).all():
        raise ValueError(f"{model}: non-finite stored endpoint")
    expected = set(frame.loc[frame.method == "R0", "index"])
    if len(expected) != 2000:
        raise ValueError(f"{model}: expected 2,000 paired images")
    counts = frame.groupby("method", sort=False).agg(images=("index", "nunique"), rows=("index", "size"))
    if not ((counts.images == 2000) & (counts.rows == 2000 * 2 * len(BUDGETS))).all():
        raise ValueError(f"{model}: incomplete image/method/condition/budget grid")
    if not {"R0", "C3", "stand_in_no_high"} <= set(counts.index):
        raise ValueError(f"{model}: a preselected method is missing")
    # A full row count plus uniqueness and 2,000 unique indices ensures each method
    # has all 18 condition/budget rows for its images, but also check shared IDs.
    for method, part in frame.groupby("method"):
        if set(part["index"]) != expected:
            raise ValueError(f"{model}: nonpaired image sets for {method}")
        if not (part.groupby("index").size() == 2 * len(BUDGETS)).all():
            raise ValueError(f"{model}: incomplete per-image grid for {method}")
    return np.array(sorted(expected))


def faithfulness(model: str, historical: pd.DataFrame):
    frame = pd.read_csv(ROOT / "outputs" / "table1" / "imagenet_s" / model / "faithfulness.csv.gz")
    ids = validate_faith(frame, model)
    selected = frame[frame.method.isin(METHODS)]
    if len(selected) != 2000 * len(METHODS) * 2 * len(BUDGETS):
        raise ValueError(f"{model}: selected-grid size mismatch")
    areas, budget_rows, integrated_rows, checks = {}, [], [], []
    for metric in ("agree", "p_target", "kl"):
        wide = selected.pivot(index="index", columns=["method", "condition", "budget"], values=metric)
        wide = wide.sort_index(axis=1)
        wide = wide.loc[ids]
        separations = {}
        for method in METHODS:
            bottom = wide[method, "drop_bottom"].loc[:, BUDGETS].to_numpy()
            top = wide[method, "drop_top"].loc[:, BUDGETS].to_numpy()
            # Original v12 top-1 SRG is LIF - MIF; the zero-budget
            # agreement is 1 for both and therefore cancels exactly.
            separations[method] = top - bottom if metric == "kl" else bottom - top
        areas[metric] = {method: area(values) for method, values in separations.items()}
        for a, b in CONTRASTS:
            diff = areas[metric][a] - areas[metric][b]
            integrated_rows.append({"model": model, "metric": metric, "a": a, "b": b,
                                    "unit": "fraction" if metric == "agree" else
                                    "probability_fraction" if metric == "p_target" else "nats_fraction",
                                    **named(diff)})
            for j, budget in enumerate(BUDGETS):
                by_image = separations[a][:, j] - separations[b][:, j]
                budget_rows.append({"model": model, "metric": metric, "a": a, "b": b,
                                    "budget": budget, **named(by_image)})
            if metric == "agree":
                old = historical[(historical.model == model) & (historical.metric == "srg")
                                 & (historical.a == a) & (historical.b == b)]
                if len(old) < 1:
                    raise ValueError(f"{model} {a}-{b}: missing original SRG paired row")
                if old[["n", "mean", "low", "high"]].drop_duplicates().shape[0] != 1:
                    raise ValueError(f"{model} {a}-{b}: conflicting original SRG rows")
                old = old.iloc[0]
                calculated = interval(diff)
                errors = [abs(calculated[i] - float(old[name])) for i, name in enumerate(("mean", "low", "high"))]
                checks.append({"model": model, "metric": "srg", "a": a, "b": b,
                               "max_absolute_error": max(errors), "n_match": int(old.n) == len(diff)})
                if max(errors) > 1e-12 or int(old.n) != len(diff):
                    raise ValueError(f"{model} {a}-{b}: original SRG contrast not reconstructed")
    return integrated_rows, budget_rows, checks


def clickme(model: str, historical: pd.DataFrame):
    frame = pd.read_csv(ROOT / "outputs" / "table1" / "clickme" / model / "alignment.csv.gz")
    if frame.model.nunique() != 1 or frame.model.iloc[0] != model:
        raise ValueError(f"{model}: ClickMe model mismatch")
    if frame.duplicated(["index", "method"]).any():
        raise ValueError(f"{model}: duplicate ClickMe keys")
    if not np.isfinite(frame[["spearman_pixel", "spearman_pixel_shuffled"]].to_numpy()).all():
        raise ValueError(f"{model}: non-finite ClickMe score")
    ids = set(range(17408))
    for method, part in frame.groupby("method"):
        if set(part["index"]) != ids:
            raise ValueError(f"{model}: incomplete ClickMe IDs for {method}")
    frame["specific"] = frame.spearman_pixel - frame.spearman_pixel_shuffled
    wide = frame[frame.method.isin(METHODS)].pivot(index="index", columns="method", values="specific")
    wide = wide.loc[np.arange(17408)]
    clusters = wide.index.to_numpy() // 64
    if len(np.unique(clusters)) != 272 or len(wide) != 272 * 64:
        raise ValueError(f"{model}: expected 272 original 64-image batches")
    rows, checks = [], []
    for a, b in CONTRASTS:
        diff = (wide[a] - wide[b]).to_numpy()
        image_mean, image_lo, image_hi = interval(diff)
        cluster_mean, cluster_lo, cluster_hi = interval(diff, clusters=clusters)
        old = historical[(historical.model == model) & (historical.metric == "clickme")
                         & (historical.a == a) & (historical.b == b)]
        if len(old) < 1:
            raise ValueError(f"{model} {a}-{b}: missing original ClickMe paired row")
        if old[["n", "mean", "low", "high"]].drop_duplicates().shape[0] != 1:
            raise ValueError(f"{model} {a}-{b}: conflicting original ClickMe rows")
        old = old.iloc[0]
        errors = [abs(image_mean - float(old["mean"])), abs(image_lo - float(old["low"])),
                  abs(image_hi - float(old["high"]))]
        checks.append({"model": model, "metric": "clickme", "a": a, "b": b,
                       "max_absolute_error": max(errors), "n_match": int(old.n) == len(diff)})
        if max(errors) > 1e-12 or int(old.n) != len(diff):
            raise ValueError(f"{model} {a}-{b}: original ClickMe contrast not reconstructed")
        if abs(image_mean - cluster_mean) > 1e-12:
            raise ValueError(f"{model}: unequal image and cluster point estimates")
        rows.append({"model": model, "a": a, "b": b, "n_images": len(diff), "n_batches": 272,
                     "mean": image_mean, "image_low": image_lo, "image_high": image_hi,
                     "batch_low": cluster_lo, "batch_high": cluster_hi,
                     "registered_verdict": old.verdict})
    return rows, checks


def main():
    if OUT == ARCHIVE:
        raise ValueError("Output directory must not overwrite archived exploratory results")
    OUT.mkdir(parents=True, exist_ok=True)
    historical = pd.read_csv(ROOT / "results" / "correct_tests.csv")
    integrated, budgets, clickme_rows, checks = [], [], [], []
    for model in MODELS:
        a, b, c = faithfulness(model, historical)
        integrated.extend(a)
        budgets.extend(b)
        checks.extend(c)
        a, c = clickme(model, historical)
        clickme_rows.extend(a)
        checks.extend(c)
    pd.DataFrame(integrated).to_csv(OUT / "deletion_areas.csv", index=False)
    pd.DataFrame(budgets).to_csv(OUT / "budget_differences.csv", index=False)
    pd.DataFrame(clickme_rows).to_csv(OUT / "clickme_bootstrap.csv", index=False)
    source_files = [ROOT / "configs" / "table1.yaml", ROOT / "results" / "correct_tests.csv"]
    for model in MODELS:
        source_files.extend((ROOT / "outputs" / "table1" / "imagenet_s" / model / "faithfulness.csv.gz",
                             ROOT / "outputs" / "table1" / "clickme" / model / "alignment.csv.gz"))
    sources = {str(path.relative_to(ROOT)).replace("\\", "/"):
               hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    report = {"source_sha256": sources,
              "historical_contrasts_verified": len(checks),
              "max_absolute_error": max(row["max_absolute_error"] for row in checks),
              "all_counts_match": all(row["n_match"] for row in checks),
              "checks": checks}
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(checks)} original paired contrasts; maximum error {report['max_absolute_error']:.3g}")
    print(pd.DataFrame(integrated).to_string(index=False))
    print(pd.DataFrame(clickme_rows).to_string(index=False))


if __name__ == "__main__":
    main()
