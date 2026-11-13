"""Recompute the reviewer-requested receiver contrasts from saved observations."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = HERE / "deit_b"
    metadata = json.loads((source / "metadata.json").read_text())
    masks = json.loads((source / "three_endpoint_metadata.json").read_text())
    names = (source / "selected_names.txt").read_text().splitlines()
    assert names == (ROOT / "experiments/cls/selected_names.txt").read_text().splitlines()
    assert len(names) == len(set(names)) == 128
    assert metadata["checkpoint"] == ["deit_base_patch16_224.fb_in1k",
                                      "b78cc5532a69df6bcad9c3a8d76653fd20b31ac6"]
    assert metadata["protocol_sha256"] == masks["addendum_protocol_sha256"] == digest(HERE / "protocol.md")
    for filename, field in (("per_source.csv.gz", "per_source_sha256"),
                            ("per_image.csv", "per_image_sha256"),
                            ("selected_names.txt", "selected_names_sha256")):
        assert digest(source / filename) == masks[field]
    checks = pd.read_csv(source / "forward_validation.csv")
    assert len(checks) == 12
    assert checks[["max_logit_error", "fixed_margin_error"]].to_numpy().max() < .002
    assert metadata["max_baseline_logit_error"] < .002
    assert metadata["max_reference_logit_or_map_error"] < .002
    grid = pd.read_csv(source / "per_source.csv.gz")
    assert len(grid) == 128 * 196
    assert not grid.duplicated(["image", "source_patch"]).any()
    assert set(grid.image) == set(names)
    assert all(sorted(group.source_patch) == list(range(196)) for _, group in grid.groupby("image"))
    image = pd.read_csv(source / "per_image.csv").set_index("image")
    methods = ("all_r0", "all_c3", "cls_attention", "cls_sum_head", "cls_net")
    assert np.isfinite(grid[["abs_margin_change", *methods]]).all().all()
    for name, group in grid.groupby("image"):
        for method in methods:
            computed = spearmanr(group.abs_margin_change, group[method]).statistic
            np.testing.assert_allclose(computed, image.loc[name, method], atol=1e-12)
    endpoints = pd.read_csv(source / "three_endpoint_per_image.csv")
    assert len(endpoints) == 128 * 5
    assert masks["eligible"] == 128 and not masks["excluded"]
    draws = np.random.default_rng(2026).integers(128, size=(10000, 128))
    contrasts = pd.read_csv(source / "three_endpoint_contrasts.csv")
    for row in contrasts.itertuples():
        wide = endpoints.pivot(index="image", columns="method", values=row.endpoint).loc[names]
        delta = (wide[row.a] - wide[row.b]).to_numpy()
        actual = [delta.mean(), *np.quantile(delta[draws].mean(axis=1), [.025, .975])]
        np.testing.assert_allclose(actual, [row.mean, row.low, row.high], atol=1e-12)
    # Assert the counterexample is retained: both endpoints favour CLS in DeiT-B.
    primary = contrasts[contrasts.a.eq("cls_sum_head")].set_index("endpoint")
    assert primary.loc["auroc", "low"] > 0
    assert primary.loc["margin_spearman", "low"] > 0
    assert primary.loc["ring_enrichment", "low"] < 0 < primary.loc["ring_enrichment", "high"]
    print("DeiT-B audit passed: 128 images, 25,088 interventions, five rankings, six paired intervals.")


if __name__ == "__main__":
    main()
