"""Post-result, no-forward comparison of CLS sensitivity and object localization.

Reads the fixed 128-image, five-score CLS output and Oxford trimaps. See
addendum_protocol.md for the timing and scope of this exploratory check.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "build/paper-v9/python-deps"))
sys.path.insert(0, str(ROOT / "research/v13/spatial"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from fastweight_vit.enrichment import measures  # noqa: E402
from fastweight_vit.models.deit import load_deit  # noqa: E402
from run_spatial import PetTest  # noqa: E402


METHODS = ("all_r0", "all_c3", "cls_attention", "cls_sum_head", "cls_net")
ENDPOINTS = ("margin_spearman", "ring_enrichment", "auroc")
CONTRASTS = (("cls_sum_head", "all_c3"), ("cls_net", "cls_sum_head"))
SEED = 2026
RESAMPLES = 10000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values, draws):
    sample_means = values[draws].mean(axis=1)
    low, high = np.quantile(sample_means, [.025, .975])
    return float(values.mean()), float(low), float(high)


def main():
    started = time.perf_counter()
    names = [line.strip() for line in (HERE / "selected_names.txt").read_text().splitlines() if line.strip()]
    if len(names) != 128 or len(set(names)) != 128:
        raise ValueError("Expected 128 unique names from the CLS experiment")
    sources = pd.read_csv(HERE / "per_source.csv.gz")
    original = pd.read_csv(HERE / "per_image.csv")
    if len(sources) != 128 * 196 or sources.duplicated(["image", "source_patch"]).any():
        raise ValueError("Incomplete or duplicate CLS per-source output")
    if len(original) != 128 or original.image.duplicated().any():
        raise ValueError("Incomplete CLS per-image output")
    if set(original.image) != set(names) or set(sources.image) != set(names):
        raise ValueError("Selection mismatch between saved CLS sources and names")
    if not np.isfinite(sources[list(METHODS) + ["abs_margin_change"]].to_numpy()).all():
        raise ValueError("Non-finite saved CLS score or effect")

    # The model is instantiated only to recover its pinned evaluation transform.
    # No weights are loaded and no image is passed through the model.
    transform_reference = load_deit("cpu", pretrained=False)
    dataset = PetTest(transform_reference)
    index = {name: i for i, name in enumerate(dataset.names)}
    if not set(names) <= set(index):
        raise ValueError("CLS subset is outside the spatial-study Oxford selection")
    foreground, valid, excluded = [], [], []
    eligible = []
    for name in names:
        _, _, _, fg, is_valid = dataset.aligned(index[name])
        n_fg = int((fg & is_valid).sum())
        n_bg = int((~fg & is_valid).sum())
        if n_fg == 0 or n_bg == 0:
            excluded.append({"image": name, "valid_object_patches": n_fg,
                             "valid_background_patches": n_bg})
            continue
        eligible.append(name)
        foreground.append(fg)
        valid.append(is_valid)
    if len(eligible) < 2:
        raise ValueError("Too few object/background-eligible images")
    foreground = torch.stack(foreground)
    valid = torch.stack(valid)

    positions = pd.MultiIndex.from_product([eligible, range(196)], names=["image", "source_patch"])
    score_grid = sources.set_index(["image", "source_patch"]).reindex(positions)
    if score_grid[list(METHODS)].isna().any().any():
        raise ValueError("A selected image is missing a source patch")
    original = original.set_index("image").loc[eligible]
    rows = []
    for method in METHODS:
        score = torch.from_numpy(score_grid[method].to_numpy().reshape(len(eligible), 196).copy()).float()
        values = measures(score, foreground, valid, foreground,
                          generator=torch.Generator().manual_seed(SEED))
        for i, name in enumerate(eligible):
            rows.append({"image": name, "method": method,
                         "margin_spearman": float(original.loc[name, method]),
                         "ring_enrichment": float(values["enrichment"][i]),
                         "auroc": float(values["auroc"][i])})
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[list(ENDPOINTS)].to_numpy()).all():
        raise ValueError("A common-sample endpoint is non-finite")
    frame.to_csv(HERE / "three_endpoint_per_image.csv", index=False)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(eligible), size=(RESAMPLES, len(eligible)))
    summaries, contrasts = [], []
    for endpoint in ENDPOINTS:
        wide = frame.pivot(index="image", columns="method", values=endpoint).loc[eligible]
        for method in METHODS:
            mean, low, high = bootstrap(wide[method].to_numpy(), draws)
            summaries.append({"endpoint": endpoint, "method": method, "n": len(eligible),
                              "mean": mean, "low": low, "high": high})
        for a, b in CONTRASTS:
            mean, low, high = bootstrap((wide[a] - wide[b]).to_numpy(), draws)
            contrasts.append({"endpoint": endpoint, "a": a, "b": b, "n": len(eligible),
                              "mean": mean, "low": low, "high": high})
    pd.DataFrame(summaries).to_csv(HERE / "three_endpoint_summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(HERE / "three_endpoint_contrasts.csv", index=False)
    metadata = {"selected": len(names), "eligible": len(eligible), "excluded": excluded,
                "no_new_model_forward": True,
                "bootstrap_resamples": RESAMPLES, "bootstrap_seed": SEED,
                "tie_noise_seed_shared_by_methods": SEED,
                "selected_names_sha256": digest(HERE / "selected_names.txt"),
                "per_source_sha256": digest(HERE / "per_source.csv.gz"),
                "per_image_sha256": digest(HERE / "per_image.csv"),
                "addendum_protocol_sha256": digest(HERE / "addendum_protocol.md"),
                "script_sha256": digest(Path(__file__)),
                "elapsed_seconds": time.perf_counter() - started}
    (HERE / "three_endpoint_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
