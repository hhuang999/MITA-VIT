"""Exploratory joint area-by-distance tabulation, using frozen cuts from analyze.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analyze import GEOMETRY, HERE, MODELS, N_BOOT, RNG_SEED, data_by_model


def main() -> None:
    metadata = json.loads((HERE / "metadata.json").read_text(encoding="utf-8"))
    rows, contrasts = [], []
    for mi, model in enumerate(MODELS):
        table, _ = data_by_model(model)
        cuts = metadata["cuts"][model]
        table["area_q"] = np.searchsorted(cuts["object_area"], table.object_area, side="left") + 1
        table["distance_q"] = np.searchsorted(cuts["object_distance"], table.object_distance, side="left") + 1
        for ci, control in enumerate(("R0", "stand_in_no_high")):
            pair = table[["C3", control, "area_q", "distance_q"]].dropna().copy()
            pair["difference"] = pair.C3 - pair[control]
            for aq in range(1, 5):
                for dq in range(1, 5):
                    cell = pair[(pair.area_q == aq) & (pair.distance_q == dq)].difference.to_numpy()
                    rows.append({"model": model, "control": control, "area_q": aq,
                                 "distance_q": dq, "n": len(cell),
                                 "mean": cell.mean() if len(cell) else np.nan})
                lo = pair[(pair.area_q == aq) & (pair.distance_q == 1)].difference.to_numpy()
                hi = pair[(pair.area_q == aq) & (pair.distance_q == 4)].difference.to_numpy()
                if len(lo) == 0 or len(hi) == 0:
                    contrasts.append({"model": model, "control": control, "area_q": aq,
                                      "n_d1": len(lo), "n_d4": len(hi),
                                      "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan})
                    continue
                rng = np.random.default_rng(RNG_SEED + 1 + mi * 100 + ci * 10 + aq)
                boot = np.empty(N_BOOT)
                for start in range(0, N_BOOT, 100):
                    end = min(start + 100, N_BOOT)
                    ld = rng.integers(0, len(lo), size=(end-start, len(lo)))
                    hd = rng.integers(0, len(hi), size=(end-start, len(hi)))
                    boot[start:end] = hi[hd].mean(axis=1) - lo[ld].mean(axis=1)
                contrasts.append({"model": model, "control": control, "area_q": aq,
                                  "n_d1": len(lo), "n_d4": len(hi),
                                  "mean": hi.mean() - lo.mean(),
                                  "ci_low": np.quantile(boot, 0.025),
                                  "ci_high": np.quantile(boot, 0.975)})
    pd.DataFrame(rows).to_csv(HERE / "joint_cells.csv", index=False, float_format="%.9f")
    pd.DataFrame(contrasts).to_csv(HERE / "joint_contrasts.csv", index=False, float_format="%.9f")
    print(f"Wrote {len(rows)} joint cells and {len(contrasts)} contrasts")


if __name__ == "__main__":
    main()
