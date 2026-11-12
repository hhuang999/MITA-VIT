"""Post-result sensitivity analysis: reciprocal donor-pair block bootstrap.

The primary image bootstrap was specified in PROTOCOL.md before the run.
This additional paired-block analysis was requested after the scores were
visible because donor_rank uses reciprocal image pairs. It samples all 192
two-image blocks and uses a ratio of sums so the one block with an excluded
image contributes its one valid image, not the same weight as a full pair.
"""

import csv
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
COMPARISONS = (("C3", "R0"), ("C3", "donor_rank"), ("C3", "ring_shuffle"),
               ("C3", "norm_only"), ("net_output", "C3"), ("C3v", "C3"))


def main():
    with (HERE / "per_image.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    maps = {(int(row["index"]), row["method"]): row for row in rows}
    indices = sorted({index for index, _ in maps})
    rng = np.random.default_rng(2026)
    draws = rng.integers(0, 192, (10000, 192))
    out = {"status": "post-result sensitivity analysis", "n_images": len(indices),
           "n_reciprocal_donor_blocks": 192, "comparisons": {}}
    for left, right in COMPARISONS:
        sum_by_block = np.zeros(192)
        count_by_block = np.zeros(192, dtype=int)
        for index in indices:
            delta = (float(maps[index, left]["enrichment"])
                     - float(maps[index, right]["enrichment"]))
            sum_by_block[index // 2] += delta
            count_by_block[index // 2] += 1
        sampled = sum_by_block[draws].sum(axis=1) / count_by_block[draws].sum(axis=1)
        out["comparisons"][f"{left}-{right}"] = {
            "paired_mean": float(sum_by_block.sum() / count_by_block.sum()),
            "pair_block_ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
            "mean_bootstrap_denominator": float(count_by_block[draws].sum(axis=1).mean()),
        }
    (HERE / "pair_block_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
