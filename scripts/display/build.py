"""Rebuild every plotted figure and generated table in the manuscript.

Run from a checkout with the project Python environment:
    python scripts/display/build.py
All inputs are saved repository evidence; this command does not rerun experiments.
"""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DISPLAY = Path(__file__).resolve().parent


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def main():
    # Keep PDF creation dates fixed across builds of the same scientific figures.
    os.environ.setdefault("SOURCE_DATE_EPOCH", "1794528000")
    run(DISPLAY / "build_registered_tables.py")
    run(DISPLAY / "build_exploratory_tables.py")
    run(DISPLAY / "build_static_figures.py")
    run(DISPLAY / "build_diagnostic_figure.py")
    run(ROOT / "experiments/endpoints/plot_curves.py",
        "--input", ROOT / "experiments/endpoints/budget_differences.csv",
        "--output", ROOT / "paper/figures/deletion_curves.pdf")
    expected = (
        "fast_weight_lens.pdf", "diagnosis_correction_bridge.pdf",
        "execution_k_sweep.pdf", "example_maps.pdf", "example_maps_more.pdf",
        "deletion_curves.pdf",
    )
    missing = [name for name in expected if not (ROOT / "paper/figures" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing manuscript figures: {missing}")
    print("All manuscript figures and generated tables rebuilt.")


if __name__ == "__main__":
    main()
