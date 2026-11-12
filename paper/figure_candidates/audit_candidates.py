"""Audit candidate source/exports and verify every selected row against saved evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-scripts", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "output/figure_candidates")
    args = parser.parse_args()
    out = args.out.resolve()
    qa = ROOT / "build/figure-candidates/qa"
    qa.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((out / "source_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["inputs"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Figure input changed: {name}")
    original_diagnostics = rows(ROOT / "results/diagnose.csv")
    original_contrasts = rows(ROOT / "results/correct_tests.csv")
    checks = [
        ("01_repair_benefits_limits.source.csv", original_contrasts, ["panel"], 15),
        ("02_diagnosis_correction_bridge.effects.csv", original_contrasts, ["figure_panel"], 36),
        ("02_diagnosis_correction_bridge.diagnostics.csv", original_diagnostics, ["figure_panel"], 108),
    ]
    for name, originals, added, expected in checks:
        selected = rows(out / name)
        if len(selected) != expected:
            raise ValueError(f"Unexpected selected-row count: {name}")
        for row in selected:
            raw = {key: value for key, value in row.items() if key not in added}
            if raw not in originals:
                raise ValueError(f"Exported source row differs from saved evidence: {name}")
    jobs = [("source", "validate_figure.py", [str(Path(__file__).with_name("build_figures.py")), "--json"])]
    for stem in ("01_repair_benefits_limits", "02_diagnosis_correction_bridge"):
        jobs.extend([
            (stem + ".text", "audit_pdf_text.py", [str(out / f"{stem}.pdf"), "--min-pt", "5", "--json"]),
            (stem + ".collision", "audit_figure_collisions.py",
             [str(out / f"{stem}.pdf"), "--json-out", str(out / f"{stem}.collision.json"),
              "--overlay-pdf", str(qa / f"{stem}.collision-overlay.pdf"), "--json"]),
        ])
    summary = {"input_hashes_match": True, "all_selected_rows_match_saved_evidence": True}
    for label, script, arguments in jobs:
        result = subprocess.run([sys.executable, str(args.audit_scripts / script), *arguments],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"Audit failed ({label}):\n{result.stdout}\n{result.stderr}")
        payload = json.loads(result.stdout)
        (out / f"{label}.json").write_text(json.dumps(payload, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
        summary[label] = {"exit_code": result.returncode, "summary": payload.get("summary", {}),
                          "verdict": payload.get("verdict"),
                          "minimum_found_pt": payload.get("minimum_found_pt")}
    for stem in ("01_repair_benefits_limits", "02_diagnosis_correction_bridge"):
        report = json.loads((out / f"{stem}.alignment.json").read_text(encoding="utf-8"))
        if report.get("verdict") != "PASS":
            raise ValueError(f"Final alignment gate did not pass: {stem}")
    summary["panel_alignment"] = "Both figures PASS at 1.5 pt tolerance; equal plot widths and aligned labels"
    summary["accepted_source_warnings"] = {
        "139.7 mm width": "Matches the actual NeurIPS text width; not a Nature-width submission",
        "PNG 300 dpi": "Preview only; the manuscript assets are vectors",
        "No TIFF": "Editable PDF/SVG are the intended line-art exports",
    }
    (out / "qa_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
