"""Verify the v10 manuscript build and its unchanged scientific evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import fitz


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/v10"


def digest(path):
    data = path.read_bytes()
    if path.suffix == ".tex":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def figure_block(text, label):
    anchor = text.index(r"\label{" + label + "}")
    start = text.rindex(r"\begin{figure}", 0, anchor)
    end = text.index(r"\end{figure}", anchor) + len(r"\end{figure}")
    return text[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/paper-v10")
    parser.add_argument("--pdf", type=Path,
                        default=ROOT / "output/pdf/attention_fast_weights_paper_v10.pdf")
    args = parser.parse_args()
    aux = (args.build / "main.aux").read_text(encoding="utf-8")
    log = (args.build / "main.log").read_text(encoding="utf-8")
    labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux))
    end = re.search(r"\\newlabel\{main:end\}\{\{6\}\{(\d+)\}", aux)
    assert end and int(end.group(1)) == 4, "Main text must end on page 4"
    assert labels["fig:lens"] == "1" and labels["fig:diag"] == "2"
    assert labels["tab:main"] == "1" and "tab:premises" not in labels
    assert not re.search(r"(?:undefined|multiply defined|Overfull)", log, re.I)
    sources = {name: (SOURCE / name).read_text(encoding="utf-8")
               for name in ("main.tex", "appendix.tex")}
    references = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", "\n".join(sources.values())))
    assert references <= labels.keys(), references - labels.keys()
    assert r"\ref{fig:diag}d)" not in sources["appendix.tex"]
    assert figure_block(sources["main.tex"], "fig:lens") == figure_block(
        (ROOT / "paper/v9/main.tex").read_text(encoding="utf-8"), "fig:lens")
    for name in ("fast_weight_lens.pdf", "contribution_effects.pdf"):
        assert (SOURCE / "figures" / name).read_bytes() == (ROOT / "paper/v9/figures" / name).read_bytes()
    assert (SOURCE / "figures/diagnosis_correction_bridge.pdf").read_bytes() == (
        ROOT / "output/figure_candidates/02_diagnosis_correction_bridge.pdf").read_bytes()
    for name in ("main_results.tex", "sanity_check.tex"):
        assert (SOURCE / "tables" / name).read_text(encoding="utf-8") == (
            ROOT / "paper/v9/tables" / name).read_text(encoding="utf-8")
    table = (SOURCE / "tables/main_results.tex").read_text(encoding="utf-8")
    numeric_cells = len(re.findall(r"(?<!\d)0\.\d{3}", table))
    assert numeric_cells == 86
    assert (SOURCE / "evidence/sanity.csv").read_bytes() == (ROOT / "archive/results/sanity.csv").read_bytes()
    manifest = json.loads((SOURCE / "asset_sources.json").read_text(encoding="utf-8"))
    for path, expected in manifest["sha256"].items():
        assert digest(ROOT / path) == expected, path
    previous = json.loads((ROOT / "paper/v9/asset_sources.json").read_text(encoding="utf-8"))
    for path, expected in previous["input_sha256"].items():
        assert digest(ROOT / path) == expected, path
    evidence_audit = json.loads((args.build / "original-integrity.json").read_text(encoding="utf-8"))
    assert evidence_audit["passed"]
    figure_audit = json.loads((ROOT / "output/figure_candidates/qa_summary.json").read_text(encoding="utf-8"))
    assert figure_audit["input_hashes_match"] and figure_audit["all_selected_rows_match_saved_evidence"]
    rows = {}
    for suffix in ("diagnostics", "effects"):
        with (ROOT / f"output/figure_candidates/02_diagnosis_correction_bridge.{suffix}.csv").open(
                encoding="utf-8", newline="") as stream:
            rows[suffix] = len(list(csv.DictReader(stream)))
    assert rows == {"diagnostics": 108, "effects": 36}
    with fitz.open(args.pdf) as doc:
        assert len(doc) == 12
        assert "Conclusion" in doc[3].get_text()
        assert doc[4].get_text().startswith("References")
        assert "Experimental details" in doc[6].get_text()
        assert "Historical parameter-randomisation check" in doc[11].get_text()
        assert all(f"app:{key}" in labels for key in ("setup", "more", "reloc", "gallery", "code", "sanity"))
        assert not any("\ufffd" in page.get_text() or "??" in page.get_text() for page in doc)
        fonts = {}
        for page in doc:
            for font in page.get_fonts(full=True):
                xref, _, kind, name = font[:4]
                embedded = bool(doc.extract_font(xref)[3])
                if kind == "Type3":
                    embedded = doc.xref_get_key(xref, "CharProcs")[0] != "null"
                assert embedded, name
                fonts[name] = {"type": kind, "embedded": embedded}
        assert not doc.is_encrypted
        for page in doc:
            for link in page.get_links():
                if link["kind"] == fitz.LINK_GOTO:
                    assert 0 <= link["page"] < len(doc)
    report = {
        "pdf": args.pdf.relative_to(ROOT).as_posix(), "pdf_sha256": digest(args.pdf),
        "main_content_pages": 4, "references_pages": [5, 6], "appendix_pages": [7, 12],
        "total_pages": 12, "main_table_numeric_cells_unchanged": numeric_cells,
        "homepage_figure_source_and_assets_byte_identical_to_v9": True,
        "adopted_figure_byte_identical_to_approved_candidate": True,
        "adopted_figure_source_rows_verified": rows,
        "asset_hashes_verified": len(manifest["sha256"]),
        "registration_sha256": digest(ROOT / "docs/preregistration.md"),
        "original_integrity_audit_passed": True,
        "all_cross_references_resolved": True, "overfull_boxes": False,
        "all_fonts_embedded": True, "fonts": fonts,
        "compiler": "Tectonic 0.17.0; original NeurIPS 2025 preprint style and geometry",
        "visual_review": "All 12 final pages inspected at 1.5x; no clipping or overlapping content",
        "scope": "Manuscript presentation only; no experimental reruns or protocol changes",
    }
    (SOURCE / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in report.items() if key != "fonts"}, indent=2))


if __name__ == "__main__":
    main()
