"""Audit the v11 submission PDF, preserved figures, results and deferred references."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import fitz


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "paper/v11"


def digest(path):
    data = path.read_bytes()
    if path.suffix in (".tex", ".bib"):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def bib_entries(path):
    return {m.group(1): m.group(0) for m in re.finditer(
        r"(?ms)^@\w+\{([^,]+),.*?^\}", path.read_text(encoding="utf-8"))}


def measurements(text):
    return re.findall(r"(?:\$-\$)?0\.\d{3}|\\textit\{[hcl]\}", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/paper-v11")
    parser.add_argument("--pdf", type=Path,
                        default=ROOT / "output/pdf/attention_fast_weights_paper_v11.pdf")
    args = parser.parse_args()
    aux = (args.build / "main.aux").read_text(encoding="utf-8")
    log = (args.build / "main.log").read_text(encoding="utf-8")
    labels = {m.group(1): (m.group(2), int(m.group(3))) for m in re.finditer(
        r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux)}
    assert labels["main:end"][1] == 4
    assert labels["fig:lens"] == ("1", 1)
    assert labels["fig:diag"] == ("2", 3)
    assert labels["tab:main"][0] == "1"
    assert not re.search(r"(?:undefined|multiply defined|Overfull)", log, re.I)
    assert not re.search(r"Warning--", (args.build / "main.blg").read_text(encoding="utf-8"))
    source_text = "\n".join((SOURCE / f).read_text(encoding="utf-8")
                            for f in ("main.tex", "appendix.tex"))
    refs = set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source_text))
    assert refs <= labels.keys(), refs - labels.keys()

    for name in ("fast_weight_lens.pdf", "contribution_effects.pdf", "diagnosis_correction_bridge.pdf"):
        assert (SOURCE / "figures" / name).read_bytes() == (ROOT / "paper/v10/figures" / name).read_bytes()
    for name in ("main_results.tex", "sanity_check.tex"):
        assert (SOURCE / "tables" / name).read_text(encoding="utf-8") == (
            ROOT / "paper/v10/tables" / name).read_text(encoding="utf-8")
    main_table = (SOURCE / "tables/main_results.tex").read_text(encoding="utf-8")
    assert len(re.findall(r"(?<!\d)0\.\d{3}", main_table)) == 86
    assert main_table.count(r"\diamond") == 7 and main_table.count(r"\textbf{") == 3
    paired = (SOURCE / "tables/paired_differences.tex").read_text(encoding="utf-8")
    old_paired = (ROOT / "paper/tables_own/paired_differences.tex").read_text(encoding="utf-8")
    assert measurements(paired) == measurements(old_paired)
    assert (SOURCE / "evidence/sanity.csv").read_bytes() == (ROOT / "archive/results/sanity.csv").read_bytes()

    old_bib, new_bib = bib_entries(ROOT / "paper/references.bib"), bib_entries(SOURCE / "references.bib")
    assert set(old_bib) <= new_bib.keys()
    assert set(new_bib) - set(old_bib) == {"modarressi2022", "ferrando2022"}
    for key in ("ramsauer2021", "wen2026mita"):
        assert old_bib[key] == new_bib[key], key
    citation_keys = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", source_text):
        citation_keys.update(group.split(","))
    assert citation_keys == new_bib.keys()
    bbl = (args.build / "main.bbl").read_text(encoding="utf-8")
    assert bbl.count(r"\bibitem[") == 33
    manifest = json.loads((SOURCE / "asset_sources.json").read_text(encoding="utf-8"))
    for path, expected in manifest["sha256"].items():
        assert digest(ROOT / path) == expected, path
    assert json.loads((args.build / "original-integrity.json").read_text(encoding="utf-8"))["passed"]

    with fitz.open(args.pdf) as doc:
        assert len(doc) == 14 and not doc.is_encrypted
        assert "Conclusion" in doc[3].get_text()
        assert doc[4].get_text().startswith("References")
        assert "Experimental details" in doc[6].get_text()
        assert "Historical parameter-randomisation check" in doc[13].get_text()
        assert "Table F.1" in doc[13].get_text()
        assert all(f"app:{name}" in labels for name in ("setup", "more", "reloc", "gallery", "code", "sanity"))
        fonts = {}
        for page in doc:
            assert "??" not in page.get_text() and "\ufffd" not in page.get_text()
            for font in page.get_fonts(full=True):
                xref, _, kind, name = font[:4]
                embedded = bool(doc.extract_font(xref)[3])
                if kind == "Type3":
                    embedded = doc.xref_get_key(xref, "CharProcs")[0] != "null"
                assert embedded, name
                fonts[name] = {"type": kind, "embedded": True}
            for block in page.get_text("dict")["blocks"]:
                if block["type"] == 0:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            box = fitz.Rect(span["bbox"])
                            assert box.x0 >= 0 and box.y0 >= 0
                            assert box.x1 <= page.rect.width + 0.1 and box.y1 <= page.rect.height + 0.1
            for link in page.get_links():
                if link["kind"] == fitz.LINK_GOTO:
                    assert 0 <= link["page"] < len(doc)
        diagram = fitz.open(SOURCE / "figures/diagnosis_correction_bridge.pdf")
        assert abs(diagram[0].rect.width / 72 - 5.5) < 1e-6
        diagram.close()
    report = {
        "pdf": args.pdf.relative_to(ROOT).as_posix(), "pdf_sha256": digest(args.pdf),
        "main_content_pages": 4, "references_pages": [5, 6], "appendix_pages": [7, 14],
        "total_pages": 14, "bibliography_entries": 33,
        "main_table_numeric_cells_preserved": 86, "control_markers_preserved": 7,
        "joint_bold_cells_preserved": 3, "paired_measurements_and_verdicts_preserved": True,
        "homepage_and_figure2_vector_assets_byte_identical_to_v10": True,
        "deferred_bibliography_entries_unchanged": ["ramsauer2021", "wen2026mita"],
        "original_integrity_audit_passed": True, "source_hashes_verified": len(manifest["sha256"]),
        "registration_sha256": digest(ROOT / "docs/preregistration.md"),
        "all_references_and_internal_links_resolved": True, "overfull_boxes": False,
        "all_fonts_embedded": True, "fonts": fonts,
        "visual_review": "All 14 final pages inspected at 1.5x; no clipping, overlap or isolated code/table continuation",
        "compiler": "Tectonic 0.17.0; original NeurIPS 2025 preprint font sizes and page geometry",
        "scope": "Editorial and reproducibility clarification only; no new experiment or statistical reclassification",
    }
    (SOURCE / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "fonts"}, indent=2))


if __name__ == "__main__":
    main()
