"""Compile an isolated, four-page placement proof; never edit the v9 manuscript."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import fitz


ROOT = Path(__file__).resolve().parents[2]


def replace_float(text, label, replacement):
    anchor = text.index(r"\label{" + label + "}")
    environment = "figure" if label.startswith("fig:") else "table"
    start = text.rindex(r"\begin{" + environment + "}", 0, anchor)
    end_token = r"\end{" + environment + "}"
    end = text.index(end_token, anchor) + len(end_token)
    return text[:start] + replacement + text[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / "build/figure-candidates/placement")
    parser.add_argument("--figures", type=Path, default=ROOT / "output/figure_candidates")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    figures = args.figures.resolve()
    preserved = [ROOT / p for p in ("paper/v9/main.tex", "paper/v9/appendix.tex",
                                    "output/pdf/attention_fast_weights_paper_v9.pdf")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in preserved}
    text = preserved[0].read_text(encoding="utf-8")
    (work / "figures").mkdir(exist_ok=True)
    for name in ("01_repair_benefits_limits", "02_diagnosis_correction_bridge"):
        shutil.copyfile(figures / f"{name}.pdf", work / "figures" / f"{name}.pdf")
    for name in ("example_maps", "example_maps_more"):
        shutil.copyfile(ROOT / "paper/figures_own" / f"{name}.pdf",
                        work / "figures" / f"{name}.pdf")
    shutil.copyfile(ROOT / "figures/execution_k_sweep.pdf", work / "figures/execution_k_sweep.pdf")
    shutil.copytree(ROOT / "paper/tables_own", work / "tables_own", dirs_exist_ok=True)
    shutil.copytree(ROOT / "paper/v9/tables", work / "v9_tables", dirs_exist_ok=True)
    graphicspath = r"\graphicspath{{figures/}}"
    text = re.sub(r"\\graphicspath\{[^\n]+", lambda _: graphicspath, text)
    text = text.replace(r"\bibliography{../references}", r"\bibliography{references}")
    text = text.replace(r"\input{tables/main_results.tex}", r"\input{v9_tables/main_results.tex}")
    first = r"""\begin{figure}[!h]
\centering
\includegraphics{01_repair_benefits_limits}
\caption{Contribution weighting (C3): (a) gains over received attention; (b) gains beyond zeroing high-norm tokens; (c) deletion boundary. Points/bars: original paired means/95\% intervals. Filled/open: higher/lower versus comparable under the registered rule. Grey: practical-margin range (0.02; deletion 0.05).}
\label{fig:lens}
\end{figure}"""
    text = replace_float(text, "fig:lens", first)
    text = text.replace(r" (Fig.~\ref{fig:lens}a)", "")
    definition = r"""\noindent Corrections are $s-\bar s$, $r_p=\sum_j s_jP(p\mid x_j)$ and $c_j=\sum_{h,i}a_{ij}^h\lVert W_O^hv_j^h\rVert$. Their controls subtract a centred Gaussian, blur by matched displacement and zero high-norm tokens, respectively."""
    text = replace_float(text, "tab:premises", definition)
    text = text.replace(r"Table~\ref{tab:premises}", r"Fig.~\ref{fig:diag}")
    second = r"""\begin{figure}[t]
\centering
\includegraphics{02_diagnosis_correction_bridge}
\caption{Diagnostic proxies (a--c) and corresponding corrections (d--f), matched by column. Points/bands/bars: original means and 95\% intervals. Circle/diamond: versus received attention/matched control. Filled/open: higher/lower versus comparable. Grey: layers 9--12 (top); $\pm0.02$ margin (bottom).}
\label{fig:diag}
\end{figure}"""
    text = replace_float(text, "fig:diag", second)
    (work / "main.tex").write_text(text, encoding="utf-8", newline="\n")
    appendix = preserved[1].read_text(encoding="utf-8")
    appendix = appendix.replace(r"\input{../tables_own/", r"\input{tables_own/")
    appendix = appendix.replace(r"\input{tables/sanity_check.tex}",
                                 r"\input{v9_tables/sanity_check.tex}")
    (work / "appendix.tex").write_text(appendix, encoding="utf-8", newline="\n")
    shutil.copyfile(ROOT / "paper/references.bib", work / "references.bib")
    shutil.copyfile(ROOT / "paper/v9/neurips_2025.sty", work / "neurips_2025.sty")
    compiled = subprocess.run([str(args.tectonic.resolve()), "-X", "compile", str(work / "main.tex"),
                               "--outdir", str(work), "--keep-logs", "--keep-intermediates"],
                              cwd=ROOT, capture_output=True, text=True)
    (work / "compile-console.txt").write_text(compiled.stdout + compiled.stderr, encoding="utf-8")
    if compiled.returncode:
        raise RuntimeError("Placement compilation failed:\n" + compiled.stdout + compiled.stderr)
    aux = (work / "main.aux").read_text(encoding="utf-8")
    match = re.search(r"\\newlabel\{main:end\}\{\{6\}\{(\d+)\}", aux)
    if not match:
        raise ValueError("Main-content end label missing")
    pages = int(match.group(1))
    print("Main-content pages:", pages)
    if pages > 4:
        raise ValueError("Candidate placement exceeds the four-page contract")
    doc = fitz.open(work / "main.pdf")
    if "Conclusion" not in doc[pages - 1].get_text():
        raise ValueError("Main-content boundary is inconsistent with the PDF")
    preview = fitz.open()
    preview.insert_pdf(doc, from_page=0, to_page=pages - 1)
    preview.set_metadata({"title": "Candidate figure placement preview; v9 unchanged"})
    preview.save(figures / "03_four_page_placement_preview.pdf")
    preview.close()
    for index in range(pages):
        doc[index].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(work / f"page-{index + 1:02}.png")
    for path, digest in before.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("The placement proof modified the preserved manuscript")
    report = {"status": "Placement proof only; no adoption", "main_content_pages": pages,
              "width_in": 5.5, "font_sizes_and_geometry": "Original NeurIPS style retained",
              "preserved_v9_sha256": {p.relative_to(ROOT).as_posix(): h for p, h in before.items()},
              "changes": ["Replace first figure", "Merge conceptual Table 1 and diagnostic Figure 2",
                          "Keep original numerical main table", "Move displaced correction formulas into Method"]}
    (figures / "placement_verification.json").write_text(json.dumps(report, indent=2) + "\n",
                                                        encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
