"""Build the manuscript with Tectonic and verify the rendered submission."""
import argparse
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path

import fitz

PAPER = Path(__file__).resolve().parents[1] / "paper"
BUILD = PAPER.parent / "build" / "paper"
STYLE_URL = "https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip"
STYLE_HASH = "67a960f35da6fecb5a0ce0202fa400459f4c8bb6c294f1a3fdd82fbbd39244c2"
FILENAME = "paper.pdf"


def prepare_source(style):
    source = BUILD / "source"
    source.mkdir(parents=True, exist_ok=True)
    for name in ("main.tex", "appendix.tex", "references.bib"):
        shutil.copyfile(PAPER / name, source / name)
    for folder, pattern in (("tables", "*.tex"), ("figures", "*.pdf")):
        (source / folder).mkdir(exist_ok=True)
        for path in (PAPER / folder).glob(pattern):
            shutil.copyfile(path, source / folder / path.name)
    target = source / "neurips_2025.sty"
    data = (style or PAPER / "neurips_2025.sty").read_bytes()
    if hashlib.sha256(data).hexdigest() != STYLE_HASH:
        raise ValueError("The style differs from the official NeurIPS 2025 file.")
    target.write_bytes(data)
    return source


def verify(pdf):
    with fitz.open(pdf) as doc:
        pages = [page.get_text() for page in doc]
        if (not pages[4].startswith("References") or
                "Discussion and conclusion" not in pages[3]):
            raise ValueError("Expected the conclusion on page 4 and references on page 5.")
        last = " ".join(pages[-1].split())
        statement = (PAPER / "appendix.tex").read_text(encoding="utf8").split(
            r"\paragraph{AI Use Statement.} ")[-1].strip()
        if not last.endswith(f"AI Use Statement. {statement} {len(pages)}"):
            raise ValueError("The printed AI statement must follow the last code listing.")
        if not 0 <= last.find("Key parts of the code") < last.find("AI Use Statement"):
            raise ValueError("Missing final key code.")
        refs = re.findall(r"(?m)^\[(\d+)\] ", "\n".join(pages[4:]))
        if list(map(int, refs)) != list(range(1, 37)):
            raise ValueError("Expected 36 consecutive references.")
        destinations = doc.resolve_names()
        links = 0
        for page in doc:
            if "??" in page.get_text():
                raise ValueError(f"Unresolved reference on page {page.number+1}")
            for link in page.get_links():
                if link["kind"] in (fitz.LINK_GOTO, fitz.LINK_NAMED):
                    destination = link["page"]
                    if not 0 <= destination < len(doc):
                        raise ValueError(f"Invalid destination on page {page.number+1}")
                    if "nameddest" in link and destinations[link["nameddest"]]["page"] != destination:
                        raise ValueError("Inconsistent named destination")
                    links += 1
            for ref, _, kind, name, *_ in page.get_fonts(full=True):
                embedded = (doc.xref_get_key(ref, "CharProcs")[0] != "null"
                            if kind == "Type3" else bool(doc.extract_font(ref)[3]))
                if not embedded:
                    raise ValueError(f"Unembedded font: {name}")
        print(f"{len(doc)} pages, four content pages, 36 references, {links} internal links.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", type=Path, help="Offline copy of the official neurips_2025.sty")
    parser.add_argument("--tectonic", default="tectonic", help="Tectonic executable")
    args = parser.parse_args()
    source = prepare_source(args.style)
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "1794528000")
    subprocess.run([args.tectonic, "-X", "compile", "main.tex", "--outdir", str(BUILD),
                    "--keep-logs", "--keep-intermediates"], cwd=source, env=env, check=True)
    log = (BUILD / "main.log").read_text(encoding="utf8")
    if re.search(r"undefined|multiply defined|Overfull", log, re.I):
        raise ValueError("Resolve the layout or reference warnings in main.log.")
    if "Warning--" in (BUILD / "main.blg").read_text(encoding="utf8"):
        raise ValueError("Resolve the BibTeX warnings in main.blg.")
    pdf = BUILD / "main.pdf"
    verify(pdf)
    shutil.copyfile(pdf, BUILD / FILENAME)
    print(BUILD / FILENAME)


if __name__ == "__main__":
    main()
