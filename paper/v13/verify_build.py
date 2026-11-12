"""Verify the submitted v13 PDF against its build, sources and experimental inputs."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=ROOT / "build/paper-v13")
    args = parser.parse_args()
    pdf = ROOT / "output/pdf/attention_fast_weights_paper_v13.pdf"
    assert sha(pdf) == sha(args.build / "main.pdf")
    assert json.loads((args.build / "baseline-audit.json").read_text())["passed"]
    original = json.loads((ROOT / "paper/v12/verification.json").read_text())
    assert sha(ROOT / original["pdf"]) == original["pdf_sha256"]
    assert sha(ROOT / "docs/preregistration.md") == original["registration_sha256"]

    aux = (args.build / "main.aux").read_text(encoding="utf8")
    log = (args.build / "main.log").read_text(encoding="utf8")
    labels = {m[1]:(m[2],int(m[3])) for m in re.finditer(
        r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{(\d+)\}", aux)}
    assert not re.search(r"undefined|multiply defined|Overfull",log,re.I)
    assert "Warning--" not in (args.build / "main.blg").read_text()
    text = "\n".join(p.read_text(encoding="utf8") for p in SOURCE.glob("*.tex"))
    assert set(re.findall(r"\\(?:eqref|ref)\{([^}]+)\}",text)) <= labels.keys()
    bib = (SOURCE / "references.bib").read_text(encoding="utf8")
    assert bib.startswith((ROOT / "paper/v12/references.bib").read_text(encoding="utf8").rstrip())
    keys = set(re.findall(r"(?m)^@\w+\{([^,]+),",bib))
    cited = {key for item in re.findall(r"\\cite\w*\{([^}]+)\}",text) for key in item.split(",")}
    assert keys == cited
    bbl=(args.build / "main.bbl").read_text(encoding="utf8")
    assert bbl.count(r"\bibitem[") == len(keys)
    assert labels["main:end"][1] == 4 and labels["fig:lens"] == ("1",1) and labels["fig:diag"] == ("2",3)
    assert labels["tab:pets"] == ("1",4)
    assert (SOURCE / "appendix.tex").read_text(encoding="utf8").rstrip().endswith(r"\end{lstlisting}")
    fonts=set()
    links=0
    with fitz.open(pdf) as doc:
        assert doc[4].get_text().startswith("References")
        assert "Experimental details" in doc[labels["app:setup"][1]-1].get_text()
        assert labels["app:code"] == ("F",len(doc)) and "Key parts of the code" in doc[-1].get_text()
        destinations = doc.resolve_names()
        for page in doc:
            assert "??" not in page.get_text() and "\ufffd" not in page.get_text()
            for ref,_,kind,name,*_ in page.get_fonts(full=True):
                assert (doc.xref_get_key(ref,"CharProcs")[0]!="null" if kind=="Type3" else bool(doc.extract_font(ref)[3])),name
                fonts.add(name)
            for link in page.get_links():
                if link["kind"] in (fitz.LINK_GOTO, fitz.LINK_NAMED):
                    assert 0<=link["page"]<len(doc)
                    if "nameddest" in link:
                        assert destinations[link["nameddest"]]["page"] == link["page"]
                    links+=1
            for b in page.get_text("blocks"):
                box=fitz.Rect(b[:4])
                assert (page.rect + (-.1,-.1,.1,.1)).contains(box), (page.number,box)
        assert links > 0
        report=dict(pdf=pdf.relative_to(ROOT).as_posix(),sha256=sha(pdf),
                    pages=len(doc),main_pages=[1,4],references_pages=[5,labels["app:setup"][1]-1],
                    appendix_pages=[labels["app:setup"][1],len(doc)],references=len(keys),embedded_fonts=len(fonts),
                    internal_page_links_checked=links,baseline_sha256=original["pdf_sha256"],
                    registration_sha256=sha(ROOT/"docs/preregistration.md"),last_appendix="F. Key parts of the code")
    (SOURCE/"verification.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8",newline="\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__":
    main()
