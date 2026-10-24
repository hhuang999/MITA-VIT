"""Download the handoff reading list and source BibTeX directly from arXiv."""

import hashlib
import json
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPERS = {
    "mita": "2602.01219",
    "clustered_attention": "2007.04825",
    "attention_flow": "2005.00928",
    "attention_not_explanation": "1902.10186",
    "attention_not_not_explanation": "1908.04626",
    "transformer_interpretability": "2012.09838",
    "eraser": "1911.03429",
    "attention_knockout": "2304.14767",
    "deit": "2012.12877",
    "dino": "2104.14294",
    "topk_attention": "2106.06899",
    "registers": "2309.16588",
    "rise": "1806.07421",
    "teaching_matters": "2212.03862",
}


def main() -> None:
    directory = ROOT / "references" / "papers"
    directory.mkdir(parents=True, exist_ok=True)
    bibliography = []
    manifest = []
    for name, arxiv_id in PAPERS.items():
        path = directory / f"{name}_{arxiv_id}.pdf"
        if not path.exists():
            with urllib.request.urlopen(f"https://arxiv.org/pdf/{arxiv_id}", timeout=90) as r:
                content = r.read()
            if not content.startswith(b"%PDF"):
                raise ValueError(f"Expected a PDF for {arxiv_id}")
            path.write_bytes(content)
            time.sleep(3)
        bib_path = directory / f"{name}_{arxiv_id}.bib"
        if not bib_path.exists():
            with urllib.request.urlopen(f"https://arxiv.org/bibtex/{arxiv_id}", timeout=60) as r:
                bib_path.write_bytes(r.read())
            time.sleep(3)
        bibtex = bib_path.read_text(encoding="utf-8")
        if not bibtex.lstrip().startswith("@"):
            raise ValueError(f"Expected BibTeX for {arxiv_id}")
        bibliography.append("\n".join(line.rstrip() for line in bibtex.strip().splitlines()))
        manifest.append({
            "name": name, "arxiv_id": arxiv_id,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        print(f"Saved {name}: {path.stat().st_size:,} bytes", flush=True)
    (ROOT / "references" / "references.bib").write_text(
        "\n\n".join(bibliography) + "\n", encoding="utf-8"
    )
    (ROOT / "references" / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
