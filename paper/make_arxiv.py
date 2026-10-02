"""Build paper/arxiv_submission.zip: main.tex, compiled main.bbl, the \\input tables, and only the
figures main.tex includes. Comment lines and trailing comments are stripped from every .tex file.
The zip is then extracted to a clean temporary folder and compiled with pdflatex only (twice).
Run after a full local build (pdflatex, bibtex, pdflatex, pdflatex) so main.bbl is current.
Usage: python paper/make_arxiv.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ZIP = PAPER / "arxiv_submission.zip"


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.splitlines():
        if line.lstrip().startswith("%"):
            continue
        out.append(re.sub(r"(?<!\\)%.*$", "", line).rstrip())
    return "\n".join(out) + "\n"


def main() -> int:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    files = {"main.tex": strip_comments(tex)}
    for rel in inputs:
        files[rel] = strip_comments((PAPER / rel).read_text(encoding="utf-8"))
    bbl = (PAPER / "main.bbl").read_text(encoding="utf-8")
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for rel, text in files.items():
            z.writestr(rel, text)
        z.writestr("main.bbl", bbl)
        for rel in figs:
            z.write(PAPER / rel, rel)

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(tmp)
            names = sorted(z.namelist())
        for _ in range(2):
            r = subprocess.run(["pdflatex", "--disable-installer", "-interaction=nonstopmode", "main.tex"], cwd=tmp,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        log = (Path(tmp) / "main.log").read_text(encoding="utf-8", errors="replace")
        errors = [l for l in log.splitlines() if l.startswith("!")]
        undefined = [l for l in log.splitlines() if "undefined" in l.lower()]
        pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
        comments_left = sum(1 for n in names if n.endswith(".tex")
                            for l in (Path(tmp) / n).read_text(encoding="utf-8").splitlines() if re.search(r"(?<!\\)%", l))
    print(f"zip: {ZIP.name} {ZIP.stat().st_size / 1024:.0f} KiB, {len(names)} files")
    for n in names:
        print("  ", n)
    print(f"clean compile: pages={pages.group(1) if pages else None}, errors={len(errors)}, undefined={len(undefined)}, "
          f"comment lines left={comments_left}")
    for l in errors + undefined:
        print("  ", l)
    return 0 if pages and not errors and not undefined else 1


if __name__ == "__main__":
    sys.exit(main())
