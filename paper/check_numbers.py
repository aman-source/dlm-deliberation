"""Check every number in the prose of paper/main.tex against the results files.

Prose = main.tex minus comments, \\input lines, \\includegraphics lines, and the bibliography.
Each number is matched against every numeric value found in the results files, at the precision the
prose uses (and as a percentage where the prose writes %). Design constants that come from the
protocol rather than from results are whitelisted below with their source.
Usage: python paper/check_numbers.py   (exit code 1 if anything is unmatched)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

# Protocol constants (PLAN.md, configs/*.yaml, HYPOTHESES.md), not results.
WHITELIST = {
    "1": "T value / NFE", "2": "count words", "3": "count words", "4": "T value / dataset count", "5": "count of contrasts",
    "6": "cumulative NFE / count", "8": "contrast count", "16": "T value", "23": "cumulative NFE", "32": "S/T value",
    "128": "S value", "13": "grid cells", "15": "ECE bins", "10": "equal-mass reliability bins", "50": "tau grid",
    "1000": "bootstrap resamples / test items per dataset", "1234": "seed", "95": "CI level", "20": "smoke items per dataset",
    "0": "interval bound / tau range", "0.5": "AUROC chance level", "12": "LLaDA scratch cells", "24": "Dream scratch cells",
    "48": "LLaDA scratch cells", "400": "original test items per dataset", "240": "original jagged test items",
    "8B": "model size", "7B": "model size",
}

NUM = re.compile(r"(?<![A-Za-z0-9.])(\$-\$|-)?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(\\%|%)?")


def collect_values() -> list[float]:
    vals: list[float] = []

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            vals.append(float(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            try:
                vals.append(float(x))
            except ValueError:
                pass

    for p in list(R.glob("*.json")) + list((R / "dream").glob("*.json")) + list((R / "causal").glob("*/*.json")) \
            + list((R / "smoke_v1_0").glob("*.json")) + list((ROOT / "paper" / "figs").glob("*.json")):
        walk(json.loads(p.read_text(encoding="utf-8")))
    for row in csv.DictReader((R / "summary.csv").open(encoding="utf-8")):
        walk(dict(row))
    return vals


def prose(tex: str) -> str:
    keep = []
    for line in tex.splitlines():
        s = line.strip()
        if s.startswith("%") or s.startswith("\\input") or s.startswith("\\includegraphics") or s.startswith("\\bibliography"):
            continue
        keep.append(re.sub(r"(?<!\\)%.*$", "", line))
    text = "\n".join(keep)
    text = text.split("\\appendix")[0] + "\n" + "\n".join(re.findall(r"\\caption\{(.*)\}", text.split("\\appendix")[-1]))
    text = re.sub(r"\\(label|ref|citep|cite|todo|section|paragraph|url)\{[^}]*\}", " ", text)
    text = re.sub(r"revision [0-9a-f]{7}", " ", text)
    return text


def matches(token: str, pct: bool, vals: list[float]) -> bool:
    x = float(token.replace(",", ""))
    dec = len(token.split(".")[1]) if "." in token else 0
    for v in vals:
        for cand in ((v * 100, abs(v) * 100) if pct else (v, abs(v))):
            if round(cand, dec) == round(x, dec) or (dec == 0 and abs(cand - x) < 0.5 and pct):
                return True
    return False


def main() -> int:
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    text = prose(tex)
    vals = collect_values()
    found, unmatched, whitelisted = 0, [], 0
    for m in NUM.finditer(text):
        sign, token, pct = m.group(1), m.group(2), m.group(3)
        after = text[m.end():m.end() + 1]
        if after in ("B",):  # model sizes like 8B
            whitelisted += 1
            continue
        raw = token.replace(",", "")
        if not pct and raw in WHITELIST:
            whitelisted += 1
            continue
        found += 1
        if not matches(token, bool(pct), vals):
            ctx = text[max(0, m.start() - 60):m.end() + 30].replace("\n", " ")
            unmatched.append(f"{'-' if sign else ''}{token}{'%' if pct else ''}  ...{ctx}...")
    print(f"numbers checked against results: {found}; whitelisted protocol constants: {whitelisted}; unmatched: {len(unmatched)}")
    for u in unmatched:
        print("  UNMATCHED:", u)
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
