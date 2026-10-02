"""Dev-sweep table per dataset from the main-setting rows and results/dev_fits.json (read-only).

Usage: python scripts/dev_table.py > results/dev_table.md
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dd.report import _temp_key, grouped, load_rows, summarize_rows  # noqa: E402

ORDER = [("C0", 0, None)] + [(c, s, t) for c in ("C1", "C2") for s in (32, 128) for t in (1, 4, 16)]


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    fits = json.loads((ROOT / "results" / "dev_fits.json").read_text(encoding="utf-8"))
    temps = fits.get("temperatures") or {}
    rows = [row for row in load_rows(ROOT) if row.get("split") == "dev"]
    groups = grouped(rows)
    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    print("# Dev sweep (protocol v1.1)\n")
    print("Mean off_label_mass. ECE: 15 bins. ECE-TS: one temperature per (model, cell), fit on dev pooled over datasets.\n")
    for dataset in datasets:
        print(f"## {dataset}\n")
        print("| model | cell | n | acc | ECE | ECE-TS | off_label | eos_pad (median) | NFE |")
        print("|---|---|---|---|---|---|---|---|---|")
        for model in models:
            for condition, scratch, steps in ORDER:
                group = groups.get((dataset, "dev", model, condition, scratch, steps))
                if not group:
                    continue
                temperature = (temps.get(_temp_key(model, condition, scratch, steps)) or {}).get("temperature")
                m = summarize_rows(group, temperature, n_resamples=200)
                eos = [r["eos_pad_fraction"] for r in group if r.get("eos_pad_fraction") is not None]
                cell = condition if condition == "C0" else f"{condition} S={scratch} T={steps}"
                print(
                    f"| {model} | {cell} | {m['n']} | {fmt(m['accuracy'])} | {fmt(m['ece'])} | {fmt(m['ece_ts'])} | "
                    f"{fmt(m['off_label_mass'])} | {fmt(statistics.median(eos)) if eos else '-'} | {fmt(m['nfe'], 0)} |"
                )
        print()
    print("## Fitted temperatures (dev)\n")
    print("| key | T | NLL | n_dev |")
    print("|---|---|---|---|")
    for key, fit in sorted(temps.items()):
        print(f"| {key} | {fmt(fit.get('temperature'))} | {fmt(fit.get('nll'))} | {fit.get('n_dev')} |")
    print("\n## Fitted tau (dev)\n")
    for name in ("tau", "tau_temperature_scaled"):
        block = fits.get(name) or {}
        print(f"- {name}: tau = {fmt(block.get('tau'))}, mean gap vs fixed frontier = {fmt(block.get('mean_gap'), 4)}, "
              f"mean NFE = {fmt(block.get('mean_nfe'), 2)}")
    print(f"\nSelection rule: {fits.get('selection_rule', 'n/a')}")


if __name__ == "__main__":
    main()
