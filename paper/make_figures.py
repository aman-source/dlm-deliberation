"""Paper summary figure, generated from results files only (no hand-typed numbers).

calibration_summary.png: for each model (rows) and dataset (columns), raw ECE of the single read
(C0) vs the mean raw ECE over all scratch cells (C1/C2), each also after per-budget temperature
scaling (one temperature per cell, fit on dev).
  LLaDA-8B-Instruct: results/summary.csv (full test set; ECE-TS = pooled per-cell temperature).
  Dream-v0-Instruct-7B: results/v13_bc.json (original test items; ECE-TS = Dream's own dev temperatures).

Run from the repo root:  python paper/make_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
DATASETS = ["boolq", "strategyqa", "arc_c", "jagged"]
DSNAME = {"boolq": "BoolQ", "strategyqa": "StrategyQA", "arc_c": "ARC-C", "jagged": "Jagged"}


def llada() -> dict:
    out = {}
    rows = [r for r in csv.DictReader((R / "summary.csv").open(encoding="utf-8")) if r["model"].startswith("llada")]
    for d in DATASETS:
        c0 = next(r for r in rows if r["dataset"] == d and r["condition"] == "C0")
        sc = [r for r in rows if r["dataset"] == d and r["condition"] != "C0"]
        out[d] = {"c0": float(c0["ece"]), "c0_ts": float(c0["ece_ts_pooled"]),
                  "scratch": float(np.mean([float(r["ece"]) for r in sc])),
                  "scratch_ts": float(np.mean([float(r["ece_ts_pooled"]) for r in sc])), "n_cells": len(sc)}
    return out


def dream() -> dict:
    tables = json.loads((R / "v13_bc.json").read_text(encoding="utf-8"))["C"]["tables"]
    out = {}
    for d in DATASETS:
        c0 = tables[f"{d}|C0"]
        sc = [v for k, v in tables.items() if k.startswith(f"{d}|") and not k.endswith("|C0")]
        out[d] = {"c0": c0["ece"], "c0_ts": c0["ece_ts"], "scratch": float(np.mean([v["ece"] for v in sc])),
                  "scratch_ts": float(np.mean([v["ece_ts"] for v in sc])), "n_cells": len(sc)}
    return out


def main() -> None:
    data = {"LLaDA-8B-Instruct (full test, 12 scratch cells)": llada(),
            "Dream-v0-Instruct-7B (original test items, 6 scratch cells)": dream()}
    bars = [("c0", "C0 raw"), ("scratch", "scratch mean raw"), ("c0_ts", "C0 + TS"), ("scratch_ts", "scratch mean + TS")]
    colors = ["#4c72b0", "#dd8452", "#9fb7dd", "#f2c19c"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), sharey=True)
    for row, (model, per_ds) in enumerate(data.items()):
        for col, d in enumerate(DATASETS):
            ax = axes[row, col]
            vals = [per_ds[d][k] for k, _ in bars]
            ax.bar(range(4), vals, color=colors, edgecolor="#333", linewidth=0.5)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(4))
            ax.set_xticklabels([lbl for _, lbl in bars], rotation=30, ha="right", fontsize=7)
            if row == 0:
                ax.set_title(DSNAME[d])
            if col == 0:
                ax.set_ylabel(f"ECE\n{model.split(' (')[0]}")
    fig.suptitle("ECE of the single read (C0) vs the mean over scratch cells, raw and after per-budget TS", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "paper" / "figs" / "calibration_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    (ROOT / "paper" / "figs" / "calibration_summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("wrote", out)


def reliability() -> None:
    """Reliability diagrams with 10 equal-mass bins (no sparse bins), LLaDA test set, C0 and C1(S=128, T=16).

    Items are sorted by top-label confidence and split into 10 groups of near-equal size; each point is
    (mean confidence, accuracy) of one group. Rows: results/raw/<dataset>_test_llada-8b-instruct_*_v3e.jsonl.
    """
    cells = [("C0", "C0_S0_Tna"), ("C1 S=128 T=16", "C1_S128_T16")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharex=True, sharey=True)
    summary = {}
    for col, d in enumerate(DATASETS):
        for row, (label, tag) in enumerate(cells):
            path = R / "raw" / f"{d}_test_llada-8b-instruct_{tag}_v3e.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            conf = np.array([max(r["final_probs"].values()) for r in rows])
            corr = np.array([bool(r["correct"]) for r in rows], dtype=float)
            order = np.lexsort((np.arange(len(conf)), conf))
            groups = np.array_split(order, 10)
            xs = [float(conf[g].mean()) for g in groups]
            ys = [float(corr[g].mean()) for g in groups]
            ax = axes[row, col]
            ax.plot([0, 1], [0, 1], color="#bbb", lw=1)
            ax.plot(xs, ys, marker="o", color="#4c72b0" if row == 0 else "#dd8452")
            ax.set_xlim(0, 1.02)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{DSNAME[d]}: {label} (n={len(rows)}, {len(groups[0])}-{len(groups[-1])}/bin)", fontsize=8)
            if row == 1:
                ax.set_xlabel("mean confidence (bin)")
            if col == 0:
                ax.set_ylabel("accuracy (bin)")
            summary[f"{d}|{label}"] = {"n": len(rows), "bin_conf": xs, "bin_acc": ys, "bin_sizes": [len(g) for g in groups]}
    fig.suptitle("Reliability, LLaDA-8B-Instruct test set, 10 equal-mass bins", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(ROOT / "paper" / "figs" / "reliability_equal_mass.png", dpi=150)
    (ROOT / "paper" / "figs" / "reliability_equal_mass.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote reliability_equal_mass.png")


if __name__ == "__main__":
    main()
    reliability()
