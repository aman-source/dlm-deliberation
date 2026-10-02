"""Section B dev report (CPU only): tau objective, dev escalation curves for both ladders vs the
fixed-budget frontier with oracle bounds, and per-dataset TS + adaptive ECE for every dev cell.

Usage: python scripts/dev_v12.py   -> results/dev_v12.md, results/figs/dev_escalation.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dd.analysis import CELLS13, LADDER_LABELS, LADDERS, TAUS, LadderData, cell_metrics, index_rows, temp_key  # noqa: E402
from dd.report import load_rows  # noqa: E402

MODEL = "llada-8b-instruct"
DATASETS = ["boolq", "strategyqa", "arc_c", "jagged"]
SHOW_TAUS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]


def f(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def cell_name(cell):
    condition, scratch, steps = cell
    return "C0" if condition == "C0" else f"{condition} S={scratch} T={steps}"


def main() -> None:
    fits = json.loads((ROOT / "results" / "dev_fits.json").read_text(encoding="utf-8"))
    pooled = fits["temperatures"]
    by_ds = fits["temperatures_by_dataset"]
    rows = load_rows(ROOT)
    dev = index_rows(rows, MODEL, "dev")
    qwen = index_rows(rows, "qwen3-8b", "dev")
    frozen = fits["tau"]["tau"]
    frozen_ts = fits["tau_temperature_scaled"]["tau"]
    out = ["# Dev report, v1.2 (CPU only; dev split; not hypothesis tests)", ""]

    out += ["## 1. Exact objective used to fit tau (Phase 3, frozen before test)", ""]
    out += [
        "For each dataset d, on the dev items that have all three pre-registered ladder levels "
        "(C0 → C1(32,4) → C1(128,16), cumulative NFE 1/6/23) and for each τ on the 50-point grid "
        "`np.linspace(0, 1, 50)`:",
        "",
        "- escalation accuracy acc_d(τ) and mean NFE n_d(τ), escalating from level k while top1 − top2 < τ;",
        "- fixed envelope E_d(n): for each distinct mean NFE among the 13 fixed LLaDA dev cells, the best "
        "accuracy at that NFE (best-at-each-NFE, *not* running-max), linearly interpolated and clamped "
        "(`dd.escalate.fixed_frontier` + `interpolate`);",
        "- gap_d(τ) = acc_d(τ) − E_d(n_d(τ)).",
        "",
        "Objective: **τ\\* = argmax_τ mean_d gap_d(τ)** (unweighted mean over the 4 datasets), ties to lower mean "
        "NFE, then lower τ (`dd.report.fit_tau_for_split`). The same objective after pooled per-cell temperature "
        "scaling of the margins gives the second τ.",
        "",
        f"Result: τ = {f(frozen, 4)} raw (mean gap {f(fits['tau']['mean_gap'], 4)}, mean NFE "
        f"{f(fits['tau']['mean_nfe'], 3)}); τ = {f(frozen_ts, 4)} after TS (mean gap "
        f"{f(fits['tau_temperature_scaled']['mean_gap'], 4)}, mean NFE {f(fits['tau_temperature_scaled']['mean_nfe'], 3)}).",
        "",
        "Note: the v1.2 H3 test uses the monotone (running-max) frontier below, which is at least as high as the "
        "envelope used for this fit.",
        "",
    ]

    out += ["## 2. Dev escalation curves vs the fixed-budget frontier (running-max), both ladders", ""]
    fig_data = {}
    for ladder_name, ladder in LADDERS.items():
        out += [f"### {LADDER_LABELS[ladder_name]}", ""]
        out += ["| dataset | n | G = mean gap (95% CI) | min gap | max gap | share τ with gap ≥ 0 | frozen τ: acc / NFE / frontier | oracle acc / NFE |",
                "|---|---|---|---|---|---|---|---|"]
        for dataset in DATASETS:
            data = LadderData(dev[dataset], ladder, None, MODEL)
            h3 = data.h3()
            point = data.at_tau(frozen)
            oracle = data.oracle()
            fig_data[(ladder_name, dataset)] = (data.summary(), data.ece_curve(), oracle, point)
            out.append(
                f"| {dataset} | {h3['n']} | {f(h3['G'], 4)} ({f(h3['lo'], 4)}, {f(h3['hi'], 4)}) | {f(h3['min_gap'], 3)} | "
                f"{f(h3['max_gap'], 3)} | {f(h3['frac_nonneg'], 2)} | {f(point['accuracy'])} / {f(point['mean_nfe'], 2)} / "
                f"{f(point['frontier'])} | {f(oracle['accuracy'])} / {f(oracle['mean_nfe'], 2)} |"
            )
        out.append("")
        out += ["Curve points (accuracy / mean NFE / frontier at that NFE) at selected τ:", ""]
        out += ["| τ | " + " | ".join(DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
        for t in SHOW_TAUS:
            cells = []
            for dataset in DATASETS:
                s = fig_data[(ladder_name, dataset)][0]
                cells.append(f"{f(s['acc'][t])} / {f(s['nfe'][t], 2)} / {f(s['frontier'][t])}")
            out.append(f"| {TAUS[t]:.3f} | " + " | ".join(cells) + " |")
        out.append("")
        out += ["Fixed-budget frontier on dev (mean NFE → running-max accuracy):", ""]
        for dataset in DATASETS:
            s = fig_data[(ladder_name, dataset)][0]
            pts = ", ".join(f"{int(x)}→{y:.3f}" for x, y in zip(s["front_x"], s["front_y"]))
            out.append(f"- {dataset}: {pts}")
        out.append("")

    out += ["## 3. Per-dataset TS and adaptive ECE, every dev cell", ""]
    out += ["ECE-TS pooled = one temperature per (model, cell) pooled over datasets; ECE-TS per-dataset = one "
            "temperature per (model, dataset, cell). Both are fit on these same dev items, so both are in-sample here.", ""]
    for dataset in DATASETS:
        out += [f"### {dataset}", "",
                "| model | cell | n | acc | ECE | adaptive ECE | ECE-TS pooled | ECE-TS per-dataset | T pooled | T per-dataset | off_label | NFE |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for model, index in ((MODEL, dev), ("qwen3-8b", qwen)):
            for cell in CELLS13:
                group = index[dataset].get(cell)
                if not group:
                    continue
                tp = (pooled.get(temp_key(model, None, cell)) or {}).get("temperature")
                td = (by_ds.get(temp_key(model, dataset, cell)) or {}).get("temperature")
                m = cell_metrics(group, tp, td)
                out.append(
                    f"| {model} | {cell_name(cell)} | {m['n']} | {f(m['accuracy'])} | {f(m['ece'])} | {f(m['adaptive_ece'])} | "
                    f"{f(m['ece_ts_pooled'])} | {f(m['ece_ts_dataset'])} | {f(tp, 2)} | {f(td, 2)} | {f(m['off_label_mass'])} | {f(m['nfe'], 0)} |"
                )
        out.append("")

    (ROOT / "results" / "dev_v12.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for row, ladder_name in enumerate(LADDERS):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            s, _ece, oracle, point = fig_data[(ladder_name, dataset)]
            cell_nfe = [1] + [t + 1 for _c in ("C1", "C2") for _s in (32, 128) for t in (1, 4, 16)]
            ax.scatter(cell_nfe, s["cell_acc"], s=14, color="#888", label="fixed cells")
            ax.step(s["front_x"], s["front_y"], where="post", color="#444", lw=1, label="frontier (running max)")
            ax.plot(s["nfe"], s["acc"], color="#1f6feb", lw=1.6, label="escalation (τ sweep)")
            ax.scatter([point["mean_nfe"]], [point["accuracy"]], color="#d1242f", zorder=5, label="frozen τ")
            ax.scatter([oracle["mean_nfe"]], [oracle["accuracy"]], marker="*", s=90, color="#1a7f37", zorder=5, label="oracle")
            ax.set_title(f"{dataset} — {ladder_name}", fontsize=9)
            ax.set_xlabel("mean NFE")
            if col == 0:
                ax.set_ylabel("dev accuracy")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    (ROOT / "results" / "figs").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "results" / "figs" / "dev_escalation.png", dpi=120)
    print("wrote results/dev_v12.md and results/figs/dev_escalation.png")


if __name__ == "__main__":
    main()
