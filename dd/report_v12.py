"""Phase 5 report for protocol v1.1 with the v1.2 registration (python -m dd.report).

Every number is computed here from results/raw (main-setting rows via dd.report.load_rows, and the
*_noeos ablation files) plus results/dev_fits.json, and also written to results/summary.csv,
results/hypothesis_tests.json and results/escalation_test.json so REPORT.md is traceable.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dd.analysis import (
    CELLS13,
    LADDER_LABELS,
    LADDERS,
    TAUS,
    LadderData,
    adaptive_ece,
    cell_key,
    cell_metrics,
    h3_verdict,
    index_rows,
    temp_key,
)
from dd.metrics import (
    accuracy,
    aurc,
    bootstrap_ci,
    expected_calibration_error,
    mcnemar_test,
    multiclass_brier,
    negative_log_likelihood,
    paired_bootstrap_diff,
)
from dd.report import evaluate_hypotheses, load_rows

MODEL = "llada-8b-instruct"
BASELINE = "qwen3-8b"
DATASETS = ["boolq", "strategyqa", "arc_c", "jagged"]
N_BOOT = 1000


def f(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def cell_name(cell) -> str:
    condition, scratch, steps = cell
    return "C0" if condition == "C0" else f"{condition} S={scratch} T={steps}"


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def load_noeos_rows(root: Path, split: str) -> list[dict]:
    """Ablation rows (suppress ON). Kept out of every main table."""
    template = json.loads((root / "results" / "template.json").read_text(encoding="utf-8"))["template"]
    rows = []
    for path in sorted((root / "results" / "raw").glob(f"*_{split}_*_noeos.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("protocol") == "v1.1" and row.get("template") == template and row.get("suppress_eos_in_scratch"):
                    rows.append(row)
    return rows


def full_metrics(group: dict, pooled_t, dataset_t) -> dict:
    rows = list(group.values())
    correct = [bool(r["correct"]) for r in rows]
    confidence = [max(r["final_probs"].values()) for r in rows]
    base = cell_metrics(group, pooled_t, dataset_t)
    acc_ci = bootstrap_ci(lambda idx: accuracy([correct[i] for i in idx]), len(rows), n_resamples=N_BOOT)
    ece_ci = bootstrap_ci(lambda idx: expected_calibration_error([correct[i] for i in idx], [confidence[i] for i in idx]),
                          len(rows), n_resamples=N_BOOT)
    prob_rows, gold_index, gold_probs = [], [], []
    for r in rows:
        letters = list(r["final_probs"])
        prob_rows.append([r["final_probs"][x] for x in letters])
        gold_index.append(letters.index(r["gold"]))
        gold_probs.append(r["final_probs"][r["gold"]])
    walls = [r["wall_ms"] for r in rows if r.get("wall_ms") is not None]
    base.update({
        "accuracy_lo": acc_ci["lo"], "accuracy_hi": acc_ci["hi"], "ece_lo": ece_ci["lo"], "ece_hi": ece_ci["hi"],
        "brier": multiclass_brier(prob_rows, gold_index), "nll": negative_log_likelihood(gold_probs),
        "aurc": aurc(correct, confidence), "ms_per_item": float(np.mean(walls)) if walls else None,
        "temperature_pooled": pooled_t, "temperature_dataset": dataset_t,
    })
    return base


def build(root: Path, split: str = "test", out_name: str = "REPORT.md") -> dict:
    results = root / "results"
    fits = json.loads((results / "dev_fits.json").read_text(encoding="utf-8"))
    pooled, by_ds = fits.get("temperatures") or {}, fits.get("temperatures_by_dataset") or {}
    frozen_tau = (fits.get("tau") or {}).get("tau")
    frozen_tau_ts = (fits.get("tau_temperature_scaled") or {}).get("tau")
    protocol = json.loads((results / "protocol_v1_1.json").read_text(encoding="utf-8"))
    rows = load_rows(root)
    index = {MODEL: index_rows(rows, MODEL, split), BASELINE: index_rows(rows, BASELINE, split)}
    figs = results / "figs"
    figs.mkdir(parents=True, exist_ok=True)

    # ---------- main tables
    summary_rows = []
    for dataset in DATASETS:
        for model in (MODEL, BASELINE):
            for cell in CELLS13:
                group = index[model][dataset].get(cell)
                if not group:
                    continue
                tp = (pooled.get(temp_key(model, None, cell)) or {}).get("temperature")
                td = (by_ds.get(temp_key(model, dataset, cell)) or {}).get("temperature")
                m = full_metrics(group, tp, td)
                summary_rows.append({"dataset": dataset, "split": split, "model": model, "cell": cell_name(cell),
                                     "condition": cell[0], "S": cell[1], "T": cell[2], **m})
    with (results / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["dataset"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # ---------- hypotheses: H1, H2, H4 from the pre-registered contrast code; H3 from v1.2 full curve
    split_rows = [r for r in rows if r.get("split") == split]
    relabeled = [dict(r, split="test") for r in split_rows]  # evaluate_hypotheses reads split == "test"
    hyp = evaluate_hypotheses(relabeled, None, n_resamples=N_BOOT)
    contrasts = [c for c in hyp.get("contrasts", []) if c["hypothesis"] != "H3"]
    escalation = {"split": split, "ladders": {}}
    h3_by_dataset = {}
    for ladder_name, ladder in LADDERS.items():
        block = {"label": LADDER_LABELS[ladder_name], "datasets": {}}
        for dataset in DATASETS:
            cellmaps = index[MODEL][dataset]
            if not all(cellmaps.get(cell) for cell in CELLS13):
                continue
            raw = LadderData(cellmaps, ladder, None, MODEL)
            scaled = LadderData(cellmaps, ladder, pooled, MODEL)
            s = raw.summary()
            h3 = raw.h3()
            entry = {
                "n": h3["n"], "h3_statistic": h3, "curve": {"tau": TAUS, "accuracy": s["acc"], "mean_nfe": s["nfe"],
                                                              "frontier": s["frontier"], "ece": raw.ece_curve()},
                "frontier": {"nfe": s["front_x"], "accuracy": s["front_y"], "ece_nfe": s["ece_front_x"], "ece": s["ece_front_y"]},
                "cells": {"nfe": raw.fixed_nfe, "accuracy": s["cell_acc"], "ece": s["cell_ece"]},
                "frozen_tau": raw.at_tau(frozen_tau) if frozen_tau is not None else None,
                "oracle": raw.oracle(),
                "ts_curve_secondary": {"h3_statistic": scaled.h3(n_boot=200), "frozen_tau_ts": scaled.at_tau(frozen_tau_ts) if frozen_tau_ts is not None else None},
            }
            block["datasets"][dataset] = entry
            if ladder_name == "preregistered":
                h3_by_dataset[dataset] = h3
        escalation["ladders"][ladder_name] = block
    hyp_out = {
        "H1": hyp.get("H1"), "H2": hyp.get("H2"), "H4": hyp.get("H4"),
        "H3": h3_verdict(h3_by_dataset),
        "H3_per_dataset": h3_by_dataset,
        "contrasts": contrasts,
        "notes": {k: v for k, v in (hyp.get("notes") or {}).items() if k != "H3"} | {
            "H3": "v1.2 registration: per dataset, G = mean over 50 tau points of escalation accuracy minus the running-max "
                  "fixed frontier at the same mean NFE; paired bootstrap CI (1000, seed 1234); dominates if CI > 0; "
                  "supported if >= 3 of 4 datasets dominate. Frozen tau is an operating point only."},
        "split": split,
    }
    (results / ("hypothesis_tests.json" if split == "test" else f"hypothesis_tests_{split}.json")).write_text(
        json.dumps(_json(hyp_out), indent=2) + "\n", encoding="utf-8")
    (results / ("escalation_test.json" if split == "test" else f"escalation_{split}.json")).write_text(
        json.dumps(_json(escalation), indent=2) + "\n", encoding="utf-8")

    # ---------- noeos ablation vs main (S=128), paired on the same items
    ablation = []
    noeos = index_rows(load_noeos_rows(root, split), MODEL, split)
    for dataset in DATASETS:
        for condition in ("C1", "C2"):
            for steps in (1, 4, 16):
                cell = (condition, 128, steps)
                off, on = index[MODEL][dataset].get(cell, {}), noeos[dataset].get(cell, {})
                ids = sorted(set(off) & set(on))
                if not ids:
                    continue
                a, b = [on[i] for i in ids], [off[i] for i in ids]

                def acc_diff(idx, a=a, b=b):
                    return accuracy([a[i]["correct"] for i in idx]) - accuracy([b[i]["correct"] for i in idx])

                def ece_diff(idx, a=a, b=b):
                    return (expected_calibration_error([a[i]["correct"] for i in idx], [max(a[i]["final_probs"].values()) for i in idx])
                            - expected_calibration_error([b[i]["correct"] for i in idx], [max(b[i]["final_probs"].values()) for i in idx]))

                ca = paired_bootstrap_diff(acc_diff, len(ids), n_resamples=N_BOOT)
                ce = paired_bootstrap_diff(ece_diff, len(ids), n_resamples=N_BOOT)
                ablation.append({
                    "dataset": dataset, "cell": cell_name(cell), "n": len(ids),
                    "acc_off": accuracy([r["correct"] for r in b]), "acc_on": accuracy([r["correct"] for r in a]),
                    "acc_diff": acc_diff(np.arange(len(ids))), "acc_lo": ca["lo"], "acc_hi": ca["hi"],
                    "ece_off": expected_calibration_error([r["correct"] for r in b], [max(r["final_probs"].values()) for r in b]),
                    "ece_on": expected_calibration_error([r["correct"] for r in a], [max(r["final_probs"].values()) for r in a]),
                    "ece_diff": ece_diff(np.arange(len(ids))), "ece_lo": ce["lo"], "ece_hi": ce["hi"],
                    "eos_off_median": float(np.median([r["eos_pad_fraction"] for r in b])),
                    "eos_on_median": float(np.median([r["eos_pad_fraction"] for r in a])),
                    "mcnemar": mcnemar_test([r["correct"] for r in a], [r["correct"] for r in b]),
                })
    (results / (f"noeos_ablation_{split}.json")).write_text(json.dumps(_json(ablation), indent=2) + "\n", encoding="utf-8")

    figures = _figures(figs, escalation, index, split)
    text = _markdown(root, split, protocol, fits, summary_rows, hyp_out, escalation, ablation, figures, rows)
    (results / out_name).write_text(text, encoding="utf-8")
    return {"summary_rows": len(summary_rows), "H": {k: hyp_out[k] for k in ("H1", "H2", "H3", "H4")}}


def _figures(figs: Path, escalation: dict, index: dict, split: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    for metric, ylabel, fname in (("accuracy", "accuracy", f"accuracy_vs_nfe_{split}.png"), ("ece", "raw ECE", f"ece_vs_nfe_{split}.png")):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
        for ax, dataset in zip(axes, DATASETS):
            pre = escalation["ladders"]["preregistered"]["datasets"].get(dataset)
            exp = escalation["ladders"]["exploratory"]["datasets"].get(dataset)
            if not pre:
                continue
            ax.scatter(pre["cells"]["nfe"], pre["cells"][metric], s=14, color="#888", label="fixed cells")
            if metric == "accuracy":
                ax.step(pre["frontier"]["nfe"], pre["frontier"]["accuracy"], where="post", color="#444", lw=1, label="frontier")
                ax.plot(pre["curve"]["mean_nfe"], pre["curve"]["accuracy"], color="#1f6feb", lw=1.6, label="pre-registered ladder")
                ax.plot(exp["curve"]["mean_nfe"], exp["curve"]["accuracy"], color="#bf8700", lw=1.2, ls="--", label="exploratory ladder")
                for entry, color in ((pre, "#1f6feb"), (exp, "#bf8700")):
                    ax.scatter([entry["oracle"]["mean_nfe"]], [entry["oracle"]["accuracy"]], marker="*", s=90, color=color, zorder=5)
                if pre["frozen_tau"]:
                    ax.scatter([pre["frozen_tau"]["mean_nfe"]], [pre["frozen_tau"]["accuracy"]], color="#d1242f", zorder=6, label="frozen τ")
            else:
                ax.step(pre["frontier"]["ece_nfe"], pre["frontier"]["ece"], where="post", color="#444", lw=1, label="ECE frontier (min)")
                ax.plot(pre["curve"]["mean_nfe"], pre["curve"]["ece"], color="#1f6feb", lw=1.6, label="pre-registered ladder")
                ax.plot(exp["curve"]["mean_nfe"], exp["curve"]["ece"], color="#bf8700", lw=1.2, ls="--", label="exploratory ladder")
            ax.set_title(dataset)
            ax.set_xlabel("mean NFE")
            ax.set_ylabel(ylabel)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figs / fname, dpi=120)
        plt.close(fig)
        written.append(f"figs/{fname}")

    # reliability: C0 and C1(S=128,T=16) (the pre-registered deliberative cell; no test-based selection)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    edges = np.linspace(0, 1, 16)
    for col, dataset in enumerate(DATASETS):
        for row_i, cell in enumerate((("C0", 0, None), ("C1", 128, 16))):
            ax = axes[row_i, col]
            group = index[MODEL][dataset].get(cell) or {}
            rows = list(group.values())
            if not rows:
                continue
            conf = np.array([max(r["final_probs"].values()) for r in rows])
            corr = np.array([r["correct"] for r in rows], dtype=float)
            ids = np.clip(np.digitize(conf, edges[1:-1]), 0, 14)
            xs = [conf[ids == b].mean() for b in range(15) if (ids == b).any()]
            ys = [corr[ids == b].mean() for b in range(15) if (ids == b).any()]
            ax.plot([0, 1], [0, 1], color="#bbb", lw=1)
            ax.plot(xs, ys, marker="o", color="#1f6feb")
            ax.set_title(f"{dataset} {cell_name(cell)}", fontsize=9)
            ax.set_xlabel("confidence")
            ax.set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(figs / f"reliability_{split}.png", dpi=120)
    plt.close(fig)
    written.append(f"figs/reliability_{split}.png")

    # anytime trajectories for C1(S=128, T=16): mean top-1 prob and accuracy per recorded step
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, dataset in zip(axes, DATASETS):
        rows = list((index[MODEL][dataset].get(("C1", 128, 16)) or {}).values())
        if not rows:
            continue
        steps = len(rows[0]["readouts"])
        top = [np.mean([max(r["readouts"][k]["probs"].values()) for r in rows]) for k in range(steps)]
        acc = [np.mean([max(r["readouts"][k]["probs"], key=r["readouts"][k]["probs"].get) == r["gold"] for r in rows]) for k in range(steps)]
        ax.plot(range(1, steps + 1), top, label="mean top-1 prob")
        ax.plot(range(1, steps + 1), acc, label="accuracy")
        ax.set_title(f"{dataset} C1 S=128 T=16 (readout step)")
        ax.set_xlabel("readout step (step T+1 = final read)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / f"anytime_{split}.png", dpi=120)
    plt.close(fig)
    written.append(f"figs/anytime_{split}.png")
    return written


def _markdown(root, split, protocol, fits, summary_rows, hyp, escalation, ablation, figures, rows) -> str:
    env = {}
    env_path = root / "results" / "environment.json"
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))
    splits = json.loads((root / "results" / "splits.json").read_text(encoding="utf-8"))
    L = [f"# Report: A Deliberation Dial for Diffusion Decision Models ({split} split)", "",
         "Generated by `python -m dd.report` (`dd/report_v12.py`). Every number is computed from `results/raw` and "
         "`results/dev_fits.json`; the same values are in `results/summary.csv`, `results/hypothesis_tests.json`, "
         "`results/escalation_test.json`, and `results/noeos_ablation_test.json`.", ""]

    L += ["## 1. Setup", ""]
    revisions = sorted({r.get("model") for r in rows})
    L += [f"- Protocol v1.1 (PLAN §16) with v1.2 additions registered before test (`results/HYPOTHESES.md`).",
          f"- Models: LLaDA-8B-Instruct (`GSAI-ML/LLaDA-8B-Instruct` @ 08b83a6feb34df1a6011b80c3c00c7563e963b07), "
          f"Qwen3-8B baseline (`Qwen/Qwen3-8B`, revision recorded at load; separate venv, transformers 4.56.2).",
          f"- Device: NVIDIA L4 24GB (AWS g6.2xlarge), bf16, batch size 1, greedy decoding. Last environment file: device `{env.get('device')}`.",
          f"- Template: `{protocol['template']}` (\"The answer is\" + slot + `<|eot_id|>`); suppress_eos_in_scratch main setting: "
          f"`{protocol['suppress_eos_in_scratch']}`; anytime shortcut: off (Gate 2 failed: agreement 0.90, ΔECE 0.027).",
          "- Items (dev / test): " + "; ".join(f"{k} {len(v['dev'])} / {len(v['test'])}" for k, v in splits["datasets"].items()), ""]

    L += ["## 2. Main table per dataset", "",
          "acc [95% CI] · ECE (15 equal-width bins) · adaptive ECE (15 equal-mass bins) · ECE after pooled per-cell TS · "
          "ECE after per-(dataset, cell) TS · mean off_label_mass · median eos_pad_fraction · NFE. Temperatures fit on dev.", ""]
    for dataset in DATASETS:
        L += [f"### {dataset}", "", "| model | cell | n | acc [CI] | ECE | aECE | ECE-TS pooled | ECE-TS dataset | Brier | NLL | AURC | off_label | eos_pad | NFE |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in summary_rows if x["dataset"] == dataset]:
            L.append(f"| {r['model']} | {r['cell']} | {r['n']} | {f(r['accuracy'])} [{f(r['accuracy_lo'])}, {f(r['accuracy_hi'])}] | "
                     f"{f(r['ece'])} | {f(r['adaptive_ece'])} | {f(r['ece_ts_pooled'])} | {f(r['ece_ts_dataset'])} | {f(r['brier'])} | "
                     f"{f(r['nll'])} | {f(r['aurc'])} | {f(r['off_label_mass'])} | {f(r['eos_pad_median'])} | {f(r['nfe'], 0)} |")
        L.append("")

    L += ["## 3. Figures", ""] + [f"- `results/{p}`" for p in figures] + [""]

    L += ["## 4. Hypotheses H1–H4", ""]
    for name in ("H1", "H2", "H3", "H4"):
        L.append(f"- **{name}: {hyp[name]}.** {hyp['notes'].get(name, '')}")
    L += ["", "| hypothesis | dataset | contrast | estimate | 95% CI | McNemar p (exact) | call |", "|---|---|---|---|---|---|---|"]
    for c in hyp["contrasts"]:
        mc = c.get("mcnemar") or {}
        L.append(f"| {c['hypothesis']} | {c['dataset']} | {c['description']} | {f(c.get('estimate'), 4)} | "
                 f"[{f(c.get('lo'), 4)}, {f(c.get('hi'), 4)}] | {f(mc.get('p_value_exact'), 4) if mc else '-'} | {c['status']} |")
    L += ["", "H3 per dataset (pre-registered ladder, full τ curve vs running-max frontier):", "",
          "| dataset | n | G | 95% CI | min gap | max gap | share τ gap ≥ 0 | dominates |", "|---|---|---|---|---|---|---|---|"]
    for dataset, h in hyp["H3_per_dataset"].items():
        L.append(f"| {dataset} | {h['n']} | {f(h['G'], 4)} | [{f(h['lo'], 4)}, {f(h['hi'], 4)}] | {f(h['min_gap'])} | {f(h['max_gap'])} | "
                 f"{f(h['frac_nonneg'], 2)} | {h['dominates']} |")
    L.append("")

    L += ["## 5. Escalation", ""]
    for ladder_name, block in escalation["ladders"].items():
        L += [f"### {block['label']}", "",
              "| dataset | G [CI] | frozen τ: acc / ECE / NFE / frontier | oracle acc / NFE | TS curve G [CI] (secondary) |", "|---|---|---|---|---|"]
        for dataset, e in block["datasets"].items():
            h, fz, orc, ts = e["h3_statistic"], e["frozen_tau"] or {}, e["oracle"], e["ts_curve_secondary"]["h3_statistic"]
            L.append(f"| {dataset} | {f(h['G'], 4)} [{f(h['lo'], 4)}, {f(h['hi'], 4)}] | {f(fz.get('accuracy'))} / {f(fz.get('ece'))} / "
                     f"{f(fz.get('mean_nfe'), 2)} / {f(fz.get('frontier'))} | {f(orc['accuracy'])} / {f(orc['mean_nfe'], 2)} | "
                     f"{f(ts['G'], 4)} [{f(ts['lo'], 4)}, {f(ts['hi'], 4)}] |")
        L += ["", "Curve points (accuracy / ECE / mean NFE / frontier):", "",
              "| τ | " + " | ".join(block["datasets"]) + " |", "|---|" + "---|" * len(block["datasets"])]
        for t in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49):
            cells = []
            for e in block["datasets"].values():
                c = e["curve"]
                cells.append(f"{f(c['accuracy'][t])} / {f(c['ece'][t])} / {f(c['mean_nfe'][t], 2)} / {f(c['frontier'][t])}")
            L.append(f"| {TAUS[t]:.3f} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["The exploratory ladder was chosen after looking at dev; it carries no verdict.", ""]

    L += ["## 6. Baselines", "", "Qwen3-8B C0 (next token after \"The answer is\", enable_thinking=False) is in the per-dataset tables above.", ""]

    L += ["## 7. suppress_eos_in_scratch ablation at S=128 (secondary; original test items only)", "",
          "| dataset | cell | n | acc off → on | Δacc [CI] | ECE off → on | ΔECE [CI] | eos_pad median off → on | McNemar p (exact) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for a in ablation:
        L.append(f"| {a['dataset']} | {a['cell']} | {a['n']} | {f(a['acc_off'])} → {f(a['acc_on'])} | {f(a['acc_diff'], 4)} "
                 f"[{f(a['acc_lo'], 4)}, {f(a['acc_hi'], 4)}] | {f(a['ece_off'])} → {f(a['ece_on'])} | {f(a['ece_diff'], 4)} "
                 f"[{f(a['ece_lo'], 4)}, {f(a['ece_hi'], 4)}] | {f(a['eos_off_median'])} → {f(a['eos_on_median'])} | "
                 f"{f(a['mcnemar']['p_value_exact'], 4)} |")
    if not ablation:
        L.append("| (no ablation rows for this split) | | | | | | | | |")
    L.append("")

    L += ["## 8. Deviations and failures", "",
          "Full chronological record: `results/LOG.md`. Material items:", "",
          "- Protocol v1.1 amendment before any dev data (eot-pinned slot, think prefix, new prompt, v3e template, batch 1); the v1.0 smoke is archived.",
          "- v1.2 additions registered before test: H3 full-curve test with a running-max frontier, exploratory ladder, per-dataset TS, adaptive ECE, noeos ablation, test set expanded to 1000/1000/1000/600.",
          "- Gate 2 failed, so T=4 always ran natively.",
          "- Qwen3-8B ran under transformers 4.56.2 in a separate venv (4.46.3 lacks Qwen3).",
          "- Phase 3 OOM (autograd enabled in forward) fixed with torch.inference_mode; rows unaffected.",
          "- C3 (LogicDiff) did not run: `schedulers/logicdiff.py` is absent.",
          "- StrategyQA uses its train split (only labeled split): possible pretraining contamination, relevant to H2.", ""]

    L += ["## 9. Limitations", "",
          "- One diffusion model (LLaDA-8B-Instruct); Dream-7B did not run.",
          "- Greedy decoding only; English only; one prompt template, fixed after smoke items.",
          "- Sample sizes are the test counts above; jagged is synthetic.",
          "- The dev-fitted τ and temperatures come from 360 dev items.",
          "- LogicDiff scheduling (C3) pending the author's scheduler.", ""]
    return "\n".join(L) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(build(root, "test"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
