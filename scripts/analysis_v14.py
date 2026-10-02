"""v1.4 EXPLORATORY CPU analyses on the ORIGINAL test items (400/400/400/240).

B1 Dream C0 minus Qwen2.5-7B C0 (Instruct as requested; base = Dream's actual initialization).
B2 LLaDA C0 minus Qwen2.5-7B-Instruct C0; Qwen3-8B C0 minus Qwen2.5-7B-Instruct C0.
   Columns: dacc (McNemar exact), dECE, d adaptive ECE, dBrier, dNLL, and ECE after each model's
   own dev-fitted temperature (and its difference).
B3 "Any thinking vs none": ECE(cell) - ECE(C0) for every C1/C2 cell; LLaDA on the full test set,
   Dream on the original test items; per dataset, the count of cells whose CI lies above 0.
Paired bootstrap 1000 resamples, seed 1234, 95% percentile CIs.

Usage: python scripts/analysis_v14.py -> results/v14_exploratory.md, results/v14_exploratory.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dd.analysis import CELLS13, adaptive_ece, scaled_probs  # noqa: E402
from dd.metrics import expected_calibration_error, mcnemar_test, multiclass_brier, negative_log_likelihood  # noqa: E402

DATASETS = ["boolq", "strategyqa", "arc_c", "jagged"]
N_BOOT, SEED = 1000, 1234
R = ROOT / "results"


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def rows_from(dir_: Path, model: str, template: str, split: str, condition="C0", scratch=0, steps=None, suppress=False) -> dict:
    """{dataset: {item_id: row}} for one cell."""
    step = "na" if steps is None else str(steps)
    out = {d: {} for d in DATASETS}
    for ds in DATASETS:
        path = dir_ / f"{ds}_{split}_{model}_{condition}_S{scratch}_T{step}_{template}{'_noeos' if suppress else ''}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[ds][r["item_id"]] = r
    return out


def vec(rows: list, temperature: float | None) -> dict:
    corr = np.array([bool(r["correct"]) for r in rows])
    conf = np.array([max(r["final_probs"].values()) for r in rows])
    conf_ts = np.array([max(scaled_probs(r, temperature).values()) for r in rows]) if temperature else conf
    probs = [[r["final_probs"][x] for x in r["final_probs"]] for r in rows]
    gold = [list(r["final_probs"]).index(r["gold"]) for r in rows]
    gp = [r["final_probs"][r["gold"]] for r in rows]
    return {"corr": corr, "conf": conf, "conf_ts": conf_ts, "probs": probs, "gold": gold, "gp": gp}


def stats(v: dict, ix: np.ndarray) -> np.ndarray:
    c, k, kt = v["corr"][ix], v["conf"][ix], v["conf_ts"][ix]
    return np.array([
        c.mean(),
        expected_calibration_error(c, k),
        adaptive_ece(c, k),
        multiclass_brier([v["probs"][i] for i in ix], [v["gold"][i] for i in ix]),
        negative_log_likelihood([v["gp"][i] for i in ix]),
        expected_calibration_error(c, kt),
    ])


NAMES = ["acc", "ECE", "aECE", "Brier", "NLL", "ECE-TS(own)"]


def paired_table(L, out, key, title, A, tA, B, tB, orig):
    L += [f"### {title}", "", "| dataset | n | Δacc [CI] (McNemar p) | ΔECE [CI] | ΔaECE [CI] | ΔBrier [CI] | ΔNLL [CI] | ECE-TS A / B | ΔECE-TS [CI] |",
          "|---|---|---|---|---|---|---|---|---|"]
    out[key] = {}
    for ds in DATASETS:
        ids = sorted(set(A[ds]) & set(B[ds]) & orig[ds])
        if not ids:
            L.append(f"| {ds} | 0 | rows missing | | | | | | |")
            continue
        va, vb = vec([A[ds][i] for i in ids], tA), vec([B[ds][i] for i in ids], tB)
        full = np.arange(len(ids))
        sa, sb = stats(va, full), stats(vb, full)
        d = sa - sb
        rng = np.random.default_rng(SEED)
        boots = np.array([stats(va, ix) - stats(vb, ix) for ix in (rng.integers(0, len(ids), len(ids)) for _ in range(N_BOOT))])
        lo, hi = np.quantile(boots, 0.025, axis=0), np.quantile(boots, 0.975, axis=0)
        p = mcnemar_test(va["corr"], vb["corr"])["p_value_exact"]
        out[key][ds] = {"n": len(ids), "A": dict(zip(NAMES, sa.tolist())), "B": dict(zip(NAMES, sb.tolist())),
                        "diff": dict(zip(NAMES, d.tolist())), "lo": dict(zip(NAMES, lo.tolist())), "hi": dict(zip(NAMES, hi.tolist())),
                        "mcnemar_p": p, "temperature_A": tA, "temperature_B": tB}
        cell = lambda j: f"{f(d[j])} [{f(lo[j])}, {f(hi[j])}]"  # noqa: E731
        L.append(f"| {ds} | {len(ids)} | {cell(0)} (p={f(p, 4)}) | {cell(1)} | {cell(2)} | {cell(3)} | {cell(4)} | "
                 f"{f(sa[5])} / {f(sb[5])} | {cell(5)} |")
    L.append("")


def any_thinking(L, out, key, label, cells_rows, base_rows, ids_by_ds):
    L += [f"### {label}", "", "| dataset | cell | ECE(cell) − ECE(C0) [CI] | CI |", "|---|---|---|---|"]
    out[key] = {"cells": {}, "counts": {}}
    for ds in DATASETS:
        count, total = 0, 0
        for cell, rows in cells_rows.items():
            ids = sorted(set(rows[ds]) & set(base_rows[ds]) & ids_by_ds[ds])
            if not ids:
                continue
            a = vec([rows[ds][i] for i in ids], None)
            b = vec([base_rows[ds][i] for i in ids], None)

            def dece(ix):
                return expected_calibration_error(a["corr"][ix], a["conf"][ix]) - expected_calibration_error(b["corr"][ix], b["conf"][ix])

            rng = np.random.default_rng(SEED)
            boots = [dece(rng.integers(0, len(ids), len(ids))) for _ in range(N_BOOT)]
            est, lo, hi = dece(np.arange(len(ids))), np.quantile(boots, 0.025), np.quantile(boots, 0.975)
            call = "above 0" if lo > 0 else ("below 0" if hi < 0 else "includes 0")
            count += lo > 0
            total += 1
            out[key]["cells"][f"{ds}|{cell}"] = {"n": len(ids), "estimate": est, "lo": float(lo), "hi": float(hi), "call": call}
            L.append(f"| {ds} | {cell} | {f(est)} [{f(lo)}, {f(hi)}] | {call} |")
        out[key]["counts"][ds] = {"above_zero": int(count), "cells": total}
    L += ["", "Cells whose CI lies above 0 (worse calibration than the single read): " +
          "; ".join(f"{ds} {v['above_zero']}/{v['cells']}" for ds, v in out[key]["counts"].items()), ""]


def main() -> None:
    splits = json.loads((R / "splits.json").read_text(encoding="utf-8"))
    orig = {ds: set(splits["datasets"][ds]["test_original"]) for ds in DATASETS}
    full = {ds: set(splits["datasets"][ds]["test"]) for ds in DATASETS}
    fits = json.loads((R / "dev_fits.json").read_text(encoding="utf-8"))["temperatures"]
    dream_fits = json.loads((R / "dream" / "dev_fits.json").read_text(encoding="utf-8"))["temperatures"]
    dream_t = json.loads((R / "dream" / "template.json").read_text(encoding="utf-8"))["template"]

    models = {
        "LLaDA-8B": (rows_from(R / "raw", "llada-8b-instruct", "v3e", "test"), fits["llada-8b-instruct|C0|0|na"]["temperature"]),
        "Qwen3-8B": (rows_from(R / "raw", "qwen3-8b", "v3e", "test"), fits["qwen3-8b|C0|0|na"]["temperature"]),
        "Dream-7B": (rows_from(R / "raw_dream", "dream-v0-instruct-7b", dream_t, "test"), dream_fits["dream-v0-instruct-7b|C0|0|na"]["temperature"]),
    }
    for name, folder in (("Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"), ("Qwen2.5-7B-base", "qwen2.5-7b-base")):
        tdir = R / "causal" / folder
        if (tdir / "dev_fits.json").exists():
            cf = json.loads((tdir / "dev_fits.json").read_text(encoding="utf-8"))
            models[name] = (rows_from(R / "raw_causal", folder, cf["template"], "test"), cf["temperature"])

    out: dict = {"temperatures": {k: v[1] for k, v in models.items()}}
    L = ["# v1.4 exploratory analyses (original test items unless stated)", "",
         "Paired bootstrap (1000, seed 1234), 95% percentile CIs; McNemar exact. Differences are A − B. "
         "ECE-TS(own) applies each model's own dev-fitted C0 temperature. Generated by `scripts/analysis_v14.py`.", "",
         "Temperatures (C0, dev-fitted): " + ", ".join(f"{k} {f(v[1], 2)}" for k, v in models.items()), ""]

    L += ["## B1. Dream-7B C0 vs its Qwen2.5-7B relatives", "",
          "Dream-v0-Instruct-7B was initialized from Qwen2.5-7B **base** (Dream blog); Qwen2.5-7B-Instruct is its sibling.", ""]
    for other in ("Qwen2.5-7B-Instruct", "Qwen2.5-7B-base"):
        if other in models:
            paired_table(L, out, f"B1|Dream-7B-{other}", f"Dream-7B C0 − {other} C0", *models["Dream-7B"], *models[other], orig)

    L += ["## B2. Context: LLaDA-8B and Qwen3-8B vs Qwen2.5-7B-Instruct", ""]
    if "Qwen2.5-7B-Instruct" in models:
        for a in ("LLaDA-8B", "Qwen3-8B"):
            paired_table(L, out, f"B2|{a}-Qwen2.5-7B-Instruct", f"{a} C0 − Qwen2.5-7B-Instruct C0", *models[a], *models["Qwen2.5-7B-Instruct"], orig)

    L += ["## B3. Any thinking vs none: ECE(C1/C2 cell) − ECE(C0)", ""]
    llada_cells = {f"{c} {s},{t}": rows_from(R / "raw", "llada-8b-instruct", "v3e", "test", c, s, t) for c, s, t in CELLS13[1:]}
    any_thinking(L, out, "B3|LLaDA", "LLaDA-8B, full test set (1000/1000/1000/600), 12 cells", llada_cells, models["LLaDA-8B"][0], full)
    dream_cells = {f"{c} 32,{t}": rows_from(R / "raw_dream", "dream-v0-instruct-7b", dream_t, "test", c, 32, t) for c in ("C1", "C2") for t in (1, 4, 16)}
    any_thinking(L, out, "B3|Dream", "Dream-7B, original test items, 6 cells", dream_cells, models["Dream-7B"][0], orig)

    (R / "v14_exploratory.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (R / "v14_exploratory.json").write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print("wrote results/v14_exploratory.md")


if __name__ == "__main__":
    main()
