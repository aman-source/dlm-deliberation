"""Compute every derived number used in the paper prose into results/paper_numbers.json.

Values that already sit verbatim in results files are referenced there; this file holds
aggregates (ranges, means, counts) so every number in paper/main.tex has a file source.
Usage: python scripts/paper_numbers.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
DS = ["boolq", "strategyqa", "arc_c", "jagged"]


def rng(values):
    return {"min": float(min(values)), "max": float(max(values))}


def rows_all_llada(R: Path) -> list:
    return [r for r in csv.DictReader((R / "summary.csv").open(encoding="utf-8")) if r["model"].startswith("llada")]


def c0_ece(R: Path, dataset: str) -> float:
    return next(float(r["ece"]) for r in rows_all_llada(R) if r["dataset"] == dataset and r["condition"] == "C0")


def main() -> None:
    out: dict = {}
    splits = json.loads((R / "splits.json").read_text(encoding="utf-8"))
    out["n_test"] = {d: len(splits["datasets"][d]["test"]) for d in DS}
    out["n_test_total"] = sum(out["n_test"].values())
    out["n_test_original"] = {d: len(splits["datasets"][d]["test_original"]) for d in DS}
    out["n_test_original_total"] = sum(out["n_test_original"].values())
    out["n_dev"] = {d: len(splits["datasets"][d]["dev"]) for d in DS}
    out["n_dev_total"] = sum(out["n_dev"].values())

    # instance-hours from the ledger (START/STOP pairs). The ledger is not in the public release;
    # there the value already recorded in paper_numbers.json is kept.
    ledger = R / "aws_usage.log"
    if not ledger.exists():
        prev = R / "paper_numbers.json"
        out["instance_hours_total"] = json.loads(prev.read_text(encoding="utf-8")).get("instance_hours_total") if prev.exists() else None
    total, open_start = 0.0, {}
    for line in (ledger.read_text(encoding="utf-8").splitlines() if ledger.exists() else []):
        p = line.split()
        if len(p) < 3:
            continue
        t = dt.datetime.strptime(p[0], "%Y-%m-%dT%H:%M:%SZ")
        if p[1] == "START":
            open_start[p[2]] = t
        elif p[1] == "STOP" and p[2] in open_start:
            total += (t - open_start.pop(p[2])).total_seconds() / 3600
    if ledger.exists():
        out["instance_hours_total"] = round(total, 2)

    hyp = json.loads((R / "hypothesis_tests.json").read_text(encoding="utf-8"))
    c = hyp["contrasts"]
    h1 = [x for x in c if x["hypothesis"] == "H1"]
    out["H1"] = {"verdict": hyp["H1"], "n_contrasts": len(h1), "n_above_zero": sum(x["lo"] > 0 for x in h1),
                 "estimate_range": rng([x["estimate"] for x in h1])}
    out["H2"] = {"verdict": hyp["H2"], "by_dataset": {x["dataset"]: {"estimate": x["estimate"], "lo": x["lo"], "hi": x["hi"],
                 "p": (x.get("mcnemar") or {}).get("p_value_exact")} for x in c if x["hypothesis"] == "H2"}}
    out["H3"] = {"verdict": hyp["H3"], "by_dataset": hyp["H3_per_dataset"],
                 "n_dominating": sum(v["dominates"] for v in hyp["H3_per_dataset"].values())}
    h4 = [x for x in c if x["hypothesis"] == "H4"]
    out["H4"] = {"verdict": hyp["H4"], "contrasts": {f"{x['dataset']}|{x['kind']}": {"estimate": x["estimate"], "lo": x["lo"], "hi": x["hi"],
                 "status": x["status"], "p": (x.get("mcnemar") or {}).get("p_value_exact")} for x in h4},
                 "n_supported": sum(x["status"] == "supported" for x in h4), "n_contrasts": len(h4)}
    out["n_hypotheses_supported"] = sum(hyp[k] == "supported" for k in ("H1", "H2", "H3", "H4"))

    # temperature scaling aggregates
    rows = list(csv.DictReader((R / "summary.csv").open(encoding="utf-8")))
    sc = [r for r in rows if r["model"].startswith("llada") and r["condition"] != "C0"]
    raw = np.array([float(r["ece"]) for r in sc])
    pool = np.array([float(r["ece_ts_pooled"]) for r in sc])
    dsts = np.array([float(r["ece_ts_dataset"]) for r in sc])
    out["ts_llada"] = {"n_cells": len(sc), "mean_raw": float(raw.mean()), "mean_pooled": float(pool.mean()), "mean_dataset": float(dsts.mean()),
                       "reduction_pooled_pct": float(100 * (1 - pool.mean() / raw.mean())),
                       "reduction_dataset_pct": float(100 * (1 - dsts.mean() / raw.mean())),
                       "share_lowered_pooled_pct": float(100 * (pool < raw).mean()), "share_lowered_dataset_pct": float(100 * (dsts < raw).mean())}
    bc = json.loads((R / "v13_bc.json").read_text(encoding="utf-8"))
    dt_ = [v for k, v in bc["C"]["tables"].items() if not k.endswith("|C0")]
    out["ts_dream"] = {"n_cells": len(dt_), "mean_raw": float(np.mean([v["ece"] for v in dt_])),
                       "mean_ts": float(np.mean([v["ece_ts"] for v in dt_]))}
    out["calibration_summary"] = json.loads((ROOT / "paper" / "figs" / "calibration_summary.json").read_text(encoding="utf-8"))

    v14 = json.loads((R / "v14_exploratory.json").read_text(encoding="utf-8"))
    for model in ("LLaDA", "Dream"):
        counts = v14[f"B3|{model}"]["counts"]
        out[f"any_scratch_{model}"] = {"above_zero": sum(v["above_zero"] for v in counts.values()),
                                       "cells": sum(v["cells"] for v in counts.values()), "by_dataset": counts}
    def onepass(key):
        b = v14[key]
        return {"dECE_range": rng([b[d]["diff"]["ECE"] for d in DS]), "dacc_range": rng([b[d]["diff"]["acc"] for d in DS]),
                "dECE_ci_includes_zero": [d for d in DS if b[d]["lo"]["ECE"] <= 0 <= b[d]["hi"]["ECE"]],
                "dacc_ci_includes_zero": [d for d in DS if b[d]["lo"]["acc"] <= 0 <= b[d]["hi"]["acc"]], "by_dataset": b}
    out["onepass_dream_vs_qwen25_instruct"] = onepass("B1|Dream-7B-Qwen2.5-7B-Instruct")
    out["onepass_dream_vs_qwen25_base"] = onepass("B1|Dream-7B-Qwen2.5-7B-base")
    out["onepass_llada_vs_qwen25_instruct"] = onepass("B2|LLaDA-8B-Qwen2.5-7B-Instruct")
    ex = json.loads((R / "v13_exploratory.json").read_text(encoding="utf-8"))
    a5 = ex["A5"]
    out["onepass_llada_vs_qwen3"] = {"dECE_range": rng([a5[d]["diff"][1] for d in DS]), "dacc": {d: a5[d]["diff"][0] for d in DS},
                                     "dacc_lo": {d: a5[d]["lo"][0] for d in DS}, "dacc_hi": {d: a5[d]["hi"][0] for d in DS}}
    out["temperatures_c0"] = v14["temperatures"]
    out["dream_vs_base_ece_includes_zero_n"] = len(out["onepass_dream_vs_qwen25_base"]["dECE_ci_includes_zero"])

    # T=32 plateau
    bcon = bc["B"]["contrasts"]
    t32_t1 = [v for k, v in bcon.items() if k.endswith("|ece|T32-T1")]
    t32_t16 = [v for k, v in bcon.items() if k.endswith("|ece|T32-T16")]
    acc_t32_t1 = [v for k, v in bcon.items() if k.endswith("|accuracy|T32-T1")]
    out["t32"] = {"ece_T32_T1_above": sum(v["call"] == "above 0" for v in t32_t1), "n": len(t32_t1),
                  "ece_T32_T1_range": rng([v["estimate"] for v in t32_t1]),
                  "ece_T32_T16_includes_zero": sum(v["call"] == "includes 0" for v in t32_t16),
                  "acc_T32_T1_includes_zero": sum(v["call"] == "includes 0" for v in acc_t32_t1)}

    # churn, AUROC, oracle stability, stated answers, trajectories
    out["margin_auroc"] = {d: {k: ex["A4"][d][k] for k in ("auroc_fix", "lo", "hi", "wrong_n", "fixed")} for d in DS}
    out["margin_auroc_range"] = rng([ex["A4"][d]["auroc_fix"] for d in DS])
    out["oracle_stability_prereg_kept_range"] = rng([v["kept"] for k, v in ex["A1_oracle"].items() if k.startswith("pre-registered")])
    out["flips_examples"] = {k: ex["A1"][k] for k in ("strategyqa|C1 32,16", "jagged|C1 128,16")}
    a1 = ex["A1"].values()
    out["flips_net_counts"] = {"cells": len(ex["A1"]), "net_ci_above_zero": sum(v["lo"] > 0 for v in a1),
                               "net_ci_below_zero": sum(v["hi"] < 0 for v in a1),
                               "net_ci_includes_zero": sum(v["lo"] <= 0 <= v["hi"] for v in a1)}
    out["dream_step_acc_gain_datasets"] = sum(bc["C"]["contrasts"][f"{d}|H1 acc"]["lo"] > 0 for d in DS)
    a2 = ex["A2"]
    c1_t16 = {k: v["share"] for k, v in a2.items() if "|C1 " in k and k.endswith(",16")}
    c2_t16 = {k: v["share"] for k, v in a2.items() if "|C2 " in k and k.endswith(",16")}
    out["stated_answer"] = {"C1_T16_share": c1_t16, "C1_T16_share_max_excl_arc128": max(v for k, v in c1_t16.items() if k != "arc_c|C1 128,16"),
                            "C1_arc128_share": c1_t16["arc_c|C1 128,16"], "C2_T16_share_range": rng(c2_t16.values()),
                            "C2_T16_agree_range": rng([a2[k]["agree"] for k in c2_t16])}
    a3 = ex["A3"]
    out["trajectory_s32_wrong_final_range"] = rng([a3[f"S32|{d}|wrong"]["mean_top1_by_step"][-1] for d in DS])
    out["trajectory_s32_wrong_step1_range"] = rng([a3[f"S32|{d}|wrong"]["mean_top1_by_step"][0] for d in DS])

    # Dream step effects and H4-style
    dc = bc["C"]["contrasts"]
    out["dream_step_acc"] = {d: dc[f"{d}|H1 acc"] for d in DS}
    out["dream_step_ece"] = {d: dc[f"{d}|H1 ECE"] for d in DS}
    out["dream_h4"] = {d: {"acc": dc[f"{d}|H4 acc"], "ece": dc[f"{d}|H4 ECE"]} for d in DS}
    out["dream_template"] = json.loads((R / "dream" / "template.json").read_text(encoding="utf-8"))
    out["llada_template_smoke_v10"] = json.loads((R / "smoke_v1_0" / "template.json").read_text(encoding="utf-8"))
    out["llada_template_v11"] = json.loads((R / "template.json").read_text(encoding="utf-8"))
    out["gate2"] = json.loads((R / "gate2.json").read_text(encoding="utf-8"))
    out["noeos"] = json.loads((R / "noeos_ablation_test.json").read_text(encoding="utf-8"))
    out["dev_fits_tau"] = {k: json.loads((R / "dev_fits.json").read_text(encoding="utf-8"))[k] for k in ("tau", "tau_temperature_scaled")}
    # anytime readouts (C1 S=128 T=16, LLaDA test) and reliability gaps (figure captions)
    anytime = {}
    for d in DS:
        rows = [json.loads(l) for l in (R / "raw" / f"{d}_test_llada-8b-instruct_C1_S128_T16_v3e.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        top = lambda k: float(np.mean([max(r["readouts"][k]["probs"].values()) for r in rows]))  # noqa: E731
        acc = lambda k: float(np.mean([max(r["readouts"][k]["probs"], key=r["readouts"][k]["probs"].get) == r["gold"] for r in rows]))  # noqa: E731
        anytime[d] = {"top1_first": top(0), "top1_final": top(-1), "acc_first": acc(0), "acc_final": acc(-1)}
    out["anytime_c1_s128_t16"] = anytime
    out["anytime_top1_first_range"] = rng([v["top1_first"] for v in anytime.values()])
    out["anytime_top1_final_range"] = rng([v["top1_final"] for v in anytime.values()])
    out["anytime_abs_acc_change_max"] = float(max(abs(v["acc_final"] - v["acc_first"]) for v in anytime.values()))
    rel = json.loads((ROOT / "paper" / "figs" / "reliability_equal_mass.json").read_text(encoding="utf-8"))
    out["reliability_mean_gap"] = {k: float(np.mean(np.array(v["bin_conf"]) - np.array(v["bin_acc"]))) for k, v in rel.items()}
    out["c0_lowest_raw_ece_exceptions"] = [f"{r['dataset']}|{r['cell']}" for r in rows_all_llada(R)
                                           if r["condition"] != "C0" and float(r["ece"]) < c0_ece(R, r["dataset"])]

    (R / "paper_numbers.json").write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print("wrote results/paper_numbers.json")


if __name__ == "__main__":
    main()
