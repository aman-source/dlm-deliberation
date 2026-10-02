"""Build results/summary.csv, figures, and results/REPORT.md from raw rows.

Numbers in the report are formatted from summary.csv, dev_fits.json,
hypothesis_tests.json, and escalation_test.json. The writer does not
restate a metric that was not saved to one of those files.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from dd.decode import flip_rate
from dd.escalate import (
    decide_items,
    fixed_frontier,
    group_ladder_items,
    interpolate,
    oracle_decisions,
    select_tau,
    summarize_decisions,
    sweep_tau,
    tau_grid,
)
from dd.metrics import (
    accuracy,
    apply_temperature_probs,
    aurc,
    bootstrap_ci,
    classify_contrast,
    expected_calibration_error,
    fit_temperature_ragged,
    multiclass_brier,
    negative_log_likelihood,
    paired_bootstrap_diff,
    reliability_bins,
    mcnemar_test,
)

PRIMARY_MODEL_PREFIX = "llada"
BASELINE_TOKEN = "qwen"


def main_setting(root: Path) -> dict:
    """Frozen template and suppression setting for protocol v1.1 (None when not yet frozen)."""
    out = {"template": None, "suppress_eos_in_scratch": None}
    template_path = root / "results" / "template.json"
    protocol_path = root / "results" / "protocol_v1_1.json"
    if template_path.exists():
        out["template"] = json.loads(template_path.read_text(encoding="utf-8")).get("template")
    if protocol_path.exists():
        out["suppress_eos_in_scratch"] = bool(json.loads(protocol_path.read_text(encoding="utf-8"))["suppress_eos_in_scratch"])
    return out


def load_rows(root: Path) -> list[dict]:
    """Main-table rows: protocol v1.1, the frozen template, and (C1/C2) the main suppression setting.

    Only files directly in results/raw are read; archived v1.0 smoke rows live in subfolders.
    """
    raw = root / "results" / "raw"
    rows = []
    if not raw.exists():
        return rows
    setting = main_setting(root)
    for path in sorted(raw.glob("*.jsonl")):
        if path.name.startswith("failures"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("protocol") != "v1.1":
                continue
            if setting["template"] is not None and row.get("template") != setting["template"]:
                continue
            if (
                setting["suppress_eos_in_scratch"] is not None
                and row.get("condition") in {"C1", "C2", "C3"}
                and bool(row.get("suppress_eos_in_scratch", False)) != setting["suppress_eos_in_scratch"]
            ):
                continue
            rows.append(row)
    return rows


def _cell_label(row: dict) -> str:
    step = "na" if row.get("T") is None else str(row["T"])
    return f"{row['condition']}|S{row['S']}|T{step}"


def _temp_key(model: str, condition: str, scratch: int, steps) -> str:
    step = "na" if steps is None else str(steps)
    return f"{model}|{condition}|{scratch}|{step}"


def _finite(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _letters_of(row: dict) -> list[str]:
    return list(row["final_probs"])


def summarize_rows(rows: list[dict], temperature: float | None, n_resamples: int) -> dict:
    correct = [bool(row["correct"]) for row in rows]
    confidence = [max(row["final_probs"].values()) for row in rows]
    prob_rows = []
    gold_index = []
    gold_probs = []
    for row in rows:
        letters = _letters_of(row)
        prob_rows.append([row["final_probs"][letter] for letter in letters])
        gold_index.append(letters.index(row["gold"]))
        gold_probs.append(row["final_probs"][row["gold"]])
    acc_ci = bootstrap_ci(lambda index: accuracy([correct[i] for i in index]), len(rows), n_resamples=n_resamples)
    ece_ci = bootstrap_ci(
        lambda index: expected_calibration_error([correct[i] for i in index], [confidence[i] for i in index]),
        len(rows),
        n_resamples=n_resamples,
    )
    ece_ts = None
    if temperature is not None and temperature > 0 and all("letter_logits" in row for row in rows):
        scaled_conf = []
        for row in rows:
            letters = list(row["letter_logits"])
            scaled = apply_temperature_probs(row["letter_logits"], letters, temperature)
            scaled_conf.append(max(scaled.values()))
        ece_ts = expected_calibration_error(correct, scaled_conf)
    walls = [row["wall_ms"] for row in rows if row.get("wall_ms") is not None]
    flips = flip_rate(rows) if rows and rows[0].get("condition") == "C1" else None
    return {
        "n": len(rows),
        "accuracy": accuracy(correct),
        "accuracy_lo": acc_ci["lo"],
        "accuracy_hi": acc_ci["hi"],
        "ece": expected_calibration_error(correct, confidence),
        "ece_lo": ece_ci["lo"],
        "ece_hi": ece_ci["hi"],
        "ece_ts": _finite(ece_ts),
        "brier": multiclass_brier(prob_rows, gold_index),
        "nll": negative_log_likelihood(gold_probs),
        "aurc": aurc(correct, confidence),
        "off_label_mass": float(np.mean([row["off_label_mass"] for row in rows])),
        "nfe": float(np.mean([row["nfe"] for row in rows])),
        "ms_per_item": float(np.mean(walls)) if walls else None,
        "flip_rate": _finite(flips),
        "temperature": temperature,
    }


def grouped(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["dataset"], row["split"], row["model"], row["condition"], row["S"], row["T"])
        groups[key].append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["item_id"])
    return groups


def temperature_table(rows: list[dict]) -> dict[str, dict]:
    """One temperature per (model, cell), fit on dev only, pooled across datasets."""
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "dev" or "letter_logits" not in row:
            continue
        pools[_temp_key(row["model"], row["condition"], row["S"], row["T"])].append(row)
    fitted = {}
    for key, group in sorted(pools.items()):
        logit_rows = []
        gold = []
        for row in group:
            letters = list(row["letter_logits"])
            logit_rows.append([row["letter_logits"][letter] for letter in letters])
            gold.append(letters.index(row["gold"]))
        fit = fit_temperature_ragged(logit_rows, gold)
        fit["n_dev"] = len(group)
        fitted[key] = fit
    return fitted


def _with_temperature(rows: list[dict], temperatures: dict[str, dict]) -> list[dict]:
    copied = []
    for row in rows:
        clone = dict(row)
        key = _temp_key(row["model"], row["condition"], row["S"], row["T"])
        fit = temperatures.get(key) or {}
        clone["temperature"] = fit.get("temperature") or 1.0
        copied.append(clone)
    return copied


def _frontier_from_summary(summary_rows: list[dict], dataset: str, split: str, model: str) -> dict:
    points = []
    for row in summary_rows:
        if row["dataset"] == dataset and row["split"] == split and row["model"] == model:
            points.append({"nfe": row["nfe"], "accuracy": row["accuracy"], "ece": row["ece"]})
    return fixed_frontier(points)


def fit_tau_for_split(rows: list[dict], temperatures: dict[str, dict], split: str, model: str) -> dict:
    """Select one tau on `split` (dev). Returns the choice, the raw curve, and the scaled curve."""
    subset = [row for row in rows if row.get("split") == split and row.get("model") == model]
    warmed = _with_temperature(subset, temperatures)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in warmed:
        by_dataset[row["dataset"]].append(row)
    # Per-dataset summaries for the frontier use the same rows.
    summary_rows = []
    for (dataset, row_split, row_model, condition, scratch, steps), group in grouped(subset).items():
        if row_split != split or row_model != model:
            continue
        key = _temp_key(row_model, condition, scratch, steps)
        temperature = (temperatures.get(key) or {}).get("temperature")
        metrics = summarize_rows(group, temperature, n_resamples=20)
        summary_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "model": model,
                "nfe": metrics["nfe"],
                "accuracy": metrics["accuracy"],
                "ece": metrics["ece"],
            }
        )
    raw_curves = {}
    scaled_curves = {}
    raw_choice = None
    scaled_choice = None
    # Mean gap across datasets. select_tau is per dataset; average the gaps here.
    for scaled in (False, True):
        curves = {}
        best = None
        for dataset, dataset_rows in sorted(by_dataset.items()):
            ladder = group_ladder_items(dataset_rows)
            if len(ladder) < len({row["item_id"] for row in dataset_rows if row["condition"] == "C0"}):
                # Keep partial ladders only when every included item has all three levels.
                pass
            if not ladder:
                continue
            frontier = _frontier_from_summary(summary_rows, dataset, split, model)
            curve = sweep_tau(ladder, scaled=scaled)
            for point in curve:
                point["dataset"] = dataset
                point["interpolated_fixed_accuracy"] = interpolate(
                    frontier["nfe"], frontier["accuracy"], point["mean_nfe"]
                )
                point["gap"] = point["accuracy"] - point["interpolated_fixed_accuracy"]
            curves[dataset] = curve
        if not curves:
            choice = {"tau": None}
        else:
            taus = tau_grid()
            choice = None
            for index, tau in enumerate(taus):
                gaps = []
                nfes = []
                for curve in curves.values():
                    point = curve[index]
                    if point["accuracy"] is None or point["gap"] is None or math.isnan(point["gap"]):
                        continue
                    gaps.append(point["gap"])
                    nfes.append(point["mean_nfe"])
                if not gaps:
                    continue
                candidate = {"tau": tau, "mean_gap": float(np.mean(gaps)), "mean_nfe": float(np.mean(nfes))}
                if choice is None or (candidate["mean_gap"], -candidate["mean_nfe"], -candidate["tau"]) > (
                    choice["mean_gap"],
                    -choice["mean_nfe"],
                    -choice["tau"],
                ):
                    choice = candidate
        if scaled:
            scaled_curves = curves
            scaled_choice = choice
        else:
            raw_curves = curves
            raw_choice = choice
    return {
        "tau": raw_choice,
        "tau_temperature_scaled": scaled_choice,
        "dev_curves": raw_curves,
        "dev_curves_temperature_scaled": scaled_curves,
        "selection_rule": (
            "On dev, maximize the unweighted mean across datasets of "
            "(escalation accuracy - accuracy of the interpolated fixed-budget upper envelope) "
            "at the escalation rule's mean NFE. Ties break toward lower mean NFE, then lower tau. "
            "A second tau is fit the same way after per-cell temperature scaling. "
            "Section 7 did not name the selection objective; this one matches the dominance claim "
            "and is frozen before test."
        ),
    }


def write_dev_fits(root: Path, n_resamples: int = 1000) -> Path:
    del n_resamples  # temperatures use a deterministic grid, not the bootstrap
    rows = load_rows(root)
    temperatures = temperature_table(rows)
    models = sorted({row["model"] for row in rows if row.get("split") == "dev" and row["model"].startswith(PRIMARY_MODEL_PREFIX)})
    tau_block = {}
    if models:
        tau_block = fit_tau_for_split(rows, temperatures, "dev", models[0])
    payload = {"temperatures": temperatures, "model": models[0] if models else None, **tau_block}
    path = root / "results" / "dev_fits.json"
    path.write_text(encoding="utf-8", data=json.dumps(payload, indent=2) + "\n")
    return path


def _index_maps(rows: list[dict]) -> dict[tuple, dict[str, dict]]:
    maps = {}
    for key, group in grouped(rows).items():
        maps[key] = {row["item_id"]: row for row in group}
    return maps


def _paired(left: dict[str, dict], right: dict[str, dict]) -> tuple[list[str], list[dict], list[dict]]:
    ids = sorted(set(left) & set(right))
    return ids, [left[item_id] for item_id in ids], [right[item_id] for item_id in ids]


def _diff_ci(left_rows: list[dict], right_rows: list[dict], kind: str, n_resamples: int) -> dict:
    """Difference left minus right. kind is 'accuracy' or 'ece'."""
    n = len(left_rows)
    if n == 0:
        return {"estimate": None, "lo": None, "hi": None, "n": 0}

    def stat(index):
        a = [left_rows[i] for i in index]
        b = [right_rows[i] for i in index]
        if kind == "accuracy":
            return accuracy([row["correct"] for row in a]) - accuracy([row["correct"] for row in b])
        return expected_calibration_error(
            [row["correct"] for row in a], [max(row["final_probs"].values()) for row in a]
        ) - expected_calibration_error(
            [row["correct"] for row in b], [max(row["final_probs"].values()) for row in b]
        )

    ci = paired_bootstrap_diff(stat, n, n_resamples=n_resamples)
    estimate = stat(np.arange(n))
    mcnemar = None
    if kind == "accuracy":
        mcnemar = mcnemar_test([row["correct"] for row in left_rows], [row["correct"] for row in right_rows])
    return {
        "estimate": estimate,
        "lo": ci["lo"],
        "hi": ci["hi"],
        "n": n,
        "mcnemar": mcnemar,
        "n_resamples": n_resamples,
        "seed": 1234,
    }


def _lookup(maps, dataset, split, model, condition, scratch, steps):
    return maps.get((dataset, split, model, condition, scratch, steps), {})


def evaluate_hypotheses(rows: list[dict], dev_fits: dict | None, n_resamples: int) -> dict:
    test_rows = [row for row in rows if row.get("split") == "test"]
    if not test_rows:
        return {"status": "not evaluated", "reason": "no test rows in results/raw", "contrasts": []}
    maps = _index_maps(test_rows)
    models = sorted({row["model"] for row in test_rows if str(row["model"]).startswith(PRIMARY_MODEL_PREFIX)})
    if not models:
        return {"status": "not evaluated", "reason": "no primary-model test rows", "contrasts": []}
    model = models[0]
    datasets = sorted({row["dataset"] for row in test_rows if row["model"] == model})
    contrasts = []

    def add(name, dataset, description, left, right, kind, direction):
        ids, left_rows, right_rows = _paired(left, right)
        diff = _diff_ci(left_rows, right_rows, kind, n_resamples)
        status = classify_contrast(diff["lo"], diff["hi"], direction)
        contrasts.append(
            {
                "hypothesis": name,
                "dataset": dataset,
                "description": description,
                "kind": kind,
                "direction": direction,
                "status": status,
                **diff,
            }
        )

    for dataset in datasets:
        for scratch in (32, 128):
            add(
                "H1",
                dataset,
                f"C1 S={scratch} ECE(T=16) - ECE(T=1)",
                _lookup(maps, dataset, "test", model, "C1", scratch, 16),
                _lookup(maps, dataset, "test", model, "C1", scratch, 1),
                "ece",
                "positive",
            )
            add(
                "H1-accuracy",
                dataset,
                f"C1 S={scratch} accuracy(T=16) - accuracy(T=1)",
                _lookup(maps, dataset, "test", model, "C1", scratch, 16),
                _lookup(maps, dataset, "test", model, "C1", scratch, 1),
                "accuracy",
                "positive",
            )
        add(
            "H2",
            dataset,
            "C1(S=128,T=16) accuracy - C0 accuracy",
            _lookup(maps, dataset, "test", model, "C1", 128, 16),
            _lookup(maps, dataset, "test", model, "C0", 0, None),
            "accuracy",
            "positive" if dataset in {"strategyqa", "jagged"} else "no_positive",
        )
        add(
            "H4",
            dataset,
            "C1 - C2 accuracy at S=128, T=16",
            _lookup(maps, dataset, "test", model, "C1", 128, 16),
            _lookup(maps, dataset, "test", model, "C2", 128, 16),
            "accuracy",
            "either",
        )
        add(
            "H4",
            dataset,
            "C1 - C2 ECE at S=128, T=16",
            _lookup(maps, dataset, "test", model, "C1", 128, 16),
            _lookup(maps, dataset, "test", model, "C2", 128, 16),
            "ece",
            "either",
        )

    h3 = _h3_contrasts(test_rows, model, dev_fits, n_resamples)
    contrasts.extend(h3)
    return {
        "model": model,
        "n_resamples": n_resamples,
        "seed": 1234,
        "contrasts": contrasts,
        "H1": _overall_h1(contrasts),
        "H2": _overall_h2(contrasts, datasets),
        "H3": _overall_h3(contrasts),
        "H4": _overall_h4(contrasts),
        "notes": {
            "H1": "Supported only when every C1 ECE(T=16)-ECE(T=1) CI at S=32 and S=128 lies entirely above 0. The accuracy companion contrasts are reported and are not the support rule.",
            "H2": "Primary cell is C1(S=128, T=16) versus C0. StrategyQA and jagged are supported only if the accuracy CI lies entirely above 0. BoolQ and ARC-Challenge match 'no significant gain' when the CI is not entirely above 0. Overall support needs all four dataset calls to be supported.",
            "H3": "One tau frozen on dev. A dataset is supported when the bootstrap CI of (escalation accuracy - interpolated fixed upper envelope) lies entirely above 0. Overall support needs at least 3 such datasets.",
            "H4": "Primary family is S=128, T=16, accuracy and ECE, two-sided, no multiplicity correction (none was pre-registered). Supported if any of those CIs excludes 0. All-inclusive CIs are inconclusive, not evidence of no difference.",
        },
    }


def _status_of(contrasts, hypothesis, dataset=None):
    found = [
        row["status"]
        for row in contrasts
        if row["hypothesis"] == hypothesis and (dataset is None or row["dataset"] == dataset) and row["n"] > 0
    ]
    return found


def _overall_h1(contrasts) -> str:
    statuses = _status_of(contrasts, "H1")
    if not statuses:
        return "not evaluated"
    if all(status == "supported" for status in statuses):
        return "supported"
    if all(status == "not supported" for status in statuses):
        return "not supported"
    return "inconclusive"


def _overall_h2(contrasts, datasets) -> str:
    needed = ["strategyqa", "jagged", "boolq", "arc_c"]
    statuses = []
    for dataset in needed:
        found = _status_of(contrasts, "H2", dataset)
        if dataset not in datasets or not found:
            return "inconclusive" if statuses else "not evaluated"
        statuses.append(found[0])
    if all(status == "supported" for status in statuses):
        return "supported"
    if any(status == "not supported" for status in statuses):
        return "not supported"
    return "inconclusive"


def _overall_h3(contrasts) -> str:
    statuses = _status_of(contrasts, "H3")
    if not statuses:
        return "not evaluated"
    supported = sum(status == "supported" for status in statuses)
    if supported >= 3:
        return "supported"
    if len(statuses) < 3:
        return "inconclusive"
    return "not supported"


def _overall_h4(contrasts) -> str:
    statuses = _status_of(contrasts, "H4")
    if not statuses:
        return "not evaluated"
    if any(status == "supported" for status in statuses):
        return "supported"
    return "inconclusive"


def _h3_contrasts(test_rows, model, dev_fits, n_resamples) -> list[dict]:
    if not dev_fits or dev_fits.get("tau") is None or dev_fits["tau"].get("tau") is None:
        return [
            {
                "hypothesis": "H3",
                "dataset": None,
                "description": "tau was not fit on dev",
                "kind": "accuracy",
                "direction": "positive",
                "status": "not evaluated",
                "estimate": None,
                "lo": None,
                "hi": None,
                "n": 0,
                "mcnemar": None,
            }
        ]
    tau = float(dev_fits["tau"]["tau"])
    temperatures = dev_fits.get("temperatures") or {}
    warmed = _with_temperature([row for row in test_rows if row["model"] == model], temperatures)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in warmed:
        by_dataset[row["dataset"]].append(row)
    contrasts = []
    for dataset, dataset_rows in sorted(by_dataset.items()):
        maps = _index_maps(dataset_rows)
        cell_maps = {key: value for key, value in maps.items() if key[0] == dataset}
        if not cell_maps:
            continue
        id_sets = [set(mapping) for mapping in cell_maps.values()]
        ids = sorted(set.intersection(*id_sets)) if id_sets else []
        ladder_ids = set(group_ladder_items(dataset_rows)[i]["item_id"] for i in range(len(group_ladder_items(dataset_rows)))) if dataset_rows else set()
        # Rebuild ladder once.
        ladder = {item["item_id"]: item for item in group_ladder_items(dataset_rows)}
        ids = [item_id for item_id in ids if item_id in ladder]
        if not ids:
            contrasts.append(
                {
                    "hypothesis": "H3",
                    "dataset": dataset,
                    "description": "escalation accuracy minus interpolated fixed envelope",
                    "kind": "accuracy",
                    "direction": "positive",
                    "status": "not evaluated",
                    "estimate": None,
                    "lo": None,
                    "hi": None,
                    "n": 0,
                    "mcnemar": None,
                    "tau": tau,
                }
            )
            continue

        def stat(index, ids=ids, cell_maps=cell_maps, ladder=ladder):
            chosen = [ids[i] for i in index]
            points = []
            for key, mapping in cell_maps.items():
                correct = [mapping[item_id]["correct"] for item_id in chosen]
                nfe = float(np.mean([mapping[item_id]["nfe"] for item_id in chosen]))
                points.append({"nfe": nfe, "accuracy": accuracy(correct), "ece": 0.0})
            frontier = fixed_frontier(points)
            decisions = decide_items([ladder[item_id] for item_id in chosen], tau, scaled=False)
            summary = summarize_decisions(decisions)
            baseline = interpolate(frontier["nfe"], frontier["accuracy"], summary["mean_nfe"])
            return summary["accuracy"] - baseline

        ci = paired_bootstrap_diff(stat, len(ids), n_resamples=n_resamples)
        estimate = stat(np.arange(len(ids)))
        contrasts.append(
            {
                "hypothesis": "H3",
                "dataset": dataset,
                "description": "escalation accuracy minus interpolated fixed envelope at the dev tau",
                "kind": "accuracy",
                "direction": "positive",
                "status": classify_contrast(ci["lo"], ci["hi"], "positive"),
                "estimate": estimate,
                "lo": ci["lo"],
                "hi": ci["hi"],
                "n": len(ids),
                "mcnemar": None,
                "tau": tau,
                "n_resamples": n_resamples,
                "seed": 1234,
            }
        )
    return contrasts


def escalation_report(rows: list[dict], dev_fits: dict | None, split: str) -> dict:
    if not dev_fits or not dev_fits.get("model"):
        return {"split": split, "available": False, "reason": "dev_fits.json has no frozen tau"}
    model = dev_fits["model"]
    temperatures = dev_fits.get("temperatures") or {}
    subset = _with_temperature(
        [row for row in rows if row.get("split") == split and row.get("model") == model],
        temperatures,
    )
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in subset:
        by_dataset[row["dataset"]].append(row)
    datasets = {}
    for dataset, dataset_rows in sorted(by_dataset.items()):
        ladder = group_ladder_items(dataset_rows)
        summary_rows = []
        for key, group in grouped(dataset_rows).items():
            metrics_nfe = float(np.mean([row["nfe"] for row in group]))
            summary_rows.append(
                {
                    "nfe": metrics_nfe,
                    "accuracy": accuracy([row["correct"] for row in group]),
                    "ece": expected_calibration_error(
                        [row["correct"] for row in group],
                        [max(row["final_probs"].values()) for row in group],
                    ),
                    "cell": _cell_label(group[0]),
                }
            )
        frontier = fixed_frontier(summary_rows)
        raw_curve = sweep_tau(ladder, scaled=False) if ladder else []
        scaled_curve = sweep_tau(ladder, scaled=True) if ladder else []
        oracle = summarize_decisions(oracle_decisions(ladder)) if ladder else {"accuracy": None, "ece": None, "mean_nfe": None, "n": 0}
        tau = (dev_fits.get("tau") or {}).get("tau")
        tau_ts = (dev_fits.get("tau_temperature_scaled") or {}).get("tau")
        datasets[dataset] = {
            "n_ladder": len(ladder),
            "fixed_cells": summary_rows,
            "frontier_nfe": frontier["nfe"],
            "frontier_accuracy": frontier["accuracy"],
            "frontier_ece": frontier["ece"],
            "curve": raw_curve,
            "curve_temperature_scaled": scaled_curve,
            "oracle": oracle,
            "at_dev_tau": _point_on_curve(raw_curve, frontier, tau),
            "at_dev_tau_temperature_scaled": _point_on_curve(scaled_curve, frontier, tau_ts),
        }
    return {"split": split, "available": bool(datasets), "model": model, "datasets": datasets}


def _point_on_curve(curve, frontier, tau) -> dict | None:
    if tau is None or not curve:
        return None
    point = min(curve, key=lambda row: abs(row["tau"] - tau))
    acc_base = interpolate(frontier["nfe"], frontier["accuracy"], point["mean_nfe"])
    ece_base = interpolate(frontier["ece_nfe"], frontier["ece"], point["mean_nfe"])
    return {
        "tau": point["tau"],
        "accuracy": point["accuracy"],
        "ece": point["ece"],
        "mean_nfe": point["mean_nfe"],
        "fixed_accuracy": acc_base,
        "fixed_ece": ece_base,
        "accuracy_gap": point["accuracy"] - acc_base,
        "ece_gap": point["ece"] - ece_base,
    }


def write_summary(root: Path, rows: list[dict], temperatures: dict[str, dict], n_resamples: int) -> list[dict]:
    summary = []
    for key, group in sorted(grouped(rows).items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], item[0][4], item[0][5] or -1)):
        dataset, split, model, condition, scratch, steps = key
        temperature = (temperatures.get(_temp_key(model, condition, scratch, steps)) or {}).get("temperature")
        metrics = summarize_rows(group, temperature, n_resamples=n_resamples)
        summary.append(
            {
                "dataset": dataset,
                "split": split,
                "model": model,
                "condition": condition,
                "S": scratch,
                "T": "" if steps is None else steps,
                "cell": _cell_label(group[0]),
                **metrics,
            }
        )
    path = root / "results" / "summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "split",
        "model",
        "condition",
        "S",
        "T",
        "cell",
        "n",
        "accuracy",
        "accuracy_lo",
        "accuracy_hi",
        "ece",
        "ece_lo",
        "ece_hi",
        "ece_ts",
        "brier",
        "nll",
        "aurc",
        "off_label_mass",
        "nfe",
        "ms_per_item",
        "flip_rate",
        "temperature",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow({field: row.get(field) for field in fields})
    return summary


def _fmt(value, digits=4) -> str:
    value = _finite(value)
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _best_mask(rows: list[dict], column: str, higher: bool) -> set[int]:
    values = [(index, row[column]) for index, row in enumerate(rows) if _finite(row.get(column)) is not None]
    if not values:
        return set()
    target = max(value for _, value in values) if higher else min(value for _, value in values)
    return {index for index, value in values if abs(value - target) <= 1e-12}


def _markdown_table(rows: list[dict]) -> str:
    columns = [
        ("cell", "cell", False, 0),
        ("acc", "accuracy", True, 4),
        ("ECE", "ece", False, 4),
        ("ECE-TS", "ece_ts", False, 4),
        ("Brier", "brier", False, 4),
        ("NLL", "nll", False, 4),
        ("AURC", "aurc", False, 4),
        ("off_label", "off_label_mass", False, 4),
        ("NFE", "nfe", False, 2),
        ("ms/item", "ms_per_item", False, 1),
    ]
    best = {name: _best_mask(rows, key, higher) for name, key, higher, _digits in columns if name != "cell"}
    header = "| " + " | ".join(name for name, *_rest in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for index, row in enumerate(rows):
        cells = []
        for name, key, _higher, digits in columns:
            if name == "cell":
                cells.append(str(row["cell"]))
                continue
            text = _fmt(row.get(key), digits)
            if index in best.get(name, set()) and text != "n/a":
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _draw_figures(root: Path, summary: list[dict], escalation: dict, rows: list[dict]) -> list[str]:
    test_summary = [row for row in summary if row["split"] == "test" and str(row["model"]).startswith(PRIMARY_MODEL_PREFIX)]
    if not test_summary:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = root / "results" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    written = []
    datasets = sorted({row["dataset"] for row in test_summary})
    written.append(_plot_metric_vs_nfe(plt, figs / "accuracy_vs_nfe.png", test_summary, escalation, datasets, "accuracy", "Accuracy"))
    written.append(_plot_metric_vs_nfe(plt, figs / "ece_vs_nfe.png", test_summary, escalation, datasets, "ece", "Raw ECE"))
    written.extend(_plot_reliability(plt, figs, rows))
    anytime = _plot_anytime(plt, figs / "anytime_c1_s128_t16.png", rows)
    if anytime:
        written.append(anytime)
    bars = _plot_c1_c2(plt, figs / "c1_vs_c2.png", test_summary)
    if bars:
        written.append(bars)
    return [path for path in written if path]


def _plot_metric_vs_nfe(plt, path: Path, summary, escalation, datasets, metric, ylabel) -> str:
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        cells = [row for row in summary if row["dataset"] == dataset]
        axis.scatter([row["nfe"] for row in cells], [row[metric] for row in cells], label="fixed cell", zorder=3)
        block = (escalation.get("datasets") or {}).get(dataset) or {}
        curve = block.get("curve") or []
        if curve and metric in {"accuracy", "ece"}:
            axis.plot([point["mean_nfe"] for point in curve], [point[metric] for point in curve], label="escalation")
        oracle = block.get("oracle") or {}
        if oracle.get("mean_nfe") is not None and oracle.get(metric) is not None:
            axis.scatter([oracle["mean_nfe"]], [oracle[metric]], marker="*", s=120, label="oracle", zorder=4)
        axis.set_title(dataset)
        axis.set_xlabel("Mean NFE")
        axis.set_ylabel(ylabel)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _plot_reliability(plt, figs: Path, rows: list[dict]) -> list[str]:
    test = [row for row in rows if row.get("split") == "test" and str(row["model"]).startswith(PRIMARY_MODEL_PREFIX)]
    if not test:
        return []
    datasets = sorted({row["dataset"] for row in test})
    written = []
    for dataset in datasets:
        subset = [row for row in test if row["dataset"] == dataset]
        c0 = [row for row in subset if row["condition"] == "C0"]
        c1 = [row for row in subset if row["condition"] == "C1"]
        best = None
        if c1:
            # Best C1 by accuracy, then lower ECE. Chosen from the test rows for the
            # diagram only; the choice is recorded in the figure filename's sibling json.
            scored = []
            for (condition, scratch, steps), group in grouped(c1).items():
                del condition
                acc = accuracy([row["correct"] for row in group])
                ece = expected_calibration_error(
                    [row["correct"] for row in group], [max(row["final_probs"].values()) for row in group]
                )
                scored.append((acc, -ece, scratch, steps, group))
            scored.sort(reverse=True)
            best = scored[0]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        _one_reliability(axes[0], c0, "C0")
        if best is None:
            axes[1].set_title("C1 absent")
        else:
            _one_reliability(axes[1], best[4], f"C1 S={best[2]} T={best[3]}")
        fig.suptitle(dataset)
        fig.tight_layout()
        path = figs / f"reliability_{dataset}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    return written


def _one_reliability(axis, rows: list[dict], title: str) -> None:
    if not rows:
        axis.set_title(f"{title} (no rows)")
        return
    bins = reliability_bins(
        [row["correct"] for row in rows],
        [max(row["final_probs"].values()) for row in rows],
    )
    xs = [0.5 * (b["lo"] + b["hi"]) for b in bins if b["count"]]
    acc = [b["accuracy"] for b in bins if b["count"]]
    conf = [b["confidence"] for b in bins if b["count"]]
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axis.bar(xs, acc, width=1 / 15, alpha=0.7, label="accuracy")
    axis.plot(xs, conf, marker="o", label="confidence")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend(fontsize=8)


def _plot_anytime(plt, path: Path, rows: list[dict]) -> str | None:
    subset = [
        row
        for row in rows
        if row.get("split") == "test"
        and row.get("condition") == "C1"
        and row.get("S") == 128
        and row.get("T") == 16
        and str(row.get("model", "")).startswith(PRIMARY_MODEL_PREFIX)
    ]
    if not subset:
        return None
    datasets = sorted({row["dataset"] for row in subset})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.6), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        group = [row for row in subset if row["dataset"] == dataset]
        steps = sorted({step["step"] for row in group for step in row.get("readouts") or []})
        top1 = []
        acc = []
        for step_id in steps:
            probs = []
            correct = []
            for row in group:
                match = next((step for step in row["readouts"] if step.get("step") == step_id), None)
                if match is None:
                    continue
                probs.append(max(match["probs"].values()))
                letters = list(match["probs"])
                pred = max(letters, key=lambda letter: (match["probs"][letter], -letters.index(letter)))
                correct.append(pred == row["gold"])
            top1.append(float(np.mean(probs)) if probs else math.nan)
            acc.append(float(np.mean(correct)) if correct else math.nan)
        axis.plot(steps, top1, marker="o", label="mean top-1")
        axis.plot(steps, acc, marker="s", label="accuracy")
        axis.set_title(dataset)
        axis.set_xlabel("Recorded step")
        axis.set_ylim(0, 1)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _plot_c1_c2(plt, path: Path, summary: list[dict]) -> str | None:
    datasets = sorted({row["dataset"] for row in summary})
    pairs = []
    for dataset in datasets:
        for scratch in (32, 128):
            for steps in (1, 4, 16):
                c1 = next((row for row in summary if row["dataset"] == dataset and row["condition"] == "C1" and row["S"] == scratch and row["T"] == steps), None)
                c2 = next((row for row in summary if row["dataset"] == dataset and row["condition"] == "C2" and row["S"] == scratch and row["T"] == steps), None)
                if c1 and c2:
                    pairs.append((dataset, scratch, steps, c1["accuracy"], c2["accuracy"]))
    if not pairs:
        return None
    fig, axes = plt.subplots(1, len(datasets), figsize=(4 * len(datasets), 3.6), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        block = [pair for pair in pairs if pair[0] == dataset]
        if not block:
            axis.set_title(dataset)
            continue
        labels = [f"S{scratch}\nT{steps}" for _dataset, scratch, steps, _c1, _c2 in block]
        x = np.arange(len(block))
        axis.bar(x - 0.15, [pair[3] for pair in block], width=0.3, label="C1")
        axis.bar(x + 0.15, [pair[4] for pair in block], width=0.3, label="C2")
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_ylim(0, 1)
        axis.set_title(dataset)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def _setup_section(root: Path, rows: list[dict]) -> str:
    lines = ["## 1. Setup", ""]
    env_path = root / "results" / "environment.json"
    if env_path.exists():
        env = json.loads(env_path.read_text(encoding="utf-8"))
        lines.append(f"- Device recorded in `results/environment.json`: `{env.get('device')}`.")
        lines.append(f"- Torch `{env.get('torch_version')}`, CUDA available `{env.get('cuda_available')}`.")
        lines.append(f"- Estimated LLaDA parameter count `{env.get('llada_n_params_estimate')}` (from the published config, not from a loaded `numel`).")
    else:
        lines.append("- `results/environment.json` is absent, so device and dtype were not recorded.")
    if rows:
        devices = sorted({row.get("device") for row in rows})
        dtypes = sorted({row.get("dtype") for row in rows})
        models = sorted({(row.get("model"), row.get("template")) for row in rows})
        lines.append(f"- Devices on raw rows: `{devices}`.")
        lines.append(f"- dtypes on raw rows: `{dtypes}`.")
        lines.append(f"- Model/template pairs on raw rows: `{models}`.")
    else:
        lines.append("- No rows in `results/raw`, so no model revision or dtype was measured.")
    template_path = root / "results" / "template.json"
    if template_path.exists():
        template = json.loads(template_path.read_text(encoding="utf-8"))
        lines.append(f"- Frozen template: `{template.get('template')}` (see `results/template.json`).")
    else:
        lines.append("- No template has been frozen. Gate 1 did not complete a C0 off-label check.")
    gate2 = root / "results" / "gate2.json"
    if gate2.exists():
        payload = json.loads(gate2.read_text(encoding="utf-8"))
        lines.append(
            f"- Anytime shortcut: `{'on' if payload.get('shortcut_enabled') else 'off'}` "
            f"(agreement `{payload.get('argmax_agreement')}`, ECE absolute difference `{payload.get('ece_abs_diff')}`)."
        )
    else:
        lines.append("- Anytime shortcut: not validated. `results/gate2.json` is absent, so the shortcut stays off.")
    splits_path = root / "results" / "splits.json"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        lines.append(f"- Split seed `{splits.get('seed')}`, format version `{splits.get('format_version')}`.")
        for name, block in splits.get("datasets", {}).items():
            if block.get("dropped"):
                lines.append(f"- `{name}`: dropped.")
            else:
                lines.append(
                    f"- `{name}`: dev `{len(block.get('dev') or [])}`, test `{len(block.get('test') or [])}`, "
                    f"available before subsample `{block.get('available')}`."
                )
    else:
        lines.append("- `results/splits.json` is absent, so item counts are not reported.")
    lines.append("")
    return "\n".join(lines)


def _hypothesis_section(hypotheses: dict) -> str:
    lines = ["## 4. Hypotheses H1 to H4", ""]
    if hypotheses.get("status") == "not evaluated" or hypotheses.get("H1") is None and not hypotheses.get("contrasts"):
        reason = hypotheses.get("reason", "no test rows")
        lines.append(f"Not evaluated. {reason}. No support call is made.")
        lines.append("")
        return "\n".join(lines)
    for name in ("H1", "H2", "H3", "H4"):
        lines.append(f"**{name}: {hypotheses.get(name, 'not evaluated')}.**")
        note = (hypotheses.get("notes") or {}).get(name)
        if note:
            lines.append("")
            lines.append(note)
        lines.append("")
        for contrast in hypotheses.get("contrasts") or []:
            if contrast["hypothesis"] != name and not (name == "H1" and contrast["hypothesis"] == "H1-accuracy"):
                continue
            mcnemar = contrast.get("mcnemar") or {}
            stat = mcnemar.get("statistic")
            p_value = mcnemar.get("p_value")
            extra = ""
            if stat is not None:
                extra = f" McNemar chi-square `{_fmt(stat)}`, p `{_fmt(p_value)}`."
            lines.append(
                f"- `{contrast['dataset']}` {contrast['description']}: status `{contrast['status']}`, "
                f"estimate `{_fmt(contrast.get('estimate'))}`, "
                f"95% CI `[{_fmt(contrast.get('lo'))}, {_fmt(contrast.get('hi'))}]`, "
                f"n `{contrast.get('n')}`."
                + extra
            )
        lines.append("")
    return "\n".join(lines)


def _escalation_section(escalation: dict) -> str:
    lines = ["## 5. Escalation results", ""]
    if not escalation.get("available"):
        lines.append(escalation.get("reason") or "Escalation was not computed.")
        lines.append("")
        return "\n".join(lines)
    for dataset, block in escalation["datasets"].items():
        point = block.get("at_dev_tau")
        oracle = block.get("oracle") or {}
        lines.append(f"### {dataset}")
        lines.append("")
        if not point:
            lines.append("No dev tau was applied.")
        else:
            lines.append(
                f"- At the dev tau `{_fmt(point['tau'])}`: escalation accuracy `{_fmt(point['accuracy'])}`, "
                f"ECE `{_fmt(point['ece'])}`, mean NFE `{_fmt(point['mean_nfe'], 2)}`."
            )
            lines.append(
                f"- Interpolated fixed envelope at that mean NFE: accuracy `{_fmt(point['fixed_accuracy'])}`, "
                f"ECE `{_fmt(point['fixed_ece'])}`."
            )
            lines.append(
                f"- Gap (escalation minus fixed): accuracy `{_fmt(point['accuracy_gap'])}`, "
                f"ECE `{_fmt(point['ece_gap'])}`."
            )
        lines.append(
            f"- Oracle: accuracy `{_fmt(oracle.get('accuracy'))}`, ECE `{_fmt(oracle.get('ece'))}`, "
            f"mean NFE `{_fmt(oracle.get('mean_nfe'), 2)}`, n `{oracle.get('n')}`."
        )
        scaled = block.get("at_dev_tau_temperature_scaled")
        if scaled:
            lines.append(
                f"- Temperature-scaled margin, dev tau `{_fmt(scaled['tau'])}`: "
                f"accuracy `{_fmt(scaled['accuracy'])}`, ECE `{_fmt(scaled['ece'])}`, "
                f"mean NFE `{_fmt(scaled['mean_nfe'], 2)}`, "
                f"accuracy gap `{_fmt(scaled['accuracy_gap'])}`."
            )
        lines.append("")
    lines.append("Full curves are in `results/escalation_test.json`.")
    lines.append("")
    return "\n".join(lines)


def build_report(root: Path, n_resamples: int = 1000) -> str:
    rows = load_rows(root)
    fits_path = root / "results" / "dev_fits.json"
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    if dev_rows and not fits_path.exists():
        write_dev_fits(root)
        log = root / "results" / "LOG.md"
        with log.open("a", encoding="utf-8") as handle:
            handle.write("\n## Dev fits\n\n`dev_fits.json` was computed at report time from dev rows only.\n")
    dev_fits = json.loads(fits_path.read_text(encoding="utf-8")) if fits_path.exists() else None
    temperatures = (dev_fits or {}).get("temperatures") or {}
    summary = write_summary(root, rows, temperatures, n_resamples=n_resamples)
    hypotheses = evaluate_hypotheses(rows, dev_fits, n_resamples=n_resamples)
    escalation = escalation_report(rows, dev_fits, "test")
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "results" / "hypothesis_tests.json").write_text(encoding="utf-8", data=json.dumps(hypotheses, indent=2) + "\n")
    (root / "results" / "escalation_test.json").write_text(encoding="utf-8", data=json.dumps(escalation, indent=2) + "\n")
    figures = []
    try:
        figures = _draw_figures(root, summary, escalation, rows)
    except Exception as exc:
        figures = []
        with (root / "results" / "LOG.md").open("a", encoding="utf-8") as handle:
            handle.write(f"\n## Figures\n\nFigure generation failed: `{type(exc).__name__}: {exc}`\n")

    test_summary = [row for row in summary if row["split"] == "test" and not str(row["model"]).lower().startswith(BASELINE_TOKEN) and BASELINE_TOKEN not in str(row["model"]).lower()]
    qwen_summary = [row for row in summary if BASELINE_TOKEN in str(row["model"]).lower() and row["split"] == "test"]
    lines = [
        "# Deliberation dial report",
        "",
        "Generated by `python -m dd.report`. Metric values below are copied from `results/summary.csv`, `results/hypothesis_tests.json`, and `results/escalation_test.json`.",
        "",
        _setup_section(root, rows),
        "## 2. Main table per dataset",
        "",
    ]
    if not test_summary:
        lines.append("No test rows for the primary model. The test table is empty. Dev rows, if any, stay in `results/summary.csv` and are not presented here as test results.")
        lines.append("")
    else:
        for dataset in sorted({row["dataset"] for row in test_summary}):
            for model in sorted({row["model"] for row in test_summary if row["dataset"] == dataset}):
                block = [row for row in test_summary if row["dataset"] == dataset and row["model"] == model]
                lines.append(f"### {dataset} / {model}")
                lines.append("")
                lines.append(_markdown_table(block))
                lines.append("")
                lines.append("Bold marks the best value in the column (higher accuracy, lower error and cost).")
                lines.append("")
    lines.append("## 3. Figures")
    lines.append("")
    if not figures:
        lines.append("No figures were written. The test sweep has no primary-model rows, or figure generation failed (see `results/LOG.md`).")
    else:
        for figure in figures:
            name = Path(figure).name
            lines.append(f"- `results/figs/{name}`")
    lines.append("")
    lines.append(_hypothesis_section(hypotheses))
    lines.append(_escalation_section(escalation))
    lines.append("## 6. Baselines")
    lines.append("")
    if not qwen_summary:
        lines.append("Qwen3-8B C0 was not run, so there is no baseline metric.")
    else:
        lines.append(_markdown_table(qwen_summary))
    lines.append("")
    lines.append("Thinking-enabled Qwen was not run. Phase 6 was not started.")
    lines.append("")
    lines.append("## 7. Deviations and failures")
    lines.append("")
    log_path = root / "results" / "LOG.md"
    if log_path.exists():
        lines.append(log_path.read_text(encoding="utf-8").rstrip())
    else:
        lines.append("`results/LOG.md` is absent.")
    lines.append("")
    lines.append("## 8. Limitations")
    lines.append("")
    dream = any("dream" in str(row.get("model", "")).lower() for row in rows)
    logicdiff = (root / "schedulers" / "logicdiff.py").exists()
    lines.append("- Single model family." if not dream else "- Dream rows are present; see the raw files for which cells ran.")
    lines.append("- Greedy decoding only (temperature 0 during denoising, no Gumbel noise).")
    lines.append("- English only.")
    if (root / "results" / "splits.json").exists():
        lines.append("- Sample sizes are the dev and test counts in `results/splits.json`.")
    else:
        lines.append("- Sample sizes were not frozen in `results/splits.json`.")
    if logicdiff:
        lines.append("- `schedulers/logicdiff.py` is present. Whether C3 ran depends on the raw rows.")
    else:
        lines.append("- LogicDiff scheduling is pending the author's `schedulers/logicdiff.py`. It did not run.")
    if not any(row.get("split") == "test" for row in rows):
        lines.append("- The test sweep did not run. Hypothesis calls above are not evaluated. `results/CLOUD.md` has the remaining commands when that file exists.")
    lines.append("")
    text = "\n".join(lines)
    (root / "results" / "REPORT.md").write_text(encoding="utf-8", data=text)
    return text


def write_cloud(root: Path, reason: str, environment: dict, projection_hours: float | None, budget_hours: float | None) -> Path:
    results = root / "results"
    existing = sorted(path.name for path in results.glob("*") if path.name != "CLOUD.md") if results.exists() else []
    if projection_hours is None:
        projection_line = "Projected full-sweep hours: not measured. No forward-pass timing exists in `results/timing.json`."
    else:
        projection_line = f"Projected full-sweep hours from the measured pass time: {projection_hours:.3f}."
    budget_line = "Device budget hours: not applicable (CPU has no Phase 3-5 budget)." if budget_hours is None else f"Device budget hours: {budget_hours}."
    text = f"""# Cloud continuation

This machine stopped before the remaining sweep.

{reason}

{projection_line}

{budget_line}

Environment snapshot:

```json
{json.dumps(environment, indent=2)}
```

Files already under `results/`:

{chr(10).join('- `' + name + '`' for name in existing) or '- (none)'}

## Commands

From the repository root, on a CUDA machine with enough memory for LLaDA-8B in bf16 (weights are about 16GB before activations; a 24GB GPU is a practical minimum):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m dd.run --config configs/smoke.yaml
python -m dd.run --config configs/smoke.yaml --validate-anytime
python -m dd.run --config configs/dev.yaml
python -m dd.run --config configs/full.yaml
python -m dd.report
```

`configs/full.yaml` refuses to start when `results/dev_fits.json` is missing, and it refuses to start when the measured projection exceeds the device budget (48 hours on CUDA, 24 hours on MPS). CPU is smoke-only. The runner appends one JSON line per finished item and skips items already in `results/raw/`.

Do not refit the template, temperatures, or tau on test. Gate 1 freezes the template. Phase 3 writes `results/dev_fits.json`.
"""
    path = results / "CLOUD.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoding="utf-8", data=text)
    return path


def main() -> int:
    """Phase 5 entry point. Protocol v1.1 + v1.2 registration: dd/report_v12.py builds REPORT.md.

    The v1.0 `build_report` above is kept for reference; its single-tau H3 was superseded before test.
    """
    from dd.report_v12 import main as main_v12

    return main_v12()


if __name__ == "__main__":
    raise SystemExit(main())
