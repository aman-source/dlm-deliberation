"""Margin escalation over the C1 ladder, the fixed-budget frontier, and the oracle.

Ladder, rerun from scratch:
  L0 = C0                         cumulative NFE 1
  L1 = C1(S=32, T=4)              cumulative NFE 6
  L2 = C1(S=128, T=16)            cumulative NFE 23

At level k < 2, escalate when top1 - top2 < tau. L2 always answers.
Tau is chosen on dev and frozen before test.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from dd.metrics import accuracy, apply_temperature_probs, expected_calibration_error

LADDER = (
    {"level": 0, "condition": "C0", "S": 0, "T": None, "nfe": 1, "cumulative_nfe": 1},
    {"level": 1, "condition": "C1", "S": 32, "T": 4, "nfe": 5, "cumulative_nfe": 6},
    {"level": 2, "condition": "C1", "S": 128, "T": 16, "nfe": 17, "cumulative_nfe": 23},
)


def margin(probs: dict[str, float]) -> float:
    ordered = sorted(probs.values(), reverse=True)
    if len(ordered) < 2:
        return float(ordered[0]) if ordered else float("nan")
    return float(ordered[0] - ordered[1])


def tau_grid(n: int = 50) -> list[float]:
    return [float(value) for value in np.linspace(0.0, 1.0, n)]


def choose_level(prob_levels: Sequence[dict[str, float]], tau: float) -> int:
    last = len(prob_levels) - 1
    for level, probs in enumerate(prob_levels):
        if level == last or margin(probs) >= tau:
            return level
    return last


def cumulative_nfe(level: int) -> int:
    return int(LADDER[level]["cumulative_nfe"])


def scale_levels(
    logit_levels: Sequence[dict[str, float]],
    letters: Sequence[str],
    temperatures: Sequence[float],
) -> list[dict[str, float]]:
    return [
        apply_temperature_probs(logits, letters, temperature)
        for logits, temperature in zip(logit_levels, temperatures)
    ]


def decide_items(
    items: Sequence[dict],
    tau: float,
    *,
    scaled: bool = False,
) -> list[dict]:
    """Each item needs `levels`: a list of three dicts with probs, letter_logits, correct, letters.

    When scaled is True, each level must also carry `temperature`.
    """
    decisions = []
    for item in items:
        levels = item["levels"]
        letters = item["letters"]
        if scaled:
            probs = scale_levels(
                [level["letter_logits"] for level in levels],
                letters,
                [level["temperature"] for level in levels],
            )
        else:
            probs = [level["probs"] for level in levels]
        level = choose_level(probs, tau)
        pred = max(probs[level], key=lambda letter: (probs[level][letter], -letters.index(letter)))
        decisions.append(
            {
                "item_id": item["item_id"],
                "level": level,
                "probs": probs[level],
                "pred": pred,
                "correct": pred == item["gold"],
                "nfe": cumulative_nfe(level),
                "gold": item["gold"],
            }
        )
    return decisions


def oracle_decisions(items: Sequence[dict]) -> list[dict]:
    """Cheapest correct level. If none are correct, stay at L0."""
    decisions = []
    for item in items:
        chosen = 0
        for level, block in enumerate(item["levels"]):
            if block["correct"]:
                chosen = level
                break
        probs = item["levels"][chosen]["probs"]
        letters = item["letters"]
        pred = max(probs, key=lambda letter: (probs[letter], -letters.index(letter)))
        decisions.append(
            {
                "item_id": item["item_id"],
                "level": chosen,
                "probs": probs,
                "pred": pred,
                "correct": pred == item["gold"],
                "nfe": cumulative_nfe(chosen),
                "gold": item["gold"],
            }
        )
    return decisions


def summarize_decisions(decisions: Sequence[dict]) -> dict:
    if not decisions:
        return {"accuracy": None, "ece": None, "mean_nfe": None, "n": 0}
    correct = [row["correct"] for row in decisions]
    confidence = [max(row["probs"].values()) for row in decisions]
    return {
        "accuracy": accuracy(correct),
        "ece": expected_calibration_error(correct, confidence),
        "mean_nfe": float(np.mean([row["nfe"] for row in decisions])),
        "n": len(decisions),
    }


def upper_envelope(points: Sequence[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """Best (highest) value at each x, sorted by x."""
    best: dict[float, float] = {}
    for x_value, y_value in points:
        key = float(x_value)
        best[key] = max(y_value, best.get(key, float("-inf")))
    xs = sorted(best)
    return xs, [best[x] for x in xs]


def interpolate(xs: Sequence[float], ys: Sequence[float], target: float) -> float:
    """Linear interpolation. Targets outside the span clamp to the nearest end.

    Clamping does not invent a frontier beyond the measured fixed budgets.
    """
    if not xs:
        return float("nan")
    if target <= xs[0]:
        return float(ys[0])
    if target >= xs[-1]:
        return float(ys[-1])
    for left in range(len(xs) - 1):
        x0, x1 = xs[left], xs[left + 1]
        if x0 <= target <= x1:
            if x1 == x0:
                return float(ys[left])
            weight = (target - x0) / (x1 - x0)
            return float(ys[left] * (1 - weight) + ys[left + 1] * weight)
    return float("nan")


def fixed_frontier(cell_rows: Sequence[dict]) -> dict:
    """cell_rows have nfe, accuracy, ece. Envelope is the best accuracy and, separately, the best (lowest) ECE."""
    acc_x, acc_y = upper_envelope([(row["nfe"], row["accuracy"]) for row in cell_rows])
    # Lowest ECE is the better calibration frontier. Store it as a lower envelope
    # by negating, then negating back.
    ece_x, neg_ece = upper_envelope([(row["nfe"], -row["ece"]) for row in cell_rows])
    ece_y = [-value for value in neg_ece]
    return {"nfe": acc_x, "accuracy": acc_y, "ece_nfe": ece_x, "ece": ece_y}


def sweep_tau(items: Sequence[dict], taus: Sequence[float] | None = None, *, scaled: bool = False) -> list[dict]:
    taus = list(taus) if taus is not None else tau_grid()
    curve = []
    for tau in taus:
        summary = summarize_decisions(decide_items(items, tau, scaled=scaled))
        summary["tau"] = float(tau)
        curve.append(summary)
    return curve


def select_tau(curve: Sequence[dict], frontier: dict) -> dict:
    """Pick the dev tau whose accuracy most exceeds the interpolated fixed frontier.

    Ties break toward the smaller mean NFE, then the smaller tau.
    This criterion was not specified in Section 7; it is the selection rule
    that matches the dominance claim, and it is fit on dev only.
    """
    best = None
    for row in curve:
        if row["accuracy"] is None or row["mean_nfe"] is None:
            continue
        baseline = interpolate(frontier["nfe"], frontier["accuracy"], row["mean_nfe"])
        gap = row["accuracy"] - baseline
        candidate = {
            "tau": row["tau"],
            "gap": gap,
            "accuracy": row["accuracy"],
            "ece": row["ece"],
            "mean_nfe": row["mean_nfe"],
            "interpolated_fixed_accuracy": baseline,
        }
        if best is None:
            best = candidate
            continue
        key = (candidate["gap"], -candidate["mean_nfe"], -candidate["tau"])
        best_key = (best["gap"], -best["mean_nfe"], -best["tau"])
        if key > best_key:
            best = candidate
    return best or {"tau": None, "gap": None}


def group_ladder_items(rows: Sequence[dict]) -> list[dict]:
    """Join C0, C1(S=32,T=4), and C1(S=128,T=16) rows that share an item id."""
    needed = {
        (0, "C0", 0, None),
        (1, "C1", 32, 4),
        (2, "C1", 128, 16),
    }
    by_item: dict[str, dict] = {}
    for row in rows:
        key = (row["condition"], row["S"], row["T"])
        level = None
        for spec_level, condition, scratch, steps in needed:
            if key == (condition, scratch, steps):
                level = spec_level
        if level is None:
            continue
        slot = by_item.setdefault(
            row["item_id"],
            {"item_id": row["item_id"], "gold": row["gold"], "letters": None, "levels": [None, None, None]},
        )
        letters = list(row["final_probs"])
        slot["letters"] = letters
        slot["levels"][level] = {
            "probs": row["final_probs"],
            "letter_logits": row["letter_logits"],
            "correct": bool(row["correct"]),
            "temperature": row.get("temperature", 1.0),
        }
    complete = []
    for slot in by_item.values():
        if all(level is not None for level in slot["levels"]):
            complete.append(slot)
    complete.sort(key=lambda slot: slot["item_id"])
    return complete
