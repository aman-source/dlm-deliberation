"""Protocol v1.2 analyses (registered before any test run, 29 Sep 2026).

- Adaptive ECE: 15 equal-mass bins on top-label confidence, reported next to equal-width ECE.
- Per-(dataset, cell) temperature scaling fit on dev, next to the pooled per-cell temperature.
- Escalation for two ladders: the pre-registered one and a dev-selected exploratory one.
- H3 primary evaluation: the full tau-sweep curve against the fixed-budget frontier.

Fixed-budget frontier (accuracy): for the 13 fixed LLaDA cells, take the best accuracy at each
mean NFE, then the running maximum over increasing NFE (a budget of n passes can always use a
cheaper cell), and interpolate linearly between those points; outside the measured NFE range the
value is clamped to the nearest end. The ECE frontier is the same with the lowest ECE and a
running minimum.

H3 statistic per dataset: G = mean over the 50 tau grid points of
    accuracy_escalation(tau) - frontier(mean_NFE_escalation(tau)),
with a 95% paired-bootstrap CI over items (1000 resamples, seed 1234) that recomputes the cell
accuracies, the frontier and the curve on every resample. A dataset "dominates" when the CI lies
entirely above 0. H3 is supported when at least 3 of 4 datasets dominate.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from dd.metrics import apply_temperature_probs, expected_calibration_error, fit_temperature_ragged

CELLS13 = [("C0", 0, None)] + [(c, s, t) for c in ("C1", "C2") for s in (32, 128) for t in (1, 4, 16)]
LADDERS = {
    "preregistered": [("C0", 0, None), ("C1", 32, 4), ("C1", 128, 16)],
    "exploratory": [("C0", 0, None), ("C1", 32, 4), ("C1", 32, 16)],
}
LADDER_LABELS = {
    "preregistered": "pre-registered: C0 -> C1(32,4) -> C1(128,16)",
    "exploratory": "exploratory (dev-selected, labeled as such): C0 -> C1(32,4) -> C1(32,16)",
}
TAUS = np.linspace(0.0, 1.0, 50)
N_BOOT = 1000
SEED = 1234


def nfe_of(cell) -> int:
    condition, _scratch, steps = cell
    return 1 if condition == "C0" else int(steps) + 1


def cumulative_nfe(ladder) -> np.ndarray:
    return np.cumsum([nfe_of(cell) for cell in ladder]).astype(float)


def adaptive_ece(correct: Sequence[bool], confidence: Sequence[float], n_bins: int = 15) -> float:
    """Equal-mass bins: items sorted by confidence (ties by input order), split into n_bins groups."""
    correct_a = np.asarray(list(correct), dtype=np.float64)
    conf_a = np.asarray(list(confidence), dtype=np.float64)
    n = correct_a.size
    if n == 0:
        return float("nan")
    order = np.lexsort((np.arange(n), conf_a))
    total = 0.0
    for chunk in np.array_split(order, min(n_bins, n)):
        if chunk.size:
            total += chunk.size / n * abs(correct_a[chunk].mean() - conf_a[chunk].mean())
    return float(total)


def cell_key(row) -> tuple:
    return (row["condition"], int(row["S"]), row["T"])


def index_rows(rows, model: str, split: str) -> dict:
    """{dataset: {cell: {item_id: row}}} for one model and split."""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row.get("model") == model and row.get("split") == split:
            out[row["dataset"]][cell_key(row)][row["item_id"]] = row
    return out


def letters_of(row) -> list[str]:
    return list(row["final_probs"])


def scaled_probs(row, temperature: float | None) -> dict:
    if not temperature:
        return row["final_probs"]
    letters = list(row["letter_logits"])
    return apply_temperature_probs(row["letter_logits"], letters, temperature)


def fit_temperatures_by_dataset(rows, model: str) -> dict:
    """Per-(dataset, cell) temperature on dev rows (secondary analysis, v1.2)."""
    pools: dict = defaultdict(list)
    for row in rows:
        if row.get("split") == "dev" and row.get("model") == model and "letter_logits" in row:
            pools[(row["dataset"], cell_key(row))].append(row)
    fits = {}
    for (dataset, cell), group in sorted(pools.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        logit_rows, gold = [], []
        for row in group:
            letters = list(row["letter_logits"])
            logit_rows.append([row["letter_logits"][letter] for letter in letters])
            gold.append(letters.index(row["gold"]))
        fit = fit_temperature_ragged(logit_rows, gold)
        fits[temp_key(model, dataset, cell)] = {"temperature": fit["temperature"], "nll": fit["nll"], "n_dev": len(group)}
    return fits


def temp_key(model: str, dataset: str | None, cell) -> str:
    condition, scratch, steps = cell
    step = "na" if steps is None else str(steps)
    return f"{model}|{condition}|{scratch}|{step}" if dataset is None else f"{model}|{dataset}|{condition}|{scratch}|{step}"


def cell_metrics(group: dict, pooled_t: float | None, dataset_t: float | None) -> dict:
    rows = list(group.values())
    correct = [bool(row["correct"]) for row in rows]
    confidence = [max(row["final_probs"].values()) for row in rows]

    def ts_ece(temperature):
        if not temperature:
            return None
        conf = []
        for row in rows:
            probs = scaled_probs(row, temperature)
            conf.append(max(probs.values()))
        return expected_calibration_error(correct, conf)

    eos = [row["eos_pad_fraction"] for row in rows if row.get("eos_pad_fraction") is not None]
    return {
        "n": len(rows),
        "accuracy": float(np.mean(correct)),
        "ece": expected_calibration_error(correct, confidence),
        "adaptive_ece": adaptive_ece(correct, confidence),
        "ece_ts_pooled": ts_ece(pooled_t),
        "ece_ts_dataset": ts_ece(dataset_t),
        "off_label_mass": float(np.mean([row["off_label_mass"] for row in rows])),
        "eos_pad_median": float(np.median(eos)) if eos else None,
        "nfe": float(np.mean([row["nfe"] for row in rows])),
    }


# ---------------------------------------------------------------- escalation


def pareto(nfe: np.ndarray, value: np.ndarray, higher_is_better: bool = True) -> tuple[np.ndarray, np.ndarray]:
    xs = np.unique(nfe)
    best = np.array([value[nfe == x].max() if higher_is_better else value[nfe == x].min() for x in xs])
    running = np.maximum.accumulate(best) if higher_is_better else np.minimum.accumulate(best)
    return xs, running


def _margin(probs: np.ndarray) -> np.ndarray:
    top2 = np.sort(probs, axis=-1)[..., -2:]
    return top2[..., 1] - top2[..., 0]


class LadderData:
    """Per-item arrays for one dataset: ladder margins/correctness/confidence and the 13 fixed cells."""

    def __init__(self, cellmaps: dict, ladder, temps: dict | None, model: str, dataset: str | None = None):
        ids = set.intersection(*[set(cellmaps.get(cell, {})) for cell in CELLS13 + list(ladder)])
        self.ids = sorted(ids)
        n = len(self.ids)
        self.cum_nfe = cumulative_nfe(ladder)
        self.fixed_nfe = np.array([nfe_of(cell) for cell in CELLS13], dtype=float)
        self.fixed_correct = np.zeros((n, len(CELLS13)))
        self.fixed_conf = np.zeros((n, len(CELLS13)))
        for j, cell in enumerate(CELLS13):
            for i, item_id in enumerate(self.ids):
                row = cellmaps[cell][item_id]
                self.fixed_correct[i, j] = row["correct"]
                self.fixed_conf[i, j] = max(row["final_probs"].values())
        self.margin = np.zeros((n, 3))
        self.correct = np.zeros((n, 3))
        self.conf = np.zeros((n, 3))
        for k, cell in enumerate(ladder):
            temperature = None
            if temps is not None:
                key = temp_key(model, dataset, cell)
                temperature = (temps.get(key) or {}).get("temperature")
            for i, item_id in enumerate(self.ids):
                row = cellmaps[cell][item_id]
                probs = scaled_probs(row, temperature)
                letters = list(probs)
                vector = np.array([probs[letter] for letter in letters])
                pred = letters[int(np.argmax(vector))]
                self.margin[i, k] = _margin(vector)
                self.correct[i, k] = pred == row["gold"]
                self.conf[i, k] = vector.max()
        # level chosen at each tau: escalate from k while margin_k < tau; L2 always answers.
        esc0 = self.margin[:, 0][:, None] < TAUS[None, :]
        esc1 = self.margin[:, 1][:, None] < TAUS[None, :]
        self.level = np.where(~esc0, 0, np.where(~esc1, 1, 2))  # (n, 50)
        rows_idx = np.arange(n)[:, None]
        self.curve_correct = self.correct[rows_idx, self.level]
        self.curve_conf = self.conf[rows_idx, self.level]
        self.curve_nfe = self.cum_nfe[self.level]

    def summary(self, index: np.ndarray | None = None, with_ece: bool = True) -> dict:
        idx = np.arange(len(self.ids)) if index is None else index
        cell_acc = self.fixed_correct[idx].mean(0)
        xs, ys = pareto(self.fixed_nfe, cell_acc, True)
        cell_ece = exs = eys = None
        if with_ece:
            cell_ece = np.array([expected_calibration_error(self.fixed_correct[idx, j], self.fixed_conf[idx, j]) for j in range(len(CELLS13))])
            exs, eys = pareto(self.fixed_nfe, cell_ece, False)
        acc = self.curve_correct[idx].mean(0)
        nfe = self.curve_nfe[idx].mean(0)
        frontier = np.interp(nfe, xs, ys)
        return {"acc": acc, "nfe": nfe, "frontier": frontier, "gap": acc - frontier, "front_x": xs, "front_y": ys,
                "ece_front_x": exs, "ece_front_y": eys, "cell_acc": cell_acc, "cell_ece": cell_ece}

    def ece_curve(self) -> np.ndarray:
        return np.array([expected_calibration_error(self.curve_correct[:, t], self.curve_conf[:, t]) for t in range(len(TAUS))])

    def h3(self, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
        n = len(self.ids)
        point = self.summary()
        estimate = float(point["gap"].mean())
        rng = np.random.default_rng(seed)
        stats = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            stats[b] = self.summary(idx, with_ece=False)["gap"].mean()
        lo, hi = np.quantile(stats, [0.025, 0.975])
        return {"G": estimate, "lo": float(lo), "hi": float(hi), "min_gap": float(point["gap"].min()),
                "max_gap": float(point["gap"].max()), "frac_nonneg": float((point["gap"] >= -1e-12).mean()),
                "dominates": bool(lo > 0), "n": n, "n_boot": n_boot, "seed": seed}

    def at_tau(self, tau: float) -> dict:
        esc0 = self.margin[:, 0] < tau
        esc1 = self.margin[:, 1] < tau
        level = np.where(~esc0, 0, np.where(~esc1, 1, 2))
        rows = np.arange(len(self.ids))
        correct = self.correct[rows, level]
        conf = self.conf[rows, level]
        nfe = self.cum_nfe[level].mean()
        s = self.summary()
        return {"tau": tau, "accuracy": float(correct.mean()), "ece": expected_calibration_error(correct, conf),
                "mean_nfe": float(nfe), "frontier": float(np.interp(nfe, s["front_x"], s["front_y"])),
                "share_by_level": [float((level == k).mean()) for k in range(3)]}

    def oracle(self) -> dict:
        any_correct = self.correct.any(1)
        level = np.where(any_correct, np.argmax(self.correct, axis=1), 0)
        rows = np.arange(len(self.ids))
        correct = self.correct[rows, level]
        return {"accuracy": float(correct.mean()), "mean_nfe": float(self.cum_nfe[level].mean()),
                "share_by_level": [float((level == k).mean()) for k in range(3)]}


def h3_verdict(per_dataset: dict) -> str:
    if len(per_dataset) < 4:
        return "not evaluated"
    return "supported" if sum(r["dominates"] for r in per_dataset.values()) >= 3 else "not supported"
