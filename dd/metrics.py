"""Accuracy, calibration, and paired tests.

Every public statistic is a pure function of arrays. Report code must call
these rather than reimplement them.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import binom, chi2


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("expected a 1-D sequence")
    return array


def accuracy(correct: Sequence[bool]) -> float:
    array = np.asarray(list(correct), dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(array.mean())


def expected_calibration_error(
    correct: Sequence[bool],
    confidence: Sequence[float],
    n_bins: int = 15,
) -> float:
    """ECE with equal-width bins on top-label confidence. Empty bins are skipped."""
    correct_a = np.asarray(list(correct), dtype=np.float64)
    conf_a = _as_float_array(confidence)
    if correct_a.size == 0:
        return float("nan")
    if correct_a.shape != conf_a.shape:
        raise ValueError("correct and confidence must have the same length")
    if np.any(conf_a < 0) or np.any(conf_a > 1):
        raise ValueError("confidence must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Rightmost edge is inclusive so confidence == 1 is counted.
    bin_ids = np.clip(np.digitize(conf_a, edges[1:-1], right=False), 0, n_bins - 1)
    total = correct_a.size
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_acc = float(correct_a[mask].mean())
        bin_conf = float(conf_a[mask].mean())
        ece += (count / total) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_bins(
    correct: Sequence[bool],
    confidence: Sequence[float],
    n_bins: int = 15,
) -> list[dict]:
    correct_a = np.asarray(list(correct), dtype=np.float64)
    conf_a = _as_float_array(confidence)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(conf_a, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "count": count,
                "accuracy": float(correct_a[mask].mean()) if count else None,
                "confidence": float(conf_a[mask].mean()) if count else None,
            }
        )
    return rows


def multiclass_brier(prob_rows: Sequence[Sequence[float]], gold_index: Sequence[int]) -> float:
    """Mean over items of sum_k (p_k - one_hot_k)^2."""
    if len(prob_rows) == 0:
        return float("nan")
    total = 0.0
    for probs, gold in zip(prob_rows, gold_index):
        vector = np.asarray(probs, dtype=np.float64)
        one_hot = np.zeros_like(vector)
        one_hot[int(gold)] = 1.0
        total += float(np.sum((vector - one_hot) ** 2))
    return total / len(prob_rows)


def negative_log_likelihood(gold_probs: Sequence[float], clip: float = 1e-12) -> float:
    probs = _as_float_array(gold_probs)
    if probs.size == 0:
        return float("nan")
    clipped = np.maximum(probs, clip)
    return float(np.mean(-np.log(clipped)))


def aurc(correct: Sequence[bool], confidence: Sequence[float]) -> float:
    """Area under the risk-coverage curve, items ranked by descending confidence.

    Coverage points are k/n for k = 1..n. The integral is trapezoidal. Ties in
    confidence are broken by original item order so the ranking is deterministic.
    """
    correct_a = np.asarray(list(correct), dtype=np.float64)
    conf_a = _as_float_array(confidence)
    n = int(correct_a.size)
    if n == 0:
        return float("nan")
    order = np.lexsort((np.arange(n), -conf_a))
    ranked = correct_a[order]
    errors = np.cumsum(1.0 - ranked)
    ks = np.arange(1, n + 1, dtype=np.float64)
    coverages = ks / n
    risks = errors / ks
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trap(risks, coverages))


def softmax_rows(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def nll_from_logits(logits: np.ndarray, gold_index: np.ndarray, temperature: float, clip: float = 1e-12) -> float:
    probs = softmax_rows(logits, temperature)
    gold_probs = probs[np.arange(len(gold_index)), gold_index]
    return negative_log_likelihood(gold_probs, clip=clip)


def fit_temperature(
    logits: np.ndarray,
    gold_index: Sequence[int],
    *,
    low: float = 0.05,
    high: float = 20.0,
    grid_size: int = 64,
    refine: int = 32,
) -> dict:
    """Minimize NLL on a log-spaced grid, then refine around the best point.

    Logits may be ragged across items (different option counts). Pass a list via
    `fit_temperature_ragged` in that case. This function requires a dense matrix.
    """
    gold = np.asarray(list(gold_index), dtype=np.int64)
    if len(logits) == 0:
        return {"temperature": None, "nll": None}

    def search(lo: float, hi: float, count: int) -> tuple[float, float]:
        grid = np.logspace(math.log10(lo), math.log10(hi), count)
        best_t = float(grid[0])
        best_nll = float("inf")
        for temperature in grid:
            value = nll_from_logits(logits, gold, float(temperature))
            if value < best_nll:
                best_nll = value
                best_t = float(temperature)
        return best_t, best_nll

    temperature, nll = search(low, high, grid_size)
    # Refine inside the neighboring decade span, staying inside [low, high].
    log_t = math.log10(temperature)
    span = (math.log10(high) - math.log10(low)) / max(grid_size - 1, 1)
    lo = max(low, 10 ** (log_t - span))
    hi = min(high, 10 ** (log_t + span))
    temperature, nll = search(lo, hi, refine)
    return {"temperature": temperature, "nll": nll}


def fit_temperature_ragged(
    logit_rows: Sequence[Sequence[float]],
    gold_index: Sequence[int],
    *,
    low: float = 0.05,
    high: float = 20.0,
    grid_size: int = 64,
    refine: int = 32,
) -> dict:
    """Same search as fit_temperature, allowing a different label count per item."""
    rows = [np.asarray(row, dtype=np.float64) for row in logit_rows]
    gold = [int(g) for g in gold_index]
    if not rows:
        return {"temperature": None, "nll": None}

    def nll_at(temperature: float) -> float:
        gold_probs = []
        for row, g in zip(rows, gold):
            probs = softmax_rows(row.reshape(1, -1), temperature)[0]
            gold_probs.append(float(probs[g]))
        return negative_log_likelihood(gold_probs)

    def search(lo: float, hi: float, count: int) -> tuple[float, float]:
        grid = np.logspace(math.log10(lo), math.log10(hi), count)
        best_t = float(grid[0])
        best_nll = float("inf")
        for temperature in grid:
            value = nll_at(float(temperature))
            if value < best_nll:
                best_nll = value
                best_t = float(temperature)
        return best_t, best_nll

    temperature, nll = search(low, high, grid_size)
    log_t = math.log10(temperature)
    span = (math.log10(high) - math.log10(low)) / max(grid_size - 1, 1)
    lo = max(low, 10 ** (log_t - span))
    hi = min(high, 10 ** (log_t + span))
    temperature, nll = search(lo, hi, refine)
    return {"temperature": temperature, "nll": nll}


def apply_temperature_probs(letter_logits: dict[str, float], letters: Sequence[str], temperature: float) -> dict[str, float]:
    row = np.array([letter_logits[letter] for letter in letters], dtype=np.float64)
    probs = softmax_rows(row.reshape(1, -1), temperature)[0]
    return {letter: float(probs[i]) for i, letter in enumerate(letters)}


def bootstrap_ci(
    stat_fn,
    n_items: int,
    *,
    n_resamples: int = 1000,
    seed: int = 1234,
    alpha: float = 0.05,
) -> dict:
    """Resample item indices with replacement. stat_fn(indices) -> float."""
    if n_items <= 0:
        return {"mean": None, "lo": None, "hi": None, "n_resamples": n_resamples, "seed": seed}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        index = rng.integers(0, n_items, size=n_items)
        stats[i] = float(stat_fn(index))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(stats.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def paired_bootstrap_diff(
    stat_fn,
    n_items: int,
    *,
    n_resamples: int = 1000,
    seed: int = 1234,
) -> dict:
    """stat_fn(indices) returns the paired difference on those items."""
    return bootstrap_ci(stat_fn, n_items, n_resamples=n_resamples, seed=seed)


def mcnemar_test(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """McNemar test on paired correctness. Chi-square with continuity correction.

    Also reports the exact binomial p-value on discordant pairs.
    """
    a = np.asarray(list(correct_a), dtype=bool)
    b = np.asarray(list(correct_b), dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired correctness vectors differ in length")
    both_right = int(np.sum(a & b))
    both_wrong = int(np.sum(~a & ~b))
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    discordant = a_only + b_only
    if discordant == 0:
        chi2_stat = 0.0
        p_chi2 = 1.0
        p_exact = 1.0
    else:
        # Continuity-corrected McNemar. SciPy 1.18 no longer exports
        # scipy.stats.mcnemar, so the statistic is computed here.
        corrected = abs(a_only - b_only) - 1
        chi2_stat = 0.0 if corrected < 0 else (corrected ** 2) / discordant
        p_chi2 = float(chi2.sf(chi2_stat, 1))
        k = min(a_only, b_only)
        p_exact = float(binom.cdf(k, discordant, 0.5) + binom.sf(discordant - k - 1, discordant, 0.5))
        p_exact = min(1.0, p_exact)
    return {
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "statistic": chi2_stat,
        "p_value": p_chi2,
        "p_value_exact": p_exact,
        "correction": "continuity",
    }


def ci_excludes_zero(lo: float | None, hi: float | None) -> bool:
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return False
    return hi < 0 or lo > 0


def classify_contrast(lo: float | None, hi: float | None, direction: str) -> str:
    """Map a 95% CI to supported / not supported / inconclusive.

    direction:
      - "positive": predicted difference > 0
      - "negative": predicted difference < 0
      - "either": any nonzero difference
      - "no_positive": predicted absence of a significant gain (H2 null datasets)
    """
    if lo is None or hi is None or (isinstance(lo, float) and math.isnan(lo)) or (isinstance(hi, float) and math.isnan(hi)):
        return "not evaluated"
    if direction == "positive":
        if lo > 0:
            return "supported"
        if hi < 0:
            return "not supported"
        return "inconclusive"
    if direction == "negative":
        if hi < 0:
            return "supported"
        if lo > 0:
            return "not supported"
        return "inconclusive"
    if direction == "either":
        if lo > 0 or hi < 0:
            return "supported"
        return "inconclusive"
    if direction == "no_positive":
        # A significant accuracy gain (CI entirely above 0) rejects the null claim.
        # A CI that includes 0, or lies entirely below 0, is consistent with
        # "no significant gain". This is not the same as a CI excluding zero.
        if lo > 0:
            return "not supported"
        return "supported"
    raise ValueError(f"unknown direction {direction}")

