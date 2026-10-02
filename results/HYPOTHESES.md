# Pre-registered hypotheses

- **H1 (calibration drift):** at fixed scratch length S > 0, raw ECE of the answer slot increases with steps T, even where accuracy increases.
- **H2 (where deliberation helps):** scratch (S > 0) improves accuracy on StrategyQA and the jaggedness set, and gives no significant gain on BoolQ and ARC-Challenge.
- **H3 (escalation wins):** the margin-based escalation rule (Section 7) dominates the fixed-budget frontier on accuracy vs mean forward passes on at least 3 of 4 datasets.
- **H4 (in-canvas vs pre-read):** in-canvas scratch and pre-read thought differ in accuracy or ECE at matched (S, T). Direction not predicted.

Statistics: paired bootstrap (1000 resamples over items) for accuracy and ECE differences, 95% CIs; McNemar test for paired accuracy. A hypothesis is "supported" only if the CI excludes zero in the predicted direction.

---

## v1.2 additions, registered before test (29 Sep 2026)

Registered and committed before any test item was run. The primary hypotheses H1–H4 above are unchanged. The operational rules that already applied (logged in `results/LOG.md` before any data) are restated here for completeness. The new additions follow them.

**Statistics everywhere:** paired bootstrap over items, 1000 resamples, seed 1234, 95% percentile CIs. McNemar (continuity-corrected, plus the exact binomial p-value) for paired accuracy.

**Unchanged operational rules:**
- **H1:** for each dataset and S ∈ {32, 128}, the contrast is ECE(C1, T=16) − ECE(C1, T=1). H1 is supported only if all 8 CIs lie entirely above 0. The accuracy companion contrasts are reported and are not part of the support rule.
- **H2:** the contrast is accuracy(C1, S=128, T=16) − accuracy(C0).
  - StrategyQA and jagged must have a CI entirely above 0.
  - BoolQ and ARC-C match "no significant gain" when the CI is not entirely above 0.
  - H2 is supported only when all four dataset calls hold.
- **H4:** C1 − C2 at S=128, T=16, for accuracy and ECE, two-sided, with no multiplicity correction. H4 is supported if any of these CIs excludes 0.

**v1.2 additions:**

1. **H3 primary evaluation (replaces the single-τ contrast).**
   - The H3 test is the full τ-sweep curve on test, exactly as PLAN §7 specifies: 50 τ values on [0, 1], pre-registered ladder C0 → C1(32,4) → C1(128,16), cumulative NFE 1 / 6 / 23, raw margins.
   - Fixed-budget frontier: the best accuracy at each mean NFE among the 13 fixed LLaDA cells, then the running maximum over increasing NFE (a budget of n passes can use any cheaper cell), interpolated linearly and clamped outside the measured range.
   - Per-dataset statistic: G = mean over the 50 τ points of [accuracy(τ) − frontier(mean NFE(τ))]. The bootstrap recomputes the cell accuracies, the frontier and the curve on every resample.
   - A dataset "dominates" when the CI of G lies entirely above 0. H3 is supported when at least 3 of 4 datasets dominate.
   - Reported with each result: the minimum and maximum gap over the curve, the fraction of τ points with gap ≥ 0, and the ECE-vs-mean-NFE curve against the ECE frontier (lowest ECE, running minimum; descriptive only).
   - The dev-frozen τ (0.0204 raw; 0.0 after temperature scaling; `results/dev_fits.json`) is one reported operating point, not the H3 test.
   - A curve after pooled per-cell temperature scaling is reported as secondary.
2. **Exploratory ladder (dev-selected, labeled as such):** C0 → C1(32,4) → C1(32,16), cumulative NFE 1 / 6 / 23. It gets the same curve, frontier, statistic and oracle, reported alongside the pre-registered ladder, and carries no verdict.
3. **Secondary: per-(dataset, cell) temperature scaling** fit on dev (NLL grid search), frozen in `results/dev_fits.json` under `temperatures_by_dataset` before test. It is reported next to the pooled per-cell temperature scaling.
4. **Secondary: adaptive ECE:** 15 equal-mass bins on top-label confidence, reported next to equal-width ECE everywhere.
5. **Secondary ablation:** C1 and C2 at S=128, T ∈ {1, 4, 16}, with `suppress_eos_in_scratch` ON, on the ORIGINAL test items only (400 / 400 / 400 / 240; `splits.json` `test_original`). Rows go to separate `*_noeos.jsonl` files (`configs/full_noeos.yaml`). The main setting stays OFF.
6. **Test set expansion:** every existing test item is kept, plus never-seen items drawn with seed 1246 (`dd/expand.py`). Test = 1000 BoolQ, 1000 StrategyQA, 1000 ARC-C, 600 jagged. Dev is unchanged. The new split was written to `results/splits.json` (`test`, `test_original`, `test_added`, `expansion`) before any test run.
   - The 360 new jagged items (90 per type) come from generator seed 1246 with id prefix `jagged2`.
   - 4 numeric items that duplicated existing content were replaced from seed 1247 (`jagged3`).
