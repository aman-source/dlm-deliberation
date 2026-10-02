# PLAN.md: A Deliberation Dial for Diffusion Decision Models

Autonomous execution plan for Claude Code. Read this whole file before writing any code.

Owner: Aman Shaik. Status: v1 (trimmed scope). Target: first arXiv version in about 3 weeks.

---

## 0. Agent operating rules (read first, follow always)

1. **Never fabricate results, numbers, citations, or dataset fields.** Every number in any report must come from a file in `results/`. If something fails, report the failure.
2. **Verify before trusting.** Items marked `VERIFY` below are things the plan author was not certain about (dataset IDs, token IDs, API shapes). Check each one, log what you found in `results/LOG.md`, and adapt.
3. **Log every deviation** from this plan in `results/LOG.md` with the reason. Deviations are fine; silent deviations are not.
4. **Never tune on test.** All thresholds, temperatures, and prompt choices are fixed on the `dev` split. Test is touched only in Phase 4 and later.
5. **Resumable by default.** Every run writes one JSON line per (item, cell) to `results/raw/*.jsonl` and skips items already present. A crash must never lose finished work.
6. **Fixed seeds everywhere** (`seed: 1234` in config). Record the seed in every output row.
7. **Gates are hard stops.** If a phase's acceptance check fails, fix it or log why it cannot be fixed. Do not proceed to the next phase with a failed gate unless the gate says "warn only".
8. **Do not ask the user questions mid-run** unless a gate fails and cannot be fixed. Make the reasonable choice, log it, continue.
9. **Keep the machine safe.** Before any full sweep, print the projected wall-clock time. If the projection exceeds the device budget (Section 9), stop after Phase 3 and write `results/CLOUD.md` instead of running.

---

## 1. Research question and claim

**Question.** In a masked diffusion LLM used as a System One decision model (typed answer read from one masked slot), does adding a jointly denoised scratch region before the answer slot, and more denoising steps, work as a controllable System 1 to System 2 dial? How do accuracy **and calibration** change along that dial, and can a confidence-based escalation rule beat fixed budgets at matched compute?

**Why this is a first (per the gap report, 28 Sep 2026).**
- djev / vLLM PR 57250 expose a step cap, pinned templates, and a pre-read `think` channel, but publish no accuracy or calibration sweep over budget.
- arXiv 2608.08791 studies dLLM confidence vs internal correctness but does not sweep steps or scratch.
- arXiv 2604.23235 sweeps denoising steps but on token infilling, not typed decisions (ECE rises 0.034 to 0.415 over steps there).
- Prophet (arXiv 2508.19982) uses top-2 confidence gap for early commit in generation, accuracy only, no calibration.
- Nobody has measured the budget/calibration trade-off for typed reads or compared in-canvas scratch against pre-read thought.

**Headline claim we will test (not assume):** a single dLLM can serve as both a fast System One reader and a slower deliberative reader, and a margin-based escalation rule gives a better accuracy/calibration vs compute frontier than any fixed budget.

---

## 2. Pre-registered hypotheses

Write these into `results/HYPOTHESES.md` at the start, unchanged. Report every one, including nulls.

- **H1 (calibration drift):** at fixed scratch length S > 0, raw ECE of the answer slot increases with steps T, even where accuracy increases.
- **H2 (where deliberation helps):** scratch (S > 0) improves accuracy on StrategyQA and the jaggedness set, and gives no significant gain on BoolQ and ARC-Challenge.
- **H3 (escalation wins):** the margin-based escalation rule (Section 7) dominates the fixed-budget frontier on accuracy vs mean forward passes on at least 3 of 4 datasets.
- **H4 (in-canvas vs pre-read):** in-canvas scratch and pre-read thought differ in accuracy or ECE at matched (S, T). Direction not predicted.

Statistics: paired bootstrap (1000 resamples over items) for accuracy and ECE differences, 95% CIs; McNemar test for paired accuracy. A hypothesis is "supported" only if the CI excludes zero in the predicted direction.

---

## 3. Repository layout (create exactly this)

```
dlm-deliberation/
  PLAN.md                  # this file
  requirements.txt
  configs/
    smoke.yaml
    dev.yaml
    full.yaml
  dd/
    __init__.py
    device.py              # device + dtype selection
    data.py                # loaders, jaggedness generator, formatting, option shuffle, splits
    labels.py              # label token resolution
    llada.py               # LLaDA wrapper: load, forward, read slot
    canvas.py              # builds token sequences per condition
    decode.py              # in-canvas scratch loop, pre-read thought, anytime readouts
    baselines.py           # Qwen3-8B letter readout (+ optional CoT)
    metrics.py             # acc, ECE, Brier, NLL, AURC, temperature scaling, bootstrap, McNemar
    escalate.py            # escalation rule + oracle + fixed-budget frontier
    run.py                 # orchestration, resume, jsonl writing
    report.py              # tables, figures, REPORT.md
  schedulers/
    logicdiff.py           # OPTIONAL, provided by the author (see Section 6.4)
  results/
    LOG.md
    HYPOTHESES.md
    raw/                   # jsonl outputs
    figs/
    REPORT.md
```

`requirements.txt`: `torch`, `transformers` (VERIFY a version compatible with LLaDA's `trust_remote_code`; pin whatever works), `accelerate`, `datasets`, `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`.

---

## 4. Models

### 4.1 Primary: LLaDA-8B-Instruct
- HF id: `GSAI-ML/LLaDA-8B-Instruct`. Load with `AutoTokenizer` and `AutoModel`, both `trust_remote_code=True`.
- Forward: `logits = model(input_ids).logits`. Bidirectional attention, no KV cache.
- Mask token: id `126336`, token string expected `<|mdm_mask|>`. **VERIFY** with `tok.convert_ids_to_tokens(126336)`. If different, find the mask token in `tok.special_tokens_map` / added tokens and log it.
- Chat format: `tok.apply_chat_template([{"role":"user","content":prompt}], add_generation_prompt=True, tokenize=False)`, then tokenize, then append the canvas tokens (Section 5.3).
- dtype: bf16 on CUDA. On Apple MPS try bf16; if it errors or produces NaNs, fall back to fp16, then fp32 (8B fp32 is about 32GB and fits in 64GB). Log the dtype used. Keep one dtype for all runs of a given sweep.

### 4.2 Optional second model: Dream-7B-Instruct (Phase 6 only)
- HF id: `Dream-org/Dream-v0-Instruct-7B` (VERIFY). Same harness if its forward returns logits over a masked canvas; mask id from its tokenizer. If its interface differs substantially, skip and log.

### 4.3 AR baseline: Qwen3-8B
- HF id: `Qwen/Qwen3-8B`. Use `enable_thinking=False` in `apply_chat_template` for the System One readout.

---

## 5. Data

### 5.1 Datasets and splits

For each dataset: fixed-seed subsample, then split into `dev` (20%) and `test` (80%). Store the chosen item IDs in `results/splits.json` so every run uses identical items.

| Name | Source | Type | Size (dev/test) | Notes |
|---|---|---|---|---|
| boolq | `google/boolq`, validation (VERIFY id; fallback `boolq`) | yes/no | 100 / 400 | state = passage truncated to 300 words; question = question |
| strategyqa | try `ChilleD/StrategyQA`, then `wics/strategy-qa` (VERIFY both) | yes/no | up to 100 / 400 | state = "(none)"; if neither loads cleanly, drop and log; do NOT substitute silently |
| arc_c | `allenai/ai2_arc`, config `ARC-Challenge`, test | choice | 100 / 400 | options from `choices.text`, gold from `answerKey` mapped via `choices.label` |
| jagged | generated (Section 5.2) | mixed | 60 / 240 | fully synthetic, exact ground truth |

Smoke mode uses 20 items per dataset from `dev` only.

### 5.2 Jaggedness set (generator in `data.py`, seed 1234)

75 items each of 4 types, targeting failure modes TypeSafe documents for Jev:
- **counting:** list of 8 to 15 words drawn from a 6-word vocab; "How many times does 'X' appear in the list?"; 4 numeric options including gold (gold in 1 to 6).
- **date_compare:** two dates, each rendered in ISO (`2024-03-05`) or long form (`March 5, 2024`), never ambiguous numeric formats; yes/no "Is the first date earlier than the second?"; balance yes/no 50/50.
- **indirection:** a chain of 3 to 4 "X's manager is Y" facts, shuffled; "Who is A's manager's manager?"; 4 name options.
- **numeric_compare:** decimals chosen to trap string comparison (for example 3.9 vs 3.11); "Which number is larger?"; 3 options.

### 5.3 Unified formatting

Every item becomes lettered multiple choice. Yes/no items become two options, "Yes" and "No".

**Option shuffle:** for each item, shuffle option order with `random.Random(seed + stable_hash(item_id))` so the gold letter is roughly uniform. Store the permutation. `stable_hash` must be process-independent: Python's built-in `hash()` on `str` is salted per process (PYTHONHASHSEED), so it would reshuffle options on every run. The code uses the first 8 bytes of SHA-256 of the UTF-8 item id, little-endian (`dd/data.py: stable_hash`); `zlib.crc32(item_id.encode())` would also be valid. (Plan bug fixed 29 Sep 2026; see `results/LOG.md`.)

**Prompt (user message), exact text:**
```
Read the state and answer the question.

State:
{state}

Question: {question}

Options:
A. {opt_A}
B. {opt_B}
...

Reply with the letter of the correct option after "Answer:".
```

### 5.4 Label token resolution (`labels.py`)
For each letter L, collect single-token variants among `["L", " L"]`. The slot probability for L is the logsumexp of its variants' logits, then a softmax across letters. Also record `off_label_mass` = 1 minus the total probability (full-vocab softmax) on all variants of all valid letters. Abort with a clear error if any letter has zero single-token variants.

---

## 6. Conditions (the dial)

Notation: S = scratch length in tokens, T = denoising steps, NFE = number of forward passes.

### 6.1 C0: System One read (S = 0)
Canvas = pinned tokens of `"Answer:"` + one masked slot. One forward pass. Read the slot. NFE = 1.

### 6.2 C1: in-canvas scratch, commit-last (main condition)
Canvas = S masked scratch tokens + pinned tokens of `"\nAnswer:"` + one masked slot.
- Steps t = 1..T: forward pass on the full sequence. Among still-masked **scratch** positions, compute confidence = max softmax over the full vocab; unmask the top k_t with their argmax tokens, where the k_t split S as evenly as possible over T steps (earlier steps take the remainder). The answer slot is never unmasked during these steps.
- **Anytime readout:** at every step, also read the slot's label distribution (free, same pass) and store it.
- After step T the scratch is fully filled; do one final pass and read the slot. NFE = T + 1.
- Greedy and deterministic: temperature 0, no Gumbel noise.
- Constraint: T <= S.

### 6.3 C2: pre-read thought (djev `think` analogue)
- Generate S thought tokens with standard LLaDA low-confidence remasking over T steps on a canvas of S masks only (no answer template present).
- Then append pinned `"\nAnswer:"` + one masked slot after the thought and do a single read. NFE = T + 1.

The only difference from C1 is whether the answer template and slot are visible in the canvas while the scratch is being denoised.

### 6.4 C3 (optional): LogicDiff scheduling
Same as C1, but scratch positions are unmasked by the author's LogicDiff logical-role schedule instead of by confidence.

**The plan author does not have the LogicDiff implementation.** Implement C3 only if `schedulers/logicdiff.py` exists and exposes:
```python
def select_positions(logits, masked_positions, step, total_steps, k, tokenizer) -> list[int]
```
If the file is missing, skip C3, and write in `LOG.md` and `REPORT.md` that it is pending the author's scheduler. Do not invent a LogicDiff approximation.

### 6.5 Grid (v1)

| Condition | S | T | NFE per item |
|---|---|---|---|
| C0 | 0 | n/a | 1 |
| C1 | 32 | 1, 4, 16 | 2, 5, 17 |
| C1 | 128 | 1, 4, 16 | 2, 5, 17 |
| C2 | 32 | 1, 4, 16 | 2, 5, 17 |
| C2 | 128 | 1, 4, 16 | 2, 5, 17 |

13 cells, 97 forward passes per item before the anytime shortcut.

---

## 7. Escalation rule (`escalate.py`)

- Ladder (all C1): L0 = C0, L1 = C1(S=32, T=4), L2 = C1(S=128, T=16).
- At level k, compute margin m = p_top1 minus p_top2 of the slot distribution. If m < tau, escalate to level k+1. L2 always answers.
- Cost accounting: escalation reruns from scratch, so cumulative NFE is L0 = 1, L0 to L1 = 6, L0 to L1 to L2 = 23.
- Sweep a single tau on a grid of 50 values in [0, 1] **on dev**. On test, report the full accuracy vs mean NFE curve and the ECE vs mean NFE curve.
- Compare against every fixed-budget cell at matched mean NFE (interpolate the fixed frontier).
- **Oracle upper bound:** per item, the cheapest level that is correct; if none, L0.
- Also report the escalation curve with per-cell temperature scaling applied before computing the margin (temperatures fit on dev).

---

## 8. Metrics (`metrics.py`)

Per (dataset, model, cell), on test:
- **Accuracy.**
- **ECE:** 15 equal-width bins on top-label confidence.
- **Brier:** multiclass, sum over labels of (p minus one-hot) squared, averaged over items.
- **NLL:** negative log probability of the gold letter, clipped at 1e-12.
- **AURC:** area under the risk-coverage curve, ranking items by top-label confidence.
- **ECE after temperature scaling:** a single temperature per (model, cell), fit on dev by minimizing NLL (log-spaced grid 0.05 to 20, then refine).
- **Mean off_label_mass:** the template-drift diagnostic.
- **Flip rate:** fraction of items whose argmax at the anytime readout changes between consecutive recorded steps (C1 only).
- **NFE and mean wall-clock ms per item.**

Paired comparisons: bootstrap CIs and McNemar, as in Section 2.

---

## 9. Compute plan and device budget

### 9.1 Device selection (`device.py`)
Use CUDA if available, else MPS, else CPU (CPU for smoke tests only). Log the device and dtype.

### 9.2 Timing gate (Phase 1)
- Measure mean forward-pass time over 20 passes at the realistic sequence length (about 600 tokens), after 3 warmup passes.
- Projected full-sweep hours = (items across all splits) × 97 × pass_time / 3600, plus baselines.
- Device budgets:
  - MPS (Mac): 24 hours of projected compute for everything in Phases 3 to 5.
  - CUDA: 48 hours.
- If the projection exceeds the budget, run Phases 1 to 3 at dev scale only, then write `results/CLOUD.md`: exact commands to run Phases 4 and 5 on a rented CUDA GPU, the projected hours there, and a note on which results already exist.
- The plan author's rough estimate is 1 to 3 s per pass on an M4 Pro under PyTorch MPS. This is unmeasured; replace it with the measured number.

### 9.3 Batching
Batch items of similar length in C0 and in the final reads, padding on the right with an attention mask. VERIFY that LLaDA's forward accepts `attention_mask`. If it does not, run batch size 1 and log it.

### 9.4 Anytime shortcut
If the Phase 2 validation passes, T = 4 cells can be read from the T = 16 run after its 4th unmasking step (the readout on forward pass 5, same NFE as native T = 4) instead of being run natively. T = 1 cells are always run natively.

---

## 10. Phases and gates

### Phase 0: setup and novelty check
1. Create the repo layout, `requirements.txt`, and a virtualenv; install.
2. Write `results/HYPOTHESES.md` (Section 2, verbatim).
3. If you have web access, search arXiv, Google Scholar, and X for: "diffusion language model typed decision", "djev evaluation steps", "diffusion LLM calibration denoising steps", "scratch canvas diffusion decision". Log findings with links in `results/LOG.md` under "Novelty check". If you find a paper that runs the C1 vs budget calibration sweep on typed decisions, **stop and report it** before spending compute.

**Gate 0:** environment imports cleanly; novelty check logged (or "no web access" logged).

### Phase 1: smoke test (20 items per dataset, dev only)
Run C0 and C1(S=32, T=4) on LLaDA.

**Gate 1 checks:**
- Every letter resolves to at least one single-token variant.
- Median off_label_mass at C0 is below 0.5. If not, try up to 3 prompt/template variants (for example `"Answer: "` with a trailing space and single-token letters without a space, or `"The answer is"`), pick the lowest-off-label variant on these smoke items, and log it. That template is then frozen.
- C0 accuracy on ARC-Challenge and BoolQ smoke items beats chance by at least 10 points (**warn only** on 20 items, but log it loudly).
- No NaNs in logits.
- Timing measured and projection printed (Section 9.2).

### Phase 2: anytime-readout validation (50 dev items: ARC-C 25 and jagged 25)
Run C1(S=32) natively at T = 4, and also at T = 16 reading step 4.

**Gate 2:** argmax agreement is at least 95% and the ECE difference is below 0.02. Pass: enable the shortcut. Fail: disable it and run T = 4 natively. Either way, log the numbers. Both outcomes are fine; this is also a small reportable finding.

### Phase 3: dev sweep
Full grid (Section 6.5) on all dev items, plus Qwen3-8B C0 readout on dev.
- Fit per-cell temperatures and the escalation tau on dev.
- Save to `results/dev_fits.json`.

**Gate 3:** all dev rows present; fits saved; budget check (Section 9.2) passed, or `CLOUD.md` written and the run stops here.

### Phase 4: test sweep
Full grid on all test items, LLaDA and Qwen3-8B C0. Frozen template, temperatures, and tau from Phase 3.

**Gate 4:** row counts match the expected count (items × cells); no NaN rows.

### Phase 5: analysis and report
Run `python -m dd.report`, which writes `results/REPORT.md` and `results/figs/` (Section 11).

**Gate 5:** every number in REPORT.md is traceable to `results/raw` or `results/summary.csv`.

### Phase 6: optional extensions (only after Gate 5)
Run in this order, each only if time remains:
1. C3 LogicDiff scheduling, if `schedulers/logicdiff.py` exists.
2. Dream-7B on the full grid.
3. Qwen3-8B with thinking enabled as a System 2 ceiling (cap thinking at 512 tokens; log the cap).
4. Option-order control: C0 on arc_c test with options reversed, to measure position bias.

---

## 11. Report spec (`report.py` writes `results/REPORT.md`)

Sections, in order:
1. **Setup:** device, dtype, model ids and revisions, template chosen in Gate 1, anytime shortcut on/off, item counts.
2. **Main table per dataset:** rows = cells; columns = acc, ECE, ECE-after-TS, Brier, NLL, AURC, off_label_mass, NFE, ms/item. Bold the best per column.
3. **Figures** (`results/figs/`):
   - accuracy vs NFE per dataset, fixed cells plus the escalation curve plus the oracle;
   - raw ECE vs NFE, same layout;
   - reliability diagrams for C0 and for the best C1 cell;
   - anytime trajectories: mean top-1 prob and accuracy vs step for C1(S=128, T=16);
   - C1 vs C2 paired bar chart at matched (S, T).
4. **Hypotheses H1 to H4:** each marked supported / not supported / inconclusive, with the exact test statistic and CI. Auto-generated from `summary.csv`; no hand-written claims.
5. **Escalation results:** accuracy and ECE at matched mean NFE vs the best fixed cell, and the gap to the oracle.
6. **Baselines:** Qwen3-8B C0 (and optional CoT), same metrics.
7. **Deviations and failures:** copied from LOG.md.
8. **Limitations:** single model family unless Dream ran; greedy decoding only; English only; sample sizes; LogicDiff pending unless it ran.

Also write `results/summary.csv` (one row per dataset × model × cell with every metric and its CI).

---

## 12. Output row schema (`results/raw/*.jsonl`)

```json
{
  "dataset": "arc_c", "split": "test", "item_id": "...", "model": "llada-8b-instruct",
  "condition": "C1", "S": 32, "T": 4, "seed": 1234, "template": "v1",
  "perm": [2, 0, 3, 1], "gold": "C",
  "readouts": [{"step": 1, "probs": {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.1}, "off_label_mass": 0.03}],
  "final_probs": {"A": 0.05, "B": 0.1, "C": 0.8, "D": 0.05},
  "pred": "C", "correct": true, "nfe": 5, "wall_ms": 812.4,
  "scratch_text": "decoded scratch or thought tokens, for qualitative analysis",
  "device": "mps", "dtype": "bfloat16"
}
```

---

## 13. Commands

```
python -m dd.run --config configs/smoke.yaml     # Phase 1
python -m dd.run --config configs/smoke.yaml --validate-anytime   # Phase 2
python -m dd.run --config configs/dev.yaml       # Phase 3
python -m dd.run --config configs/full.yaml      # Phase 4
python -m dd.report                              # Phase 5
```

Each config sets: model list, datasets, split, cells, batch size, seed, dtype preference, device budget hours, anytime shortcut flag (set automatically from Gate 2 if left as `auto`).

---

## 14. Definition of done

- [ ] `results/REPORT.md` exists with all 8 sections, every number sourced from `results/`.
- [ ] All four hypotheses reported with statistics.
- [ ] `results/LOG.md` lists every VERIFY outcome, deviation, and gate result.
- [ ] Raw jsonl complete for dev and test (or `CLOUD.md` explains what remains and how to run it).
- [ ] A final summary to the user: 5 lines max. What was run, the headline number for escalation vs best fixed budget, H1 to H4 outcomes, and anything that failed.

---

## 15. Notes for the paper (do not act on these; context only)

- Working title: "A Deliberation Dial for Diffusion Decision Models: Scratch Canvases, Denoising Steps, and Calibrated Escalation."
- The contribution is the measurement and the escalation rule, not a new model. A clean null on H1 or H3 is still publishable if the sweep is careful.
- Before posting, redo the novelty search from Phase 0. This field moved daily in September 2026.

---

## 16. Protocol amendment v1.1 (29 Sep 2026, before any dev or test data was seen)

Decided by the project lead after the v1.0 smoke on 20 dev items per dataset. It supersedes the matching parts of Sections 5.3, 6.1–6.3, 9.3 and Phase 1. H1–H4 are unchanged.

1. **Slot fix.** Every canvas pins `<|eot_id|>` immediately after the answer slot, so the slot is no longer the sequence's final position. Every readout records the top-5 full-vocabulary slot tokens (`slot_topk`).
2. **Prompt** (all conditions, identical). The last line of the Section 5.3 prompt becomes: `If you have space to think, think step by step first. Then finish with "The answer is" followed by the letter.` `results/items.jsonl` stays frozen; the prompt is rendered at run time from the frozen state, question and shuffled options (`dd.data.prompt_for`).
3. **Template.** "The answer is" (v3e) for all conditions, unless the re-smoke shows the v1 wording with eot (v1e) has lower median C0 off_label_mass. Pre-committed rule: the template with the lowest median C0 off_label_mass wins. Gate: it must be < 0.5; otherwise stop and report the top-5 slot tokens.
4. **Canvases.**
   - C0: prompt + "The answer is" + [slot] + `<|eot_id|>`.
   - C1: prompt + "Let me think step by step.\n" + S scratch + "\nThe answer is" + [slot] + `<|eot_id|>`.
   - C2: prompt + "Let me think step by step.\n" + S thought tokens, denoised with no template or slot visible, then + "\nThe answer is" + [slot] + `<|eot_id|>`, read once.
   - The only C1/C2 difference is whether the template is visible during denoising (H4 unchanged).
5. **suppress_eos_in_scratch** (main setting, the same for C1 and C2). Pre-committed rule: ON if either condition's median `eos_pad_fraction` with the flag off exceeds 0.3 on the re-smoke; otherwise OFF.
6. **Batch size 1** for every run. Batching is not tested (supersedes Section 9.3).
7. The v1.0 smoke C2 rows (thought = "Answer: X" plus EOS padding) are kept in `results/raw/appendix_c2_naive/` for an appendix note and excluded from the main tables. The other v1.0 smoke rows and gate files are archived under `results/raw/smoke_v1_0/` and `results/smoke_v1_0/`.
8. **Re-smoke** (`configs/resmoke.yaml`) uses the same 20 dev items per dataset: C0 with v3e and v1e, then C1(32,4) and C2(32,4) with the winning template, suppression off and on. Output: `results/protocol_v1_1.json`.
