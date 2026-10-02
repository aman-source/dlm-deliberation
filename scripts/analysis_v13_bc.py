"""v1.3 EXPLORATORY tables for B (LLaDA S=32, T=32) and C (Dream-v0-Instruct-7B), original test items.

B: C1/C2 at S=32, T in {1,4,16,32} on the original test items, with H1-style contrasts
   ECE(T=32) - ECE(T=1) and accuracy(T=32) - accuracy(T=1).
C: Dream C0 and C1/C2 at S=32, T in {1,4,16}, temperatures fit on Dream's own dev run;
   H1-style contrast ECE(T=16) - ECE(T=1) (C1, S=32) and H4-style C1 - C2 at S=32, T=16.
Paired bootstrap 1000, seed 1234; McNemar exact for accuracy.

Usage: python scripts/analysis_v13_bc.py  -> results/v13_bc.md, results/v13_bc.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dd.analysis import adaptive_ece, scaled_probs, temp_key  # noqa: E402
from dd.metrics import expected_calibration_error, mcnemar_test  # noqa: E402

DATASETS = ["boolq", "strategyqa", "arc_c", "jagged"]
N_BOOT, SEED = 1000, 1234


def f(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def load(dir_: Path, model: str, template: str, split: str = "test") -> dict:
    """{dataset: {(cond,S,T): {item_id: row}}} for unsuppressed protocol-v1.1 rows."""
    out: dict = {d: {} for d in DATASETS}
    for path in sorted(dir_.glob(f"*_{split}_{model}_*_{template}.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("protocol") != "v1.1" or r.get("suppress_eos_in_scratch"):
                continue
            out[r["dataset"]].setdefault((r["condition"], int(r["S"]), r["T"]), {})[r["item_id"]] = r
    return out


def conf(r, t=None):
    return max(scaled_probs(r, t).values())


def metrics(group: list, t_pool=None) -> dict:
    corr = [bool(r["correct"]) for r in group]
    cf = [conf(r) for r in group]
    eos = [r["eos_pad_fraction"] for r in group if r.get("eos_pad_fraction") is not None]
    return {
        "n": len(group), "accuracy": float(np.mean(corr)), "ece": expected_calibration_error(corr, cf),
        "adaptive_ece": adaptive_ece(corr, cf),
        "ece_ts": expected_calibration_error(corr, [conf(r, t_pool) for r in group]) if t_pool else None,
        "off_label": float(np.mean([r["off_label_mass"] for r in group])),
        "eos_median": float(np.median(eos)) if eos else None, "nfe": float(np.mean([r["nfe"] for r in group])),
    }


def contrast(left: dict, right: dict, kind: str) -> dict:
    ids = sorted(set(left) & set(right))
    a, b = [left[i] for i in ids], [right[i] for i in ids]

    def stat(ix):
        x, y = [a[i] for i in ix], [b[i] for i in ix]
        if kind == "accuracy":
            return np.mean([r["correct"] for r in x]) - np.mean([r["correct"] for r in y])
        return (expected_calibration_error([r["correct"] for r in x], [conf(r) for r in x])
                - expected_calibration_error([r["correct"] for r in y], [conf(r) for r in y]))

    rng = np.random.default_rng(SEED)
    boots = [stat(rng.integers(0, len(ids), len(ids))) for _ in range(N_BOOT)]
    out = {"n": len(ids), "estimate": float(stat(np.arange(len(ids)))),
           "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))}
    if kind == "accuracy":
        out["mcnemar_p"] = mcnemar_test([r["correct"] for r in a], [r["correct"] for r in b])["p_value_exact"]
    out["call"] = "above 0" if out["lo"] > 0 else ("below 0" if out["hi"] < 0 else "includes 0")
    return out


def section_b(L: list, out: dict) -> None:
    splits = json.loads((ROOT / "results" / "splits.json").read_text(encoding="utf-8"))
    rows = load(ROOT / "results" / "raw", "llada-8b-instruct", "v3e")
    fits = json.loads((ROOT / "results" / "dev_fits.json").read_text(encoding="utf-8"))["temperatures"]
    L += ["## B. LLaDA-8B-Instruct, C1/C2 at S=32 with T up to 32 (one token per step), original test items", "",
          "T=32 has no dev-fitted temperature (it was not in the dev grid), so ECE-TS is n/a for it.", ""]
    out["B"] = {"tables": {}, "contrasts": {}}
    for ds in DATASETS:
        orig = set(splits["datasets"][ds]["test_original"])
        L += [f"### {ds}", "", "| cell | n | acc | ECE | aECE | ECE-TS pooled | off_label | eos_pad | NFE |", "|---|---|---|---|---|---|---|---|---|"]
        for cond in ("C1", "C2"):
            for steps in (1, 4, 16, 32):
                group = [r for i, r in rows[ds].get((cond, 32, steps), {}).items() if i in orig]
                if not group:
                    continue
                t = (fits.get(temp_key("llada-8b-instruct", None, (cond, 32, steps))) or {}).get("temperature")
                m = metrics(group, t)
                out["B"]["tables"][f"{ds}|{cond} 32,{steps}"] = m
                L.append(f"| {cond} 32,{steps} | {m['n']} | {f(m['accuracy'])} | {f(m['ece'])} | {f(m['adaptive_ece'])} | {f(m['ece_ts'])} | "
                         f"{f(m['off_label'])} | {f(m['eos_median'])} | {f(m['nfe'], 0)} |")
        L.append("")
    L += ["### H1-style contrasts at S=32 (T=32 minus T=1; T=32 minus T=16), original test items", "",
          "| dataset | cond | contrast | estimate | 95% CI | McNemar p | CI |", "|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        orig = set(splits["datasets"][ds]["test_original"])
        for cond in ("C1", "C2"):
            g = {t: {i: r for i, r in rows[ds].get((cond, 32, t), {}).items() if i in orig} for t in (1, 16, 32)}
            for lo_t in (1, 16):
                for kind in ("ece", "accuracy"):
                    c = contrast(g[32], g[lo_t], kind)
                    out["B"]["contrasts"][f"{ds}|{cond}|{kind}|T32-T{lo_t}"] = c
                    L.append(f"| {ds} | {cond} | {kind}(T=32) − {kind}(T={lo_t}) | {f(c['estimate'], 4)} | [{f(c['lo'], 4)}, {f(c['hi'], 4)}] | "
                             f"{f(c.get('mcnemar_p'), 4) if 'mcnemar_p' in c else '-'} | {c['call']} |")
    L.append("")


def section_c(L: list, out: dict) -> None:
    raw = ROOT / "results" / "raw_dream"
    fits_path = ROOT / "results" / "dream" / "dev_fits.json"
    if not raw.exists() or not fits_path.exists():
        L += ["## C. Dream-v0-Instruct-7B", "", "Dream rows not available yet.", ""]
        return
    template = json.loads((ROOT / "results" / "dream" / "template.json").read_text(encoding="utf-8"))["template"]
    fits = json.loads(fits_path.read_text(encoding="utf-8"))["temperatures"]
    rows = load(raw, "dream-v0-instruct-7b", template)
    dev = load(raw, "dream-v0-instruct-7b", template, split="dev")
    L += [f"## C. Dream-v0-Instruct-7B (template `{template}`), original test items", "",
          "Temperatures fit on Dream's own dev run (same dev items). Same protocol as LLaDA (think prefix, end-of-turn pinned after "
          "the slot, batch 1, greedy, suppression off). Dream logits are shifted right by one, as in Dream's own generation code.", ""]
    out["C"] = {"template": template, "tables": {}, "dev_tables": {}, "contrasts": {}}
    cells = [("C0", 0, None)] + [(c, 32, t) for c in ("C1", "C2") for t in (1, 4, 16)]
    for ds in DATASETS:
        L += [f"### {ds}", "", "| cell | n | acc | ECE | aECE | ECE-TS (Dream dev) | off_label | eos_pad | NFE | dev acc |", "|---|---|---|---|---|---|---|---|---|---|"]
        for cell in cells:
            group = list(rows[ds].get(cell, {}).values())
            if not group:
                continue
            t = (fits.get(temp_key("dream-v0-instruct-7b", None, cell)) or {}).get("temperature")
            m = metrics(group, t)
            dg = list(dev[ds].get(cell, {}).values())
            dacc = float(np.mean([r["correct"] for r in dg])) if dg else None
            name = "C0" if cell[0] == "C0" else f"{cell[0]} 32,{cell[2]}"
            out["C"]["tables"][f"{ds}|{name}"] = m | {"temperature": t, "dev_accuracy": dacc}
            L.append(f"| {name} | {m['n']} | {f(m['accuracy'])} | {f(m['ece'])} | {f(m['adaptive_ece'])} | {f(m['ece_ts'])} | "
                     f"{f(m['off_label'])} | {f(m['eos_median'])} | {f(m['nfe'], 0)} | {f(dacc)} |")
        L.append("")
    L += ["### H1-style (C1 S=32: T=16 minus T=1) and H4-style (C1 − C2 at S=32, T=16) contrasts for Dream", "",
          "| dataset | contrast | estimate | 95% CI | McNemar p | CI |", "|---|---|---|---|---|---|"]
    for ds in DATASETS:
        g = rows[ds]
        specs = [
            ("H1 ECE", g.get(("C1", 32, 16), {}), g.get(("C1", 32, 1), {}), "ece", "C1 S=32 ECE(T=16) − ECE(T=1)"),
            ("H1 acc", g.get(("C1", 32, 16), {}), g.get(("C1", 32, 1), {}), "accuracy", "C1 S=32 acc(T=16) − acc(T=1)"),
            ("H1 ECE C2", g.get(("C2", 32, 16), {}), g.get(("C2", 32, 1), {}), "ece", "C2 S=32 ECE(T=16) − ECE(T=1)"),
            ("H4 acc", g.get(("C1", 32, 16), {}), g.get(("C2", 32, 16), {}), "accuracy", "C1 − C2 acc at S=32, T=16"),
            ("H4 ECE", g.get(("C1", 32, 16), {}), g.get(("C2", 32, 16), {}), "ece", "C1 − C2 ECE at S=32, T=16"),
        ]
        for key, left, right, kind, label in specs:
            if not left or not right:
                continue
            c = contrast(left, right, kind)
            out["C"]["contrasts"][f"{ds}|{key}"] = c
            L.append(f"| {ds} | {label} | {f(c['estimate'], 4)} | [{f(c['lo'], 4)}, {f(c['hi'], 4)}] | "
                     f"{f(c.get('mcnemar_p'), 4) if 'mcnemar_p' in c else '-'} | {c['call']} |")
    L.append("")


def main() -> None:
    L = ["# v1.3 B and C (exploratory; original test items 400/400/400/240)", "",
         "Paired bootstrap (1000, seed 1234), 95% percentile CIs; McNemar exact. Generated by `scripts/analysis_v13_bc.py`.", ""]
    out: dict = {}
    section_b(L, out)
    section_c(L, out)
    (ROOT / "results" / "v13_bc.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ROOT / "results" / "v13_bc.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote results/v13_bc.md")


if __name__ == "__main__":
    main()
