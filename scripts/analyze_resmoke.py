"""Summarize the protocol v1.1 re-smoke from results/ (read-only). Usage: python scripts/analyze_resmoke.py"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dd.run import cell_file  # noqa: E402

DATASETS = ["boolq", "strategyqa", "arc_c", "jagged"]
MODEL = "llada-8b-instruct"


def rows(dataset, condition, template, suppress=False):
    scratch, steps = (0, None) if condition == "C0" else (32, 4)
    path = cell_file(dataset, "dev", MODEL, condition, scratch, steps, template, suppress_eos=suppress and condition != "C0")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    protocol = json.loads((ROOT / "results" / "protocol_v1_1.json").read_text(encoding="utf-8"))
    winner = protocol["template"]
    main_flag = protocol["suppress_eos_in_scratch"]
    items = {json.loads(l)["item_id"]: json.loads(l) for l in (ROOT / "results" / "items.jsonl").read_text(encoding="utf-8").splitlines()}

    print("== median off_label_mass")
    for template in ("v3e", "v1e"):
        values = [r["off_label_mass"] for d in DATASETS for r in rows(d, "C0", template)]
        print(f"C0 {template}: {statistics.median(values):.4f}")
    for condition in ("C1", "C2"):
        for flag in (False, True):
            values = [r["off_label_mass"] for d in DATASETS for r in rows(d, condition, winner, flag)]
            print(f"{condition} {winner} suppress={'on' if flag else 'off'}: {statistics.median(values):.4f}")

    print("== top-5 slot tokens, C0", winner, "(first item of boolq, arc_c, jagged)")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", revision="08b83a6feb34df1a6011b80c3c00c7563e963b07", trust_remote_code=True)
    for dataset in ("boolq", "arc_c", "jagged"):
        row = rows(dataset, "C0", winner)[0]
        top = [(tok.convert_ids_to_tokens(i), round(p, 4)) for i, p in row["readouts"][0]["slot_topk"]]
        print(row["item_id"], "gold", row["gold"], "off_label", round(row["off_label_mass"], 4), top)

    print("== eos_pad_fraction median / max")
    for condition in ("C1", "C2"):
        for flag in (False, True):
            values = [r["eos_pad_fraction"] for d in DATASETS for r in rows(d, condition, winner, flag)]
            print(f"{condition} suppress={'on' if flag else 'off'}: median {statistics.median(values):.4f} max {max(values):.4f}")

    print(f"== scratch samples (chosen setting: suppress={'on' if main_flag else 'off'})")
    for condition in ("C1", "C2"):
        for dataset in ("boolq", "arc_c", "jagged"):
            row = rows(dataset, condition, winner, main_flag)[0]
            print(f"--- {condition} {row['item_id']} eos_pad_fraction={row['eos_pad_fraction']}")
            print(repr(row["scratch_text"]))

    print("== accuracy (20 items each)")
    header = ["dataset", "chance", f"C0 v3e", f"C0 v1e"] + [f"{c} {w}".strip() for c in ("C1", "C2") for w in ("off", "on")]
    print(" | ".join(header))
    for dataset in DATASETS:
        base = rows(dataset, "C0", winner)
        chance = statistics.fmean(1 / items[r["item_id"]]["n_options"] for r in base)
        cells = [statistics.fmean(r["correct"] for r in rows(dataset, "C0", t)) for t in ("v3e", "v1e")]
        cells += [statistics.fmean(r["correct"] for r in rows(dataset, c, winner, f)) for c in ("C1", "C2") for f in (False, True)]
        print(" | ".join([dataset, f"{chance:.3f}"] + [f"{v:.2f}" for v in cells]))
    print("== protocol decision:", json.dumps({k: protocol[k] for k in ("template", "suppress_eos_in_scratch", "passed")}))


if __name__ == "__main__":
    main()
