"""v1.4 exploratory autoregressive C0 baselines (Qwen2.5-7B-Instruct, and Qwen2.5-7B base = Dream's init).

Same protocol as the Qwen3-8B C0 baseline: the user prompt rendered with the model's chat template
(enable_thinking=False is passed; templates without thinking ignore it), the template prefix
appended, and the next-token distribution read over the letter variants (logsumexp per letter,
softmax across letters); off_label_mass recorded. No generation.

Per model:
  1. Gate 1 template rule on the 20-per-dataset smoke items (dev): prefix "The answer is" (v3e)
     vs "Answer:" (v1e); lowest median off_label_mass wins; it must be < 0.5.
  2. Dev run on all dev items with the winner; one temperature fit on dev (NLL grid, pooled).
  3. Test run on the ORIGINAL test items (splits.json test_original).
Rows: results/raw_causal/; frozen files: results/causal/<model>/. Resumable.

Usage (on the GPU box): python -m dd.causal_baseline --model qwen2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from dd.baselines import load_causal_model, read_causal_batch
from dd.canvas import TEMPLATES
from dd.data import load_items, prompt_for, select_items
from dd.metrics import fit_temperature_ragged
from dd.run import append_row, base_row, cell_file, finish_row, read_done

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["boolq", "strategyqa", "arc_c", "jagged"]
MODELS = {
    "qwen2.5-7b-instruct": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "qwen2.5-7b-base": ("Qwen/Qwen2.5-7B", "d149729398750b98c0af14eb82c78cfe92750796"),
}
CANDIDATES = ["v3e", "v1e"]  # c0_prefix "The answer is" / "Answer:"; the slot suffix does not apply causally
C0 = {"condition": "C0", "S": 0, "T": None}
SEED = 1234


def raw_path(dataset: str, split: str, model: str, template: str) -> Path:
    path = cell_file(dataset, split, model, "C0", 0, None, template)
    return ROOT / "results" / "raw_causal" / path.name


def run_items(bundle, model: str, items: list[dict], split: str, template: str) -> list[dict]:
    path = raw_path(items[0]["dataset"], split, model, template)
    done = read_done(path)
    prefix = TEMPLATES[template]["c0_prefix"]
    dtype = str(bundle.dtype).replace("torch.", "")
    for item in items:
        if item["item_id"] in done:
            continue
        start = time.perf_counter()
        result = read_causal_batch(bundle, [prompt_for(item)], [item["letters"]], prefix)[0]
        wall = (time.perf_counter() - start) * 1000.0
        row = finish_row(base_row(item, split, model, C0, template, bundle.device.type, dtype, SEED), result, wall)
        row["slot_topk"] = None
        append_row(path, row)
        done[item["item_id"]] = row
    return [done[item["item_id"]] for item in items]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    args = parser.parse_args()
    hf_id, revision = MODELS[args.model]
    out_dir = ROOT / "results" / "causal" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = json.loads((ROOT / "results" / "splits.json").read_text(encoding="utf-8"))
    items = load_items(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle, thinking = load_causal_model(hf_id, args.model, device, torch.bfloat16, revision=revision)
    print(f"loaded {hf_id} @ {bundle.revision} ({thinking})")

    template_path = out_dir / "template.json"
    if template_path.exists():
        winner = json.loads(template_path.read_text(encoding="utf-8"))["template"]
    else:
        medians = {}
        for template in CANDIDATES:
            offs = []
            for ds in DATASETS:
                rows = run_items(bundle, args.model, select_items(items, splits, ds, "dev", limit=20), "dev", template)
                offs += [r["off_label_mass"] for r in rows]
            medians[template] = statistics.median(offs)
        winner = min(medians, key=lambda k: (medians[k], k != CANDIDATES[0]))
        payload = {"template": winner, "prefix": TEMPLATES[winner]["c0_prefix"], "median_off_label_mass": medians[winner],
                   "candidates": medians, "gate_passed": medians[winner] < 0.5,
                   "rule": "lowest median C0 off_label_mass on 20 smoke items per dataset; gate < 0.5"}
        template_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("template:", payload)
        if not payload["gate_passed"]:
            print("Gate 1 template rule not met; stopping.")
            return 3

    dev_rows = []
    for ds in DATASETS:
        dev_rows += run_items(bundle, args.model, select_items(items, splits, ds, "dev"), "dev", winner)
    logit_rows = [[r["letter_logits"][x] for x in r["letter_logits"]] for r in dev_rows]
    gold = [list(r["letter_logits"]).index(r["gold"]) for r in dev_rows]
    fit = fit_temperature_ragged(logit_rows, gold)
    fits = {"model": args.model, "hf_id": hf_id, "revision": bundle.revision, "template": winner,
            "temperature": fit["temperature"], "nll": fit["nll"], "n_dev": len(dev_rows), "thinking_flag": thinking}
    (out_dir / "dev_fits.json").write_text(json.dumps(fits, indent=2) + "\n", encoding="utf-8")
    print("dev fit:", fits)

    for ds in DATASETS:
        run_items(bundle, args.model, select_items(items, splits, ds, "test_original"), "test", winner)
    print("done", args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
