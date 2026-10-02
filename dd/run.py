"""Orchestrate resumable cell runs.

One JSON line per finished (item, cell) is appended to results/raw and skipped
on resume. Test is never used to choose a template, a temperature, or tau.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import yaml

from dd.baselines import load_causal_model, read_causal_batch
from dd.canvas import TEMPLATES, build_canvas, encode_prompt
from dd.data import PROTOCOL, ensure_items, load_items, prompt_for, select_items
from dd.decode import nfe_for, run_c0, run_c1, run_c2
from dd.device import (
    available_ram_bytes,
    budget_hours_for,
    cuda_free_bytes,
    model_fits,
    preferred_dtype,
    select_device,
    write_environment,
)
from dd.labels import ALL_LETTERS, resolve_end_tokens, resolve_letter_tokens, verify_mask_token
from dd.llada import AttentionMaskRejected, load_masked_model, load_tokenizer, probe_dtype, repo_revision
from dd.metrics import accuracy, expected_calibration_error
from dd.report import write_cloud

FORWARDS_PER_ITEM = 97
# C1 readout "step" t is taken on forward pass t, before that pass's unmasking. The T=4 cell's
# final read sees 4 finished unmask steps on pass 5, so the matching T=16 readout is step 5
# (same NFE, same number of completed steps), not step 4.
ANYTIME_READ_STEP = 5
ROOT = Path(__file__).resolve().parents[1]
# Protocol v1.1: default template before the re-smoke picks between v3e and v1e.
DEFAULT_TEMPLATE = "v3e"
PROTOCOL_FILE = ROOT / "results" / "protocol_v1_1.json"


def append_log(text: str) -> None:
    path = ROOT / "results" / "LOG.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["seed"] = int(config.get("seed", 1234))
    for cell in config["cells"]:
        if cell["T"] == "null":
            cell["T"] = None
        if cell["condition"] != "C0":
            if cell["T"] is None or int(cell["T"]) > int(cell["S"]):
                raise SystemExit(f"cell {cell} violates T <= S")
    return config


def chunks(items: list, size: int):
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def cell_file(
    dataset: str, split: str, model: str, condition: str, scratch: int, steps, template: str, suppress_eos: bool = False
) -> Path:
    step_label = "na" if steps is None else str(steps)
    suffix = "_noeos" if suppress_eos else ""
    name = f"{dataset}_{split}_{model}_{condition}_S{scratch}_T{step_label}_{template}{suffix}.jsonl"
    return ROOT / "results" / "raw" / name


def cell_path(bundle, dataset: str, split: str, condition: str, scratch: int, steps, template: str) -> Path:
    """Resume file for one cell. C1/C2 rows made with suppress_eos_in_scratch go to *_noeos files."""
    suppress = bool(getattr(bundle, "suppress_eos_in_scratch", False)) and condition != "C0"
    return cell_file(dataset, split, bundle.name, condition, int(scratch), steps, template, suppress_eos=suppress)


def resolve_suppress(config: dict) -> bool:
    """`suppress_eos_in_scratch`: true/false, or `protocol` = the main setting the re-smoke rule chose."""
    value = config.get("suppress_eos_in_scratch", False)
    if value == "protocol":
        if not PROTOCOL_FILE.exists():
            raise SystemExit("suppress_eos_in_scratch: protocol, but results/protocol_v1_1.json is missing; run the re-smoke first")
        return bool(json.loads(PROTOCOL_FILE.read_text(encoding="utf-8"))["suppress_eos_in_scratch"])
    return bool(value)


def read_done(path: Path) -> dict[str, dict]:
    """Finished rows by item id. A torn final line from a crash is skipped, so that item reruns."""
    done = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        lines = handle.read().splitlines()
    for number, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines) - 1:
                append_log(f"\nSkipped a torn final line in `{path.name}`; that item will rerun.\n")
                continue
            raise
        done[row["item_id"]] = row
    return done


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def frozen_template(config: dict | None = None) -> str:
    """Frozen template if any; otherwise the config's first template candidate (a model-specific
    default, e.g. v3e_dream), otherwise DEFAULT_TEMPLATE (LLaDA)."""
    path = ROOT / "results" / "template.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["template"]
    if config and config.get("template_candidates"):
        return list(config["template_candidates"])[0]
    return DEFAULT_TEMPLATE


def shortcut_enabled(config: dict) -> bool:
    flag = config.get("anytime_shortcut", "auto")
    if flag in (True, "on", "true"):
        return True
    if flag in (False, "off", "false", None):
        return False
    gate = ROOT / "results" / "gate2.json"
    if not gate.exists():
        return False
    return bool(json.loads(gate.read_text(encoding="utf-8")).get("shortcut_enabled"))


def resident_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return cuda_free_bytes(device)
    return available_ram_bytes()


# Forward passes per item by scratch length S for the PLAN Section 6.5 grid:
# C0 = 1 pass at S=0; C1 and C2 at T in {1, 4, 16} give 2 + 5 + 17 = 24 passes each, so 48 per S > 0.
PASSES_PER_ITEM_BY_S = {0: 1, 32: 48, 128: 48}
assert sum(PASSES_PER_ITEM_BY_S.values()) == FORWARDS_PER_ITEM
# The budget gate uses the p90-length timing (conservative); p50 is reported alongside.
PROJECTION_BASIS = "p90"


def projection(n_items: int, seconds_by_s: dict) -> float:
    """Hours for n_items over the full grid, given measured seconds per pass at each S."""
    per_item = sum(passes * float(seconds_by_s[str(scratch)]) for scratch, passes in PASSES_PER_ITEM_BY_S.items())
    return n_items * per_item / 3600.0


def count_items(splits: dict, which: str) -> int:
    total = 0
    for block in splits["datasets"].values():
        if block.get("dropped"):
            continue
        total += len(block.get(which) or [])
    return total


def argmax_letter(probs: dict[str, float]) -> str:
    letters = list(probs)
    return max(letters, key=lambda letter: (probs[letter], -letters.index(letter)))


def base_row(item: dict, split: str, model: str, cell: dict, template: str, device: str, dtype: str, seed: int) -> dict:
    return {
        "dataset": item["dataset"],
        "split": split,
        "item_id": item["item_id"],
        "model": model,
        "condition": cell["condition"],
        "S": int(cell["S"]),
        "T": cell["T"],
        "seed": seed,
        "template": template,
        "perm": item["perm"],
        "gold": item["gold"],
        "device": device,
        "dtype": dtype,
        "read_source": "native",
        "protocol": PROTOCOL,
    }


def finish_row(base: dict, result: dict, wall_ms: float) -> dict:
    row = dict(base)
    row.update(
        {
            "readouts": result["readouts"],
            "final_probs": result["final_probs"],
            "letter_logits": result["letter_logits"],
            # Final-read drift diagnostic. Gate 1 and summarize_rows read this key.
            "off_label_mass": result["off_label_mass"],
            "pred": result["pred"],
            "correct": result["pred"] == base["gold"],
            "nfe": result["nfe"],
            "wall_ms": wall_ms,
            "scratch_text": result.get("scratch_text", ""),
            # C1/C2 only: share of scratch tokens that are end-type ids (eos, pad, end-of-turn).
            "eos_pad_fraction": result.get("eos_pad_fraction"),
            "suppress_eos_in_scratch": result.get("suppress_eos_in_scratch", False),
        }
    )
    return row


def _time_forward(bundle, ids: list[int], warmup: int = 3, passes: int = 20) -> float:
    batch = torch.tensor([ids], dtype=torch.long)
    attn = torch.ones_like(batch)
    for _ in range(warmup):
        bundle.forward(batch, attn)
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(passes):
        bundle.forward(batch, attn)
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / passes


def prompt_length_items(tokenizer, items: list[dict], splits: dict) -> dict:
    """Items at the p50 and p90 chat-templated prompt length over every dev+test item."""
    wanted = set()
    for block in splits["datasets"].values():
        if not block.get("dropped"):
            wanted.update(block.get("dev") or [])
            wanted.update(block.get("test") or [])
    lengths = [(len(encode_prompt(tokenizer, prompt_for(item))), item) for item in items if item["item_id"] in wanted]
    ordered = sorted(length for length, _ in lengths)
    picked = {}
    for name, q in (("p50", 0.50), ("p90", 0.90)):
        target = ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]
        length, item = min(lengths, key=lambda pair: (abs(pair[0] - target), pair[1]["item_id"]))
        picked[name] = {"target_prompt_tokens": target, "prompt_tokens": length, "item_id": item["item_id"], "item": item}
    return picked


def measure_timing(bundle, items: list[dict], splits: dict, template: str = "v1") -> dict:
    """Pass time at the p50 and p90 prompt lengths plus the real canvas for S = 0, 32, 128.

    S=0 is the C0 canvas. S>0 is the C1 canvas (prompt + S masks + answer prefix + slot),
    the longest sequence either C1 or C2 feeds the model at that S.
    """
    picked = prompt_length_items(bundle.tokenizer, items, splits)
    per_length = {}
    for name, info in picked.items():
        block = {
            "target_prompt_tokens": info["target_prompt_tokens"],
            "prompt_tokens": info["prompt_tokens"],
            "item_id": info["item_id"],
            "seconds_by_s": {},
            "seq_len_by_s": {},
        }
        for scratch in PASSES_PER_ITEM_BY_S:
            condition = "C0" if scratch == 0 else "C1"
            canvas = build_canvas(bundle.tokenizer, prompt_for(info["item"]), condition, scratch, bundle.mask_id, template=template)
            block["seconds_by_s"][str(scratch)] = _time_forward(bundle, canvas.input_ids)
            block["seq_len_by_s"][str(scratch)] = len(canvas.input_ids)
        per_length[name] = block
    return {
        "per_length": per_length,
        "passes_per_item_by_s": {str(k): v for k, v in PASSES_PER_ITEM_BY_S.items()},
        "projection_basis": PROJECTION_BASIS,
        "warmup": 3,
        "measured_passes": 20,
        "device": bundle.device.type,
        "dtype": str(bundle.dtype).replace("torch.", ""),
    }


def write_timing(payload: dict) -> dict:
    path = ROOT / "results" / "timing.json"
    path.write_text(encoding="utf-8", data=json.dumps(payload, indent=2) + "\n")
    return payload


def projections_from_timing(timing: dict, splits: dict) -> dict:
    n_dev = count_items(splits, "dev")
    n_all = n_dev + count_items(splits, "test")
    by_length = {}
    for name, block in timing["per_length"].items():
        by_length[name] = {
            "dev_hours": projection(n_dev, block["seconds_by_s"]),
            "full_hours": projection(n_all, block["seconds_by_s"]),
        }
    basis = timing.get("projection_basis", PROJECTION_BASIS)
    return {
        "by_length": by_length,
        "basis": basis,
        "dev_hours": by_length[basis]["dev_hours"],
        "full_hours": by_length[basis]["full_hours"],
    }


def log_logicdiff() -> None:
    path = ROOT / "schedulers" / "logicdiff.py"
    if not path.exists():
        append_log(
            "\n## LogicDiff\n\n"
            "`schedulers/logicdiff.py` is absent. C3 is skipped. "
            "No approximation of the author's scheduler was implemented.\n"
        )


def dataset_items(config: dict, items: list[dict], splits: dict) -> dict[str, list[dict]]:
    limit = config.get("smoke_n") if config.get("phase") in {"smoke", "resmoke"} else None
    selected = {}
    for name in config["datasets"]:
        block = splits["datasets"].get(name)
        if not block or block.get("dropped") or not block.get(config["split"]):
            append_log(f"\nDataset `{name}` is dropped or has no `{config['split']}` ids. Skipping.\n")
            continue
        # `split_ids` picks the id list (e.g. test_original for the v1.2 noeos ablation); rows keep `split`.
        selected[name] = select_items(items, splits, name, config.get("split_ids", config["split"]), limit=limit)
    return selected


def run_masked_batch(bundle, batch, cell, template: str):
    letters = [item["letters"] for item in batch]
    if bundle.kind == "causal" or cell["condition"] == "C0" and getattr(bundle, "kind", "masked") == "causal":
        raise RuntimeError("causal batches are handled separately")
    canvases = [
        build_canvas(
            bundle.tokenizer,
            prompt_for(item),
            cell["condition"],
            int(cell["S"]),
            bundle.mask_id,
            template=template,
        )
        for item in batch
    ]
    if cell["condition"] == "C0":
        return run_c0(bundle.forward, canvases, bundle.letter_tokens, letters, bundle.mask_id, bundle.pad_id)
    if cell["condition"] == "C1":
        return run_c1(
            bundle.forward,
            canvases,
            bundle.letter_tokens,
            letters,
            bundle.mask_id,
            bundle.pad_id,
            int(cell["T"]),
            tokenizer=bundle.tokenizer,
            end_ids=list(bundle.end_tokens),
            suppress_end_in_scratch=bool(getattr(bundle, "suppress_eos_in_scratch", False)),
        )
    if cell["condition"] == "C2":
        return run_c2(
            bundle.forward,
            canvases,
            bundle.letter_tokens,
            letters,
            bundle.mask_id,
            bundle.pad_id,
            int(cell["T"]),
            bundle.tokenizer,
            template,
            end_ids=list(bundle.end_tokens),
            suppress_end_in_scratch=bool(getattr(bundle, "suppress_eos_in_scratch", False)),
        )
    raise RuntimeError(f"condition {cell['condition']} is not implemented in this runner")


def execute_native(bundle, items, split, cell, template, batch_size, seed: int) -> Path:
    if not items:
        return cell_path(bundle, "none", split, cell["condition"], cell["S"], cell["T"], template)
    path = cell_path(bundle, items[0]["dataset"], split, cell["condition"], cell["S"], cell["T"], template)
    done = read_done(path)
    pending = [item for item in items if item["item_id"] not in done]
    dtype_name = str(bundle.dtype).replace("torch.", "")
    device_name = bundle.device.type
    current_batch = batch_size if bundle.attention_mask_ok else 1
    index = 0
    while index < len(pending):
        batch = pending[index : index + current_batch]
        start = time.perf_counter()
        try:
            results = run_masked_batch(bundle, batch, cell, template)
        except AttentionMaskRejected:
            if current_batch == 1:
                raise
            append_log(
                "\nLLaDA forward rejected `attention_mask`. "
                "Falling back to batch size 1 for the rest of this process.\n"
            )
            bundle.attention_mask_ok = False
            current_batch = 1
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        share = elapsed_ms / max(len(batch), 1)
        for item, result in zip(batch, results):
            append_row(
                path,
                finish_row(base_row(item, split, bundle.name, cell, template, device_name, dtype_name, seed), result, share),
            )
        index += len(batch)
    return path


def synthesize_from_t16(bundle, items, split, cell, template, seed: int) -> None:
    """Write nominal T=4 rows from readout ANYTIME_READ_STEP of an already finished T=16 run."""
    source = cell_path(bundle, items[0]["dataset"], split, "C1", cell["S"], 16, template)
    target = cell_path(bundle, items[0]["dataset"], split, "C1", cell["S"], 4, template)
    source_rows = read_done(source)
    done = read_done(target)
    for item in items:
        if item["item_id"] in done or item["item_id"] not in source_rows:
            continue
        origin = source_rows[item["item_id"]]
        readout = next(
            (step for step in origin["readouts"] if step.get("step") == ANYTIME_READ_STEP and not step.get("final")),
            None,
        )
        if readout is None:
            continue
        pred = argmax_letter(readout["probs"])
        row = base_row(item, split, bundle.name, {"condition": "C1", "S": cell["S"], "T": 4}, template, origin["device"], origin["dtype"], seed)
        row.update(
            {
                "readouts": [readout],
                "final_probs": readout["probs"],
                "letter_logits": readout["letter_logits"],
                "pred": pred,
                "correct": pred == item["gold"],
                "nfe": nfe_for("C1", 4),
                "wall_ms": None,
                "scratch_text": origin.get("scratch_text", ""),
                "off_label_mass": readout["off_label_mass"],
                "eos_pad_fraction": origin.get("eos_pad_fraction"),
                "suppress_eos_in_scratch": origin.get("suppress_eos_in_scratch", False),
                "read_source": f"anytime_step_{ANYTIME_READ_STEP}",
                "source_cell": "C1 S=%s T=16 readout step %d" % (cell["S"], ANYTIME_READ_STEP),
            }
        )
        append_row(target, row)


def model_accepts(config_model: dict, cell: dict) -> bool:
    allowed = config_model.get("conditions")
    return allowed is None or cell["condition"] in allowed


def run_grid(config: dict, bundles: list, work: dict[str, list[dict]], template: str, shortcut: bool) -> None:
    cells = list(config["cells"])
    if shortcut:
        cells = sorted(cells, key=lambda cell: (cell["condition"], int(cell["S"]), -int(cell["T"] or 0)))
    for bundle in bundles:
        for name, items in work.items():
            if not items:
                continue
            for cell in cells:
                if not model_accepts(bundle.spec, cell):
                    continue
                if cell["condition"] == "C0" and bundle.kind == "causal":
                    run_causal_cell(bundle, items, config["split"], cell, template, config["seed"])
                    continue
                if bundle.kind != "masked":
                    continue
                if (
                    shortcut
                    and cell["condition"] == "C1"
                    and cell["T"] == 4
                ):
                    synthesize_from_t16(bundle, items, config["split"], cell, template, config["seed"])
                    missing = _missing_shortcut(bundle, items, config["split"], cell, template)
                    if missing:
                        append_log(
                            f"\nAnytime shortcut could not cover {len(missing)} `{name}` items "
                            f"for C1 S={cell['S']} T=4. Those items are run natively.\n"
                        )
                        execute_native(bundle, missing, config["split"], cell, template, config.get("batch_size", 1), config["seed"])
                    continue
                execute_native(bundle, items, config["split"], cell, template, config.get("batch_size", 1), config["seed"])


def _missing_shortcut(bundle, items, split, cell, template) -> list:
    target = read_done(cell_path(bundle, items[0]["dataset"], split, "C1", cell["S"], 4, template))
    return [item for item in items if item["item_id"] not in target]


def run_causal_cell(bundle, items, split, cell, template, seed: int) -> None:
    path = cell_path(bundle, items[0]["dataset"], split, "C0", 0, None, template)
    done = read_done(path)
    pending = [item for item in items if item["item_id"] not in done]
    # Causal read: the next token after the prefix. The <|eot_id|> slot suffix has no causal analogue.
    prefix = TEMPLATES[template]["c0_prefix"]
    dtype_name = str(bundle.dtype).replace("torch.", "")
    for batch in chunks(pending, 1):
        start = time.perf_counter()
        results = read_causal_batch(bundle, [prompt_for(item) for item in batch], [item["letters"] for item in batch], prefix)
        wall = (time.perf_counter() - start) * 1000.0
        for item, result in zip(batch, results):
            append_row(
                path,
                finish_row(
                    base_row(item, split, bundle.name, {"condition": "C0", "S": 0, "T": None}, template, bundle.device.type, dtype_name, seed),
                    result,
                    wall,
                ),
            )


class Bundle:
    def __init__(self, spec, kind, model):
        self.spec = spec
        self.kind = kind
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)


def load_bundles(config: dict, device: torch.device, kinds: tuple[str, ...] = ("masked", "causal")) -> list:
    """Load the configured models of the given kinds. Two 8B models do not fit one 24GB GPU together,
    so main() loads the masked model, runs it, frees it, and only then loads the causal baseline."""
    bundles = []
    dtypes = preferred_dtype(device, config.get("dtype", "bfloat16"))
    suppress = resolve_suppress(config)
    for spec in config["models"]:
        kind = spec["kind"]
        if kind not in kinds:
            continue
        if kind == "masked":
            def load_fn(dtype, spec=spec):
                return load_masked_model(
                    spec["hf_id"],
                    spec["name"],
                    device,
                    dtype,
                    mask_id=int(spec.get("mask_id", 126336)),
                    expected_mask_token=spec.get("expected_mask_token", "<|mdm_mask|>"),
                    revision=spec.get("revision"),
                    logit_shift=bool(spec.get("logit_shift", False)),
                )

            model, mask_report, dtype = probe_dtype(load_fn, dtypes)
            append_log(
                "\n## Mask token VERIFY\n\n```json\n"
                + json.dumps(mask_report, indent=2)
                + "\n```\n\n"
                + f"Letter variants: `{json.dumps(model.letter_tokens)}`\n"
                + f"Frozen dtype: `{dtype}`.\n"
                + "End-type tokens (counted in eos_pad_fraction; suppressed in scratch only when enabled): "
                + f"`{json.dumps({str(k): v for k, v in model.end_tokens.items()})}`\n"
                + f"suppress_eos_in_scratch: `{suppress}`.\n"
            )
            bundle = Bundle(spec, "masked", model)
            bundle.suppress_eos_in_scratch = suppress
            bundles.append(bundle)
        elif kind == "causal":
            model, thinking = load_causal_model(spec["hf_id"], spec["name"], device, dtypes[0])
            append_log(f"\n## Causal baseline\n\n`{spec['hf_id']}` thinking flag: {thinking}. dtype `{dtypes[0]}`.\n")
            bundles.append(Bundle(spec, "causal", model))
        else:
            raise SystemExit(f"unknown model kind {kind}")
    return bundles


def verify_tokenizer_only(config: dict) -> None:
    """Letter and mask checks that do not download weights."""
    for spec in config["models"]:
        if spec["kind"] != "masked":
            continue
        tokenizer = load_tokenizer(spec["hf_id"], spec.get("revision"))
        report = verify_mask_token(
            tokenizer,
            int(spec.get("mask_id", 126336)),
            spec.get("expected_mask_token", "<|mdm_mask|>"),
        )
        try:
            letters = resolve_letter_tokens(tokenizer, ALL_LETTERS)
            letter_error = None
        except Exception as exc:
            letters = None
            letter_error = f"{type(exc).__name__}: {exc}"
        try:
            end_tokens = {str(k): v for k, v in resolve_end_tokens(tokenizer).items()}
        except Exception as exc:
            end_tokens = f"{type(exc).__name__}: {exc}"
        revision = spec.get("revision") or repo_revision(spec["hf_id"])
        append_log(
            "\n## Tokenizer VERIFY (no weights)\n\n"
            + f"Revision: `{revision}`\n\n```json\n"
            + json.dumps({"mask": report, "letters": letters, "letter_error": letter_error, "end_tokens": end_tokens}, indent=2)
            + "\n```\n"
        )


def gate1_from_rows(config: dict, template: str, model_name: str = "llada-8b-instruct") -> dict:
    """Warn-only accuracy check and the off-label median on smoke C0 rows."""
    dataset_reports = {}
    off_values = []
    for name in config["datasets"]:
        path = cell_file(name, config["split"], model_name, "C0", 0, None, template)
        rows = list(read_done(path).values())
        if not rows:
            continue
        correct = [row["correct"] for row in rows]
        acc = accuracy(correct)
        # Chance is 1/n_options. Smoke items in one dataset can share n_options;
        # use the mean so mixed jagged types stay honest.
        items = load_items(ROOT)
        by_id = {item["item_id"]: item for item in items}
        chance = statistics.fmean(1.0 / by_id[row["item_id"]]["n_options"] for row in rows)
        dataset_reports[name] = {
            "n": len(rows),
            "accuracy": acc,
            "chance": chance,
            "beats_chance_by_10_points": acc >= chance + 0.10,
        }
        off_values.extend(row["off_label_mass"] for row in rows if "off_label_mass" in row)
        if name in {"arc_c", "boolq"} and acc < chance + 0.10:
            append_log(
                f"\n**Gate 1 warning:** C0 accuracy on `{name}` is {acc:.4f} "
                f"versus chance {chance:.4f} (n={len(rows)}). "
                "The plan marks this warn-only on 20 items.\n"
            )
    median_off = statistics.median(off_values) if off_values else None
    return {"datasets": dataset_reports, "median_off_label_mass": median_off}


def maybe_search_templates(config: dict, bundle, work, gate: dict) -> str:
    median_off = gate.get("median_off_label_mass")
    if median_off is None:
        return "v1"
    if median_off < 0.5:
        payload = {"template": "v1", "median_off_label_mass": median_off, "candidates": {"v1": median_off}}
        (ROOT / "results" / "template.json").write_text(encoding="utf-8", data=json.dumps(payload, indent=2) + "\n")
        append_log(f"\n## Gate 1 template\n\nFroze `v1`. Median C0 off_label_mass `{median_off:.4f}` is below 0.5.\n")
        return "v1"
    append_log(
        f"\n## Gate 1 template\n\nMedian C0 off_label_mass `{median_off:.4f}` is not below 0.5. "
        "Trying v2 and v3 on the same smoke items.\n"
    )
    candidates = {"v1": median_off}
    for template in ("v2", "v3"):
        for items in work.values():
            if not items:
                continue
            execute_native(bundle, items, config["split"], {"condition": "C0", "S": 0, "T": None}, template, 1, config["seed"])
        values = []
        for name in work:
            path = cell_file(name, config["split"], bundle.name, "C0", 0, None, template)
            values.extend(row["off_label_mass"] for row in read_done(path).values())
        candidates[template] = statistics.median(values) if values else math.inf
    winner = min(candidates, key=lambda key: candidates[key])
    payload = {"template": winner, "median_off_label_mass": candidates[winner], "candidates": candidates}
    (ROOT / "results" / "template.json").write_text(encoding="utf-8", data=json.dumps(payload, indent=2) + "\n")
    append_log(f"\nFroze `{winner}` with median off_label_mass `{candidates[winner]}`. Candidates: `{candidates}`.\n")
    return winner


def run_anytime(config: dict, bundle, items: list[dict], splits: dict, template: str) -> int:
    arc = select_items(items, splits, "arc_c", "dev", limit=25) if not splits["datasets"].get("arc_c", {}).get("dropped") else []
    jagged = select_items(items, splits, "jagged", "dev", limit=25) if not splits["datasets"].get("jagged", {}).get("dropped") else []
    sample = arc + jagged
    if len(sample) < 50:
        append_log(f"\n## Gate 2\n\nAnytime sample has {len(sample)} items (expected 50). Computing the gate on what is available.\n")
    cell4 = {"condition": "C1", "S": 32, "T": 4}
    cell16 = {"condition": "C1", "S": 32, "T": 16}
    by_dataset: dict[str, list] = {}
    for item in sample:
        by_dataset.setdefault(item["dataset"], []).append(item)
    for group in by_dataset.values():
        execute_native(bundle, group, "dev", cell4, template, 1, config["seed"])
        execute_native(bundle, group, "dev", cell16, template, 1, config["seed"])
    native = {}
    long = {}
    for dataset in by_dataset:
        native.update(read_done(cell_path(bundle, dataset, "dev", "C1", 32, 4, template)))
        long.update(read_done(cell_path(bundle, dataset, "dev", "C1", 32, 16, template)))
    paired = []
    for item_id, row in native.items():
        if item_id not in long:
            continue
        step4 = next((step for step in long[item_id]["readouts"] if step.get("step") == ANYTIME_READ_STEP and not step.get("final")), None)
        if step4 is None:
            continue
        paired.append((row, step4, long[item_id]["gold"]))
    if not paired:
        append_log("\n## Gate 2\n\nNo paired rows. Shortcut stays off.\n")
        payload = {"n": 0, "passed": False, "shortcut_enabled": False}
        (ROOT / "results" / "gate2.json").write_text(encoding="utf-8", data=json.dumps(payload, indent=2) + "\n")
        return 2
    agreement = statistics.fmean(row["pred"] == argmax_letter(step["probs"]) for row, step, _gold in paired)
    ece_native = expected_calibration_error(
        [row["correct"] for row, _step, _gold in paired],
        [max(row["final_probs"].values()) for row, _step, _gold in paired],
    )
    ece_step = expected_calibration_error(
        [argmax_letter(step["probs"]) == gold for _row, step, gold in paired],
        [max(step["probs"].values()) for _row, step, _gold in paired],
    )
    diff = abs(ece_native - ece_step)
    passed = agreement >= 0.95 and diff < 0.02
    payload = {
        "n": len(paired),
        "argmax_agreement": agreement,
        "ece_native_T4": ece_native,
        "ece_T16_step4": ece_step,
        "ece_abs_diff": diff,
        "passed": passed,
        "shortcut_enabled": passed,
        "rule": "agreement >= 0.95 and absolute ECE difference < 0.02",
    }
    (ROOT / "results" / "gate2.json").write_text(encoding="utf-8", data=json.dumps(payload, indent=2) + "\n")
    append_log("\n## Gate 2\n\n```json\n" + json.dumps(payload, indent=2) + "\n```\n")
    return 0


def _cell_rows(bundle, work, condition, scratch, steps, template) -> list[dict]:
    rows = []
    for name in work:
        rows.extend(read_done(cell_path(bundle, name, "dev", condition, scratch, steps, template)).values())
    return rows


def run_resmoke(config: dict, bundle, work: dict[str, list[dict]]) -> int:
    """Protocol v1.1 re-smoke with the lead's pre-committed rules (29 Sep 2026).

    1. C0 with v3e and v1e on the same smoke items. Template = lower median C0 off_label_mass.
       Gate: it must be < 0.5, otherwise stop and report the top-5 slot tokens.
    2. C1(32,4) and C2(32,4) with the winner, suppress_eos_in_scratch off, then on.
    3. Main suppression setting: ON for both if either condition's median eos_pad_fraction with the
       flag off is > 0.3; otherwise OFF. Same setting for C1 and C2.
    """
    seed = config["seed"]
    c0 = {"condition": "C0", "S": 0, "T": None}
    medians = {}
    candidates = list(config.get("template_candidates", ["v3e", "v1e"]))
    for template in candidates:
        for items in work.values():
            execute_native(bundle, items, "dev", c0, template, 1, seed)
        values = [row["off_label_mass"] for row in _cell_rows(bundle, work, "C0", 0, None, template)]
        medians[template] = statistics.median(values)
    winner = min(medians, key=lambda key: (medians[key], key != candidates[0]))
    template_gate = medians[winner] < 0.5
    (ROOT / "results" / "template.json").write_text(encoding="utf-8", data=json.dumps(
        {"template": winner, "median_off_label_mass": medians[winner], "candidates": medians,
         "rule": "lowest median C0 off_label_mass; gate < 0.5", "protocol": PROTOCOL}, indent=2) + "\n")
    append_log(f"\n## Re-smoke template (protocol v1.1)\n\nMedian C0 off_label_mass: `{medians}`. Winner `{winner}`; gate < 0.5: `{template_gate}`.\n")
    if not template_gate:
        payload = {"passed": False, "reason": "no template reached median C0 off_label_mass < 0.5", "medians": medians}
        (ROOT / "results" / "resmoke.json").write_text(encoding="utf-8", data=json.dumps(payload, indent=2) + "\n")
        return 3

    eos = {}
    run_rule = bool(config.get("resmoke_suppress_rule", True))
    for suppress in ((False, True) if run_rule else (False,)):
        bundle.suppress_eos_in_scratch = suppress
        for condition in ("C1", "C2"):
            cell = {"condition": condition, "S": 32, "T": 4}
            for items in work.values():
                execute_native(bundle, items, "dev", cell, winner, 1, seed)
            values = [row["eos_pad_fraction"] for row in _cell_rows(bundle, work, condition, 32, 4, winner)]
            eos[f"{condition}_{'on' if suppress else 'off'}"] = {"median": statistics.median(values), "max": max(values)}
    # v1.3 Dream: the rule is not applied; suppression stays OFF like LLaDA's main setting.
    main_suppress = (eos["C1_off"]["median"] > 0.3 or eos["C2_off"]["median"] > 0.3) if run_rule else False
    bundle.suppress_eos_in_scratch = main_suppress

    nan_rows = 0
    for condition, scratch, steps in (("C0", 0, None), ("C1", 32, 4), ("C2", 32, 4)):
        for row in _cell_rows(bundle, work, condition, scratch, steps, winner):
            values = list(row["final_probs"].values()) + [row["off_label_mass"]]
            nan_rows += any(not math.isfinite(value) for value in values)
    gate1 = gate1_from_rows(config, winner, bundle.name)
    passed = template_gate and nan_rows == 0
    protocol = {
        "protocol": PROTOCOL,
        "template": winner,
        "suppress_eos_in_scratch": main_suppress,
        "eos_pad_fraction": eos,
        "c0_off_label_medians": medians,
        "suppress_rule": "ON for C1 and C2 if either has median eos_pad_fraction > 0.3 with the flag off",
        "nan_rows": nan_rows,
        "gate1": gate1,
        "passed": passed,
    }
    PROTOCOL_FILE.write_text(encoding="utf-8", data=json.dumps(protocol, indent=2) + "\n")
    (ROOT / "results" / "resmoke.json").write_text(encoding="utf-8", data=json.dumps(protocol, indent=2) + "\n")
    append_log("\n## Re-smoke decisions (protocol v1.1)\n\n```json\n" + json.dumps(protocol, indent=2) + "\n```\n")
    return 0 if passed else 3


def stop_for_budget(reason: str, environment: dict, projection_hours: float | None, budget: float | None) -> int:
    append_log(f"\n## Budget stop\n\n{reason}\n")
    write_cloud(ROOT, reason, environment, projection_hours=projection_hours, budget_hours=budget)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deliberation-dial sweep.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-anytime", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)

    import os

    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config = load_config(Path(args.config))
    if config["seed"] != 1234:
        append_log(f"\nConfig seed is {config['seed']}. The plan fixes seed 1234. Using the config value and recording it.\n")
    device = select_device()
    budget = budget_hours_for(device, config.get("device_budget_hours"))
    environment = write_environment(
        ROOT / "results" / "environment.json",
        {"budget_hours": budget, "config": str(args.config), "phase": config.get("phase")},
    )
    append_log(
        f"\n## Run\n\n- config: `{args.config}`\n- phase: `{config.get('phase')}`\n"
        f"- device: `{device.type}`\n- budget hours: `{budget}`\n"
    )
    log_logicdiff()
    prepared = ensure_items(ROOT, seed=config["seed"], load_external=True)
    if not prepared["reused"]:
        append_log("\n## Dataset VERIFY\n\n```json\n" + json.dumps(prepared["payload"]["load_log"], indent=2) + "\n```\n")
        append_log(
            f"\nJagged date_compare yes/no: {prepared['payload'].get('jagged_date_yes')} / "
            f"{prepared['payload'].get('jagged_date_n')}. "
            "75 items cannot split 50/50; the generator uses 38 yes and 37 no.\n"
        )
    splits = prepared["payload"]
    items = load_items(ROOT)
    if args.prepare_only:
        append_log("\n`--prepare-only` stops before any model load.\n")
        return 0

    phase = config.get("phase")
    if phase in {"dev", "test"} and device.type == "cpu":
        return stop_for_budget(
            "CPU is for smoke tests only (Section 9.1). Dev and test sweeps were not started.",
            environment,
            projection_hours=None,
            budget=budget,
        )

    fits = model_fits(resident_bytes(device))
    if not fits:
        reason = (
            f"Refusing to download 8B weights. device={device.type}, "
            f"resident_bytes={resident_bytes(device)}, "
            f"bf16 weight estimate={environment.get('bf16_weight_bytes_estimate')} "
            f"with overhead 1.15. Gate 1 cannot run a forward pass on this machine."
        )
        append_log(f"\n## Gate 1\n\nFailed. {reason}\n")
        try:
            verify_tokenizer_only(config)
        except Exception as exc:
            append_log(f"\nTokenizer VERIFY failed: `{type(exc).__name__}: {exc}`\n")
        (ROOT / "results" / "gate1.json").write_text(encoding="utf-8", data=
            json.dumps({"passed": False, "reason": reason}, indent=2) + "\n"
        )
        return stop_for_budget(reason, environment, projection_hours=None, budget=budget) or 2

    if int(config.get("batch_size", 1)) != 1:
        raise SystemExit("protocol v1.1 fixes batch size 1 for every run")
    bundles = load_bundles(config, device, kinds=("masked",))
    template = frozen_template(config)
    if not (ROOT / "results" / "template.json").exists() and phase not in {"smoke", "resmoke"}:
        append_log(f"\nNo frozen template. Using `{DEFAULT_TEMPLATE}` and recording the deviation.\n")

    masked = next((bundle for bundle in bundles if bundle.kind == "masked"), None)
    timing_path = ROOT / "results" / "timing.json"
    if masked is not None and not timing_path.exists():
        timing = write_timing(measure_timing(masked, items, splits, template))
        proj = projections_from_timing(timing, splits)
        append_log(
            "\n## Timing gate\n\nPass time at the p50 and p90 prompt lengths plus the canvas for S = 0, 32, 128. "
            f"The budget check uses `{proj['basis']}`.\n\n```json\n"
            + json.dumps({"timing": timing, "projections": proj}, indent=2)
            + "\n```\n"
        )
        for name, block in proj["by_length"].items():
            print(f"[{name}] projected dev hours {block['dev_hours']:.3f}, full-sweep hours {block['full_hours']:.3f}")
        print(f"Device budget hours: {budget}")

    projected_full = None
    projected_dev = None
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if "per_length" not in timing:
            raise SystemExit("results/timing.json is from the old fixed-600-token gate; delete it and rerun Phase 1")
        proj = projections_from_timing(timing, splits)
        projected_full = proj["full_hours"]
        projected_dev = proj["dev_hours"]
        print(f"Projected full-sweep hours ({proj['basis']}): {projected_full:.3f} (budget {budget})")

    if phase == "resmoke":
        if masked is None:
            raise SystemExit("the re-smoke needs the masked model")
        return run_resmoke(config, masked, dataset_items(config, items, splits))
    if args.validate_anytime:
        if masked is None:
            raise SystemExit("anytime validation needs the masked model")
        return run_anytime(config, masked, items, splits, template)

    if phase == "dev" and projected_dev is not None and budget is not None and projected_dev > budget:
        return stop_for_budget(
            f"Dev sweep projects to {projected_dev:.2f} h, over the {budget:.0f} h budget. Not started.",
            environment,
            projected_full,
            budget,
        )
    if phase == "test":
        if not (ROOT / "results" / "dev_fits.json").exists():
            append_log("\n## Gate 4\n\nRefusing the test sweep: `results/dev_fits.json` is missing. Tau and temperatures would otherwise be fit on test.\n")
            return 2
        if projected_full is not None and budget is not None and projected_full > budget:
            return stop_for_budget(
                f"Full sweep projects to {projected_full:.2f} h, over the {budget:.0f} h budget. "
                "Phases 1-3 stay the finished portion. Test was not started.",
                environment,
                projected_full,
                budget,
            )

    work = dataset_items(config, items, splits)
    # Shortcut rows come from the T=16 file with the same suppression flag (cell_path), so they match.
    shortcut = shortcut_enabled(config) and phase != "smoke"
    if phase != "smoke":
        append_log(f"\nAnytime shortcut for this run: `{shortcut}`. Batch size 1.\n")
    run_grid(config, bundles, work, template if phase != "smoke" else "v1", shortcut=shortcut)
    if any(spec["kind"] == "causal" for spec in config["models"]):
        # Free the masked model before loading the causal baseline (two 8B models do not fit on an L4).
        import gc

        bundles = None
        masked = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        causal = load_bundles(config, device, kinds=("causal",))
        run_grid(config, causal, work, template, shortcut=False)

    if phase == "smoke" and masked is not None:
        gate = gate1_from_rows(config, "v1")
        maybe_search_templates(config, masked, work, gate)
        gate["passed_model_forwards"] = True
        (ROOT / "results" / "gate1.json").write_text(encoding="utf-8", data=json.dumps(gate, indent=2) + "\n")
        append_log("\n## Gate 1\n\n```json\n" + json.dumps(gate, indent=2) + "\n```\n")

    if phase == "dev":
        from dd.report import write_dev_fits

        fits_path = write_dev_fits(ROOT)
        append_log(f"\n## Gate 3\n\nDev fits written to `{fits_path}`.\n")
        if projected_full is not None and budget is not None and projected_full > budget:
            return stop_for_budget(
                f"Dev sweep finished. Full sweep projects to {projected_full:.2f} h, "
                f"over the {budget:.0f} h budget, so Phase 4 was not started.",
                environment,
                projected_full,
                budget,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
