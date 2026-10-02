"""v1.2 test-set expansion (registered before any test run). Run once: python -m dd.expand

Keeps every existing dev and test item. Adds never-seen items, chosen by a fixed seed, so that
test = 1000 BoolQ, 1000 StrategyQA, 1000 ARC-C (or all remaining if fewer), 600 jagged.
Dev is unchanged. New jagged items come from the same generator with a different seed and the
id prefix "jagged2"; any item whose (state, question) duplicates an existing jagged item is dropped,
and a type that falls short is topped up from a second fixed-seed batch (prefix "jagged3").

splits.json keeps `test_original` (the v1.0/v1.1 test ids), `test_added`, and `test`
(original then added). items.jsonl gains the new items; existing lines are byte-identical.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from dd.data import SEED, generate_jagged, load_external_datasets, verify_jagged_item

ROOT = Path(__file__).resolve().parents[1]
EXPANSION_SEED = SEED + 12  # 1246: "v1.2"; fixed before any new item was drawn
TEST_TARGETS = {"boolq": 1000, "strategyqa": 1000, "arc_c": 1000, "jagged": 600}
JAGGED_PER_TYPE = 90  # 360 new = 600 - 240; 90 per type keeps the four types balanced


def main() -> int:
    results = ROOT / "results"
    splits_path = results / "splits.json"
    items_path = results / "items.jsonl"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if "expansion" in splits:
        raise SystemExit("splits.json already has an expansion block; refusing to redraw")
    lines = items_path.read_text(encoding="utf-8").splitlines()
    existing = [json.loads(line) for line in lines if line.strip()]
    existing_by_id = {item["item_id"]: item for item in existing}

    external = load_external_datasets(SEED)
    pools = {name: {item["item_id"]: item for item in loaded["items"]} for name, loaded in external.items()}

    # Determinism check: every frozen item must regenerate identically.
    for item in existing:
        if item["dataset"] == "jagged":
            continue
        regenerated = pools[item["dataset"]].get(item["item_id"])
        if regenerated != item:
            raise SystemExit(f"frozen item {item['item_id']} does not regenerate identically; aborting")

    new_items: list[dict] = []
    record = {}
    for name in ("boolq", "strategyqa", "arc_c"):
        block = splits["datasets"][name]
        used = set(block["dev"]) | set(block["test"])
        remaining = sorted(set(pools[name]) - used)
        rng = random.Random(EXPANSION_SEED)
        rng.shuffle(remaining)
        need = max(0, TEST_TARGETS[name] - len(block["test"]))
        added = remaining[:need]
        new_items.extend(pools[name][item_id] for item_id in added)
        record[name] = {"available_unused": len(remaining), "added": len(added)}
        block["test_original"] = list(block["test"])
        block["test_added"] = added
        block["test"] = list(block["test"]) + added

    # Jagged: fresh generator seed and id prefix; drop content duplicates of existing items.
    seen = {(item["state"], item["question"]) for item in existing if item["dataset"] == "jagged"}
    fresh = generate_jagged(EXPANSION_SEED, per_type=JAGGED_PER_TYPE, id_prefix="jagged2")
    jagged_added, dropped = [], []
    for item in fresh:
        verify_jagged_item(item)
        key = (item["state"], item["question"])
        if key in seen:
            dropped.append(item["item_id"])
            continue
        seen.add(key)
        jagged_added.append(item)
    # Top up any type that lost items to de-duplication from a second fixed-seed batch of the same
    # type (id prefix "jagged3"), so the four types stay at JAGGED_PER_TYPE each.
    def jagged_type(item):
        return item["meta"]["type"]

    counts = {}
    for item in jagged_added:
        counts[jagged_type(item)] = counts.get(jagged_type(item), 0) + 1
    topped = []
    if any(counts.get(kind, 0) < JAGGED_PER_TYPE for kind in ("counting", "date_compare", "indirection", "numeric_compare")):
        extra = generate_jagged(EXPANSION_SEED + 1, per_type=JAGGED_PER_TYPE, id_prefix="jagged3")
        for item in extra:
            kind = jagged_type(item)
            if counts.get(kind, 0) >= JAGGED_PER_TYPE:
                continue
            verify_jagged_item(item)
            key = (item["state"], item["question"])
            if key in seen:
                dropped.append(item["item_id"])
                continue
            seen.add(key)
            jagged_added.append(item)
            topped.append(item["item_id"])
            counts[kind] = counts.get(kind, 0) + 1
    block = splits["datasets"]["jagged"]
    need = TEST_TARGETS["jagged"] - len(block["test"])
    jagged_added = jagged_added[:need]
    new_items.extend(jagged_added)
    block["test_original"] = list(block["test"])
    block["test_added"] = [item["item_id"] for item in jagged_added]
    block["test"] = list(block["test"]) + block["test_added"]
    record["jagged"] = {"generated": len(fresh), "dropped_duplicates": dropped, "topped_up": topped, "added": len(jagged_added), "per_type": counts}

    clash = [item["item_id"] for item in new_items if item["item_id"] in existing_by_id]
    if clash:
        raise SystemExit(f"new ids collide with existing items: {clash[:5]}")

    splits["expansion"] = {
        "version": "v1.2",
        "seed": EXPANSION_SEED,
        "test_targets": TEST_TARGETS,
        "jagged_per_type": JAGGED_PER_TYPE,
        "record": record,
        "note": "Added before any test run; dev unchanged; test_original is the v1.1 test set.",
    }
    splits["n_items_written"] = len(existing) + len(new_items)

    all_items = existing + new_items
    all_items.sort(key=lambda item: (item["dataset"], item["item_id"]))
    new_lines = [json.dumps(item, ensure_ascii=False) for item in all_items]
    # Existing lines must survive byte-for-byte.
    if not set(lines) <= set(new_lines):
        raise SystemExit("an existing items.jsonl line changed; aborting")
    items_path.write_text(encoding="utf-8", data="\n".join(new_lines) + "\n")
    splits_path.write_text(encoding="utf-8", data=json.dumps(splits, indent=2) + "\n")
    summary = {name: {"dev": len(b["dev"]), "test": len(b["test"]), "test_original": len(b["test_original"])}
               for name, b in splits["datasets"].items()}
    print(json.dumps({"summary": summary, "record": record}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
