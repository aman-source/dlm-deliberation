"""Dataset loaders, the jaggedness generator, prompts, and frozen splits.

Option order uses SHA-256 of the item id. Python's built-in hash() is salted
per process, so it cannot be the shuffle seed. That deviation is logged by
the caller.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from datetime import date
from pathlib import Path

SEED = 1234
FORMAT_VERSION = 1

COUNT_VOCAB = ("apple", "river", "stone", "cloud", "maple", "tiger")
NAMES = (
    "Ava",
    "Ben",
    "Cara",
    "Drew",
    "Eli",
    "Faye",
    "Gus",
    "Hana",
    "Ivan",
    "Jade",
    "Kyle",
    "Lina",
    "Milo",
    "Nora",
    "Omar",
    "Pia",
    "Quinn",
    "Rae",
    "Seth",
    "Tess",
)
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

PROMPT_TEMPLATE = """Read the state and answer the question.

State:
{state}

Question: {question}

Options:
{options}

Reply with the letter of the correct option after "Answer:"."""


# Protocol v1.1 (29 Sep 2026): the last prompt line changes for every condition. items.jsonl stays
# frozen (its `prompt` field is the v1.0 rendering); prompts are rendered at run time from the
# frozen state, question, and shuffled options, so options, permutations, and gold never change.
PROTOCOL = "v1.1"
PROMPT_LAST_LINE_V10 = 'Reply with the letter of the correct option after "Answer:".'
PROMPT_LAST_LINE_V11 = (
    'If you have space to think, think step by step first. '
    'Then finish with "The answer is" followed by the letter.'
)


def prompt_for(item: dict) -> str:
    """The protocol v1.1 user prompt for a frozen item."""
    prompt = render_prompt(item["state"], item["question"], item["options"])
    if not prompt.endswith(PROMPT_LAST_LINE_V10):
        raise AssertionError("prompt template changed; cannot swap the v1.1 last line")
    return prompt[: -len(PROMPT_LAST_LINE_V10)] + PROMPT_LAST_LINE_V11


def stable_hash(item_id: str) -> int:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def shuffle_options(item_id: str, options: list[str], gold_text: str, seed: int = SEED) -> dict:
    rng = random.Random(seed + stable_hash(item_id))
    indexed = list(enumerate(options))
    rng.shuffle(indexed)
    perm = [index for index, _ in indexed]
    shuffled = [text for _, text in indexed]
    if gold_text not in shuffled:
        raise ValueError(f"gold text {gold_text!r} missing from options for {item_id}")
    gold_letter = chr(ord("A") + shuffled.index(gold_text))
    return {
        "options": shuffled,
        "perm": perm,
        "gold": gold_letter,
        "gold_text": gold_text,
    }


def render_prompt(state: str, question: str, options: list[str]) -> str:
    lines = [f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options)]
    return PROMPT_TEMPLATE.format(state=state, question=question, options="\n".join(lines))


def make_item(
    dataset: str,
    item_id: str,
    state: str,
    question: str,
    options: list[str],
    gold_text: str,
    seed: int = SEED,
    meta: dict | None = None,
) -> dict:
    shuffled = shuffle_options(item_id, options, gold_text, seed=seed)
    prompt = render_prompt(state, question, shuffled["options"])
    return {
        "dataset": dataset,
        "item_id": item_id,
        "state": state,
        "question": question,
        "options": shuffled["options"],
        "perm": shuffled["perm"],
        "gold": shuffled["gold"],
        "gold_text": shuffled["gold_text"],
        "prompt": prompt,
        "n_options": len(shuffled["options"]),
        "letters": [chr(ord("A") + i) for i in range(len(shuffled["options"]))],
        "meta": meta or {},
    }


def truncate_words(text: str, n_words: int = 300) -> str:
    words = text.split()
    if len(words) <= n_words:
        return text.strip()
    return " ".join(words[:n_words])


def _content_id(prefix: str, *parts: str) -> str:
    raw = "\n".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def _yes_no_item(dataset: str, item_id: str, state: str, question: str, answer_yes: bool, seed: int) -> dict:
    return make_item(
        dataset,
        item_id,
        state,
        question,
        ["Yes", "No"],
        "Yes" if answer_yes else "No",
        seed=seed,
    )


def generate_jagged(seed: int = SEED, per_type: int = 75, id_prefix: str = "jagged") -> list[dict]:
    """75 items of each of four types. Ground truth is computed, then checked."""
    rng = random.Random(seed)
    items: list[dict] = []
    items.extend(_counting_items(rng, per_type, seed, id_prefix))
    items.extend(_date_items(rng, per_type, seed, id_prefix))
    items.extend(_indirection_items(rng, per_type, seed, id_prefix))
    items.extend(_numeric_items(rng, per_type, seed, id_prefix))
    if len(items) != per_type * 4:
        raise RuntimeError(f"expected {per_type * 4} jagged items, built {len(items)}")
    return items


def _counting_items(rng: random.Random, n: int, seed: int, id_prefix: str = "jagged") -> list[dict]:
    items = []
    for i in range(n):
        length = rng.randint(8, 15)
        gold_count = rng.randint(1, min(6, length))
        target = rng.choice(COUNT_VOCAB)
        others = [word for word in COUNT_VOCAB if word != target]
        words = [target] * gold_count + [rng.choice(others) for _ in range(length - gold_count)]
        rng.shuffle(words)
        if words.count(target) != gold_count:
            raise RuntimeError("counting generator lost the gold count")
        distractor_pool = [c for c in range(0, length + 1) if c != gold_count]
        rng.shuffle(distractor_pool)
        options = [str(gold_count)] + [str(c) for c in distractor_pool[:3]]
        state = " ".join(words)
        question = f"How many times does '{target}' appear in the list?"
        item_id = f"{id_prefix}-counting-{i:03d}"
        items.append(
            make_item(
                "jagged",
                item_id,
                state,
                question,
                options,
                str(gold_count),
                seed=seed,
                meta={"type": "counting", "target": target, "gold_count": gold_count},
            )
        )
    return items


def _render_date(value: date, style: str) -> str:
    if style == "iso":
        return value.isoformat()
    return f"{MONTHS[value.month - 1]} {value.day}, {value.year}"


def _date_items(rng: random.Random, n: int, seed: int, id_prefix: str = "jagged") -> list[dict]:
    """Yes/no balance is as even as an odd count allows: 38 yes and 37 no when n=75."""
    n_yes = (n + 1) // 2
    items = []
    for i in range(n):
        answer_yes = i < n_yes
        first = date(rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(1, 28))
        second = date(rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(1, 28))
        # Reject ties. Cap the loop so a bad seed cannot spin forever.
        for _ in range(20):
            if first != second:
                break
            second = date(rng.randint(1990, 2030), rng.randint(1, 12), rng.randint(1, 28))
        if first == second:
            second = date(first.year, first.month, 28 if first.day < 28 else 1)
        earlier, later = (first, second) if first < second else (second, first)
        if answer_yes:
            shown_first, shown_second = earlier, later
        else:
            shown_first, shown_second = later, earlier
        style_a = rng.choice(("iso", "long"))
        style_b = rng.choice(("iso", "long"))
        text_a = _render_date(shown_first, style_a)
        text_b = _render_date(shown_second, style_b)
        state = f"First date: {text_a}. Second date: {text_b}."
        question = "Is the first date earlier than the second?"
        item_id = f"{id_prefix}-date_compare-{i:03d}"
        items.append(
            _yes_no_item("jagged", item_id, state, question, answer_yes, seed)
            | {
                "meta": {
                    "type": "date_compare",
                    "first": shown_first.isoformat(),
                    "second": shown_second.isoformat(),
                }
            }
        )
    return items


def _indirection_items(rng: random.Random, n: int, seed: int, id_prefix: str = "jagged") -> list[dict]:
    items = []
    for i in range(n):
        n_facts = rng.choice((3, 4))
        people = rng.sample(NAMES, n_facts + 1)
        facts = [f"{people[j]}'s manager is {people[j + 1]}" for j in range(n_facts)]
        rng.shuffle(facts)
        start = people[0]
        gold = people[2]
        outsiders = [name for name in NAMES if name not in people]
        chain_others = [name for name in people if name != gold]
        rng.shuffle(chain_others)
        rng.shuffle(outsiders)
        distractors = (chain_others + outsiders)[:3]
        if len(set(distractors + [gold])) != 4:
            raise RuntimeError("indirection options are not unique")
        state = ". ".join(facts) + "."
        question = f"Who is {start}'s manager's manager?"
        item_id = f"{id_prefix}-indirection-{i:03d}"
        items.append(
            make_item(
                "jagged",
                item_id,
                state,
                question,
                [gold] + distractors,
                gold,
                seed=seed,
                meta={"type": "indirection", "chain": people, "n_facts": n_facts},
            )
        )
    return items


def _numeric_items(rng: random.Random, n: int, seed: int, id_prefix: str = "jagged") -> list[dict]:
    """Decimals where the fractional part, read as an integer, ranks the numbers backwards.

    Example from the plan: 3.9 versus 3.11. Numerically 3.9 is larger. Reading the
    fractional parts as integers says 11 > 9.
    """
    items = []
    for i in range(n):
        whole = rng.randint(1, 9)
        big_tenth = rng.randint(2, 9)
        # Two-digit fraction: integer value exceeds the tenths digit, but the
        # number itself is smaller (3.9 vs 3.11, 2.2 vs 2.11).
        lo = max(big_tenth + 1, 10)
        hi = big_tenth * 10 - 1
        fractional = rng.randint(lo, hi)
        larger = f"{whole}.{big_tenth}"
        smaller = f"{whole}.{fractional}"
        third_frac = rng.randint(0, big_tenth - 1)
        third = f"{whole}.{third_frac}"
        if third in {larger, smaller} or float(third) >= float(larger):
            third = f"{max(whole - 1, 0)}.5"
        if third in {larger, smaller}:
            third = "0.25"
        if not (float(larger) > float(smaller) and float(larger) > float(third)):
            raise RuntimeError(f"numeric gold is not the maximum: {larger}, {smaller}, {third}")
        if int(smaller.split(".")[1]) <= int(larger.split(".")[1]):
            raise RuntimeError("numeric pair does not trap integer-fraction comparison")
        state = f"The numbers are {larger}, {smaller}, and {third}."
        question = "Which number is larger?"
        item_id = f"{id_prefix}-numeric_compare-{i:03d}"
        items.append(
            make_item(
                "jagged",
                item_id,
                state,
                question,
                [larger, smaller, third],
                larger,
                seed=seed,
                meta={"type": "numeric_compare", "larger": larger, "smaller": smaller, "third": third},
            )
        )
    return items


def verify_jagged_item(item: dict) -> None:
    """Recompute the gold answer from the prompt state. Raises if it disagrees."""
    kind = item["meta"]["type"]
    gold_text = item["gold_text"]
    if kind == "counting":
        target = item["meta"]["target"]
        words = item["state"].split()
        if str(words.count(target)) != gold_text:
            raise AssertionError(f"{item['item_id']} counting mismatch")
    elif kind == "date_compare":
        match = re.search(r"First date: (.+)\. Second date: (.+)\.", item["state"])
        if not match:
            raise AssertionError("date state did not match the expected rendering")
        first = _parse_rendered_date(match.group(1))
        second = _parse_rendered_date(match.group(2))
        expected = "Yes" if first < second else "No"
        if expected != gold_text:
            raise AssertionError(f"{item['item_id']} date mismatch")
    elif kind == "indirection":
        facts = re.findall(r"(\w+)'s manager is (\w+)", item["state"])
        graph = {src: dst for src, dst in facts}
        question = item["question"]
        start = question[len("Who is ") :].split("'s manager's manager?")[0]
        manager = graph[start]
        managers_manager = graph[manager]
        if managers_manager != gold_text:
            raise AssertionError(f"{item['item_id']} indirection mismatch")
    elif kind == "numeric_compare":
        numbers = [float(piece) for piece in re.findall(r"\d+\.\d+", item["state"])]
        best = max(numbers)
        if float(gold_text) != best:
            raise AssertionError(f"{item['item_id']} numeric mismatch")
    else:
        raise AssertionError(f"unknown jagged type {kind}")
    letter_index = ord(item["gold"]) - ord("A")
    if item["options"][letter_index] != gold_text:
        raise AssertionError(f"{item['item_id']} gold letter does not point at the gold text")


def _parse_rendered_date(text: str) -> date:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        year, month, day = (int(part) for part in text.split("-"))
        return date(year, month, day)
    match = re.fullmatch(r"([A-Z][a-z]+) (\d{1,2}), (\d{4})", text)
    if not match:
        raise AssertionError(f"ambiguous or unknown date rendering: {text}")
    month = MONTHS.index(match.group(1)) + 1
    return date(int(match.group(3)), month, int(match.group(2)))


def load_external_datasets(seed: int = SEED) -> dict[str, dict]:
    """Load BoolQ, StrategyQA, and ARC-Challenge. Failures are returned, not invented."""
    return {
        "boolq": _load_boolq(seed),
        "strategyqa": _load_strategyqa(seed),
        "arc_c": _load_arc(seed),
    }


def _load_boolq(seed: int) -> dict:
    from datasets import load_dataset

    errors = []
    dataset = None
    source = None
    for name, kwargs in (
        ("google/boolq", {"split": "validation"}),
        ("boolq", {"split": "validation"}),
    ):
        try:
            dataset = load_dataset(name, **kwargs)
            source = name
            break
        except Exception as exc:  # dataset ids are a VERIFY item; record the failure
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if dataset is None:
        return {"items": [], "source": None, "error": " | ".join(errors), "split_used": None}
    items = []
    for row in dataset:
        question = str(row["question"]).strip()
        passage = truncate_words(str(row["passage"]))
        answer = row["answer"]
        if isinstance(answer, str):
            answer_yes = answer.strip().lower() == "yes" or answer.strip().lower() == "true"
        else:
            answer_yes = bool(answer)
        item_id = _content_id("boolq", question)
        items.append(_yes_no_item("boolq", item_id, passage, question, answer_yes, seed))
    return {"items": items, "source": source, "error": None, "split_used": "validation", "attempts": errors}


def _load_strategyqa(seed: int) -> dict:
    from datasets import load_dataset

    attempts = []
    for name in ("ChilleD/StrategyQA", "wics/strategy-qa"):
        try:
            dataset = load_dataset(name)
        except Exception as exc:
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        labeled = _strategyqa_labeled_rows(dataset)
        if labeled is None:
            attempts.append(f"{name}: loaded but no split had boolean answers")
            continue
        split_name, rows = labeled
        items = []
        for row in rows:
            question = str(row["question"]).strip()
            qid = str(row.get("qid") or _content_id("strategyqa", question))
            answer = row["answer"]
            if isinstance(answer, str):
                answer_yes = answer.strip().lower() in {"yes", "true"}
            else:
                answer_yes = bool(answer)
            items.append(
                _yes_no_item("strategyqa", f"strategyqa-{qid}", "(none)", question, answer_yes, seed)
            )
        return {
            "items": items,
            "source": name,
            "error": None,
            "split_used": split_name,
            "attempts": attempts,
        }
    return {
        "items": [],
        "source": None,
        "error": " | ".join(attempts) if attempts else "no StrategyQA source attempted",
        "split_used": None,
        "attempts": attempts,
    }


def _strategyqa_labeled_rows(dataset) -> tuple[str, list] | None:
    # Prefer train when it is labeled. Some mirrors ship an unlabeled test split.
    preferred = [name for name in ("train", "validation", "test") if name in dataset]
    preferred += [name for name in dataset.keys() if name not in preferred]
    for name in preferred:
        rows = list(dataset[name])
        if not rows or "answer" not in rows[0] or "question" not in rows[0]:
            continue
        if rows[0]["answer"] is None:
            continue
        return name, rows
    return None


def _load_arc(seed: int) -> dict:
    from datasets import load_dataset

    try:
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    except Exception as exc:
        return {
            "items": [],
            "source": "allenai/ai2_arc",
            "error": f"{type(exc).__name__}: {exc}",
            "split_used": "test",
        }
    items = []
    for row in dataset:
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        key = str(row["answerKey"])
        if key not in labels:
            raise RuntimeError(f"ARC answerKey {key} not in labels for {row.get('id')}")
        gold_text = texts[labels.index(key)]
        item_id = f"arc-{row['id']}"
        items.append(
            make_item(
                "arc_c",
                item_id,
                "(none)",
                str(row["question"]).strip(),
                [str(text) for text in texts],
                str(gold_text),
                seed=seed,
                meta={"answerKey": key, "source_labels": labels},
            )
        )
    return {"items": items, "source": "allenai/ai2_arc:ARC-Challenge", "error": None, "split_used": "test"}


def assign_splits(
    items_by_dataset: dict[str, list[dict]],
    *,
    seed: int = SEED,
    targets: dict[str, int] | None = None,
) -> dict:
    """Shuffle item ids with Random(seed) per dataset, then take a 20/80 dev/test cut.

    `targets` is the total subsample size (dev + test) before the cut. Jagged uses
    every generated item. StrategyQA uses up to 500 labeled rows.
    """
    targets = targets or {"boolq": 500, "strategyqa": 500, "arc_c": 500, "jagged": 300}
    splits: dict[str, dict] = {}
    for name, items in items_by_dataset.items():
        if not items:
            splits[name] = {"dev": [], "test": [], "dropped": True}
            continue
        rng = random.Random(seed)
        ids = sorted({item["item_id"] for item in items})
        if len(ids) != len(items):
            raise RuntimeError(f"duplicate item ids in {name}")
        rng.shuffle(ids)
        total = min(targets.get(name, len(ids)), len(ids))
        chosen = ids[:total]
        n_dev = total // 5
        splits[name] = {
            "dev": chosen[:n_dev],
            "test": chosen[n_dev:],
            "available": len(ids),
            "subsample": total,
            "dropped": False,
        }
    return splits


def ensure_items(root: Path, seed: int = SEED, load_external: bool = True) -> dict:
    """Write results/splits.json and results/items.jsonl once.

    Existing files with the same format version are reused so a resumed run
    cannot redraw the split.
    """
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    splits_path = results / "splits.json"
    items_path = results / "items.jsonl"
    if splits_path.exists() and items_path.exists():
        payload = json.loads(splits_path.read_text(encoding="utf-8"))
        if payload.get("format_version") == FORMAT_VERSION and payload.get("seed") == seed:
            return {"reused": True, "splits_path": str(splits_path), "items_path": str(items_path), "payload": payload}

    by_dataset: dict[str, list[dict]] = {"jagged": generate_jagged(seed)}
    load_log = []
    for item in by_dataset["jagged"]:
        verify_jagged_item(item)
    if load_external:
        external = load_external_datasets(seed)
        for name, loaded in external.items():
            load_log.append(
                {
                    "dataset": name,
                    "source": loaded.get("source"),
                    "split_used": loaded.get("split_used"),
                    "n_items": len(loaded.get("items") or []),
                    "error": loaded.get("error"),
                    "attempts": loaded.get("attempts"),
                }
            )
            if loaded.get("items"):
                by_dataset[name] = loaded["items"]
            else:
                by_dataset[name] = []
    else:
        for name in ("boolq", "strategyqa", "arc_c"):
            by_dataset.setdefault(name, [])

    splits = assign_splits(by_dataset, seed=seed)
    chosen = {
        name: set(block["dev"]) | set(block["test"])
        for name, block in splits.items()
    }
    kept = []
    for name, items in by_dataset.items():
        for item in items:
            if item["item_id"] in chosen.get(name, set()):
                kept.append(item)
    kept.sort(key=lambda item: (item["dataset"], item["item_id"]))

    with items_path.open("w", encoding="utf-8") as handle:
        for item in kept:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_yes = sum(1 for item in by_dataset["jagged"] if item["meta"]["type"] == "date_compare" and item["gold_text"] == "Yes")
    n_dates = sum(1 for item in by_dataset["jagged"] if item["meta"]["type"] == "date_compare")
    payload = {
        "seed": seed,
        "format_version": FORMAT_VERSION,
        "hash": "sha256",
        "datasets": splits,
        "load_log": load_log,
        "jagged_date_yes": n_yes,
        "jagged_date_n": n_dates,
        "n_items_written": len(kept),
    }
    splits_path.write_text(encoding="utf-8", data=json.dumps(payload, indent=2) + "\n")
    return {"reused": False, "splits_path": str(splits_path), "items_path": str(items_path), "payload": payload}


def load_items(root: Path) -> list[dict]:
    path = root / "results" / "items.jsonl"
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(json.loads(line))
    return items


def select_items(items: list[dict], splits: dict, dataset: str, split: str, limit: int | None = None) -> list[dict]:
    wanted = splits["datasets"][dataset][split]
    if limit is not None:
        wanted = wanted[:limit]
    by_id = {item["item_id"]: item for item in items if item["dataset"] == dataset}
    missing = [item_id for item_id in wanted if item_id not in by_id]
    if missing:
        raise KeyError(f"{dataset} {split} missing {len(missing)} items, first {missing[:3]}")
    return [by_id[item_id] for item_id in wanted]
