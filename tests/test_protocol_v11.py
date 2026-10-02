"""Protocol v1.1: prompt last line, eot-pinned canvases with the think prefix, report filters."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from dd.canvas import append_answer_slot, build_canvas
from dd.data import PROMPT_LAST_LINE_V10, PROMPT_LAST_LINE_V11, prompt_for, render_prompt
from dd.decode import run_c1, run_c2

MASK, EOT = 0, 99


class V11Tokenizer:
    table = {
        "Let me think step by step.\n": [20, 21],
        "The answer is": [30],
        "\nThe answer is": [31],
        "Answer:": [40],
        "\nAnswer:": [41],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self.table.get(text, [7]))

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "USER " + messages[0]["content"]

    def convert_tokens_to_ids(self, token):
        return {"<|eot_id|>": EOT}[token]

    def convert_ids_to_tokens(self, index):
        return {EOT: "<|eot_id|>"}.get(index, f"t{index}")

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)


PROMPT = [7]  # V11Tokenizer encodes any prompt to one token


class PromptTests(unittest.TestCase):
    def test_prompt_swaps_only_the_last_line(self):
        item = {"state": "(none)", "question": "Q?", "options": ["Yes", "No"]}
        old = render_prompt(item["state"], item["question"], item["options"])
        new = prompt_for(item)
        self.assertTrue(old.endswith(PROMPT_LAST_LINE_V10))
        self.assertTrue(new.endswith(PROMPT_LAST_LINE_V11))
        self.assertEqual(old[: -len(PROMPT_LAST_LINE_V10)], new[: -len(PROMPT_LAST_LINE_V11)])
        self.assertIn('finish with "The answer is" followed by the letter.', new)

    def test_frozen_items_render(self):
        path = Path(__file__).resolve().parents[1] / "results" / "items.jsonl"
        with path.open(encoding="utf-8") as handle:
            item = json.loads(handle.readline())
        self.assertTrue(prompt_for(item).endswith(PROMPT_LAST_LINE_V11))


class CanvasV11Tests(unittest.TestCase):
    def test_c0_pins_eot_after_slot(self):
        canvas = build_canvas(V11Tokenizer(), "p", "C0", 0, MASK, template="v3e")
        self.assertEqual(canvas.input_ids, PROMPT + [30, MASK, EOT])
        self.assertEqual(canvas.slot_index, 2)

    def test_c1_layout(self):
        canvas = build_canvas(V11Tokenizer(), "p", "C1", 3, MASK, template="v3e")
        self.assertEqual(canvas.input_ids, PROMPT + [20, 21] + [MASK] * 3 + [31, MASK, EOT])
        self.assertEqual(canvas.scratch_positions, [3, 4, 5])
        self.assertEqual(canvas.slot_index, 7)

    def test_c2_layout_and_append(self):
        tok = V11Tokenizer()
        canvas = build_canvas(tok, "p", "C2", 3, MASK, template="v3e")
        self.assertEqual(canvas.input_ids, PROMPT + [20, 21] + [MASK] * 3)
        self.assertIsNone(canvas.slot_index)
        filled = type(canvas)(input_ids=PROMPT + [20, 21, 5, 5, 5], slot_index=None, scratch_positions=[3, 4, 5], condition="C2")
        read = append_answer_slot(tok, filled, MASK, template="v3e")
        self.assertEqual(read.input_ids, PROMPT + [20, 21, 5, 5, 5, 31, MASK, EOT])
        self.assertEqual(read.slot_index, 7)

    def test_v1_templates_unchanged(self):
        # Protocol v1.0 layouts must stay byte-identical for the archived smoke rows.
        canvas = build_canvas(V11Tokenizer(), "p", "C1", 2, MASK, template="v1")
        self.assertEqual(canvas.input_ids, PROMPT + [MASK, MASK, 41, MASK])


def logits_fn(batch, attn):
    del attn
    logits = torch.zeros(batch.shape[0], batch.shape[1], 128)
    logits[:, :, 5] = 3.0   # the scratch fill token
    logits[:, :, 11] = 2.0  # letter A
    logits[:, :, 13] = 1.0  # letter B
    return logits


class DecodeV11Tests(unittest.TestCase):
    def test_pinned_prefix_template_and_eot_survive_c1(self):
        tok = V11Tokenizer()
        canvas = build_canvas(tok, "p", "C1", 4, MASK, template="v3e")
        row = run_c1(logits_fn, [canvas], {"A": [11], "B": [13]}, [["A", "B"]], MASK, 1, steps=2, tokenizer=tok)[0]
        for pos, token in enumerate(canvas.input_ids):
            if pos not in canvas.scratch_positions:
                self.assertEqual(row["filled_ids"][pos], token, pos)
        self.assertEqual(row["filled_ids"][canvas.slot_index], MASK)
        self.assertEqual(row["filled_ids"][-1], EOT)
        self.assertEqual(len(row["readouts"][0]["slot_topk"]), 5)

    def test_c2_reads_with_eot_after_slot(self):
        tok = V11Tokenizer()
        canvas = build_canvas(tok, "p", "C2", 4, MASK, template="v3e")
        row = run_c2(logits_fn, [canvas], {"A": [11], "B": [13]}, [["A", "B"]], MASK, 1, steps=2, tokenizer=tok, template="v3e")[0]
        self.assertEqual(row["filled_ids"][-1], EOT)
        self.assertEqual(row["filled_ids"][-2], MASK)
        self.assertEqual(row["filled_ids"][1:3], [20, 21])


class ReportFilterTests(unittest.TestCase):
    def test_load_rows_keeps_only_main_setting(self):
        from dd.report import load_rows

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results" / "raw" / "smoke_v1_0").mkdir(parents=True)
            (root / "results" / "template.json").write_text(json.dumps({"template": "v3e"}), encoding="utf-8")
            (root / "results" / "protocol_v1_1.json").write_text(json.dumps({"suppress_eos_in_scratch": True}), encoding="utf-8")
            rows = [
                {"item_id": "keep_c0", "protocol": "v1.1", "template": "v3e", "condition": "C0"},
                {"item_id": "keep_c1", "protocol": "v1.1", "template": "v3e", "condition": "C1", "suppress_eos_in_scratch": True},
                {"item_id": "drop_flag", "protocol": "v1.1", "template": "v3e", "condition": "C2", "suppress_eos_in_scratch": False},
                {"item_id": "drop_template", "protocol": "v1.1", "template": "v1e", "condition": "C0"},
                {"item_id": "drop_v10", "template": "v3e", "condition": "C0"},
            ]
            (root / "results" / "raw" / "a.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            (root / "results" / "raw" / "smoke_v1_0" / "b.jsonl").write_text(json.dumps(rows[0] | {"item_id": "drop_archive"}) + "\n", encoding="utf-8")
            kept = {row["item_id"] for row in load_rows(root)}
        self.assertEqual(kept, {"keep_c0", "keep_c1"})

    def test_resolve_suppress_protocol(self):
        from dd import run

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol_v1_1.json"
            with mock.patch.object(run, "PROTOCOL_FILE", path):
                with self.assertRaises(SystemExit):
                    run.resolve_suppress({"suppress_eos_in_scratch": "protocol"})
                path.write_text(json.dumps({"suppress_eos_in_scratch": True}), encoding="utf-8")
                self.assertTrue(run.resolve_suppress({"suppress_eos_in_scratch": "protocol"}))
                self.assertFalse(run.resolve_suppress({"suppress_eos_in_scratch": False}))


if __name__ == "__main__":
    unittest.main()
