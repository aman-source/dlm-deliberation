"""End-type tokens, eos_pad_fraction, suppress_eos_in_scratch, and the length-aware timing projection."""

from __future__ import annotations

import unittest

import torch

from dd.canvas import Canvas
from dd.decode import end_fraction, run_c1, run_c2
from dd.labels import resolve_end_tokens

MASK, END, WORD, PAD = 0, 9, 3, 1
LETTERS = {"A": [4], "B": [6]}


class FakeTokenizer:
    eos_token_id = 9
    pad_token_id = 9
    all_special_tokens = ["<|endoftext|>", "<|startoftext|>"]
    added_tokens_encoder = {
        "<|endoftext|>": 9,
        "<|startoftext|>": 8,
        "<|eot_id|>": 12,
        "<|end_header_id|>": 13,
        "<|reserved_token_1|>": 14,
        "<|mdm_mask|>": 0,
    }

    def convert_ids_to_tokens(self, index):
        return {v: k for k, v in self.added_tokens_encoder.items()}.get(index, f"t{index}")

    def convert_tokens_to_ids(self, token):
        return self.added_tokens_encoder[token]

    def encode(self, text, add_special_tokens=False):
        return {"\nAnswer:": [2]}.get(text, [7] * max(1, len(text.split())))

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "USER " + messages[0]["content"]


def logits_fn(batch, attn):
    """The end token (9) is the top prediction everywhere, including the slot."""
    del attn
    rows, width = batch.shape
    logits = torch.zeros(rows, width, 16)
    logits[:, :, END] = 5.0
    logits[:, :, WORD] = 3.0  # best non-end token
    logits[:, :, 4] = 2.0  # letter A
    logits[:, :, 6] = 0.5  # letter B
    return logits


def c1_canvas():
    # prompt 10,10; scratch 2..5; pinned "\nAnswer:" at 6; slot 7
    return Canvas(input_ids=[10, 10, 0, 0, 0, 0, 2, 0], slot_index=7, scratch_positions=[2, 3, 4, 5], prompt_len=2, condition="C1")


def c2_canvas():
    return Canvas(input_ids=[10, 10, 0, 0, 0, 0], slot_index=None, scratch_positions=[2, 3, 4, 5], prompt_len=2, condition="C2")


class EndTokenTests(unittest.TestCase):
    def test_resolves_eos_pad_and_end_of_turn_only(self):
        found = resolve_end_tokens(FakeTokenizer(), extra_ids=(None, 9))
        self.assertEqual(found, {9: "<|endoftext|>", 12: "<|eot_id|>"})

    def test_end_fraction(self):
        self.assertEqual(end_fraction([9, 3, 12, 3], [9, 12]), 0.5)
        self.assertIsNone(end_fraction([], [9]))

    def _run(self, condition, suppress):
        kwargs = {"end_ids": [END, 12], "suppress_end_in_scratch": suppress}
        if condition == "C1":
            return run_c1(logits_fn, [c1_canvas()], LETTERS, [["A", "B"]], MASK, PAD, steps=2, tokenizer=FakeTokenizer(), **kwargs)[0]
        return run_c2(logits_fn, [c2_canvas()], LETTERS, [["A", "B"]], MASK, PAD, steps=2, tokenizer=FakeTokenizer(), template="v1", **kwargs)[0]

    def test_flag_off_scratch_fills_with_end_tokens(self):
        for condition in ("C1", "C2"):
            row = self._run(condition, suppress=False)
            scratch = [row["filled_ids"][pos] for pos in (2, 3, 4, 5)]
            self.assertEqual(scratch, [END] * 4, condition)
            self.assertEqual(row["eos_pad_fraction"], 1.0, condition)
            self.assertFalse(row["suppress_eos_in_scratch"])

    def test_flag_on_suppresses_identically_in_c1_and_c2(self):
        c1 = self._run("C1", suppress=True)
        c2 = self._run("C2", suppress=True)
        for row in (c1, c2):
            self.assertEqual([row["filled_ids"][pos] for pos in (2, 3, 4, 5)], [WORD] * 4)
            self.assertEqual(row["eos_pad_fraction"], 0.0)
            self.assertTrue(row["suppress_eos_in_scratch"])

    def test_suppression_never_touches_the_answer_slot(self):
        # The slot also has the end token on top. Its read (letters and off_label_mass) must be
        # identical with and without suppression, because suppression applies to scratch only.
        off = self._run("C1", suppress=False)
        on = self._run("C1", suppress=True)
        for a, b in zip(off["readouts"], on["readouts"]):
            self.assertEqual(a["probs"], b["probs"])
            self.assertAlmostEqual(a["off_label_mass"], b["off_label_mass"])
        self.assertGreater(on["off_label_mass"], 0.5)  # end token still dominates the slot


class TimingTests(unittest.TestCase):
    def test_projection_weights_passes_by_scratch_length(self):
        from dd.run import PASSES_PER_ITEM_BY_S, projection

        self.assertEqual(PASSES_PER_ITEM_BY_S, {0: 1, 32: 48, 128: 48})
        hours = projection(3600, {"0": 1.0, "32": 2.0, "128": 3.0})
        self.assertAlmostEqual(hours, 1 * 1.0 + 48 * 2.0 + 48 * 3.0)

    def test_prompt_length_items_picks_p50_and_p90(self):
        from dd.run import prompt_length_items

        from dd.canvas import encode_prompt
        from dd.data import prompt_for

        items = [
            {"item_id": f"x{i}", "state": " ".join(["w"] * (i + 1)), "question": "Q?", "options": ["Yes", "No"]}
            for i in range(100)
        ]
        splits = {"datasets": {"d": {"dev": [f"x{i}" for i in range(20)], "test": [f"x{i}" for i in range(20, 100)]}}}
        picked = prompt_length_items(FakeTokenizer(), items, splits)
        # Lengths are measured on the protocol v1.1 prompt; one extra state word = one extra token.
        base = len(encode_prompt(FakeTokenizer(), prompt_for(items[0])))
        self.assertEqual(picked["p50"]["prompt_tokens"], base + 50)
        self.assertEqual(picked["p90"]["prompt_tokens"], base + 89)


if __name__ == "__main__":
    unittest.main()
