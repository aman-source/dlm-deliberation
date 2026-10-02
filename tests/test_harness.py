"""Logic tests that do not load LLaDA and do not write results/."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from dd.canvas import build_canvas
from dd.data import assign_splits, generate_jagged, render_prompt, shuffle_options, verify_jagged_item
from dd.decode import _select_positions, flip_rate, kt_schedule, nfe_for, run_c0, run_c1, run_c2
from dd.device import model_fits
from dd.escalate import (
    cumulative_nfe,
    decide_items,
    interpolate,
    margin,
    oracle_decisions,
    tau_grid,
    upper_envelope,
)
from dd.labels import resolve_letter_tokens, slot_distribution
from dd.metrics import (
    accuracy,
    aurc,
    classify_contrast,
    expected_calibration_error,
    fit_temperature,
    mcnemar_test,
    multiclass_brier,
    negative_log_likelihood,
)
from dd.report import build_report


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        table = {
            "Answer:": [2],
            "\nAnswer:": [3],
            "Answer: ": [2, 4],
            "The answer is": [5],
            "\nThe answer is": [6],
        }
        if text in table:
            return list(table[text])
        return [7]

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "USER " + messages[0]["content"]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in ids)

    def convert_ids_to_tokens(self, index):
        return {0: "<|mdm_mask|>"}.get(index, f"t{index}")


class LetterTokenizer:
    def encode(self, text, add_special_tokens=False):
        mapping = {"A": [11], " A": [12], "B": [13], " B": [14], "C": [15], " D": [16]}
        if text not in mapping:
            raise AssertionError(f"unexpected variant {text!r}")
        return list(mapping[text])


class ScheduleTests(unittest.TestCase):
    def test_even_split_and_remainder(self):
        self.assertEqual(kt_schedule(32, 4), [8, 8, 8, 8])
        self.assertEqual(kt_schedule(32, 1), [32])
        self.assertEqual(kt_schedule(128, 16), [8] * 16)
        self.assertEqual(kt_schedule(10, 3), [4, 3, 3])
        with self.assertRaises(ValueError):
            kt_schedule(4, 5)

    def test_nfe_table(self):
        self.assertEqual(nfe_for("C0", None), 1)
        self.assertEqual(nfe_for("C1", 1), 2)
        self.assertEqual(nfe_for("C1", 4), 5)
        self.assertEqual(nfe_for("C1", 16), 17)
        self.assertEqual(nfe_for("C2", 16), 17)
        self.assertEqual(cumulative_nfe(0), 1)
        self.assertEqual(cumulative_nfe(1), 6)
        self.assertEqual(cumulative_nfe(2), 23)

    def test_tie_break_is_earlier_position(self):
        # Return values are indices into the masked-position list.
        # Positions 5 and 2 share confidence 0.4; position 2 is earlier, so its index (1) wins.
        self.assertEqual(_select_positions([5, 2, 9], [0.4, 0.4, 0.1], 2), [1, 0])


class LabelTests(unittest.TestCase):
    def test_logsumexp_then_softmax(self):
        letters = resolve_letter_tokens(LetterTokenizer(), ("A", "B"))
        logits = torch.full((20,), -100.0)
        logits[11] = 0.0
        logits[12] = 0.0
        logits[13] = 0.0
        read = slot_distribution(logits, letters, ["A", "B"])
        self.assertAlmostEqual(read["probs"]["A"], 2 / 3, places=5)
        self.assertAlmostEqual(read["probs"]["B"], 1 / 3, places=5)
        self.assertLess(read["off_label_mass"], 1e-4)
        self.assertEqual(read["pred"], "A")

    def test_missing_letter_aborts(self):
        class Empty:
            def encode(self, text, add_special_tokens=False):
                return [1, 2]

        with self.assertRaises(Exception):
            resolve_letter_tokens(Empty(), ("A",))


class DecodeTests(unittest.TestCase):
    def test_c1_fills_scratch_and_leaves_the_slot_masked(self):
        mask_id = 0
        # prompt tokens 10, 10; scratch at 2,3,4,5; pinned answer at 6; slot at 7
        from dd.canvas import Canvas

        canvas = Canvas(
            input_ids=[10, 10, 0, 0, 0, 0, 2, 0],
            slot_index=7,
            scratch_positions=[2, 3, 4, 5],
            prompt_len=2,
            condition="C1",
        )
        letters = {"A": [4], "B": [6]}

        def logits_fn(batch, attn):
            del attn
            rows, width = batch.shape
            logits = torch.zeros(rows, width, 16)
            for row in range(rows):
                for pos in range(width):
                    if int(batch[row, pos]) == mask_id:
                        logits[row, pos, 3] = 1.0 + pos
                logits[row, 7, 4] = 3.0
                logits[row, 7, 6] = 1.0
            return logits

        result = run_c1(logits_fn, [canvas], letters, [["A", "B"]], mask_id, pad_id=1, steps=2, tokenizer=TinyTokenizer())[0]
        self.assertEqual(result["nfe"], 3)
        self.assertEqual(len(result["readouts"]), 3)
        self.assertEqual(result["pred"], "A")
        for pos in canvas.scratch_positions:
            self.assertNotEqual(result["filled_ids"][pos], mask_id)
            self.assertEqual(result["filled_ids"][pos], 3)
        self.assertEqual(result["filled_ids"][7], mask_id)
        self.assertTrue(result["readouts"][-1].get("final"))

    def test_c0_single_pass(self):
        from dd.canvas import Canvas

        canvas = Canvas(input_ids=[10, 2, 0], slot_index=2, scratch_positions=[], condition="C0")

        def logits_fn(batch, attn):
            del attn
            logits = torch.zeros(batch.shape[0], batch.shape[1], 16)
            logits[:, 2, 4] = 2.0
            logits[:, 2, 6] = 0.0
            return logits

        result = run_c0(logits_fn, [canvas], {"A": [4], "B": [6]}, [["A", "B"]], 0, 1)[0]
        self.assertEqual(result["nfe"], 1)
        self.assertEqual(result["pred"], "A")

    def test_c2_appends_the_slot_after_thought(self):
        from dd.canvas import Canvas

        canvas = Canvas(input_ids=[10, 0, 0], slot_index=None, scratch_positions=[1, 2], prompt_len=1, condition="C2")

        def logits_fn(batch, attn):
            del attn
            logits = torch.zeros(batch.shape[0], batch.shape[1], 16)
            for row in range(batch.shape[0]):
                for pos in range(batch.shape[1]):
                    if int(batch[row, pos]) == 0:
                        logits[row, pos, 3] = 4.0
            # answer slot is the last position after the prefix is appended
            logits[:, -1, 4] = 2.0
            logits[:, -1, 6] = 0.2
            return logits

        result = run_c2(
            logits_fn,
            [canvas],
            {"A": [4], "B": [6]},
            [["A", "B"]],
            0,
            1,
            steps=1,
            tokenizer=TinyTokenizer(),
            template="v1",
        )[0]
        self.assertEqual(result["nfe"], 2)
        self.assertEqual(result["pred"], "A")
        self.assertEqual(result["filled_ids"][-1], 0)


class CanvasTests(unittest.TestCase):
    def test_c1_layout(self):
        tok = TinyTokenizer()
        canvas = build_canvas(tok, "Question?", "C1", 4, mask_id=0, template="v1")
        self.assertEqual(canvas.input_ids[-1], 0)
        self.assertEqual(len(canvas.scratch_positions), 4)
        self.assertNotIn(canvas.slot_index, canvas.scratch_positions)
        self.assertEqual(canvas.input_ids[canvas.slot_index - 1], 3)


class DataTests(unittest.TestCase):
    def test_jagged_ground_truth_and_balance(self):
        items = generate_jagged(1234)
        self.assertEqual(len(items), 300)
        counts = {}
        yes = 0
        for item in items:
            verify_jagged_item(item)
            counts[item["meta"]["type"]] = counts.get(item["meta"]["type"], 0) + 1
            self.assertIn(item["gold_text"], item["options"])
            self.assertTrue(item["prompt"].startswith("Read the state and answer the question."))
            self.assertIn('Reply with the letter of the correct option after "Answer:".', item["prompt"])
            if item["meta"]["type"] == "date_compare" and item["gold_text"] == "Yes":
                yes += 1
            if item["meta"]["type"] == "counting":
                self.assertGreaterEqual(item["meta"]["gold_count"], 1)
                self.assertLessEqual(item["meta"]["gold_count"], 6)
                self.assertEqual(len(item["options"]), 4)
            if item["meta"]["type"] == "numeric_compare":
                self.assertEqual(len(item["options"]), 3)
        self.assertEqual(counts, {"counting": 75, "date_compare": 75, "indirection": 75, "numeric_compare": 75})
        self.assertEqual(yes, 38)

    def test_shuffle_is_stable_and_not_python_hash(self):
        first = shuffle_options("arc-1", ["red", "blue", "green"], "blue", seed=1234)
        second = shuffle_options("arc-1", ["red", "blue", "green"], "blue", seed=1234)
        self.assertEqual(first, second)
        self.assertEqual(first["options"][ord(first["gold"]) - ord("A")], "blue")

    def test_prompt_layout(self):
        text = render_prompt("(none)", "Is it so?", ["Yes", "No"])
        self.assertIn("State:\n(none)\n\nQuestion: Is it so?\n\nOptions:\nA. Yes\nB. No\n\nReply with", text)

    def test_splits_are_disjoint_20_80(self):
        items = generate_jagged(1234)
        splits = assign_splits({"jagged": items, "boolq": []}, seed=1234)
        block = splits["jagged"]
        self.assertEqual(len(block["dev"]), 60)
        self.assertEqual(len(block["test"]), 240)
        self.assertEqual(len(set(block["dev"]) & set(block["test"])), 0)
        self.assertTrue(splits["boolq"]["dropped"])


class MetricTests(unittest.TestCase):
    def test_ece_extremes_and_brier_nll(self):
        self.assertAlmostEqual(expected_calibration_error([True, True], [1.0, 1.0]), 0.0)
        self.assertAlmostEqual(expected_calibration_error([False, False], [1.0, 1.0]), 1.0)
        self.assertAlmostEqual(multiclass_brier([[0.5, 0.5]], [0]), 0.5)
        self.assertAlmostEqual(negative_log_likelihood([1.0]), 0.0)
        self.assertAlmostEqual(negative_log_likelihood([0.0]), -__import__("math").log(1e-12))
        self.assertAlmostEqual(aurc([True, True], [0.2, 0.9]), 0.0)
        self.assertAlmostEqual(accuracy([True, False, True]), 2 / 3)

    def test_temperature_fit_prefers_one_when_already_calibrated_logits(self):
        # Gold is class 0 and logit 0 is larger. Temperature stays finite and NLL is defined.
        logits = [[2.0, 0.0, 0.0], [0.2, 2.0, 0.0], [0.0, 0.0, 3.0]]
        fit = fit_temperature(logits, [0, 1, 2])
        self.assertGreater(fit["temperature"], 0)
        self.assertLess(fit["nll"], 1.0)

    def test_classify_and_mcnemar(self):
        self.assertEqual(classify_contrast(0.1, 0.2, "positive"), "supported")
        self.assertEqual(classify_contrast(-0.2, -0.1, "positive"), "not supported")
        self.assertEqual(classify_contrast(-0.1, 0.2, "positive"), "inconclusive")
        self.assertEqual(classify_contrast(-0.1, 0.2, "no_positive"), "supported")
        self.assertEqual(classify_contrast(0.1, 0.2, "no_positive"), "not supported")
        self.assertEqual(classify_contrast(0.1, 0.2, "either"), "supported")
        self.assertEqual(classify_contrast(-0.1, 0.05, "either"), "inconclusive")
        result = mcnemar_test([True, True, False, False], [False, True, True, False])
        self.assertEqual(result["a_only_correct"], 1)
        self.assertEqual(result["b_only_correct"], 1)
        self.assertGreater(result["p_value"], 0.5)

    def test_flip_rate(self):
        rows = [
            {"readouts": [{"probs": {"A": 0.8, "B": 0.2}}, {"probs": {"A": 0.1, "B": 0.9}}]},
            {"readouts": [{"probs": {"A": 0.7, "B": 0.3}}, {"probs": {"A": 0.6, "B": 0.4}}]},
        ]
        self.assertAlmostEqual(flip_rate(rows), 0.5)


class EscalationTests(unittest.TestCase):
    def test_tau_ends_and_oracle(self):
        self.assertEqual(len(tau_grid()), 50)
        self.assertEqual(tau_grid()[0], 0.0)
        self.assertEqual(tau_grid()[-1], 1.0)
        item = {
            "item_id": "x",
            "gold": "A",
            "letters": ["A", "B"],
            "levels": [
                {"probs": {"A": 0.4, "B": 0.6}, "letter_logits": {"A": 0.0, "B": 0.4}, "correct": False},
                {"probs": {"A": 0.8, "B": 0.2}, "letter_logits": {"A": 1.0, "B": 0.0}, "correct": True},
                {"probs": {"A": 0.9, "B": 0.1}, "letter_logits": {"A": 2.0, "B": 0.0}, "correct": True},
            ],
        }
        stay = decide_items([item], tau=0.0)[0]
        self.assertEqual(stay["level"], 0)
        self.assertEqual(stay["nfe"], 1)
        go = decide_items([item], tau=1.0)[0]
        self.assertEqual(go["level"], 2)
        self.assertEqual(go["nfe"], 23)
        self.assertAlmostEqual(margin({"A": 0.8, "B": 0.2}), 0.6)
        oracle = oracle_decisions([item])[0]
        self.assertEqual(oracle["level"], 1)
        self.assertEqual(oracle["nfe"], 6)
        xs, ys = upper_envelope([(1, 0.4), (1, 0.6), (5, 0.7)])
        self.assertEqual(xs, [1, 5])
        self.assertEqual(ys, [0.6, 0.7])
        self.assertAlmostEqual(interpolate(xs, ys, 3), 0.65)
        self.assertAlmostEqual(interpolate(xs, ys, 9), 0.7)

    def test_memory_guard(self):
        self.assertFalse(model_fits(8 * 1024**3))
        self.assertTrue(model_fits(32 * 1024**3))


class ReportTests(unittest.TestCase):
    def test_empty_report_does_not_invent_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            text = build_report(root, n_resamples=10)
            self.assertIn("Not evaluated", text)
            self.assertIn("Qwen3-8B C0 was not run", text)
            self.assertIn("No test rows", text)
            self.assertFalse((root / "results" / "figs").exists() and any((root / "results" / "figs").iterdir()))

    def test_synthetic_rows_round_trip_into_the_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "results" / "raw"
            raw.mkdir(parents=True)
            rows = []
            for index, correct in enumerate([True, False, True, True]):
                prob_c = 0.8 if correct else 0.3
                rows.append(
                    {
                        "dataset": "arc_c",
                        "split": "test",
                        "item_id": f"arc-{index}",
                        "model": "llada-8b-instruct",
                        "condition": "C0",
                        "S": 0,
                        "T": None,
                        "seed": 1234,
                        "template": "v1",
                        "protocol": "v1.1",
                        "perm": [0, 1],
                        "gold": "A" if correct else "B",
                        "readouts": [{"step": 1, "probs": {"A": prob_c, "B": 1 - prob_c}, "off_label_mass": 0.1}],
                        "final_probs": {"A": prob_c, "B": 1 - prob_c},
                        "letter_logits": {"A": prob_c, "B": 1 - prob_c},
                        "pred": "A" if correct else "B",
                        "correct": correct,
                        "nfe": 1,
                        "wall_ms": 10.0,
                        "scratch_text": "",
                        "off_label_mass": 0.1,
                        "device": "cpu",
                        "dtype": "float32",
                    }
                )
            with (raw / "arc_c_test_llada_C0.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            text = build_report(root, n_resamples=20)
            summary = (root / "results" / "summary.csv").read_text()
            self.assertIn("0.75", summary)
            self.assertIn("0.7500", text)
            self.assertIn("not evaluated", text.lower())


if __name__ == "__main__":
    unittest.main()
