"""Hand-checkable metric values and regressions for the 29 Sep audit fixes. No weights, no results/ writes."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dd.data import stable_hash
from dd.decode import _confidence_and_token
from dd.labels import ALL_LETTERS, resolve_letter_tokens
from dd.metrics import aurc, expected_calibration_error, multiclass_brier, negative_log_likelihood


class ToyMetricTests(unittest.TestCase):
    def test_ece_exactly_calibrated_bins_is_zero(self):
        # 10 items at 0.8 with 8 right, 10 items at 0.6 with 6 right.
        confidence = [0.8] * 10 + [0.6] * 10
        correct = [True] * 8 + [False] * 2 + [True] * 6 + [False] * 4
        self.assertAlmostEqual(expected_calibration_error(correct, confidence), 0.0, places=12)

    def test_ece_large_sampled_calibrated_set_is_near_zero(self):
        rng = np.random.default_rng(1234)
        confidence = rng.uniform(0.0, 1.0, size=200_000)
        correct = rng.uniform(0.0, 1.0, size=confidence.size) < confidence
        self.assertLess(expected_calibration_error(correct, confidence), 0.01)

    def test_ece_known_miscalibration(self):
        # All at 0.9 confidence, half right: ECE = |0.5 - 0.9| = 0.4.
        self.assertAlmostEqual(expected_calibration_error([True, False] * 5, [0.9] * 10), 0.4)

    def test_brier_two_items_by_hand(self):
        # Item 1: (0.7-1)^2 + 0.2^2 + 0.1^2 = 0.14. Item 2: (0.4-1)^2 + 0.6^2 = 0.72. Mean 0.43.
        self.assertAlmostEqual(multiclass_brier([[0.7, 0.2, 0.1], [0.4, 0.6]], [0, 0]), 0.43)

    def test_nll_two_items_by_hand(self):
        expected = (-math.log(0.7) - math.log(0.4)) / 2  # 0.636483...
        self.assertAlmostEqual(negative_log_likelihood([0.7, 0.4]), expected)
        self.assertAlmostEqual(expected, 0.6364828, places=6)

    def test_aurc_sorted_example_by_hand(self):
        # Ranked by confidence: right, right, wrong, right.
        # Risk at coverage 1/4..4/4 = 0, 0, 1/3, 1/4. Trapezoid over coverage:
        # 0.25 * [(0+0)/2 + (0+1/3)/2 + (1/3+1/4)/2] = 0.1145833...
        expected = 0.25 * ((0 + 0) / 2 + (0 + 1 / 3) / 2 + (1 / 3 + 1 / 4) / 2)
        self.assertAlmostEqual(aurc([True, True, False, True], [0.9, 0.8, 0.7, 0.6]), expected)
        # Input order must not matter; ranking is by confidence.
        self.assertAlmostEqual(aurc([True, False, True, True], [0.6, 0.7, 0.9, 0.8]), expected)

    def test_aurc_perfect_ranking_is_lower_than_inverted(self):
        good = aurc([True, True, False, False], [0.9, 0.8, 0.2, 0.1])
        bad = aurc([False, False, True, True], [0.9, 0.8, 0.2, 0.1])
        self.assertLess(good, bad)


class AuditRegressionTests(unittest.TestCase):
    def test_stable_hash_is_process_independent(self):
        # Fixed value: would change if hash() (salted per process) were used.
        self.assertEqual(stable_hash("arc-Mercury_7175875"), stable_hash("arc-Mercury_7175875"))
        self.assertEqual(stable_hash("x"), int.from_bytes(__import__("hashlib").sha256(b"x").digest()[:8], "little"))

    def test_letter_e_resolves(self):
        class FiveLetters:
            def encode(self, text, add_special_tokens=False):
                return [ord(text.strip()) + (100 if text.startswith(" ") else 0)]

        resolved = resolve_letter_tokens(FiveLetters())
        self.assertEqual(tuple(resolved), ALL_LETTERS)
        self.assertIn("E", resolved)

    def test_scratch_fill_never_chooses_the_mask_token(self):
        logits = torch.tensor([0.0, 5.0, 1.0])  # mask id 1 has the top logit
        confidence, token = _confidence_and_token(logits, mask_id=1)
        self.assertEqual(token, 2)
        self.assertAlmostEqual(confidence, float(torch.softmax(logits, 0)[2]))

    def test_finish_row_keeps_off_label_mass(self):
        from dd import run

        base = {"gold": "A", "item_id": "x"}
        result = {
            "readouts": [],
            "final_probs": {"A": 0.9, "B": 0.1},
            "letter_logits": {"A": 1.0, "B": 0.0},
            "off_label_mass": 0.25,
            "pred": "A",
            "nfe": 1,
        }
        self.assertEqual(run.finish_row(base, result, 1.0)["off_label_mass"], 0.25)

    def test_read_done_skips_a_torn_final_line(self):
        from dd import run

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell.jsonl"
            path.write_text(json.dumps({"item_id": "a"}) + "\n" + '{"item_id": "b", "rea', encoding="utf-8")
            with mock.patch.object(run, "append_log"):
                done = run.read_done(path)
        self.assertEqual(set(done), {"a"})


if __name__ == "__main__":
    unittest.main()
