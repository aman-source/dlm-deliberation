"""v1.2 analyses: adaptive ECE, Pareto frontier, ladder curves, H3 statistic, oracle."""

from __future__ import annotations

import unittest

import numpy as np

from dd.analysis import CELLS13, LADDERS, TAUS, LadderData, adaptive_ece, h3_verdict, pareto


class AdaptiveEceTests(unittest.TestCase):
    def test_equal_mass_bins_by_hand(self):
        # 4 items, 2 bins: low bin conf {0.2, 0.4} acc 0; high bin conf {0.6, 0.8} acc 1.
        value = adaptive_ece([False, False, True, True], [0.2, 0.4, 0.6, 0.8], n_bins=2)
        self.assertAlmostEqual(value, 0.5 * abs(0 - 0.3) + 0.5 * abs(1 - 0.7))

    def test_calibrated_is_near_zero(self):
        rng = np.random.default_rng(0)
        conf = rng.uniform(size=100_000)
        self.assertLess(adaptive_ece(rng.uniform(size=conf.size) < conf, conf), 0.01)


class ParetoTests(unittest.TestCase):
    def test_running_max_over_budget(self):
        xs, ys = pareto(np.array([1, 2, 2, 5, 17]), np.array([0.6, 0.8, 0.7, 0.75, 0.9]))
        self.assertEqual(list(xs), [1, 2, 5, 17])
        self.assertEqual(list(ys), [0.6, 0.8, 0.8, 0.9])


def make_rows(ladder_probs, gold="A"):
    """cellmaps for 2 items: every fixed cell copies the C0 row; ladder cells use the given probs."""
    cellmaps = {}
    for cell in CELLS13 + LADDERS["preregistered"]:
        cellmaps[cell] = {}
    for item, levels in enumerate(ladder_probs):
        for cell in CELLS13:
            probs = levels[0]
            cellmaps[cell][f"i{item}"] = {"final_probs": probs, "letter_logits": probs, "gold": gold,
                                          "correct": max(probs, key=probs.get) == gold}
        for k, cell in enumerate(LADDERS["preregistered"]):
            probs = levels[k]
            cellmaps[cell][f"i{item}"] = {"final_probs": probs, "letter_logits": probs, "gold": gold,
                                          "correct": max(probs, key=probs.get) == gold}
    return cellmaps


class LadderTests(unittest.TestCase):
    def setUp(self):
        wrong_unsure = {"A": 0.45, "B": 0.55}
        right_sure = {"A": 0.9, "B": 0.1}
        self.data = LadderData(make_rows([[wrong_unsure, right_sure, right_sure], [right_sure, right_sure, right_sure]]),
                               LADDERS["preregistered"], None, "m")

    def test_tau_extremes(self):
        self.assertTrue((self.data.level[:, 0] == 0).all())   # tau = 0 never escalates
        self.assertTrue((self.data.level[:, -1] == 2).all())  # tau = 1 always reaches L2
        self.assertEqual(list(self.data.cum_nfe), [1, 6, 23])

    def test_escalation_fixes_unsure_item(self):
        point = self.data.at_tau(0.5)
        self.assertEqual(point["accuracy"], 1.0)
        self.assertAlmostEqual(point["mean_nfe"], (6 + 1) / 2)

    def test_oracle_and_h3_shape(self):
        oracle = self.data.oracle()
        self.assertEqual(oracle["accuracy"], 1.0)
        self.assertAlmostEqual(oracle["mean_nfe"], 3.5)
        result = self.data.h3(n_boot=50)
        self.assertEqual(len(TAUS), 50)
        self.assertIn("G", result)
        self.assertLessEqual(result["lo"], result["hi"])

    def test_verdict_rule(self):
        dom = {"dominates": True}
        nod = {"dominates": False}
        self.assertEqual(h3_verdict({"a": dom, "b": dom, "c": dom, "d": nod}), "supported")
        self.assertEqual(h3_verdict({"a": dom, "b": dom, "c": nod, "d": nod}), "not supported")


if __name__ == "__main__":
    unittest.main()
