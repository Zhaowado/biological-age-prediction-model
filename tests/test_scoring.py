import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitness_ba import score_record
from fitness_ba.annual import score_annual


FEMALE = {
    "oneid": "example_f",
    "sex": "F",
    "age": 23.458387235877534,
    "grip": 26.7,
    "jump": 23.7,
    "vital": 2344.0,
    "step": 65.0,
    "reach": 7.3,
    "balance": 4.0,
    "reaction": 0.56,
    "situp": 27.0,
}


class ScoringTests(unittest.TestCase):
    def test_frozen_complete_result(self):
        result = score_record(FEMALE)
        self.assertEqual(result["status"], "evaluated")
        self.assertAlmostEqual(result["FAI"], 62.5233860388398, places=10)
        self.assertAlmostEqual(result["BA"], 28.91964568676594, places=10)
        self.assertFalse(result["BA_low_confidence"])

    def test_ba_formula(self):
        result = score_record(FEMALE)
        factor = 0.43608481236232405
        expected = FEMALE["age"] + factor * (result["FAI"] - 50)
        self.assertAlmostEqual(result["BA"], expected, places=12)

    def test_supported_partial_pattern_is_flagged(self):
        record = dict(FEMALE)
        record.pop("jump")
        result = score_record(record)
        self.assertEqual(result["status"], "evaluated")
        self.assertTrue(result["BA_low_confidence"])
        self.assertEqual(result["BA_method"], "partial_bridge_6")
        self.assertIsNotNone(result["BA_imputation_interval_low"])

    def test_unsupported_pattern_is_not_fabricated(self):
        result = score_record({"sex": "F", "age": 30, "reaction": 0.5})
        self.assertEqual(result["status"], "unsupported_missingness_pattern")
        self.assertIsNone(result["BA"])

    def test_annual_uses_last_valid_item_not_last_row_only(self):
        first = dict(FEMALE, oneid="annual_example", exam_date="2024-01-01")
        later = {"oneid": "annual_example", "sex": "F", "age": FEMALE["age"], "exam_date": "2024-09-01", "grip": 30.0}
        long, wide = score_annual(pd.DataFrame([first, later]))
        expected = score_record(dict(FEMALE, grip=30.0))
        self.assertEqual(len(long), 1)
        self.assertAlmostEqual(long.iloc[0]["BA"], expected["BA"], places=10)
        self.assertAlmostEqual(wide.iloc[0]["2024"], expected["BA"], places=10)


if __name__ == "__main__":
    unittest.main()
