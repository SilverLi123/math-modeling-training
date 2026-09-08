"""问题1数据管线的确定性单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from q1_lmm_baseline import (
    load_and_prepare,
    normalize_excel_date,
    parse_gestational_week,
    subject_group_folds,
)


class ParsingTests(unittest.TestCase):
    def test_gestational_week(self) -> None:
        self.assertEqual(parse_gestational_week("23w"), 23.0)
        self.assertAlmostEqual(parse_gestational_week("11w+6"), 11 + 6 / 7)
        with self.assertRaises(ValueError):
            parse_gestational_week("11w+7")

    def test_mixed_excel_dates(self) -> None:
        self.assertEqual(normalize_excel_date(20230429), "2023-04-29")
        self.assertEqual(normalize_excel_date("2023-04-29"), "2023-04-29")
        self.assertEqual(normalize_excel_date(pd.Timestamp("2023-04-29")), "2023-04-29")


class RealWorkbookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.audit = load_and_prepare(PROJECT_ROOT / "附件.xlsx")

    def test_expected_scale(self) -> None:
        self.assertEqual(len(self.data), 1082)
        self.assertEqual(self.data["subject_id"].nunique(), 267)
        self.assertEqual(self.audit["y_fraction"]["records_at_or_above_4pct"], 937)

    def test_no_silent_model_row_deletion(self) -> None:
        self.assertEqual(self.audit["rows_removed_for_q1_lmm"], 0)
        self.assertFalse(self.data[["subject_id", "gestational_week", "bmi", "y_fraction"]].isna().any().any())

    def test_repeat_measure_structure(self) -> None:
        self.assertEqual(self.audit["same_occasion_repeat_groups"], 18)
        self.assertEqual(self.audit["records_per_subject"], {"min": 1, "median": 4.0, "max": 8})

    def test_subject_group_folds_are_complete_and_disjoint(self) -> None:
        folds = subject_group_folds(self.data["subject_id"], n_splits=5, seed=20250904)
        self.assertEqual(len(folds), 5)
        self.assertEqual(sum(len(fold) for fold in folds), 267)
        self.assertEqual(len(set().union(*folds)), 267)
        for index, fold in enumerate(folds):
            for other in folds[index + 1 :]:
                self.assertTrue(fold.isdisjoint(other))


if __name__ == "__main__":
    unittest.main(verbosity=2)
