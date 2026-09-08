import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q4_abnormality_baseline import FEATURES, folds, load_data, metrics, pipeline, threshold_from_training_scores


class Q4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.audit = load_data()

    def test_label_counts_and_parse(self):
        self.assertEqual(len(self.data), 605)
        self.assertEqual(self.audit["positive_subjects_any"], 44)
        self.assertEqual(self.audit["nonempty_ab_rows"], int(self.data.a_any.sum()))
        self.assertTrue((self.data.a_any == ((self.data.a13 + self.data.a18 + self.data.a21) > 0)).all())

    def test_grouped_folds_are_disjoint_and_complete(self):
        seen = set()
        for _, valid in folds(self.data, 20250904):
            current = set(self.data.iloc[valid].subject_id.astype(str))
            self.assertFalse(seen & current)
            seen |= current
        self.assertEqual(len(seen), 147)

    def test_threshold_is_finite(self):
        value = threshold_from_training_scores(np.array([0, 0, 1, 1]), np.array([-1., 0., 2., 3.]))
        self.assertTrue(np.isfinite(value))
        self.assertTrue(len(FEATURES) >= 3)

    def test_elasticnet_uses_predeclared_convergence_tolerance(self):
        self.assertEqual(pipeline(0.3).named_steps["model"].tol, 1e-2)

    def test_uncalibrated_rule_score_does_not_request_brier(self):
        summary = metrics(np.array([0, 1]), np.array([-0.5, 3.0]), np.array([1.0, 1.0]), include_brier=False)
        self.assertNotIn("brier", summary)


if __name__ == "__main__":
    unittest.main()
