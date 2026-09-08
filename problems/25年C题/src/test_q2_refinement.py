import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from q2_nonlinear_calibration import (
    apply_calibrator, apply_design, fit_calibrator, make_design,
)
from q2_probability import prepare


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = prepare()

    def test_spline_transform_is_batch_invariant(self):
        train = self.data[self.data.subject_id != self.data.subject_id.iloc[0]].copy()
        _, transform = make_design(train, "week_spline_df3")
        frame = self.data.iloc[:3]
        batch = apply_design(frame, transform)
        single = apply_design(frame.iloc[:1], transform)
        np.testing.assert_allclose(batch[:1], single)

    def test_spline_design_has_expected_rank(self):
        design, _ = make_design(self.data, "week_spline_df3")
        self.assertEqual(design.shape[1], 5)
        self.assertEqual(np.linalg.matrix_rank(design), design.shape[1])

    def test_calibrator_identity_on_calibrated_synthetic_data(self):
        probability = np.r_[np.repeat(0.2, 100), np.repeat(0.8, 100)]
        outcome = np.r_[np.r_[np.ones(20), np.zeros(80)],
                        np.r_[np.ones(80), np.zeros(20)]]
        subjects = pd.Series(np.arange(len(outcome)).astype(str))
        calibrator = fit_calibrator(outcome, probability, subjects)
        calibrated = apply_calibrator(calibrator, probability)
        self.assertTrue(np.isfinite(calibrated).all())
        self.assertAlmostEqual(calibrator["intercept"], 0, places=5)
        self.assertAlmostEqual(calibrator["slope"], 1, places=5)
        self.assertTrue(np.all((calibrated > 0) & (calibrated < 1)))

    def test_early_subset_has_declared_subject_count(self):
        early = self.data[self.data.first_observed_week <= 12]
        self.assertEqual(early.subject_id.nunique(), 69)

    def test_calibrator_accepts_valid_projected_boundary_optimum(self):
        probability = np.tile(np.array([0.2, 0.8]), 100)
        outcome = np.tile(np.array([0, 1]), 100)
        subjects = pd.Series(np.arange(len(outcome)).astype(str))
        calibrator = fit_calibrator(outcome, probability, subjects)
        self.assertTrue(calibrator["boundary_warning"])
        self.assertIn("slope", calibrator["boundary_parameters"])
        self.assertLess(calibrator["projected_gradient_max"], 1e-4)


if __name__ == "__main__":
    unittest.main()
