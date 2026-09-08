import unittest
import numpy as np
from q1_spline_comparison import ROOT, fit, predict, design
from q1_lmm_baseline import load_and_prepare


class SplineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = load_and_prepare(ROOT / '附件.xlsx')
        cls.train = cls.data[cls.data.bmi < 40].copy()
        cls.model = fit(cls.train, 3)

    def test_training_centers_only(self):
        np.testing.assert_allclose(self.model[2].to_numpy(),
            self.train[['gestational_week', 'bmi']].mean().to_numpy())

    def test_prediction_batch_invariant_and_extrapolation_finite(self):
        frame = self.data.iloc[[0, 1]].copy()
        frame.iloc[0, frame.columns.get_loc('bmi')] = 55
        batch = predict(self.model, frame)
        self.assertTrue(np.isfinite(batch).all())
        np.testing.assert_allclose(batch[:1], predict(self.model, frame.iloc[:1]))

    def test_design_full_rank(self):
        x = design(self.model, self.train)
        self.assertEqual(np.linalg.matrix_rank(x), x.shape[1])


if __name__ == '__main__':
    unittest.main()
