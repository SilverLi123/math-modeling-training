"""Unit tests for the ordered-DP banding/time tool (synthetic data only)."""
import unittest

import numpy as np

from q2_dp_tool import plan, pareto, late_risk


def synthetic():
    rng = np.random.default_rng(1)
    n = 60
    bmi = np.sort(rng.uniform(21, 46, n))
    weeks = np.arange(10, 25.01, 0.5)
    # slower attainment for high BMI: p rises with week, falls with bmi
    z = (weeks[None, :] - (11.0 + (bmi[:, None] - 21.0) * 0.45)) / 2.0
    p = 1.0 / (1.0 + np.exp(-z))
    return bmi, p, weeks


class DPPlanTests(unittest.TestCase):
    def test_late_risk_piecewise(self):
        self.assertEqual(late_risk(12), 0.0)
        self.assertAlmostEqual(late_risk(28), 1.0)
        self.assertGreater(late_risk(30), 1.0)

    def test_plan_shape_and_monotonic_weeks(self):
        bmi, p, weeks = synthetic()
        res = plan(bmi, p, weeks, max_segments=3, w_late=1.0)
        self.assertFalse(res['infeasible'])
        self.assertLessEqual(len(res['segments']), 3)
        weeks_used = [s['week'] for s in res['segments']]
        self.assertEqual(weeks_used, sorted(weeks_used))
        los = [s['bmi_lo'] for s in res['segments']]
        his = [s['bmi_hi'] for s in res['segments']]
        self.assertTrue(all(los[i] <= his[i] for i in range(len(los))))
        for i in range(1, len(los)):
            self.assertGreaterEqual(los[i], his[i - 1])
        total = res['total_cost']
        self.assertGreater(total, 0.0)

    def test_coverage_constraint_respected(self):
        bmi, p, weeks = synthetic()
        res = plan(bmi, p, weeks, max_segments=2, w_late=1.0,
                   coverage_q=0.85)
        if not res['infeasible']:
            for s in res['segments']:
                self.assertGreaterEqual(s['mean_coverage'], 0.85 - 1e-9)

    def test_impossible_coverage_reports_infeasible(self):
        bmi, p, weeks = synthetic()
        res = plan(bmi, p, weeks, max_segments=2, w_late=1.0,
                   coverage_q=0.9999)
        self.assertTrue(res['infeasible'] or len(res['segments']) == 0)

    def test_pareto_table(self):
        bmi, p, weeks = synthetic()
        rows, table = pareto(bmi, p, weeks, max_segments=2,
                             weights=[0.5, 1.0, 2.0])
        self.assertEqual(len(table), 3)
        for row in rows:
            if not row['infeasible']:
                ws = [s['week'] for s in row['segments']]
                self.assertEqual(ws, sorted(ws))


if __name__ == '__main__':
    unittest.main()
