import sys
import unittest
from itertools import combinations, product
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q23_explore import load, optimize
from q23_policy_summary import collapse


class Q23Tests(unittest.TestCase):
    def test_collapsing_preserves_assignments(self):
        cuts, weeks, counts = [25., 30., 35.], [12., 12., 25., 25.], [25, 30, 35, 40]
        newcuts, newweeks, newcounts = collapse(cuts, weeks, counts)
        values = np.array([20., 25., 28., 30., 35., 40.])
        self.assertTrue(np.array_equal(np.array(weeks)[np.searchsorted(cuts, values, side='right')],
                                       np.array(newweeks)[np.searchsorted(newcuts, values, side='right')]))
        self.assertEqual(newcounts, [55, 75])
        self.assertEqual(newcuts, [30.])

    @classmethod
    def setUpClass(cls):
        cls.data, cls.audit = load()
        cls.raw = pd.read_excel(Path(__file__).resolve().parents[1] / '附件.xlsx',
                                sheet_name='\u7537\u80ce\u68c0\u6d4b\u6570\u636e').set_index('\u5e8f\u53f7')

    def test_baseline_covariates_come_from_first_sample(self):
        first = self.data.drop_duplicates('baseline_sample_id').set_index('baseline_sample_id')
        for target, source in [('baseline_age', '\u5e74\u9f84'), ('baseline_height', '\u8eab\u9ad8'),
                               ('baseline_gravidity', '\u6000\u5b55\u6b21\u6570'),
                               ('baseline_parity', '\u751f\u4ea7\u6b21\u6570')]:
            expected = pd.to_numeric(self.raw.loc[first.index, source].astype(str).str.replace('\u2265', '', regex=False))
            self.assertTrue(np.array_equal(expected.to_numpy(), first[target].to_numpy()))
        natural = '\u81ea\u7136\u53d7\u5b55'
        assisted = int((self.raw.loc[first.index, 'IVF\u598a\u5a20'] != natural).sum())
        self.assertEqual(int(first.baseline_assisted.sum()), assisted)

    def test_dp_honors_support_and_does_not_split_tied_bmi(self):
        probabilities = np.array([[.20, .80], [.20, .90], [.20, .70], [.20, .60]])
        support = np.ones_like(probabilities)
        one = optimize(np.array([20., 20., 30., 30.]), probabilities, support, 1,
                       np.zeros(2), minimum=2, support_min=1)
        two = optimize(np.array([20., 20., 30., 30.]), probabilities, support, 2,
                       np.zeros(2), minimum=2, support_min=1)
        blocked = optimize(np.array([20., 20., 30., 30.]), probabilities, support, 1,
                           np.zeros(2), minimum=2, support_min=5)
        self.assertEqual(one[1][0][2], 1)
        self.assertEqual([group[:2] for group in two[1]], [(0, 2), (2, 4)])
        self.assertIsNone(blocked)

    def test_dp_matches_exhaustive_search(self):
        rng = np.random.default_rng(20250904)
        bmis = np.array([20., 20., 23., 25., 25., 30.])
        for _ in range(12):
            p = rng.uniform(.1, .95, (6, 3))
            support = rng.integers(0, 2, (6, 3))
            delay = np.array([0., .02, .1])
            for k in (1, 2, 3):
                best = np.inf
                for cuts in combinations([2, 3, 5], k-1):
                    bounds = (0, *cuts, 6)
                    for times in product(range(3), repeat=k):
                        cost = 0.
                        for a, b, t in zip(bounds[:-1], bounds[1:], times):
                            if b-a < 2 or support[a:b, t].sum() < 1:
                                cost = np.inf
                                break
                            cost += (1-p[a:b, t]+delay[t]).sum()
                        best = min(best, cost/6)
                actual = optimize(bmis, p, support, k, delay, minimum=2, support_min=1)
                if np.isinf(best):
                    self.assertIsNone(actual)
                else:
                    self.assertAlmostEqual(actual[0], best, places=12)


if __name__ == '__main__':
    unittest.main()
