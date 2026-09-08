import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q4_chromosomes import load_data, split, TARGETS, DEFAULT_SEED, model, FEATURES


class ChromosomeTests(unittest.TestCase):
    def test_all_outer_and_inner_splits_are_subject_disjoint(self):
        data, _ = load_data()
        for target in TARGETS:
            seen = []
            for fold, (left, right) in enumerate(split(data, target, DEFAULT_SEED), 1):
                self.assertFalse(set(data.iloc[left].subject_id) & set(data.iloc[right].subject_id))
                seen.extend(right.tolist())
                train = data.iloc[left]
                for il, ir in split(train, target, DEFAULT_SEED+fold):
                    self.assertFalse(set(train.iloc[il].subject_id) & set(train.iloc[ir].subject_id))
                    self.assertEqual(train.iloc[il][target].nunique(), 2)
                    self.assertEqual(train.iloc[ir][target].nunique(), 2)
            self.assertEqual(sorted(seen), list(range(len(data))))

    def test_preprocessing_is_training_only_and_probabilities_valid(self):
        data, _ = load_data()
        left, right = split(data, 'a13', DEFAULT_SEED)[0]
        train, valid = data.iloc[left], data.iloc[right].copy()
        fitted = model().fit(train[FEATURES], train.a13)
        self.assertTrue(np.allclose(fitted.steps[0][1].statistics_, train[FEATURES].median()))
        before = fitted.steps[1][1].mean_.copy()
        valid.loc[:, 'bmi'] = 1000.
        p = fitted.predict_proba(valid[FEATURES])[:, 1]
        self.assertTrue(np.array_equal(before, fitted.steps[1][1].mean_))
        self.assertTrue(np.all(np.isfinite(p) & (p >= 0) & (p <= 1)))


if __name__ == '__main__':
    unittest.main()
