"""Synthetic example for the ordered-DP tool (gate-compliant: no real-model
recommendations are produced or claimed)."""
from pathlib import Path

import numpy as np

from q2_dp_tool import plan, pareto
from q1_lmm_baseline import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q2_dp_tool'
SEED = 20250904


def synthetic():
    rng = np.random.default_rng(SEED)
    n = 120
    bmi = np.sort(rng.uniform(21, 46, n))
    weeks = np.arange(10, 25.01, 0.5)
    z = (weeks[None, :] - (11.0 + (bmi[:, None] - 21.0) * 0.45)) / 2.0
    p = 1.0 / (1.0 + np.exp(-z))
    return bmi, p, weeks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bmi, p, weeks = synthetic()
    rows, table = pareto(bmi, p, weeks, max_segments=3,
                         weights=[0.5, 1.0, 2.0], coverage_q=0.85)
    summary = {'scope': ('synthetic feasibility demo ONLY; the project gate '
                         'forbids recommending real test weeks until a '
                         'probability model passes decision gates'),
               'pareto_by_w_late': table,
               'plans': [{'w_late': w, **r} for w, r in zip([0.5, 1.0, 2.0], rows)]}
    write_json(OUT / 'example.json', summary)
    print('synthetic DP example written (gate-compliant)')


if __name__ == '__main__':
    main()
