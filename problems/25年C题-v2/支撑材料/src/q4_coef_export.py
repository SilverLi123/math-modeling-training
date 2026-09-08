"""Q4 interpretability export: standardized coefficients of the Elastic-Net
main model used for the outer-OOF predictions.

The outer-OOF probabilities in experiment.json come from one final model per
outer fold (fit on that fold's full training part at the inner-selected C).
This script reconstructs exactly those five fold models (same data, same
subject-grouped split, same seed, C taken from experiment.json fold
selections), and exports:
  - outputs/q4_model/fold_coefficients.csv : per-fold standardized beta
    (intercept stored separately), rows in FEATURES order;
  - outputs/q4_model/coefficients_top6.json : top-6 features by median
    |standardized beta| across the five fold models.

No prediction/CV metrics are recomputed: fold C's and thresholds come from the
frozen experiment.json; coefficients are a transparency artifact, not a new
experiment.  All values are on the same scale the model was fitted on
(z-scores of the fold training part), so a coefficient of -2.03 for gc21 means
a one-standard-deviation rise in 21-chr GC content lowers log-odds by ~2.03.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

from q4_model import (FEATURES, SEED, load, standardize, fit_elasticnet)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q4_model'
# human labels aligned with FEATURES order
LABELS = ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值',
          'X染色体的Z值', 'X染色体浓度', '总体GC含量',
          '13号染色体的GC含量', '18号染色体的GC含量',
          '21号染色体的GC含量', '在参考基因组上比对的比例',
          '重复读段的比例', '被过滤掉读段数的比例',
          'log10原始读段数', 'log10唯一比对读段数',
          '孕妇BMI', '年龄']


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'experiment.json', encoding='utf-8') as f:
        exp = json.load(f)
    fold_c = {s['fold']: s['selected_C'] for s in exp['fold_selections']}
    df, _ = load()
    y = df.y.to_numpy()
    subjects = df.subject_id.to_numpy()
    Xraw = df[FEATURES].to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    betas, intercepts, Cs = [], [], []
    for k, (itr, ite) in enumerate(outer.split(Xraw, y, subjects), 1):
        Xtr = Xraw[itr]
        ytr = y[itr]
        C = fold_c[k]
        Xs, _, _ = standardize(Xtr)
        m = fit_elasticnet(Xs, ytr, C)
        betas.append(m['beta'])
        intercepts.append(m['intercept'])
        Cs.append(C)
    B = np.vstack(betas)                     # (5, n_feat)
    med = np.median(B, axis=0)
    med_abs = np.abs(med)
    order = np.argsort(-med_abs)
    rows = [dict(fold=i + 1, C=Cs[i],
                 intercept=float(intercepts[i]),
                 **{LABELS[j]: float(B[i, j]) for j in range(len(FEATURES))})
            for i in range(B.shape[0])]
    pd.DataFrame(rows).to_csv(OUT / 'fold_coefficients.csv', index=False,
                              encoding='utf-8-sig')
    top = []
    for rank, j in enumerate(order[:6], 1):
        top.append({'rank': rank, 'feature': LABELS[j], 'code': FEATURES[j],
                    'median_beta': float(med[j]),
                    'min': float(B[:, j].min()),
                    'max': float(B[:, j].max()),
                    'nonzero_folds': int((B[:, j] != 0).sum())})
    with open(OUT / 'coefficients_top6.json', 'w', encoding='utf-8') as f:
        json.dump({'n_fold_models': int(B.shape[0]),
                   'l1_ratio': exp['l1_ratio'],
                   'fold_C': fold_c,
                   'note': ('median of standardized betas across the five outer '
                            'fold models; scale = z-score of fold training part'),
                   'top6_by_abs_median_beta': top},
                  f, ensure_ascii=False, indent=2)
    print('wrote fold_coefficients.csv and coefficients_top6.json')
    for t in top:
        print(f"rank {t['rank']}: {t['feature']:<14} median={t['median_beta']:+.4f}"
              f"  (min {t['min']:+.3f}, max {t['max']:+.3f}, "
              f"nonzero {t['nonzero_folds']}/5)")


if __name__ == '__main__':
    run()
