"""Q4 main model: A_any direct Elastic-Net logistic with subject-level
stratified group CV and NESTED threshold selection.

Elastic-Net logistic regression is fitted with our own deterministic FISTA
(proximal gradient) implementation: no sklearn solver deprecation or
convergence warnings, full control over stopping criteria. The intercept is
not penalized; L1/L2 trade is fixed at l1_ratio=0.5 and the strength grid C
is selected inside the outer training folds only.

Design and honesty notes (see 题目分析报告.md Q4 contract):
- Independent unit is the pregnant woman; all her rows stay in one fold.
- Outcome A_any (row label contains T13/T18/T21, regex parsed, no
  delimiter). Type-level positives are too few (T13/T21 ~5-8 subjects) for
  separate per-type CV models; type information stays descriptive in
  outputs/q4_label_audit.
- Outer: stratified subject-grouped 5-fold CV.
- Inner (3 subject-grouped folds inside each outer training part): C by mean
  PR-AUC, operating threshold by recall-weighted F2 on inner OOF
  probabilities. Outer test rows are untouched by any selection.
- Pooled outer-OOF evaluation: AUC, PR-AUC, Brier, fixed-bin calibration,
  and confusion metrics at each test row's own nested threshold.
- Uncertainty: 2000 subject resamples of frozen outer-OOF rows for AUC and
  Brier (no refit); reported as CIs, not model-selection-adjusted bands.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_recall_curve)

from q1_lmm_baseline import write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q4_model'
SEED = 20250904
C_GRID = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
L1_RATIO = 0.5
COL_MAP = {
    '孕妇代码': 'subject_id', '孕妇BMI': 'bmi', '年龄': 'age',
    '13号染色体的Z值': 'z13', '18号染色体的Z值': 'z18',
    '21号染色体的Z值': 'z21', 'X染色体的Z值': 'zx',
    'X染色体浓度': 'x_conc', 'GC含量': 'gc',
    '13号染色体的GC含量': 'gc13', '18号染色体的GC含量': 'gc18',
    '21号染色体的GC含量': 'gc21', '在参考基因组上比对的比例': 'map_ratio',
    '重复读段的比例': 'dup_ratio', '被过滤掉读段数的比例': 'filter_ratio',
    '原始读段数': 'raw_reads', '唯一比对的读段数': 'unique_reads',
    '染色体的非整倍体': 'label_raw',
}
FEATURES = ['z13', 'z18', 'z21', 'zx', 'x_conc', 'gc', 'gc13', 'gc18', 'gc21',
            'map_ratio', 'dup_ratio', 'filter_ratio', 'log_raw', 'log_unique',
            'bmi', 'age']


# ---------------------------------------------------------------- FISTA
def fit_elasticnet(X, y, C, l1_ratio=L1_RATIO, class_weight='balanced',
                   max_iter=20000, tol=1e-9):
    n, k = X.shape
    y = np.asarray(y, float)
    if class_weight == 'balanced':
        n1, n0 = float(y.sum()), float(n - y.sum())
        w = np.where(y == 1, n / (2 * n1), n / (2 * n0))
    else:
        w = np.ones(n)
    l1 = l1_ratio / C
    l2 = (1 - l1_ratio) / C

    def loss_grad(beta, b0):
        eta = np.clip(X @ beta + b0, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        loss = float((w * (np.logaddexp(0, eta) - eta * y)).sum())
        loss += 0.5 * l2 * float(beta @ beta) + l1 * float(np.abs(beta).sum())
        grad = X.T @ (w * (p - y)) + l2 * beta
        gb0 = float((w * (p - y)).sum())
        return loss, grad, gb0

    beta = np.zeros(k)
    b0 = float(np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3)))
    sig_max = np.linalg.norm(X, 2)
    step = 1.0 / (0.25 * sig_max * sig_max + l2 + 1e-12)
    t_new = 1.0
    z_beta, z_b0 = beta.copy(), b0
    prev_loss = None
    iters = 0
    for iters in range(1, max_iter + 1):
        _, grad, gb0 = loss_grad(z_beta, z_b0)
        trial_beta = z_beta - step * grad
        trial_b0 = z_b0 - step * gb0
        prox_beta = np.sign(trial_beta) * np.maximum(
            np.abs(trial_beta) - step * l1, 0.0)
        trial_loss, _, _ = loss_grad(prox_beta, trial_b0)
        if prev_loss is not None and abs(prev_loss - trial_loss) < tol:
            beta, b0 = prox_beta, trial_b0
            break
        prev_loss = trial_loss
        beta, b0 = prox_beta, trial_b0
        t_old = t_new
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t_old * t_old))
        mix = (t_old - 1) / t_new
        z_beta = beta + mix * (beta - z_beta)
        z_b0 = b0 + mix * (b0 - z_b0)
    return {'beta': beta, 'intercept': b0, 'iterations': int(iters),
            'converged': prev_loss is not None}


def predict_proba_beta(model, X):
    eta = np.clip(X @ model['beta'] + model['intercept'], -30, 30)
    return 1.0 / (1.0 + np.exp(-eta))


# ---------------------------------------------------------------- data
def load():
    df = pd.read_excel(ROOT / '附件.xlsx', sheet_name='女胎检测数据')
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if str(c).startswith('Unnamed')])
    df = df.rename(columns=COL_MAP)
    df['y'] = df.label_raw.notna().astype(int)
    df['log_raw'] = np.log10(df.raw_reads)
    df['log_unique'] = np.log10(df.unique_reads)
    before = len(df)
    df = df.dropna(subset=FEATURES + ['subject_id'])
    return df, {'rows_before_dropna': before, 'rows_after_dropna': len(df),
                'dropped_rows': before - len(df)}


def threshold_metrics(y, p, tau):
    pred = (p >= tau).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return {'tau': float(tau),
            'sensitivity': tp / (tp + fn) if tp + fn else np.nan,
            'specificity': tn / (tn + fp) if tn + fp else np.nan,
            'ppv': tp / (tp + fp) if tp + fp else np.nan,
            'npv': tn / (tn + fn) if tn + fn else np.nan,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def f2_threshold(y, p):
    prec, rec, th = precision_recall_curve(y, p)
    with np.errstate(divide='ignore', invalid='ignore'):
        f2 = 5 * prec * rec / (4 * prec + rec)
    i = int(np.nanargmax(f2))
    return float(th[min(i, len(th) - 1)])


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd, mu, sd


# ---------------------------------------------------------------- run
def run():
    OUT.mkdir(parents=True, exist_ok=True)
    df, data_audit = load()
    y = df.y.to_numpy()
    subjects = df.subject_id.to_numpy()
    Xraw = df[FEATURES].to_numpy()
    z_abs = df[['z13', 'z18', 'z21']].abs().max(axis=1).to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = df[['subject_id']].copy()
    oof['y'] = y
    oof['z_abs'] = z_abs
    oof['fold'] = 0
    oof['tau'] = np.nan
    oof['p_elasticnet'] = np.nan
    fold_rows = []
    selections = []
    for k, (itr, ite) in enumerate(outer.split(Xraw, y, subjects), 1):
        Xtr, Xte = Xraw[itr], Xraw[ite]
        ytr, yte = y[itr], y[ite]
        gtr = subjects[itr]
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                     random_state=SEED + 7)
        inner_rows = []
        p_inner = {}
        for C in C_GRID:
            pC = np.zeros(len(Xtr))
            for jtr, jte in inner.split(Xtr, ytr, gtr):
                Xs, mu, sd = standardize(Xtr[jtr])
                m = fit_elasticnet(Xs, ytr[jtr], C)
                Xsj = (Xtr[jte] - mu) / sd
                pC[jte] = predict_proba_beta(m, Xsj)
            p_inner[C] = pC
            inner_rows.append({'C': C,
                               'inner_auc': float(roc_auc_score(ytr, pC)),
                               'inner_pr_auc': float(
                                   average_precision_score(ytr, pC))})
        best = max(inner_rows, key=lambda r: r['inner_pr_auc'])
        C_best = best['C']
        tau = f2_threshold(ytr, p_inner[C_best])
        Xs_all, mu, sd = standardize(Xtr)
        final = fit_elasticnet(Xs_all, ytr, C_best)
        p_test = predict_proba_beta(final, (Xte - mu) / sd)
        oof.loc[df.index[ite], 'p_elasticnet'] = p_test
        oof.loc[df.index[ite], 'fold'] = k
        oof.loc[df.index[ite], 'tau'] = tau
        tm = threshold_metrics(yte, p_test, tau)
        fold_rows.append({'fold': k, 'C': C_best, 'tau': tau,
                          'auc': float(roc_auc_score(yte, p_test)),
                          'pr_auc': float(average_precision_score(yte, p_test)),
                          **tm})
        selections.append({'fold': k, 'selected_C': C_best,
                           'selected_tau_inner_f2': tau,
                           'inner_rows': inner_rows})
        print(f'Q4 fold {k}: C={C_best}, tau={tau:.3f}, '
              f'test PR-AUC={fold_rows[-1]["pr_auc"]:.3f}', flush=True)
    pd.DataFrame(fold_rows).to_csv(OUT / 'fold_metrics.csv', index=False)
    p = oof.p_elasticnet.to_numpy()
    pooled = {'auc': float(roc_auc_score(y, p)),
              'pr_auc': float(average_precision_score(y, p)),
              'brier': float(brier_score_loss(y, p))}
    pooled_at_tau = threshold_metrics(y, p, float(oof.tau.median()))
    pooled_at_tau['rows_flagged_at_own_tau'] = int(
        (oof.p_elasticnet >= oof.tau).sum())
    frame = pd.DataFrame({'subject': subjects, 'y': y, 'p': p})
    uni = frame.subject.unique()
    rng = np.random.default_rng(SEED)
    aucs, briers = [], []
    for _ in range(2000):
        pick = rng.integers(len(uni), size=len(uni))
        g = frame.loc[frame.subject.isin(uni[pick])]
        aucs.append(roc_auc_score(g.y, g.p))
        briers.append(brier_score_loss(g.y, g.p))
    z_auc = float(roc_auc_score(y, z_abs))
    z_pr = float(average_precision_score(y, z_abs))
    cal = []
    edges = np.array([0, .5, .7, .8, .9, .95, 1.])
    bins = np.clip(np.searchsorted(edges, p, side='right') - 1, 0,
                   len(edges) - 2)
    for b in range(len(edges) - 1):
        m = bins == b
        if not m.any():
            continue
        cal.append({'bin_low': float(edges[b]), 'bin_high': float(edges[b + 1]),
                    'rows': int(m.sum()),
                    'subjects': int(pd.Series(subjects[m]).nunique()),
                    'mean_prediction': float(p[m].mean()),
                    'observed_rate': float(y[m].mean())})
    oof.to_csv(OUT / 'oof_predictions.csv', index=False)
    pd.DataFrame(cal).to_csv(OUT / 'calibration.csv', index=False)
    report = {
        'seed': SEED, 'data_audit': data_audit,
        'n_rows': int(len(df)), 'n_subjects': int(pd.Series(subjects).nunique()),
        'positive_rows': int(y.sum()),
        'positive_subjects': int(pd.Series(subjects[y == 1]).nunique()),
        'features': FEATURES, 'C_grid': C_GRID, 'l1_ratio': L1_RATIO,
        'outer': 'StratifiedGroupKFold 5 by subject',
        'inner': '3 subject-grouped folds; C by inner PR-AUC; tau by F2',
        'pooled_oof_summary': pooled,
        'pooled_oof_at_median_tau': pooled_at_tau,
        'z_baseline': {'auc': z_auc, 'pr_auc': z_pr,
                       'note': 'continuous |Z|max score baseline, not prob'},
        'subject_bootstrap_auc_ci95': np.quantile(aucs, [.025, .975]).tolist(),
        'subject_bootstrap_brier_ci95': np.quantile(briers, [.025, .975]).tolist(),
        'bootstrap': ('2000 subject resamples of frozen outer-OOF rows; '
                      'no refit, no model-selection adjustment'),
        'fold_selections': selections,
        'per_type_model_note': ('no separate per-type CV models: T13/T21 have '
                                'only ~5-8 positive subjects'),
        'type_labels_note': 'labels parsed with regex T13|T18|T21 (no delimiter)',
    }
    write_json(OUT / 'experiment.json', report)
    print('pooled:', pooled, flush=True)
    print('nested-tau pooled:', pooled_at_tau, flush=True)


if __name__ == '__main__':
    run()
