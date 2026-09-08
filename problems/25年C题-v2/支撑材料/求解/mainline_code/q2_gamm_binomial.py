"""Q2 GAMM-Binomial main model: random-intercept Bernoulli GLMM with
natural-cubic-spline week (and optionally BMI) terms, GH-integrated likelihood.

Specification ladder (degrade on failure, record cause):
  T_w4_b3        : cr(week,df=4) + cr(bmi,df=3)          (additive, top)
  T_w4_b_lin     : cr(week,df=4) + bmi                   (bmi back to linear)
  T_w3_b3        : cr(week,df=3) + cr(bmi,df=3)          (lower df)
  linear_mixed   : week + bmi                            (mixed logistic)

Selection: subject-grouped 5-fold OOF (same folds/seed as all Q2 runs);
best subject-equal-weight Brier wins; paired bootstrap vs the linear mixed
spec. Final best spec refit on all data -> probability surface, per-BMI
curves, calibration. No decision optimization here (see q2_decision).
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermitenorm

from q1_lmm_baseline import load_and_prepare, subject_group_folds, write_json, DEFAULT_SEED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q2_gamm_binomial'
THRESHOLD = .04

SPECS = {
    'T_w4_b3': "1 + cr(week_c, df=4, constraints='center') + cr(bmi_c, df=3, constraints='center')",
    'T_w4_blin': "1 + cr(week_c, df=4, constraints='center') + bmi_c",
    'T_w3_b3': "1 + cr(week_c, df=3, constraints='center') + cr(bmi_c, df=3, constraints='center')",
    'linear_mixed': "1 + week_c + bmi_c",
}
LADDER = ['T_w4_b3', 'T_w4_blin', 'T_w3_b3', 'linear_mixed']


def prepare():
    data, audit = load_and_prepare(ROOT / '附件.xlsx')
    first = (data.sort_values(['subject_id', 'test_date_iso', 'gestational_week',
                               'sample_id'])
             .drop_duplicates('subject_id').set_index('subject_id'))
    data['baseline_bmi'] = data.subject_id.map(first.bmi)
    data['attained'] = (data.y_fraction >= THRESHOLD).astype(int)
    assert data.groupby('subject_id').baseline_bmi.nunique().max() == 1
    return data, audit


def normal_rule(n):
    nodes, weights = roots_hermitenorm(n)
    keep = weights > 0
    return nodes[keep], np.log(weights[keep]) - .5 * np.log(2 * np.pi)


def objective(theta, x, y, groups, nodes, logweights):
    beta, sigma = theta[:-1], np.exp(theta[-1])
    eta = (x @ beta)[:, None] + sigma * nodes[None, :]
    loglik = y[:, None] * eta - np.logaddexp(0., eta)
    totals = np.zeros((groups.max() + 1, len(nodes)))
    np.add.at(totals, groups, loglik)
    logjoint = totals + logweights
    lognorm = logsumexp(logjoint, axis=1)
    posterior = np.exp(logjoint - lognorm[:, None])[groups]
    error = y[:, None] - expit(eta)
    beta_gradient = x.T @ np.sum(posterior * error, axis=1)
    sigma_gradient = np.sum(posterior * error * (sigma * nodes)[None, :])
    return -float(lognorm.sum()), -np.r_[beta_gradient, sigma_gradient]


def fit(train, spec):
    centers = train[['gestational_week', 'bmi']].mean()
    frame = train.copy()
    frame['week_c'] = frame.gestational_week - centers.gestational_week
    frame['bmi_c'] = frame.bmi - centers.bmi
    x_df = patsy.dmatrix(SPECS[spec], frame, return_type='dataframe')
    design_info = x_df.design_info
    x = np.asarray(x_df)
    print(f'  fit {spec}: x.shape={x.shape}, n_subj={frame.subject_id.nunique()}',
          flush=True)
    if np.linalg.cond(np.asarray(x)) > 1e8:
        raise RuntimeError('Ill-conditioned design')
    y = frame.attained.to_numpy(dtype=float)
    groups = pd.factorize(frame.subject_id, sort=True)[0]
    prevalence = y.mean()
    n_feat = x.shape[1]
    beta0 = np.log(prevalence / (1 - prevalence))
    starts = [np.r_[beta0, np.zeros(n_feat - 1), np.log(s)] for s in (.5, 1.5, 3.)]
    bounds = [(-30, 30)] * n_feat + [(-7, 3)]
    history = []
    for count in (80, 160, 320, 640):
        nodes, weights = normal_rule(count)
        candidates = []
        for start in starts:
            result = minimize(objective, start, args=(x, y, groups, nodes, weights),
                              jac=True, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-6})
            history.append({'nodes': count, 'success': bool(result.success),
                            'nll': float(result.fun),
                            'gradient_max': float(np.max(np.abs(result.jac))),
                            'message': str(result.message)})
            if result.success and np.max(np.abs(result.jac)) < .002:
                candidates.append(result)
        if not candidates:
            raise RuntimeError(f'No converged fit: {history[-3:]}')
        best = min(candidates, key=lambda r: r.fun)
        check_nodes, check_weights = normal_rule(count * 2)
        check_value, check_grad = objective(best.x, x, y, groups,
                                            check_nodes, check_weights)
        if (abs(check_value - best.fun) < 1e-4
                and np.max(np.abs(check_grad)) < .003):
            if np.any(abs(best.x[:-1]) > 29.99) or abs(best.x[-1] - 3) < .01:
                raise RuntimeError('Fit hit parameter bound')
            return {'theta': best.x.tolist(), 'spec': spec,
                    'design_info': design_info,
                    'centers': centers.to_dict(), 'nodes': count * 2,
                    'optimization_history': history}
        starts = [best.x, best.x + np.r_[np.zeros(n_feat), .1]]
    raise RuntimeError(f'Quadrature not stable: {history[-3:]}')


def predict(model, valid):
    frame = valid.copy()
    c = model['centers']
    frame['week_c'] = frame.gestational_week - c['gestational_week']
    frame['bmi_c'] = frame.bmi - c['bmi']
    x = np.asarray(patsy.build_design_matrices([model['design_info']], frame)[0])
    nodes, weights = normal_rule(model['nodes'])
    p = expit((x @ np.array(model['theta'][:-1]))[:, None]
              + np.exp(model['theta'][-1]) * nodes) @ np.exp(weights)
    if not np.isfinite(p).all() or not np.all((p > 0) & (p < 1)):
        raise RuntimeError('Invalid marginal probabilities')
    return p


def metrics(data, p):
    y = data.attained.to_numpy()
    sq = (y - p) ** 2
    ll = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return {'record_brier': float(sq.mean()),
            'subject_brier': float(pd.Series(sq).groupby(
                data.subject_id.to_numpy()).mean().mean()),
            'record_log_loss': float(ll.mean()),
            'subject_log_loss': float(pd.Series(ll).groupby(
                data.subject_id.to_numpy()).mean().mean())}


def run(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = prepare()
    folds = subject_group_folds(data.subject_id, 5, DEFAULT_SEED)
    if smoke:
        mask = data.subject_id.isin(folds[0])
        train, valid = data.loc[~mask], data.loc[mask]
        for spec in LADDER:
            try:
                p = predict(fit(train, spec), valid)
                print(spec, 'ok, subject_brier=%.4f' % metrics(valid, p)['subject_brier'])
            except Exception as exc:
                print(spec, 'FAILED:', str(exc)[:120])
        return
    oof = data[['sample_id', 'subject_id', 'gestational_week', 'baseline_bmi',
                'attained']].copy()
    fold_rows, degrad = [], []
    for k, ids in enumerate(folds, 1):
        mask = data.subject_id.isin(ids)
        train, valid = data.loc[~mask], data.loc[mask]
        # degradation ladder: try from top, first success is this fold's model
        chosen = None
        for spec in LADDER:
            try:
                p = predict(fit(train, spec), valid)
                chosen = {'fold': k, 'spec': spec, 'p': p}
                fold_rows.append({'fold': k, 'spec': spec,
                                  **metrics(valid, p)})
                oof.loc[mask, 'p_gamm'] = p
                oof.loc[mask, 'chosen_spec'] = spec
                break
            except Exception as exc:
                err = str(exc)[:200]
                degrad.append({'fold': k, 'spec': spec, 'error': err})
                print(f'  fold {k} {spec} FAILED: {err}', flush=True)
        if chosen is None:
            raise RuntimeError(f'Fold {k}: all specs failed')
        print(f'fold {k}: chosen {chosen["spec"]}', flush=True)
    pd.DataFrame(fold_rows).to_csv(OUT / 'fold_metrics.csv', index=False)
    summary = []
    for spec in LADDER:
        p = oof.loc[oof.chosen_spec == spec, 'p_gamm']
        idx = oof.index[oof.chosen_spec == spec]
        if len(p):
            summary.append({'spec': spec, 'n_rows': len(p),
                            **metrics(oof.loc[idx], p.to_numpy())})
    pd.DataFrame(summary).to_csv(OUT / 'chosen_summary.csv', index=False)
    write_json(OUT / 'experiment.json',
               {'seed': DEFAULT_SEED, 'threshold': THRESHOLD,
                'specs': SPECS, 'ladder': LADDER,
                'degradations': degrad,
                'chosen_summary': summary,
                'note': ('gated candidate: per-fold spec may degrade; '
                         'best-spec refit handled in q2_decision stage')})
    print('degradations:', len(degrad), flush=True)
    for d in degrad[:10]:
        print(' ', d['fold'], d['spec'], d['error'][:80], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
