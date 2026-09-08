"""Q2 candidate experiment: prior-known covariates on the marginal attainment
probability model (week + first-record BMI baseline, +age, +conception type).

Reuses q2_probability's Gauss-Hermite random-intercept Bernoulli GLMM
framework but generalizes it to an arbitrary feature list, so the original
pipeline files remain untouched. Same subject-grouped outer folds and seed as
the accepted Q2 baseline. Static covariates are taken from each subject's
first record only (age, conception type) and never from future records.

This is a candidate screening run: reported metrics are outer-OOF; the
paired subject bootstrap resamples frozen OOF losses (no refit) and does not
represent full refit uncertainty. No decision optimization is run here.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermitenorm

from q2_probability import prepare as q2_prepare
from q2_probability import THRESHOLD, metrics
from q1_lmm_baseline import subject_group_folds, write_json, DEFAULT_SEED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q2_covariates'
SPECS = {
    'week_bmi': ['gestational_week', 'baseline_bmi'],
    'week_bmi_age': ['gestational_week', 'baseline_bmi', 'age_years'],
    'week_bmi_age_conception': ['gestational_week', 'baseline_bmi', 'age_years', 'non_natural'],
}
BOOT_REPLICATES = 2000


def normal_rule(n):
    nodes, weights = roots_hermitenorm(n)
    keep = weights > 0
    return nodes[keep], np.log(weights[keep]) - .5 * np.log(2 * np.pi)


def prepare():
    """Same male-fetus preparation as q2_probability, plus static covariates
    (first-record age and non-natural conception flag) and the target label."""
    data, audit = q2_prepare()
    first = (data.sort_values(['subject_id', 'test_date_iso', 'gestational_week', 'sample_id'])
             .drop_duplicates('subject_id').set_index('subject_id'))
    data['age_years'] = data.subject_id.map(first.age_years)
    data['non_natural'] = data.subject_id.map(
        (first.conception_type != '自然受孕').astype(int))
    assert data.groupby('subject_id').age_years.nunique().max() == 1
    assert data.groupby('subject_id').non_natural.nunique().max() == 1
    return data, audit


def design(data, features, center, scale):
    return np.column_stack([np.ones(len(data)),
                            (data[features].to_numpy() - center) / scale])


def objective(theta, x, y, groups, nodes, logweights):
    beta, sigma = theta[:-1], np.exp(theta[-1])
    z = x @ beta
    eta = z[:, None] + sigma * nodes[None, :]
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


def fit(data, features):
    center = data[features].mean().to_numpy()
    scale = data[features].std(ddof=0).to_numpy()
    if np.any(scale <= 0):
        raise ValueError('Constant feature in training set')
    x = design(data, features, center, scale)
    y = data.attained.to_numpy(dtype=float)
    groups = pd.factorize(data.subject_id, sort=True)[0]
    prevalence = y.mean()
    if not 0 < prevalence < 1:
        raise ValueError('Training requires both outcomes')
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
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-6})
            history.append({'nodes': count, 'success': bool(result.success),
                            'nll': float(result.fun),
                            'gradient_max': float(np.max(np.abs(result.jac))),
                            'message': str(result.message)})
            if result.success and np.max(np.abs(result.jac)) < .002:
                candidates.append(result)
        if not candidates:
            raise RuntimeError(f'No converged fit: {history}')
        best = min(candidates, key=lambda r: r.fun)
        check_nodes, check_weights = normal_rule(count * 2)
        check_value, check_grad = objective(best.x, x, y, groups, check_nodes, check_weights)
        eta = x @ best.x[:-1]
        sigma = np.exp(best.x[-1])
        p0 = expit(eta[:, None] + sigma * nodes) @ np.exp(weights)
        p1 = expit(eta[:, None] + sigma * check_nodes) @ np.exp(check_weights)
        if (abs(check_value - best.fun) < 1e-4
                and float(np.max(abs(p0 - p1))) < 1e-6
                and np.max(abs(check_grad)) < .003):
            if np.any(abs(best.x[:-1]) > 29.99) or abs(best.x[-1] - 3) < .01:
                raise RuntimeError('Fit hit upper parameter bound')
            return {'theta': best.x.tolist(), 'features': features,
                    'center': center.tolist(), 'scale': scale.tolist(),
                    'sigma': float(sigma), 'nodes': count * 2,
                    'quadrature_nll_difference': float(abs(check_value - best.fun)),
                    'quadrature_probability_difference': float(np.max(abs(p0 - p1))),
                    'checked_gradient_max': float(np.max(abs(check_grad))),
                    'optimization_history': history}
        starts = [best.x, best.x + np.array([0.] * len(best.x)) ]
        starts[-1][-1] += .1
    raise RuntimeError(f'Quadrature not stable: {history}')


def predict(model, data):
    x = design(data, model['features'], np.array(model['center']),
               np.array(model['scale']))
    nodes, weights = normal_rule(model['nodes'])
    p = expit((x @ np.array(model['theta'][:-1]))[:, None]
              + model['sigma'] * nodes) @ np.exp(weights)
    if not np.isfinite(p).all() or not np.all((p > 0) & (p < 1)):
        raise RuntimeError('Invalid marginal probabilities')
    return p


def run(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = prepare()
    folds = subject_group_folds(data.subject_id, 5, DEFAULT_SEED)
    if smoke:
        mask = data.subject_id.isin(folds[0])
        spec = 'week_bmi_age_conception'
        model = fit(data.loc[~mask], SPECS[spec])
        write_json(OUT / 'smoke.json',
                   {'model': model,
                    'heldout_metrics': metrics(data.loc[mask],
                                               predict(model, data.loc[mask]))})
        print('smoke ok', flush=True)
        return
    oof = data[['sample_id', 'subject_id', 'gestational_week', 'baseline_bmi',
                'age_years', 'non_natural', 'first_observed_week', 'attained']].copy()
    fold_metrics, histories = [], []
    # Guard: specs whose binary covariate lacks both classes in some training
    # fold are reported as infeasible rather than silently altered.
    for k, ids in enumerate(folds, 1):
        mask = data.subject_id.isin(ids)
        train, valid = data.loc[~mask], data.loc[mask]
        for spec, features in SPECS.items():
            if 'non_natural' in features and train.non_natural.nunique() < 2:
                fold_metrics.append({'fold': k, 'spec': spec, 'status': 'infeasible_binary_fold'})
                continue
            model = fit(train, features)
            pred = predict(model, valid)
            oof.loc[mask, spec] = pred
            fold_metrics.append({'fold': k, 'spec': spec, 'status': 'ok',
                                 **metrics(valid, pred)})
            histories.append({'fold': k, 'spec': spec, 'fit': model})
        print(f'covariate fold {k} done', flush=True)
    pd.DataFrame(fold_metrics).to_csv(OUT / 'fold_metrics.csv', index=False)
    summary = []
    for spec in SPECS:
        if spec in oof.columns and not oof[spec].isna().all():
            summary.append({'spec': spec, **metrics(data, oof[spec].to_numpy())})
    pd.DataFrame(summary).to_csv(OUT / 'comparison.csv', index=False)
    # Descriptive fixed-bin calibration per spec (same bins as Q2 baseline).
    cal_rows = []
    edges = np.array([0, .5, .7, .8, .9, .95, 1.])
    for spec in SPECS:
        if spec not in oof.columns or oof[spec].isna().all():
            continue
        sub = oof.dropna(subset=[spec])
        bins = np.clip(np.searchsorted(edges, sub[spec], side='right') - 1,
                       0, len(edges) - 2)
        for b in range(len(edges) - 1):
            g = sub.loc[bins == b]
            if not len(g):
                continue
            cal_rows.append({'spec': spec, 'bin_low': edges[b], 'bin_high': edges[b + 1],
                             'rows': len(g), 'subjects': g.subject_id.nunique(),
                             'mean_prediction': g[spec].mean(),
                             'observed_rate': g.attained.mean()})
    pd.DataFrame(cal_rows).to_csv(OUT / 'calibration.csv', index=False)
    # Frozen-OOF paired subject bootstrap vs the accepted week_bmi baseline.
    loss = pd.DataFrame({'subject': data.subject_id})
    base = (data.attained - oof.week_bmi) ** 2
    rng = np.random.default_rng(DEFAULT_SEED)
    boot = {}
    for spec in SPECS:
        if spec == 'week_bmi' or spec not in oof.columns:
            continue
        d = pd.Series((data.attained - oof[spec]) ** 2 - base).groupby(data.subject_id).mean().to_numpy()
        boot[spec] = np.quantile([d[rng.integers(len(d), size=len(d))].mean()
                                  for _ in range(BOOT_REPLICATES)], [.025, .975]).tolist()
    report = {'seed': DEFAULT_SEED, 'threshold': THRESHOLD,
              'input_sha256': audit['input_sha256'],
              'n_records': len(data), 'n_subjects': data.subject_id.nunique(),
              'specs': SPECS, 'comparison': summary,
              'subject_brier_difference_ci95_vs_week_bmi': boot,
              'bootstrap': (f'{BOOT_REPLICATES} subject resamples of frozen OOF '
                            'Brier differences vs week_bmi; no refit'),
              'covariate_rule': ('age and conception type taken from the '
                                 'subject first record only; no future information'),
              'fold_fits': histories,
              'note': 'candidate screening only; no decision optimization'}
    write_json(OUT / 'experiment.json', report)
    print(summary, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
