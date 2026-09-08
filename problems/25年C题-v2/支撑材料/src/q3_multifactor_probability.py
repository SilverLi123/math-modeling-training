"""Q3 multifactor probability model: A (BMI+age+conception) vs B (height+weight+age).
Same GH random-intercept framework, same 5-fold subject CV. Compares OOF Brier
against the Q2 week+bmi baseline to quantify multifactor gain (or lack thereof).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermitenorm

from q1_lmm_baseline import load_and_prepare, subject_group_folds, write_json, DEFAULT_SEED
from q2_gamm_binomial import normal_rule, objective, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q3_multifactor'
THRESHOLD = .04


def prepare():
    data, audit = load_and_prepare(ROOT / '附件.xlsx')
    first = (data.sort_values(['subject_id', 'test_date_iso', 'gestational_week',
                               'sample_id'])
             .drop_duplicates('subject_id').set_index('subject_id'))
    data['baseline_bmi'] = data.subject_id.map(first.bmi)
    data['baseline_age'] = data.subject_id.map(first.age_years)
    data['baseline_height'] = data.subject_id.map(first.height_cm)
    data['baseline_weight'] = data.subject_id.map(first.weight_kg)
    data['non_natural'] = data.subject_id.map(
        (first.conception_type != '自然受孕').astype(int))
    data['attained'] = (data.y_fraction >= THRESHOLD).astype(int)
    return data, audit


def design(data, features, center, scale):
    return np.column_stack([np.ones(len(data)),
                            (data[features].to_numpy() - center) / scale])


def fit_gh(data, features):
    center = data[features].mean().to_numpy()
    scale = data[features].std(ddof=0).to_numpy()
    if np.any(scale <= 0):
        raise ValueError('Constant feature')
    x = design(data, features, center, scale)
    y = data.attained.to_numpy(dtype=float)
    groups = pd.factorize(data.subject_id, sort=True)[0]
    prevalence = y.mean()
    n_feat = x.shape[1]
    beta0 = np.log(prevalence / (1 - prevalence))
    starts = [np.r_[beta0, np.zeros(n_feat - 1), np.log(s)] for s in (.5, 1.5, 3.)]
    bounds = [(-30, 30)] * n_feat + [(-7, 3)]
    for count in (80, 160, 320, 640):
        nodes, weights = normal_rule(count)
        candidates = []
        for start in starts:
            result = minimize(objective, start, args=(x, y, groups, nodes, weights),
                              jac=True, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-6})
            if result.success and np.max(np.abs(result.jac)) < .002:
                candidates.append(result)
        if not candidates:
            raise RuntimeError(f'No converged fit at nodes={count}')
        best = min(candidates, key=lambda r: r.fun)
        check_nodes, check_weights = normal_rule(count * 2)
        check_value, check_grad = objective(best.x, x, y, groups,
                                            check_nodes, check_weights)
        if (abs(check_value - best.fun) < 1e-4
                and np.max(np.abs(check_grad)) < .003):
            return {'theta': best.x.tolist(), 'features': features,
                    'center': center.tolist(), 'scale': scale.tolist(),
                    'nodes': count * 2}
        starts = [best.x, best.x + np.r_[np.zeros(n_feat), .1]]
    raise RuntimeError('Quadrature not stable')


def predict(model, data):
    x = design(data, model['features'], np.array(model['center']),
               np.array(model['scale']))
    nodes, weights = normal_rule(model['nodes'])
    return expit((x @ np.array(model['theta'][:-1]))[:, None]
                 + np.exp(model['theta'][-1]) * nodes) @ np.exp(weights)


SPECS = {
    'Q2_baseline': ['gestational_week', 'baseline_bmi'],
    'Q3_A_bmi_age_ivf': ['gestational_week', 'baseline_bmi', 'baseline_age', 'non_natural'],
    'Q3_B_hw_age': ['gestational_week', 'baseline_height', 'baseline_weight', 'baseline_age'],
}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = prepare()
    folds = subject_group_folds(data.subject_id, 5, DEFAULT_SEED)
    oof = data[['sample_id', 'subject_id', 'gestational_week', 'baseline_bmi',
                'attained']].copy()
    fold_rows = []
    for k, ids in enumerate(folds, 1):
        mask = data.subject_id.isin(ids)
        train, valid = data.loc[~mask], data.loc[mask]
        for spec, features in SPECS.items():
            try:
                m = fit_gh(train, features)
                p = predict(m, valid)
                oof.loc[mask, spec] = p
                fold_rows.append({'fold': k, 'spec': spec, **metrics(valid, p)})
                print(f'fold {k} {spec}: sbrier={metrics(valid,p)["subject_brier"]:.4f}', flush=True)
            except Exception as exc:
                fold_rows.append({'fold': k, 'spec': spec, 'status': f'FAILED: {str(exc)[:80]}'})
                print(f'fold {k} {spec} FAILED', flush=True)
    pd.DataFrame(fold_rows).to_csv(OUT / 'fold_metrics.csv', index=False)
    summary = []
    for spec in SPECS:
        if spec in oof.columns and not oof[spec].isna().all():
            summary.append({'spec': spec, **metrics(oof, oof[spec].to_numpy())})
    pd.DataFrame(summary).to_csv(OUT / 'comparison.csv', index=False)
    # bootstrap vs baseline
    base = (data.attained - oof.Q2_baseline) ** 2
    boot = {}
    rng = np.random.default_rng(DEFAULT_SEED)
    for spec in ['Q3_A_bmi_age_ivf', 'Q3_B_hw_age']:
        if spec not in oof.columns:
            continue
        d = pd.Series((data.attained - oof[spec]) ** 2 - base).groupby(
            data.subject_id).mean().to_numpy()
        boot[spec] = np.quantile([d[rng.integers(len(d), size=len(d))].mean()
                                  for _ in range(2000)], [.025, .975]).tolist()
    write_json(OUT / 'experiment.json',
               {'seed': DEFAULT_SEED, 'specs': SPECS, 'summary': summary,
                'brier_diff_ci95_vs_baseline': boot,
                'bootstrap': '2000 subject resamples of frozen OOF Brier differences'})
    print('summary:', summary, flush=True)
    print('bootstrap:', boot, flush=True)


if __name__ == '__main__':
    run()
