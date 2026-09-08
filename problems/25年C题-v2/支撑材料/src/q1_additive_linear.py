"""Q1 additive-linear control experiment.

Separates two effects that the earlier spline comparison confounded:
(1) dropping the week x bmi interaction and (2) adding nonlinear smooth terms.
Same subject-grouped outer folds/seed as the accepted LMM baseline and the
spline comparison. Candidate specs (all random-intercept, logit scale):
- LMM    : 1 + week_c * bmi_c            (accepted baseline)
- AddLin : 1 + week_c + bmi_c            (additive linear control)
- Spline : 1 + cr(week_c) + cr(bmi_c)    (df selected by inner folds, reuse)

Existing q1_comparison outputs are not touched; this experiment writes its
own outputs/manifest. OOF predictions for unseen subjects use fixed effects
only (random effect set to zero), same as the baselines.
"""
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from q1_lmm_baseline import load_and_prepare, subject_group_folds, regression_metrics, write_json
from q1_spline_comparison import fit as fit_spline, predict as predict_spline, \
    select_df, subject_mse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q1_additive'
SEED = 20250904


def fit_addlin(train):
    centers = train[['gestational_week', 'bmi']].mean()
    frame = train.copy()
    frame['week_c'] = frame.gestational_week - centers.gestational_week
    frame['bmi_c'] = frame.bmi - centers.bmi
    x = patsy.dmatrix('1 + week_c + bmi_c', frame, return_type='dataframe')
    if np.linalg.cond(x) > 1e7:
        raise RuntimeError('Ill-conditioned design')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sm.MixedLM(frame.y_logit.to_numpy(), np.asarray(x),
                            groups=frame.subject_id).fit(reml=False, method='lbfgs',
                                                         maxiter=1000, disp=False)
    if not result.converged or caught:
        raise RuntimeError(f'Fit not clean: {[str(w.message) for w in caught]}')
    return result, x.design_info, centers


def design_addlin(model, frame):
    _, info, centers = model
    frame = frame.copy()
    frame['week_c'] = frame.gestational_week - centers.gestational_week
    frame['bmi_c'] = frame.bmi - centers.bmi
    return np.asarray(patsy.build_design_matrices([info], frame)[0])


def predict_addlin(model, frame):
    pred = design_addlin(model, frame) @ model[0].fe_params
    if not np.isfinite(pred).all():
        raise RuntimeError('Nonfinite prediction')
    return pred


def run(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = load_and_prepare(ROOT / '附件.xlsx')
    folds = subject_group_folds(data.subject_id, 5, SEED)
    if smoke:
        mask = data.subject_id.isin(folds[0])
        m = fit_addlin(data.loc[~mask])
        print(regression_metrics(data.loc[mask].y_logit.to_numpy(),
                                 predict_addlin(m, data.loc[mask])))
        return
    predictions = data[['sample_id', 'subject_id', 'gestational_week', 'bmi',
                        'y_logit']].copy()
    rows, selections = [], []
    for fold, ids in enumerate(folds, 1):
        mask = data.subject_id.isin(ids)
        train, valid = data.loc[~mask], data.loc[mask]
        df, scores = select_df(train, SEED + fold)
        selections.append({'outer_fold': fold, 'selected_spline_df': df,
                           'inner_scores': scores})
        predictions.loc[mask, 'fold'] = fold
        lmm = fit_spline(train, 0)
        add = fit_addlin(train)
        spl = fit_spline(train, df)
        preds = {'LMM': predict_spline(lmm, valid),
                 'AddLin': predict_addlin(add, valid),
                 'Spline': predict_spline(spl, valid)}
        for label, pred in preds.items():
            predictions.loc[mask, label] = pred
            rows.append({'fold': fold, 'model': label,
                         **regression_metrics(valid.y_logit.to_numpy(), pred),
                         'subject_rmse': np.sqrt(subject_mse(valid, pred))})
        print(f'Fold {fold}: spline df={df}', flush=True)
    predictions.to_csv(OUT / 'oof_predictions.csv', index=False)
    pd.DataFrame(rows).to_csv(OUT / 'fold_metrics.csv', index=False)
    summary = []
    for label in ['LMM', 'AddLin', 'Spline']:
        summary.append({'model': label,
                        **regression_metrics(data.y_logit.to_numpy(),
                                             predictions[label].to_numpy()),
                        'subject_rmse': np.sqrt(subject_mse(data, predictions[label].to_numpy()))})
    pd.DataFrame(summary).to_csv(OUT / 'comparison.csv', index=False)
    losses = pd.DataFrame({'subject': data.subject_id,
                           'LMM': (data.y_logit - predictions.LMM) ** 2,
                           'AddLin': (data.y_logit - predictions.AddLin) ** 2,
                           'Spline': (data.y_logit - predictions.Spline) ** 2
                           }).groupby('subject').mean()
    rng = np.random.default_rng(SEED)
    boot_add = []
    boot_spline_vs_add = []
    for _ in range(2000):
        sample = losses.iloc[rng.integers(len(losses), size=len(losses))]
        boot_add.append(np.sqrt(sample.AddLin.mean()) - np.sqrt(sample.LMM.mean()))
        boot_spline_vs_add.append(np.sqrt(sample.Spline.mean())
                                  - np.sqrt(sample.AddLin.mean()))
    write_json(OUT / 'experiment.json',
               {'seed': SEED, 'outer_folds': 5, 'inner_folds': 3,
                'selections': selections,
                'delta_subject_rmse_addlin_vs_lmm_ci95':
                    np.quantile(boot_add, [.025, .975]).tolist(),
                'delta_subject_rmse_spline_vs_addlin_ci95':
                    np.quantile(boot_spline_vs_add, [.025, .975]).tolist(),
                'bootstrap': ('2000 paired subject resamples of frozen OOF '
                              'RMSE differences; no refit'),
                'comparison': summary,
                'note': ('AddLin isolates removing the week x bmi interaction; '
                         'Spline vs AddLin isolates nonlinearity; both are '
                         'frozen-OOF comparisons without model-selection '
                         'adjustment')})
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
