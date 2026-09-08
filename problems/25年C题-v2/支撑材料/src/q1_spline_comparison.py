"""Q1: nested subject CV for an unpenalized natural-spline mixed model."""
from pathlib import Path
import argparse
import warnings
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from q1_lmm_baseline import load_and_prepare, subject_group_folds, regression_metrics, write_json

ROOT = Path(__file__).resolve().parents[1]
SEED = 20250904
CANDIDATES = (3, 4)


def fit(train, df=0):
    centers = train[['gestational_week', 'bmi']].mean()
    frame = train.copy()
    frame['week_c'] = frame.gestational_week - centers.gestational_week
    frame['bmi_c'] = frame.bmi - centers.bmi
    formula = ('1 + week_c * bmi_c' if df == 0 else
               f"1 + cr(week_c, df={df}, constraints='center') + cr(bmi_c, df={df}, constraints='center')")
    x = patsy.dmatrix(formula, frame, return_type='dataframe')
    if np.linalg.cond(x) > 1e7:
        raise RuntimeError('Ill-conditioned design')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = sm.MixedLM(frame.y_logit.to_numpy(), np.asarray(x), groups=frame.subject_id).fit(
            reml=False, method='lbfgs', maxiter=1000, disp=False)
    if not result.converged or caught:
        raise RuntimeError(f'Fit not clean: converged={result.converged}, warnings={[str(w.message) for w in caught]}')
    return result, x.design_info, centers


def design(model, frame):
    _, info, centers = model
    frame = frame.copy()
    frame['week_c'] = frame.gestational_week - centers.gestational_week
    frame['bmi_c'] = frame.bmi - centers.bmi
    return np.asarray(patsy.build_design_matrices([info], frame)[0])


def predict(model, frame):
    result = design(model, frame) @ model[0].fe_params
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite prediction')
    return result


def subject_mse(frame, pred):
    return float(pd.Series((frame.y_logit.to_numpy()-pred)**2).groupby(frame.subject_id.to_numpy()).mean().mean())


def select_df(train, seed):
    scores = []
    folds = subject_group_folds(train.subject_id, 3, seed)
    for df in CANDIDATES:
        predicted = np.full(len(train), np.nan)
        for ids in folds:
            mask = train.subject_id.isin(ids).to_numpy()
            predicted[mask] = predict(fit(train.loc[~mask], df), train.loc[mask])
        scores.append({'df': df, 'subject_mse': subject_mse(train, predicted)})
    return min(scores, key=lambda r: (r['subject_mse'], r['df']))['df'], scores


def run(smoke=False):
    data, _ = load_and_prepare(ROOT / '附件.xlsx')
    out = ROOT / 'outputs' / 'q1_comparison'
    out.mkdir(parents=True, exist_ok=True)
    folds = subject_group_folds(data.subject_id, 5, SEED)
    if smoke:
        mask = data.subject_id.isin(folds[0])
        results = {}
        for df in (0, 3):
            model = fit(data.loc[~mask], df)
            results[str(df)] = regression_metrics(data.loc[mask].y_logit.to_numpy(), predict(model, data.loc[mask]))
        write_json(out / 'smoke.json', results)
        print(results)
        return
    predictions = data[['sample_id','subject_id','gestational_week','bmi','y_logit']].copy()
    rows, selections = [], []
    for fold, ids in enumerate(folds, 1):
        mask = data.subject_id.isin(ids)
        train, valid = data.loc[~mask], data.loc[mask]
        df, scores = select_df(train, SEED+fold)
        selections.append({'outer_fold': fold, 'selected_df': df, 'inner_scores': scores})
        predictions.loc[mask, 'fold'] = fold
        for label, complexity in [('LMM',0), ('Spline',df)]:
            model = fit(train, complexity)
            pred = predict(model, valid)
            predictions.loc[mask,label] = pred
            rows.append({'fold':fold,'model':label,'df':complexity,
                         **regression_metrics(valid.y_logit.to_numpy(),pred),
                         'subject_rmse':np.sqrt(subject_mse(valid,pred))})
        print(f'Fold {fold}: selected spline df={df}', flush=True)
    predictions.to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
    summary = []
    for label in ['LMM','Spline']:
        summary.append({'model':label,**regression_metrics(data.y_logit.to_numpy(),predictions[label].to_numpy()),
                        'subject_rmse':np.sqrt(subject_mse(data,predictions[label].to_numpy()))})
    pd.DataFrame(summary).to_csv(out/'comparison.csv',index=False)
    # Paired subject bootstrap of fixed OOF errors: evaluation uncertainty,
    # not refit/model-selection uncertainty.
    losses = pd.DataFrame({'subject':data.subject_id,
                          'LMM':(data.y_logit-predictions.LMM)**2,
                          'Spline':(data.y_logit-predictions.Spline)**2}).groupby('subject').mean()
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(2000):
        sample = losses.iloc[rng.integers(len(losses),size=len(losses))]
        boot.append(np.sqrt(sample.Spline.mean())-np.sqrt(sample.LMM.mean()))
    final_df, final_scores = select_df(data, SEED+100)
    model = fit(data,final_df)
    effects=[]
    for variable in ['gestational_week','bmi']:
        grid=pd.DataFrame({'gestational_week':np.repeat(data.gestational_week.median(),100),
                           'bmi':np.repeat(data.bmi.median(),100)})
        grid[variable]=np.linspace(data[variable].min(),data[variable].max(),100)
        x=design(model,grid)
        covariance=np.asarray(model[0].cov_params())[:x.shape[1],:x.shape[1]]
        se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',x,covariance,x),0))
        fitted=predict(model,grid)
        effects.append(pd.DataFrame({'variable':variable,'x':grid[variable],
                                     'fit':fitted,'low':fitted-1.96*se,'high':fitted+1.96*se}))
    pd.concat(effects).to_csv(out/'effects.csv',index=False)
    write_json(out/'experiment.json',{'seed':SEED,'candidates':CANDIDATES,'outer_folds':5,'inner_folds':3,
        'selections':selections,'final_df':final_df,'final_inner_scores':final_scores,
        'delta_subject_rmse_ci95':np.quantile(boot,[.025,.975]).tolist(),
        'bootstrap':'2000 paired subject resamples of frozen OOF errors; no refitting',
        'effect_interval':'pointwise Wald 95%, conditional on chosen basis, not simultaneous or post-selection adjusted',
        'spline':'unpenalized additive natural cubic regression spline; linear tail extrapolation',
        'comparison':summary})
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    run(args.smoke)
