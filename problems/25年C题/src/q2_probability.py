"""Q2 random-intercept Bernoulli GLMM, marginal prediction, subject CV.

Normal random intercept integrated by Gauss-Hermite quadrature. Inputs and
preprocessing frozen per training fold. No individual random-effect estimates
are used for unseen subjects. The time surface is observational, not causal.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, roots_hermitenorm
from q1_lmm_baseline import load_and_prepare, subject_group_folds, write_json, DEFAULT_SEED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q2_probability'
FEATURES = ['gestational_week', 'baseline_bmi']
THRESHOLD = .04


def prepare():
    data, audit = load_and_prepare(ROOT / '附件.xlsx')
    first = (data.sort_values(['subject_id', 'test_date_iso', 'gestational_week', 'sample_id'])
             .drop_duplicates('subject_id').set_index('subject_id'))
    data['baseline_bmi'] = data.subject_id.map(first.bmi)
    data['first_observed_week'] = data.subject_id.map(first.gestational_week)
    data['baseline_sample_id'] = data.subject_id.map(first.sample_id)
    data['attained'] = (data.y_fraction >= THRESHOLD).astype(int)
    assert data.groupby('subject_id').baseline_bmi.nunique().max() == 1
    return data, audit


def normal_rule(n):
    nodes, weights = roots_hermitenorm(n)
    keep = weights > 0
    return nodes[keep], np.log(weights[keep]) - .5 * np.log(2*np.pi)


def objective(theta, x, y, groups, nodes, logweights):
    """Negative integrated likelihood and its analytic gradient."""
    beta, sigma = theta[:-1], np.exp(theta[-1])
    z = x @ beta
    eta = z[:, None] + sigma * nodes[None, :]
    loglik = y[:, None]*eta - np.logaddexp(0., eta)
    totals = np.zeros((groups.max()+1, len(nodes)))
    np.add.at(totals, groups, loglik)
    logjoint = totals + logweights
    lognorm = logsumexp(logjoint, axis=1)
    posterior = np.exp(logjoint-lognorm[:, None])[groups]
    error = y[:, None]-expit(eta)
    beta_gradient = x.T @ np.sum(posterior*error, axis=1)
    sigma_gradient = np.sum(posterior*error*(sigma*nodes)[None, :])
    return -float(lognorm.sum()), -np.r_[beta_gradient, sigma_gradient]


def matrix(data, center, scale):
    return np.column_stack([np.ones(len(data)), (data[FEATURES].to_numpy()-center)/scale])


def fit(data):
    center = data[FEATURES].mean().to_numpy()
    scale = data[FEATURES].std(ddof=0).to_numpy()
    if np.any(scale <= 0):
        raise ValueError('Constant feature in training set')
    x = matrix(data, center, scale)
    y = data.attained.to_numpy(dtype=float)
    groups = pd.factorize(data.subject_id, sort=True)[0]
    prevalence = y.mean()
    if not 0 < prevalence < 1:
        raise ValueError('Training requires both outcomes')
    starts = [np.r_[np.log(prevalence/(1-prevalence)), 0., 0., np.log(s)] for s in (.5, 1.5, 3.)]
    history = []
    for count in (80, 160, 320, 640):
        nodes, weights = normal_rule(count)
        candidates = []
        for start in starts:
            result = minimize(objective, start, args=(x,y,groups,nodes,weights), jac=True,
                              method='L-BFGS-B', bounds=[(-30,30)]*3+[(-7,3)],
                              options={'maxiter':1000, 'ftol':1e-12, 'gtol':1e-6})
            history.append({'nodes':count,'success':bool(result.success),'nll':float(result.fun),
                            'gradient_max':float(np.max(np.abs(result.jac))), 'message':str(result.message)})
            if result.success and np.max(np.abs(result.jac)) < .002:
                candidates.append(result)
        if not candidates:
            raise RuntimeError(f'No converged fit: {history}')
        best = min(candidates, key=lambda r:r.fun)
        check_nodes, check_weights = normal_rule(count*2)
        check_value, check_grad = objective(best.x,x,y,groups,check_nodes,check_weights)
        # Integrate prediction independently at doubled order too.
        eta = x @ best.x[:-1]
        sigma = np.exp(best.x[-1])
        p0 = expit(eta[:,None]+sigma*nodes) @ np.exp(weights)
        p1 = expit(eta[:,None]+sigma*check_nodes) @ np.exp(check_weights)
        error = abs(check_value-best.fun)
        prediction_error = float(np.max(abs(p0-p1)))
        if error < 1e-4 and prediction_error < 1e-6 and np.max(abs(check_grad)) < .003:
            if np.any(abs(best.x[:-1]) > 29.99) or abs(best.x[-1]-3) < .01:
                raise RuntimeError('Fit hit upper parameter bound')
            return {'theta':best.x.tolist(),'center':center.tolist(),'scale':scale.tolist(),
                    'sigma':float(sigma),'nodes':count*2,'nll':float(check_value),
                    'quadrature_nll_difference':float(error),'quadrature_probability_difference':prediction_error,
                    'checked_gradient_max':float(np.max(abs(check_grad))),
                    'variance_lower_boundary':bool(best.x[-1]<-6.99),'optimization_history':history}
        starts = [best.x, best.x+np.array([0,0,0,.1])]
    raise RuntimeError(f'Quadrature not stable: {history}')


def predict(model, data):
    x = matrix(data, np.array(model['center']), np.array(model['scale']))
    nodes, weights = normal_rule(model['nodes'])
    p = expit((x @ np.array(model['theta'][:-1]))[:,None]+model['sigma']*nodes) @ np.exp(weights)
    if not np.isfinite(p).all() or not np.all((p>0)&(p<1)):
        raise RuntimeError('Invalid marginal probabilities')
    return p


def metrics(data, p):
    y=data.attained.to_numpy()
    sq=(y-p)**2
    ll=-(y*np.log(p)+(1-y)*np.log1p(-p))
    return {'record_brier':float(sq.mean()),
            'subject_brier':float(pd.Series(sq).groupby(data.subject_id.to_numpy()).mean().mean()),
            'record_log_loss':float(ll.mean()),
            'subject_log_loss':float(pd.Series(ll).groupby(data.subject_id.to_numpy()).mean().mean()),
            'mean_probability':float(np.mean(p)),'observed_rate':float(y.mean())}


def calibration(data):
    rows=[]
    edges=np.array([0,.5,.7,.8,.9,.95,1.])
    for name in ['mixed_logistic','training_rate']:
        bins=np.clip(np.searchsorted(edges,data[name],side='right')-1,0,len(edges)-2)
        for b in range(len(edges)-1):
            mask=bins==b
            if not mask.any():
                continue
            g=data.loc[mask]
            rows.append({'model':name,'bin_low':edges[b],'bin_high':edges[b+1],
                         'rows':len(g),'subjects':g.subject_id.nunique(),
                         'mean_prediction':g[name].mean(),'observed_rate':g.attained.mean()})
    return pd.DataFrame(rows)


def surface(model,data):
    weeks=np.arange(77,176)/7
    bmis=np.arange(np.floor(data.baseline_bmi.min()),np.ceil(data.baseline_bmi.max())+.1,.5)
    grid=pd.DataFrame([(w,b) for b in bmis for w in weeks],columns=FEATURES)
    grid['probability']=predict(model,grid)
    supported=[]
    for w,b in grid[FEATURES].itertuples(index=False,name=None):
        nearby=data[(abs(data.gestational_week-w)<=1)&(abs(data.baseline_bmi-b)<=2)]
        supported.append(nearby.subject_id.nunique())
    grid['local_subjects']=supported
    grid['in_bmi_range']=grid.baseline_bmi.between(data.baseline_bmi.min(),data.baseline_bmi.max())
    grid['supported']=(grid.local_subjects>=10)&grid.in_bmi_range
    crossings=[]
    for b,g in grid.groupby('baseline_bmi'):
        for target in (.90,.95):
            theoretical=g[g.probability>=target]
            observed=g[(g.probability>=target)&g.supported]
            crossings.append({'baseline_bmi':b,'target':target,
                'first_model_crossing_week':theoretical.gestational_week.min() if len(theoretical) else None,
                'first_supported_crossing_week':observed.gestational_week.min() if len(observed) else None,
                'status':'supported_grid_crossing' if len(observed) else 'no_supported_crossing',
                'note':'exploratory threshold crossing, not recommended test time'})
    return grid,pd.DataFrame(crossings)


def run(smoke=False):
    OUT.mkdir(parents=True,exist_ok=True)
    data,audit=prepare()
    folds=subject_group_folds(data.subject_id,5,DEFAULT_SEED)
    if smoke:
        mask=data.subject_id.isin(folds[0])
        model=fit(data.loc[~mask])
        result={'model':model,'heldout_metrics':metrics(data.loc[mask],predict(model,data.loc[mask]))}
        write_json(OUT/'smoke.json',result)
        print(result,flush=True)
        return
    oof=data[['sample_id','subject_id','gestational_week','baseline_bmi','baseline_sample_id',
              'first_observed_week','attained']].copy()
    histories,fold_metrics=[],[]
    for k,ids in enumerate(folds,1):
        mask=data.subject_id.isin(ids)
        train,valid=data.loc[~mask],data.loc[mask]
        model=fit(train)
        pred=predict(model,valid)
        rate=float(train.attained.mean())
        subject_rate=float(train.groupby('subject_id').attained.mean().mean())
        oof.loc[mask,'fold']=k
        oof.loc[mask,'mixed_logistic']=pred
        oof.loc[mask,'training_rate']=rate
        oof.loc[mask,'training_subject_rate']=subject_rate
        for name,p in [('mixed_logistic',pred),('training_rate',np.repeat(rate,len(valid))),
                       ('training_subject_rate',np.repeat(subject_rate,len(valid)))]:
            fold_metrics.append({'fold':k,'model':name,**metrics(valid,p)})
        histories.append({'fold':k,'fit':model})
        print(f'Q2 fold {k}: sigma={model["sigma"]:.4f}, nodes={model["nodes"]}',flush=True)
    assert not oof.isna().any().any()
    oof.to_csv(OUT/'oof_predictions.csv',index=False)
    pd.DataFrame(fold_metrics).to_csv(OUT/'fold_metrics.csv',index=False)
    summary=[{'model':name,**metrics(data,oof[name].to_numpy())}
             for name in ['mixed_logistic','training_rate','training_subject_rate']]
    pd.DataFrame(summary).to_csv(OUT/'comparison.csv',index=False)
    calibration(oof).to_csv(OUT/'calibration.csv',index=False)
    time_rows=[]
    for lo,hi in [(11,13),(13,16),(16,20),(20,26),(26,30)]:
        subset=oof[oof.gestational_week.between(lo,hi,inclusive='left')]
        if len(subset):
            time_rows.append({'week_low':lo,'week_high':hi,'rows':len(subset),
                'subjects':subset.subject_id.nunique(),'observed_rate':subset.attained.mean(),
                'mixed_logistic':subset.mixed_logistic.mean(),
                'training_rate':subset.training_rate.mean(),
                'training_subject_rate':subset.training_subject_rate.mean()})
    pd.DataFrame(time_rows).to_csv(OUT/'time_calibration.csv',index=False)
    final=fit(data)
    grid,crossings=surface(final,data)
    grid.to_csv(OUT/'surface.csv',index=False)
    crossings.to_csv(OUT/'threshold_crossings.csv',index=False)
    # Paired subject resampling of frozen OOF errors, not model-refit inference.
    loss=pd.DataFrame({'subject':data.subject_id,
        'delta':(data.attained-oof.mixed_logistic)**2-(data.attained-oof.training_subject_rate)**2})
    loss=loss.groupby('subject').delta.mean().to_numpy()
    rng=np.random.default_rng(DEFAULT_SEED)
    boot=[loss[rng.integers(len(loss),size=len(loss))].mean() for _ in range(2000)]
    report={'seed':DEFAULT_SEED,'threshold':THRESHOLD,'input_sha256':audit['input_sha256'],
        'n_records':len(data),'n_subjects':data.subject_id.nunique(),'comparison':summary,
        'early_observation_subjects':data.loc[data.first_observed_week<=12,'subject_id'].nunique(),
        'fold_fits':histories,'final_fit':final,
        'subject_brier_difference_ci95':np.quantile(boot,[.025,.975]).tolist(),
        'bootstrap':'2000 subject resamples of frozen OOF loss differences vs training subject rate; no refit',
        'support_rule':'at least 10 distinct subjects within +/-1 week and +/-2 BMI; exploratory, not clinical',
        'surface_scope':'full-data observational probability, no causal or scheduling validation',
        'baseline_bmi':'earliest date, then week, then sample ID; proxy may not describe earlier BMI',
        'first_record_y_available_to_model':False,
        'calibration':'fixed probability bins, record-weighted descriptive rates; no independence-based intervals'}
    write_json(OUT/'experiment.json',report)
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args().smoke)
