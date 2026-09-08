"""Fixed-specification Q3 comparison and explicitly model-dependent Q2/Q3 DP."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from q1_lmm_baseline import DEFAULT_SEED, subject_group_folds, write_json
from q2_probability import ROOT, prepare, normal_rule, metrics
from q2_nonlinear_calibration import _fit_integrated

OUT = ROOT / 'outputs' / 'q23_exploration'
SPECS = {
    'q2': ['gestational_week', 'baseline_bmi'],
    'q3_basic': ['gestational_week', 'baseline_bmi', 'baseline_age', 'baseline_height'],
    'q3_full': ['gestational_week', 'baseline_bmi', 'baseline_age', 'baseline_height',
                'baseline_assisted', 'baseline_gravidity', 'baseline_parity'],
}
WEEKS = np.arange(77, 176) / 7
SCENARIOS = {'step_only': (0., .1), 'low_delay': (.005, .1),
             'medium_delay': (.02, .1), 'high_delay': (.05, .1)}


def load():
    data, audit = prepare()
    raw = pd.read_excel(ROOT / '附件.xlsx', sheet_name='男胎检测数据', header=0)
    raw = raw.set_index('序号')
    first = raw.loc[data.baseline_sample_id]
    for dest, source in [('baseline_age', '年龄'), ('baseline_height', '身高'),
                         ('baseline_gravidity', '怀孕次数'), ('baseline_parity', '生产次数')]:
        values = first[source].astype(str).str.replace('≥', '', regex=False)
        data[dest] = pd.to_numeric(values, errors='raise').to_numpy()
    conception = first['IVF妊娠']
    if not conception.isin(['\u81ea\u7136\u53d7\u5b55', 'IUI\uff08\u4eba\u5de5\u6388\u7cbe\uff09', 'IVF\uff08\u8bd5\u7ba1\u5a74\u513f\uff09']).all():
        raise ValueError(f'Unknown conception categories: {conception.unique()}')
    data['baseline_assisted'] = (conception != '自然受孕').astype(float).to_numpy()
    return data, audit


def fit(data, spec):
    cols = SPECS[spec]
    median = data[cols].median().to_numpy()
    values = data[cols].fillna(dict(zip(cols, median))).to_numpy()
    center, scale = values.mean(0), values.std(0)
    keep = scale > 1e-10
    x = np.c_[np.ones(len(data)), ((values-center)[:, keep]/scale[keep])]
    if np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError('Rank deficient Q3 design')
    model = _fit_integrated(x, data.attained.to_numpy(float), data.subject_id)
    return model | {'cols': cols, 'median': median.tolist(), 'center': center.tolist(),
                    'scale': scale.tolist(), 'keep': keep.tolist()}


def predict(model, data):
    values = data[model['cols']].fillna(dict(zip(model['cols'], model['median']))).to_numpy()
    keep = np.array(model['keep'])
    x = np.c_[np.ones(len(data)), ((values-model['center'])[:, keep]/np.array(model['scale'])[keep])]
    nodes, weights = normal_rule(model['nodes'])
    p = expit((x @ np.array(model['theta'][:-1]))[:, None] + model['sigma']*nodes) @ np.exp(weights)
    if not np.all(np.isfinite(p) & (p > 0) & (p < 1)):
        raise ValueError('Invalid marginal predictions')
    return p


def optimize(bmis, probabilities, support, k, delay, minimum=25, support_min=10):
    """Exact contiguous DP; one individual one weight; tied BMI never split.

    support[i,t] indicates individual i observed within +/-1 week of t.
    Segment needs support_min distinct training subjects near chosen week.
    """
    n, nt = probabilities.shape
    prefix = np.vstack([np.zeros(nt), np.cumsum(1-probabilities+delay, axis=0)])
    coverage = np.vstack([np.zeros(nt), np.cumsum(support, axis=0)])
    costs = np.full((n+1, n+1), np.inf)
    times = np.zeros((n+1, n+1), dtype=int)
    boundaries = np.r_[0, np.flatnonzero(np.diff(bmis) > 0)+1, n]
    for end in boundaries[1:]:
        for start in boundaries[boundaries <= end-minimum]:
            risk = prefix[end]-prefix[start]
            risk = np.where(coverage[end]-coverage[start] >= support_min, risk, np.inf)
            times[start, end] = int(np.argmin(risk))
            costs[start, end] = risk[times[start, end]]
    dp = np.full((k+1, n+1), np.inf); dp[0, 0] = 0
    previous = np.full((k+1, n+1), -1, dtype=int)
    for count in range(1, k+1):
        for end in boundaries[1:]:
            values = dp[count-1, boundaries]+costs[boundaries, end]
            best = int(np.argmin(values))
            dp[count, end] = values[best]; previous[count, end] = boundaries[best]
    if not np.isfinite(dp[k, n]):
        return None
    groups = []; end = n
    for count in range(k, 0, -1):
        start = previous[count, end]
        groups.append((int(start), int(end), int(times[start, end])))
        end = start
    return float(dp[k, n]/n), groups[::-1]


def strategy(model, train, valid, fold, spec):
    subjects = train.drop_duplicates('subject_id').sort_values('baseline_bmi').reset_index(drop=True)
    # Each subject contributes once per candidate week, regardless of repeat count.
    grid = subjects.loc[subjects.index.repeat(len(WEEKS))].copy()
    grid['gestational_week'] = np.tile(WEEKS, len(subjects))
    probabilities = predict(model, grid).reshape(len(subjects), -1)
    support = np.array([np.any(abs(train.loc[train.subject_id == sid, 'gestational_week'].to_numpy()[:, None]-WEEKS) <= 1, axis=0)
                        for sid in subjects.subject_id])
    bmis = subjects.baseline_bmi.to_numpy()
    held = valid.drop_duplicates('subject_id').copy()
    rows = []
    for scenario, (weekly, jump) in SCENARIOS.items():
        delay = weekly*(WEEKS-11)+jump*(WEEKS >= 13)
        for k in (1, 3, 4):
            result = optimize(bmis, probabilities, support, k, delay)
            if result is None:
                rows.append({'fold': fold, 'model': spec, 'scenario': scenario, 'k': k, 'status': 'infeasible'})
                continue
            training_risk, groups = result
            cutpoints = [(bmis[end-1]+bmis[end])/2 for _, end, _ in groups[:-1]]
            assignment = np.searchsorted(cutpoints, held.baseline_bmi, side='right')
            week_indices = np.array([g[2] for g in groups])[assignment]
            frame = held.copy(); frame['gestational_week'] = WEEKS[week_indices]
            held_risk = float(np.mean(1-predict(model, frame)+delay[week_indices]))
            rows.append({'fold': fold, 'model': spec, 'scenario': scenario, 'k': k,
                         'status': 'model_dependent', 'train_expected_risk': training_risk,
                         'heldout_expected_risk': held_risk, 'heldout_subjects': len(held),
                         'heldout_outside_bmi_range': int((~held.baseline_bmi.between(bmis.min(), bmis.max())).sum()),
                         'cutpoints': json.dumps(cutpoints), 'weeks': json.dumps([float(WEEKS[t]) for _, _, t in groups]),
                         'counts': json.dumps([end-start for start, end, _ in groups])})
    return rows


def run(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = load()
    folds = subject_group_folds(data.subject_id, 5, DEFAULT_SEED)
    output = data[['sample_id', 'subject_id', 'baseline_sample_id', 'gestational_week', 'attained']].copy()
    fits, policies = [], []
    for fold, ids in enumerate(folds[:1] if smoke else folds, 1):
        mask = data.subject_id.isin(ids); train, valid = data.loc[~mask], data.loc[mask]
        for spec in SPECS:
            model = fit(train, spec)
            output.loc[mask, spec] = predict(model, valid)
            output.loc[mask, 'fold'] = fold
            fits.append({'fold': fold, 'spec': spec, 'fit': model})
            if not smoke:
                policies.extend(strategy(model, train, valid, fold, spec))
        print(f'Q2/Q3 fold {fold} finished', flush=True)
    if smoke:
        write_json(OUT/'smoke.json', {'fits': fits, 'rows': len(data)})
        return
    if output.isna().any().any():
        raise ValueError('Incomplete OOF')
    output.to_csv(OUT/'oof.csv', index=False)
    pd.DataFrame(policies).to_csv(OUT/'policies.csv', index=False)
    comparison = [{'model': spec, **metrics(data, output[spec].to_numpy())} for spec in SPECS]
    rng = np.random.default_rng(DEFAULT_SEED)
    intervals = {}
    for spec in list(SPECS)[1:]:
        delta = ((data.attained-output[spec])**2-(data.attained-output.q2)**2).groupby(data.subject_id).mean().to_numpy()
        draws = delta[rng.integers(len(delta), size=(2000, len(delta)))].mean(axis=1)
        intervals[spec] = {'mean': float(delta.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist()}
    pd.DataFrame(comparison).to_csv(OUT/'comparison.csv', index=False)
    write_json(OUT/'experiment.json', {'comparison': comparison, 'paired_subject_brier': intervals,
        'fits': fits, 'input_sha256': audit['input_sha256'], 'seed': DEFAULT_SEED,
        'scenarios': SCENARIOS, 'specifications': SPECS,
        'warning': 'Expected strategy risks are model-dependent, not observed counterfactual outcomes. No best specification chosen on OOF.',
        'bootstrap': '2000 resamples of frozen subject OOF differences, no refit',
        'support': '>=10 training subjects within +/-1 week in each BMI segment; minimum group 25',
        'gravidity': '>=3 coded as 3 (top-coded proxy, not exact count)'})
    print(json.dumps({'comparison': comparison, 'intervals': intervals}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
