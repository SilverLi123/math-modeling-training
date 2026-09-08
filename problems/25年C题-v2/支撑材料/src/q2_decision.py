"""Q2 decision layer: risk-optimal test weeks and ordered BMI banding
using the fitted GAMM-Binomial probability model.

Pure decision-stage computation: takes the fitted model's p(t,bmi) surface,
applies scenario-based risk optimisation, ordered DP banding, and outputs
recommendations for research review. Per project gate, these are research
results, not clinical recommendations.

Outputs -> outputs/q2_decision/
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from q2_gamm_binomial import prepare, fit, predict, OUT as GAMM_OUT
from q2_dp_tool import plan as dp_plan
from q1_lmm_baseline import write_json, DEFAULT_SEED

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q2_decision'
W_GRID = np.arange(10.0, 25.01, 0.1)
BMI_GRID = np.arange(21.0, 47.01, 0.5)
LATE_WEIGHTS = [0.5, 1.0, 2.0]
MIN_BAND = 30
MAX_BANDS = 6
TARGETS = [0.85, 0.90, 0.95]


def late_risk(w):
    if w <= 12:
        return 0.0
    if w <= 28:
        return (w - 12) / 16.0
    return 1.0 + (w - 28) / 8.0


def build_surface(model):
    grid = pd.DataFrame(
        [(w, b) for b in BMI_GRID for w in W_GRID],
        columns=['gestational_week', 'bmi'])
    grid['probability'] = predict(model, grid)
    return grid


def risk_optimal(surface, w_late):
    """For each BMI, find the week minimising total scenario risk."""
    surface = surface.copy()
    surface['late'] = surface.gestational_week.map(late_risk)
    surface['risk'] = (1 - surface.probability) + w_late * surface.late
    idx = surface.groupby('bmi').risk.idxmin()
    return surface.loc[idx, ['bmi', 'gestational_week', 'probability',
                             'risk']].reset_index(drop=True)


def coverage_threshold(surface, q):
    """For each BMI, find the earliest week with p >= q."""
    results = []
    for bmi, g in surface.groupby('bmi'):
        g = g.sort_values('gestational_week')
        hit = g[g.probability >= q]
        results.append({'bmi': bmi,
                        'week': float(hit.gestational_week.iloc[0]) if len(hit)
                        else None,
                        'achieved': len(hit) > 0})
    return pd.DataFrame(results)


def dp_banding(surface, model_data, w_late, q=0.90, min_band=MIN_BAND):
    """Ordered DP banding using the model's p(t,bmi) surface."""
    # For each subject, get their p at each candidate week from the surface
    # (interpolated). The DP tool takes a (n_subjects, n_weeks) matrix.
    subjects = model_data.drop_duplicates('subject_id')
    bmis = subjects.baseline_bmi.to_numpy()
    p_mat = np.zeros((len(bmis), len(W_GRID)))
    for j, w in enumerate(W_GRID):
        row = surface[np.isclose(surface.gestational_week, w)]
        p_mat[:, j] = np.interp(bmis, row.bmi, row.probability)
    result = dp_plan(bmis, p_mat, W_GRID, MAX_BANDS, w_late=w_late,
                     coverage_q=None)
    # Enforce minimum band size: merge small bands into neighbours.
    segs = result.get('segments', [])
    merged = []
    for s in segs:
        if merged and (s['n'] < min_band or
                       (merged and abs(s['week'] - merged[-1]['week']) < 0.5)):
            merged[-1]['bmi_hi'] = s['bmi_hi']
            merged[-1]['n'] += s['n']
        else:
            merged.append(s)
    if merged:
        merged[0]['n'] = sum(1 for b in bmis
                             if merged[0]['bmi_lo'] <= b <= merged[0]['bmi_hi'])
    return {'segments': merged, 'total_cost': result.get('total_cost'),
            'infeasible': result.get('infeasible', False),
            'n_segments': len(merged)}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1) Fit GAMM-Binomial on all data (final model for decision stage)
    data, audit = prepare()
    print('Fitting final GAMM-Binomial on full data...', flush=True)
    model = fit(data, 'T_w4_b3')
    surface = build_surface(model)
    surface.to_csv(OUT / 'probability_surface.csv', index=False)

    # 2) Per-BMI curves at selected levels
    curves = []
    for B in [25, 30, 35, 40]:
        g = surface[np.isclose(surface.bmi, B)].sort_values('gestational_week')
        g['bmi'] = B
        curves.append(g)
    pd.concat(curves).to_csv(OUT / 'bmi_curves.csv', index=False)

    # 3) Coverage thresholds t90/t85/t95
    cov_dfs = []
    for q in TARGETS:
        cdf = coverage_threshold(surface, q)
        cdf['target'] = q
        cov_dfs.append(cdf)
    cov_all = pd.concat(cov_dfs)
    cov_all.to_csv(OUT / 'coverage_thresholds.csv', index=False)
    for q in TARGETS:
        cdf = cov_all[cov_all.target == q]
        ok = cdf[cdf.achieved]
        if len(ok):
            print(f'  coverage {q:.0%}: {len(ok)}/{len(cdf)} BMI levels '
                  f'achievable, mean week {ok.week.mean():.1f}')

    # 4) Risk-optimal t*(BMI) per λ scenario
    risk_results = {}
    for wl in LATE_WEIGHTS:
        rr = risk_optimal(surface, wl)
        risk_results[wl] = rr
        rr.to_csv(OUT / f'risk_optimal_w{wl}.csv', index=False)
        print(f'  w_late={wl}: mean t*={rr.gestational_week.mean():.1f}, '
              f'range=[{rr.gestational_week.min():.1f}, {rr.gestational_week.max():.1f}]')

    # 5) Ordered DP banding per λ scenario
    dp_results = {}
    for wl in LATE_WEIGHTS:
        dp = dp_banding(surface, data, wl)
        dp_results[wl] = dp
        print(f'  DP w_late={wl}: {dp["n_segments"]} segments, '
              f'weeks={[s["week"] for s in dp["segments"]]}')

    # 6) Comparison with baseline KM+greedy (from 2025/ 论文)
    baseline = {'boundaries': [32.1, 34.7], 'weeks': [16.6, 16.6, 22.1],
                'source': '2025 baseline paper (KM + greedy SSE)'}
    new_main = {'boundaries': sorted(set(
                    s['bmi_hi'] for s in dp_results[1.0]['segments'][:-1])),
                'weeks': [s['week'] for s in dp_results[1.0]['segments']],
                'source': 'this study (GAMM-Binomial + ordered DP)'}

    write_json(OUT / 'experiment.json', {
        'seed': DEFAULT_SEED,
        'model_spec': 'T_w4_b3 (cr(week,df=4) + cr(bmi,df=3) random-intercept Binomial GLMM)',
        'input_sha256': audit['input_sha256'],
        'n_subjects': data.subject_id.nunique(),
        'late_weights': LATE_WEIGHTS,
        'dp_results': {str(k): v for k, v in dp_results.items()},
        'risk_optimal_summary': {
            str(w): {'mean_t': float(r.gestational_week.mean()),
                     'range': [float(r.gestational_week.min()),
                               float(r.gestational_week.max())]}
            for w, r in risk_results.items()},
        'coverage_thresholds': {str(q): {
            'achievable_count': int(cov_all[(cov_all.target == q)].achieved.sum()),
            'total': len(cov_all[cov_all.target == q])}
            for q in TARGETS},
        'baseline_comparison': {'old': baseline, 'new': new_main},
        'gate_note': ('probability model did not pass the original '
                      'applicability gate; these are research results per '
                      'user instruction to proceed with model-level rebuild'),
        'surfaces': 'probability_surface.csv (p(t,bmi) on 10-25w × 21-47 grid)',
    })
    print('[q2_decision] complete')


if __name__ == '__main__':
    run()
