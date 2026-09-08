"""Q2/Q3 检测误差敏感性 + 基于 t90(BMI) 的有序分段（P1/P2 收尾两项）。

Part A (--banding, fast):
  从全数据拟合的 T_w4_b3 概率曲面取 t90(BMI)（不可达水平记 25 周上限），
  以孕妇数为权重做一维有序 K 段 DP（K=1..4，每段至少 30 名孕妇），
  最小化段内加权 SSE；报告各 K 的边界与段内统一孕周（段内 t90 均值）。
  输出 outputs/q2_decision/t90_banding.{csv,json}。

Part B (--mc, slow):
  测量误差情景：sigma_hat = Q1 LMM-quad 残差(logit) sd 经 delta 法折算到
  Y 尺度的中位数（约 1.8pp，与 A 版组内残差口径一致）；情景 = {sigma_hat, 2x}。
  每次重复：Y' = clip(Y + N(0, sigma))，D' = 1(Y'>=4%)，重拟合
  Q2 主规格 T_w4_b3（全数据），重算 90% 覆盖（可行数/均值）、风险 t*(λ=1)
  摘要与 K=3 分段切点；Q3 部分（sigma_hat，R3 次）重拟合三个线性规格，
  记录“基线不差于 A/B”的结论是否保持。
  输出 outputs/q2_decision/error_sensitivity.json + mc_details.csv。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from q1_lmm_baseline import write_json, DEFAULT_SEED
from q2_gamm_binomial import prepare, fit, predict, metrics
from q2_decision import build_surface, coverage_threshold, risk_optimal, W_GRID, BMI_GRID

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q2_decision'
BANDS_OLD = [(20.703, 32.1), (32.1, 34.7), (34.7, 46.9)]
T_CAP = 25.0
MIN_SUBJ = 30


# ---------------------------------------------------------------- sigma
def estimate_sigma(data, q1_fit):
    """Q1 LMM-quad 残差(logit) sd，经 delta 法折算到 Y 尺度的中位数。"""
    res = q1_fit['result']
    sd_logit = float(np.sqrt(res.scale))
    y = data.y_fraction.to_numpy()
    per_record = sd_logit * y * (1 - y)
    return sd_logit, float(np.median(per_record))


# ---------------------------------------------------------------- banding
def subject_weights(data):
    """每个 BMI 网格水平的【孕妇数】（唯一 subject，按最近水平归属，
    BMI<21 或 >47 归入端点水平）。"""
    subj = data.drop_duplicates('subject_id')
    levels = BMI_GRID.copy()
    n = len(levels)
    idx = np.clip(np.round((subj.baseline_bmi.to_numpy() - levels[0]) / 0.5).astype(int),
                  0, n - 1)
    w = np.bincount(idx, minlength=n).astype(int)
    return levels, w


def seg_sse(v, w, i, j):
    vv, ww = v[i:j + 1], w[i:j + 1]
    mu = float((vv * ww).sum() / ww.sum())
    return float((ww * (vv - mu) ** 2).sum()), float((vv * ww).sum() / ww.sum())


def dp_segments(levels, w, v, K):
    n = len(levels)
    # prefix weights
    INF = float('inf')
    cost = [[INF] * n for _ in range(n)]
    mu = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            cost[i][j], mu[i][j] = seg_sse(v, w, i, j)
    # dp[k][j] = minimal cost covering 0..j with k segments; also back-pointers
    dp = [[INF] * n for _ in range(K + 1)]
    back = [[-1] * n for _ in range(K + 1)]
    # enforce min subjects per segment
    def wsum(i, j):
        return float(w[i:j + 1].sum())
    for j in range(n):
        if wsum(0, j) >= MIN_SUBJ:
            dp[1][j] = cost[0][j]
    for k in range(2, K + 1):
        for j in range(n):
            for i in range(j):
                if dp[k - 1][i] < INF and wsum(i + 1, j) >= MIN_SUBJ:
                    c = dp[k - 1][i] + cost[i + 1][j]
                    if c < dp[k][j]:
                        dp[k][j] = c
                        back[k][j] = i
    # best k: elbow rule — smallest k whose marginal SSE gain drops below
    # 10% of the K=1 SSE (i.e., adding one more band no longer helps much)
    sse1 = dp[1][n - 1] if np.isfinite(dp[1][n - 1]) else INF
    best_k, best_c = None, INF
    for k in range(1, K + 1):
        if dp[k][n - 1] < best_c:
            best_k, best_c = k, dp[k][n - 1]
    chosen = K
    for k in range(2, K + 1):
        if np.isfinite(dp[k - 1][n - 1]) and (dp[k - 1][n - 1] - dp[k][n - 1]) < 0.10 * sse1:
            chosen = k - 1
            break
    best_k = chosen
    # reconstruct for each k
    def segs_for(k):
        j = n - 1
        segs = []
        for kk in range(k, 0, -1):
            i = back[kk][j] if kk > 1 else -1
            nsub = int(w[i + 1:j + 1].sum())
            segs.append({'bmi_lo': round(float(levels[i + 1]), 1),
                         'bmi_hi': round(float(levels[j]), 1),
                         'subjects': nsub,
                         'week': round(float(mu[i + 1][j]), 1)})
            j = i
        segs.reverse()
        return segs
    table = []
    for k in range(1, K + 1):
        if dp[k][n - 1] < INF and np.isfinite(dp[k][n - 1]):
            s = segs_for(k)
            table.append({'K': k, 'weighted_sse': round(dp[k][n - 1], 2),
                          'bounds': [x['bmi_lo'] for x in s[1:]],
                          'band_weeks': [x['week'] for x in s],
                          'segments': s})
    return best_k, best_c, table


def banding_part(data, q1_fit):
    model = fit(data, 'T_w4_b3')
    surface = build_surface(model)
    cov = coverage_threshold(surface, 0.90)
    levels, w = subject_weights(data)
    v = []
    for lv in levels:
        row = cov[cov.bmi == lv]
        v.append(float(row.week.iloc[0]) if len(row) and row.achieved.iloc[0] else T_CAP)
    v = np.array(v)
    best_k, best_c, table = dp_segments(levels, w, v, 4)
    rows = []
    for ktab in table:
        rows.append(ktab)
    pd.DataFrame(rows).to_csv(OUT / 't90_banding.csv', index=False)
    # old-band means for reference
    cov_all = cov
    old = []
    for lo, hi in BANDS_OLD:
        ok = cov_all[(cov_all.bmi >= lo) & (cov_all.bmi <= hi) & cov_all.achieved]
        old.append({'band': f'{lo}-{hi}', 'mean90': round(float(ok.week.mean()), 2),
                    'n_ok': int(len(ok))})
    payload = {'sigma_none_reference': {
                   'achievable': int(cov_all.achieved.sum()),
                   'mean90_week': round(float(cov_all[cov_all.achieved].week.mean()), 2)},
               't90_cap_for_infeasible': T_CAP,
               'min_subjects_per_band': MIN_SUBJ,
               'k_table': rows, 'chosen_K': best_k,
               'old_band_mean90': old,
               'note': ('t90 capped at 25w for BMI levels that never reach p>=0.9 '
                        'within 10-25w window; segment cost = subject-weighted '
                        'within-band SSE of t90')}
    write_json(OUT / 't90_banding.json', payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1), flush=True)
    return surface


# ---------------------------------------------------------------- MC
def one_surface_metrics(surface, data):
    cov = coverage_threshold(surface, 0.90)
    ok = cov[cov.achieved]
    t90_mean = float(ok.week.mean()) if len(ok) else np.nan
    t90_n = int(len(ok))
    band_means = []
    for lo, hi in BANDS_OLD:
        b = ok[(ok.bmi >= lo) & (ok.bmi <= hi)]
        band_means.append(round(float(b.week.mean()), 2) if len(b) else None)
    rr = risk_optimal(surface, 1.0)
    levels, w = subject_weights(data)
    v = []
    for lv in levels:
        row = cov[cov.bmi == lv]
        v.append(float(row.week.iloc[0]) if len(row) and row.achieved.iloc[0] else T_CAP)
    v = np.array(v)
    _, _, ktab = dp_segments(levels, w, v, 4)
    k3 = next((t for t in ktab if t['K'] == 3), None)
    return {'t90_mean': round(t90_mean, 2), 't90_feasible': t90_n,
            'band_mean90': band_means,
            'risk_mean_t': round(float(rr.gestational_week.mean()), 2),
            'risk_min_t': round(float(rr.gestational_week.min()), 2),
            'risk_max_t': round(float(rr.gestational_week.max()), 2),
            'k3_bounds': k3['bounds'] if k3 else None,
            'k3_band_weeks': k3['band_weeks'] if k3 else None}


def perturb(data, sigma, rng):
    d = data.copy()
    y = d.y_fraction.to_numpy()
    yp = np.clip(y + rng.normal(0, sigma, size=len(y)), 1e-4, 0.9)
    d['y_fraction'] = yp
    d['attained'] = (yp >= 0.04).astype(int)
    return d


def mc_part(r_q2, r_q3, seed=DEFAULT_SEED):
    # 用 q3 的 prepare（superset：含 baseline_bmi/age/height/weight/non_natural）
    from q3_multifactor_probability import prepare as q3_prepare
    data, _ = q3_prepare()
    # sigma estimate via Q1 LMM-quad
    from q1_lmm_gamm import prepare as q1_prepare, fit_spec
    d1, _ = q1_prepare('logit')
    m1 = fit_spec(d1, 'LMM-quad')
    sd_logit, sigma_hat = estimate_sigma(data, m1)
    scenarios = [{'sigma': round(sigma_hat, 4), 'label': 'sigma_hat', 'R': r_q2},
                 {'sigma': round(2 * sigma_hat, 4), 'label': 'sigma_2x', 'R': r_q2}]
    rng = np.random.default_rng(seed)
    details = []
    q3_rows = []
    for sc in scenarios:
        s = sc['sigma']
        for r in range(sc['R']):
            dp = perturb(data, s, rng)
            try:
                model = fit(dp, 'T_w4_b3')
                surface = build_surface(model)
                met = one_surface_metrics(surface, data)
                met.update({'sigma': s, 'rep': r})
                details.append(met)
                print(f"[Q2] sigma={s} rep={r}: {met}", flush=True)
            except Exception as exc:
                details.append({'sigma': s, 'rep': r, 'error': str(exc)[:120]})
                print(f"[Q2] sigma={s} rep={r} FAILED {str(exc)[:80]}", flush=True)
        # Q3 linear specs under base sigma only
        if sc['label'] == 'sigma_hat':
            from q3_multifactor_probability import fit_gh as q3_fit, predict as q3_predict, SPECS as Q3SPECS
            for r in range(r_q3):
                dp = perturb(data, s, rng)
                briers = {}
                try:
                    for spec, feats in Q3SPECS.items():
                        m3 = q3_fit(dp, feats)
                        briers[spec] = round(float(metrics(dp, q3_predict(m3, dp))['subject_brier']), 4)
                    q3_rows.append({'sigma': s, 'rep': r, **briers,
                                    'baseline_best': bool(
                                        briers['Q2_baseline'] <= briers['Q3_A_bmi_age_ivf'] and
                                        briers['Q2_baseline'] <= briers['Q3_B_hw_age'])})
                    print(f"[Q3] rep={r}: {briers}", flush=True)
                except Exception as exc:
                    q3_rows.append({'sigma': s, 'rep': r, 'error': str(exc)[:120]})
                    print(f"[Q3] rep={r} FAILED {str(exc)[:80]}", flush=True)
    dd = pd.DataFrame(details)
    dd.to_csv(OUT / 'mc_details.csv', index=False)
    summ = (dd.dropna(subset=['t90_mean']).groupby('sigma')
              .agg(t90_mean=('t90_mean', 'mean'), t90_sd=('t90_mean', 'std'),
                   t90_min=('t90_mean', 'min'), t90_max=('t90_mean', 'max'),
                   feasible_min=('t90_feasible', 'min'),
                   risk_mean=('risk_mean_t', 'mean'),
                   risk_max=('risk_max_t', 'max'))
              .round(3).reset_index().to_dict(orient='records'))
    q3_df = pd.DataFrame([r for r in q3_rows if 'baseline_best' in r])
    if len(q3_df):
        q3_summary = {'n_reps': int(len(q3_df)),
                      'baseline_best_rate': round(float(q3_df.baseline_best.mean()), 3),
                      'mean_brier': {c: round(float(q3_df[c].mean()), 4)
                                     for c in ['Q2_baseline', 'Q3_A_bmi_age_ivf', 'Q3_B_hw_age']
                                     if c in q3_df}}
    else:
        q3_summary = {}
    payload = {'sigma_logit_sd': sd_logit, 'sigma_hat_y': round(sigma_hat, 4),
               'design': ('Y perturbed with N(0, sigma) per record (clipped), '
                          'D recomputed, T_w4_b3 refit on full data; Q3 linear '
                          'specs refit at sigma_hat'),
               'scenarios': scenarios, 'q2_summary': summ,
               'q3_summary': q3_summary}
    write_json(OUT / 'error_sensitivity.json', payload)
    print(json.dumps(payload, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--banding', action='store_true')
    ap.add_argument('--mc', action='store_true')
    ap.add_argument('--r-q2', type=int, default=20)
    ap.add_argument('--r-q3', type=int, default=15)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.banding:
        data, _ = prepare()
        banding_part(data, None)
    if args.mc:
        mc_part(args.r_q2, args.r_q3)
