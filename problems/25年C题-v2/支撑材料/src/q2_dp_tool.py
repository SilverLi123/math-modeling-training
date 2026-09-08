"""Ordered DP tool: contiguous BMI bands and non-decreasing test weeks.

PURE ALGORITHM / TOOLING ONLY — project gate (题目分析报告.md): do NOT use
this to publish recommended test weeks until a probability model passes the
decision gates. Exercised only on synthetic data in tests and the example.

Problem: subjects ordered by BMI; partition into at most K contiguous bands
and pick one test week per band from a grid, weeks non-decreasing across
bands (higher BMI bands are never advised earlier). Cost per subject j in a
band tested at week w is

    cost_j(w) = (1 - p_j(w)) + w_late * L(w),

with p_j(w) the model-supplied marginal observable-at-threshold probability
and L(w) the piecewise late-detection risk of the problem statement. An
optional per-band constraint requires mean_j p_j(w) >= coverage_q; bands
with no feasible week are reported (never silently relaxed).

Solver: DP over prefix i (1..n), number of bands k, and last week index w.
dp[i,k,w] = min over band start a<i and previous week index wp<=w of
dp[a,k-1,wp] + seg(a,i,w), where seg is the total band cost and feasibility
of seg(a,i,w) is checked against coverage. A prefix minimum over wp makes
transitions O(1) per (a,i,w), total O(n^2*K*m). Intended n ~ hundreds.
"""
import numpy as np


def late_risk(w):
    if w <= 12:
        return 0.0
    if w <= 28:
        return (w - 12) / 16.0
    return 1.0 + (w - 28) / 8.0


def plan(bmi, p, weeks, max_segments, w_late=1.0, coverage_q=None):
    """Return optimal ordered banding plan (see module docstring).

    Returns dict with segments (list of dicts), total_cost, flags; or with
    infeasible=True when no solution exists under the coverage constraint.
    """
    bmi = np.asarray(bmi, float)
    p = np.asarray(p, float)
    weeks = np.asarray(weeks, float)
    if bmi.ndim != 1 or p.ndim != 2 or p.shape[0] != len(bmi) or \
            p.shape[1] != len(weeks):
        raise ValueError('Shape mismatch: p must be (n_subjects, n_weeks)')
    if not np.all(np.diff(weeks) > 0):
        raise ValueError('weeks must be strictly increasing')
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError('p must lie strictly inside (0,1)')
    order = np.argsort(bmi, kind='stable')
    bmi_s = bmi[order]
    p_s = p[order]
    n, m = p_s.shape
    K = max(1, min(int(max_segments), n))
    INF = np.inf
    late = np.array([late_risk(w) for w in weeks])
    # band sums of p per week for segment means
    cum = np.zeros((n + 1, m))
    cum[1:] = np.cumsum(p_s, axis=0)
    late_pen = w_late * late
    # dp[a][k-1][w] used as prefix minima per a: we keep arrays sized K+1
    dp = np.full((n + 1, K + 1, m), INF)
    best_start = np.full((n + 1, K + 1, m), -1, dtype=int)
    for w in range(m):
        dp[0, 0, w] = 0.0  # empty prefix, zero bands, any week baseline
    best_start[0, 0, :] = 0
    for i in range(1, n + 1):
        for k in range(1, min(K, i) + 1):
            # prefix minima over previous week for each start a
            # pref[a][w] = min_{wp<=w} dp[a, k-1, wp]
            pref = np.empty((i, m))
            for a in range(i):
                row = dp[a, k - 1]
                pref[a] = np.minimum.accumulate(row)
            for w in range(m):
                # per-subject term in this band for week w is constant across a
                for a in range(i):
                    if not np.isfinite(pref[a, w]):
                        continue
                    width = i - a
                    if width == 0:
                        continue
                    mean_p = (cum[i, w] - cum[a, w]) / width
                    if coverage_q is not None and mean_p < coverage_q:
                        continue
                    seg = width * ((1.0 - mean_p) + late_pen[w])
                    cand = pref[a, w] + seg
                    if cand < dp[i, k, w]:
                        dp[i, k, w] = cand
                        best_start[i, k, w] = a
    # pick the best final state
    best = None
    best_val = INF
    for k in range(1, K + 1):
        for w in range(m):
            if dp[n, k, w] < best_val:
                best_val = dp[n, k, w]
                best = (k, w)
    if best is None:
        return {'segments': [], 'total_cost': None, 'infeasible': True,
                'coverage_relaxed': False, 'n_subjects': n}
    k, w = best
    # backtrack weeks and starts
    weeks_idx = []
    starts = []
    i, kk, ww = n, k, w
    while kk > 0:
        starts.append(int(best_start[i, kk, ww]))
        weeks_idx.append(int(ww))
        a = best_start[i, kk, ww]
        # previous week = argmin over wp<=ww of dp[a,kk-1,wp]
        row = dp[a, kk - 1]
        wp = int(np.argmin(np.minimum.accumulate(row)[: ww + 1]) if False
                 else np.argmin(row[: ww + 1]))
        i, kk, ww = a, kk - 1, wp
        if i == 0:
            break
    starts = starts[::-1]
    weeks_idx = weeks_idx[::-1]
    segments = []
    edges = starts + [n]
    for s_i in range(len(starts)):
        a, b = starts[s_i], edges[s_i + 1]
        band = np.arange(a, b)
        mean_p = (cum[b] - cum[a])[weeks_idx[s_i]] / (b - a)
        segments.append({
            'bmi_lo': float(bmi_s[a]),
            'bmi_hi': float(bmi_s[b - 1]),
            'week': float(weeks[weeks_idx[s_i]]),
            'n': int(b - a),
            'mean_coverage': float(mean_p)})
    return {'segments': segments, 'total_cost': float(best_val),
            'infeasible': False, 'coverage_relaxed': False,
            'n_subjects': n, 'n_segments': len(segments)}


def pareto(bmi, p, weeks, max_segments, weights, coverage_q=None):
    """Sweep late-detection-risk weights as scenario parameters.

    Returns list of plan results (one per weight) plus a compact table of
    (w_late, total_cost, first/last band weeks) for Pareto-style reporting.
    """
    rows = []
    for w in weights:
        plan_i = plan(bmi, p, weeks, max_segments, w_late=float(w),
                      coverage_q=coverage_q)
        rows.append(plan_i)
    table = []
    for w, plan_i in zip(weights, rows):
        table.append({'w_late': float(w),
                      'total_cost': plan_i.get('total_cost'),
                      'weeks': [s['week'] for s in plan_i.get('segments', [])],
                      'infeasible': plan_i.get('infeasible', False)})
    return rows, table
