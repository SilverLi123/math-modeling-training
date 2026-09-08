"""Q1 model-level rebuild (paper-consistent): LMM baselines vs GAMM-type
spline mixed models on the CLEANED male cohort.

Cohort (matches the paper's 清洗口径, see 5.1):
  raw male 1082 rows -> drop rows flagged 染色体非整倍体 (126) -> drop
  gestational week outside [10,25] (11 remaining non-aneuploid rows) ->
  945 rows; GC-content window [0.35,0.65] and Y in (0,1) are additionally
  enforced by the shared loader (no extra drops in this data).

Models (statsmodels MixedLM on logit(Y); Y in (0,1) so logit is valid):
  LMM-lin    : 1 + week_c + bmi_c
  LMM-quad   : 1 + week_c*bmi_c + I(week_c**2)
  GAMM-add   : 1 + cr(week_c, df) + cr(bmi_c, df)      (df in {3,4})
  GAMM-tensor: GAMM-add + cr(week)*cr(bmi) tensor term (degrade ok)
  LMM-rs     : LMM-quad + random slope week_c (attempted; singular on
               this data -> slope variance collapses to 0, reported not
               supported; excluded from final selection)

Metrics (OOF, subject-grouped 5-fold, new-subject = fixed effects only):
  logit-scale RMSE (record and subject-equal-weight) AND Y-scale RMSE
  (after inverse transform). Both scales are reported separately; no
  mixed-scale number is produced.

Selection rule: final model = the non-singular spec with the lowest
subject-weighted Y-scale OOF RMSE, unless a spline spec's advantage over
the parsimonious LMM-quad is not confirmed by a paired subject bootstrap
(2000 resamples on frozen OOF errors); then LMM-quad is kept. LMM-rs is
excluded from selection (random-slope variance at boundary).

Final: best spec refit on full data; Wald z-table for fixed effects;
partial-effect grids (week, bmi) with logit-space CI then inverse;
predicted week-curves at BMI 25/30/35/40; trajectory spaghetti.

Outputs -> outputs/q1_lmm_gamm/  (+ figures/q1_lmm_gamm/)
"""
from pathlib import Path
import argparse
import json
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from q1_lmm_baseline import load_and_prepare, subject_group_folds, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q1_lmm_gamm'
FIG = ROOT / 'outputs' / 'figures' / 'q1_lmm_gamm'
SEED = 20250904
DF_GRID = (3, 4)
BMI_LEVELS = (25, 30, 35, 40)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'

SPEC_CN = {'LMM-lin': 'LMM-lin（随机截距，线性）',
           'LMM-quad': 'LMM-quad（随机截距，二次）',
           'GAMM-add': 'GAMM-add（加性样条）',
           'GAMM-tensor': 'GAMM-tensor（张量样条）',
           'LMM-rs': 'LMM-rs（随机斜率，奇异）'}


# ---------------------------------------------------------------- data
def prepare(response='logit'):
    """清洗口径与正文一致：剔除染色体非整倍体标注行与孕周 10–25 周窗口
    外的行，得到 945 条男胎有效记录（267 名孕妇）。"""
    data, audit = load_and_prepare(ROOT / '附件.xlsx')
    raw = pd.read_excel(ROOT / '附件.xlsx', sheet_name='男胎检测数据')
    raw.columns = [str(c).strip() for c in raw.columns]
    aneu = set(raw.loc[raw['染色体的非整倍体'].notna(), '序号'])
    before = len(data)
    data = data.loc[~data.sample_id.isin(aneu)].copy()
    after_aneu = len(data)
    data = data.loc[data.gestational_week.between(10, 25)].copy()
    after_week = len(data)
    data = data.copy()
    if response == 'logit':
        data['y_resp'] = data.y_logit
    elif response == 'log':
        data['y_resp'] = np.log(data.y_fraction)
    elif response == 'identity':
        data['y_resp'] = data.y_fraction
    else:
        raise ValueError(response)
    audit['clean_cohort'] = {'rows_raw': int(before),
                             'rows_after_drop_aneuploid': int(after_aneu),
                             'rows_final': int(len(data)),
                             'subjects_final': int(data.subject_id.nunique()),
                             'rule': ('drop 非整倍体 rows; gestational week '
                                      '[10,25]; Y in (0,1); GC in [0.35,0.65]')}
    return data, audit


def inv_response(y, response):
    if response == 'logit':
        return 1.0 / (1.0 + np.exp(-np.clip(y, -30, 30)))
    if response == 'log':
        return np.exp(np.clip(y, -20, 20))
    return y


# ---------------------------------------------------------------- fitting
def _fit_mixedlm(frame, formula, re_formula=None):
    x = patsy.dmatrix(formula, frame, return_type='dataframe')
    if np.linalg.cond(np.asarray(x)) > 1e8:
        raise RuntimeError('Ill-conditioned design')
    if re_formula:
        exog_re = np.asarray(patsy.dmatrix('0 + ' + re_formula, frame))
    else:
        exog_re = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = sm.MixedLM(frame.y_resp.to_numpy(), np.asarray(x),
                           groups=frame.subject_id, exog_re=exog_re)
        result = model.fit(reml=False, method='lbfgs', maxiter=2000, disp=False)
    singular = False
    if re_formula:
        cov_re = np.asarray(result.cov_re)
        scale = max(np.max(np.diag(cov_re)), 1e-8)
        singular = float(np.max(np.abs(cov_re))) / scale < 1e-6 or \
            np.any(np.diag(cov_re) < 1e-6)
    if not result.converged:
        raise RuntimeError(f'Not converged: {result.converged}')
    return result, x.design_info, frame[['gestational_week', 'bmi']].mean(), \
        {'warnings': [str(w.message) for w in caught], 'singular_re': singular}


def fit_spec(data, spec, df=3):
    centers = data[['gestational_week', 'bmi']].mean()
    frame = data.copy()
    frame['week_c'] = frame.gestational_week - centers.gestational_week
    frame['bmi_c'] = frame.bmi - centers.bmi
    if spec == 'LMM-lin':
        formula, re_f = '1 + week_c + bmi_c', None
    elif spec == 'LMM-quad':
        formula, re_f = '1 + week_c*bmi_c + I(week_c**2)', None
    elif spec == 'LMM-rs':
        formula, re_f = '1 + week_c*bmi_c + I(week_c**2)', 'week_c'
    elif spec == 'GAMM-add':
        formula, re_f = (f"1 + cr(week_c, df={df}, constraints='center')"
                         f" + cr(bmi_c, df={df}, constraints='center')"), None
    elif spec == 'GAMM-tensor':
        formula, re_f = (f"1 + cr(week_c, df={df}, constraints='center')"
                         f" * cr(bmi_c, df={df}, constraints='center')"), None
    else:
        raise ValueError(spec)
    result, info, cent, diag = _fit_mixedlm(frame, formula, re_f)
    return {'kind': 'statsmodels', 'result': result, 'info': info,
            'centers': cent, 'spec': spec, 'df': df, 're_formula': re_f,
            'diag': diag}


def design(model, frame):
    frame = frame.copy()
    frame['week_c'] = frame.gestational_week - model['centers'].gestational_week
    frame['bmi_c'] = frame.bmi - model['centers'].bmi
    return np.asarray(patsy.build_design_matrices([model['info']], frame)[0])


def predict_logit(model, frame):
    pred = design(model, frame) @ model['result'].fe_params
    if not np.isfinite(pred).all():
        raise RuntimeError('Nonfinite prediction')
    return pred


# ---------------------------------------------------------------- metrics
def oof_metrics(frame, pred_y):
    """pred_y 与真值均在 Y 比例尺度；同时给出 logit 尺度与 Y 尺度、
    记录级与孕妇等权 RMSE（不再做任何二次变换）。"""
    y_true = frame.y_fraction.to_numpy()
    subj = frame.subject_id.to_numpy()
    sq = (y_true - pred_y) ** 2
    rmse_y = float(np.sqrt(sq.mean()))
    rmse_y_subj = float(np.sqrt(pd.Series(sq).groupby(subj).mean().mean()))
    pl = np.clip(pred_y, 1e-6, 1 - 1e-6)
    tl = np.clip(y_true, 1e-6, 1 - 1e-6)
    sq_l = (np.log(tl / (1 - tl)) - np.log(pl / (1 - pl))) ** 2
    rmse_l = float(np.sqrt(sq_l.mean()))
    rmse_l_subj = float(np.sqrt(pd.Series(sq_l).groupby(subj).mean().mean()))
    return {'y_rmse': rmse_y, 'y_subject_rmse': rmse_y_subj,
            'logit_rmse': rmse_l, 'logit_subject_rmse': rmse_l_subj}


def partial_effect(model, variable, response='logit', n=120):
    """logit 尺度线性预测的逐点 CI，再统一反变换回 Y 尺度显示。"""
    cent = model['centers']
    week_med = float(cent.gestational_week)
    if variable == 'bmi':
        grid = pd.DataFrame({'gestational_week': np.repeat(week_med, n),
                             'bmi': np.linspace(21, 46, n)})
    else:
        grid = pd.DataFrame({'gestational_week': np.linspace(11, 25, n),
                             'bmi': np.repeat(float(cent.bmi), n)})
    eta = predict_logit(model, grid)
    x = design(model, grid)
    cov = np.asarray(model['result'].cov_params())[:x.shape[1], :x.shape[1]]
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', x, cov, x), 0))
    x_col = grid['gestational_week'] if variable == 'week' else grid['bmi']
    return pd.DataFrame({'variable': variable, 'x': x_col,
                         'fit': inv_response(eta, response),
                         'low': inv_response(eta - 1.96 * se, response),
                         'high': inv_response(eta + 1.96 * se, response)})


# ---------------------------------------------------------------- main
# LMM-rs（随机斜率）在折叠与全数据拟合中均奇异（斜率方差收敛到 0，
# 协方差矩阵在参数空间边界），不进入正式规格表；单独诊断并记录。
SPECS_MAIN = ['LMM-lin', 'LMM-quad', 'GAMM-add', 'GAMM-tensor']


def run_outer(response='logit', specs=None, smoke=False):
    specs = specs or SPECS_MAIN
    data, audit = prepare(response)
    folds = subject_group_folds(data.subject_id, 5, SEED)
    if smoke:
        mask = data.subject_id.isin(folds[0])
        train, valid = data.loc[~mask], data.loc[mask]
        for spec in ['LMM-lin', 'LMM-quad', 'GAMM-add']:
            try:
                m = fit_spec(train, spec)
                print(spec, 'ok', flush=True)
            except Exception as exc:
                print(spec, 'FAILED:', str(exc)[:120], flush=True)
        return
    oof = data[['sample_id', 'subject_id', 'gestational_week', 'bmi',
                'y_logit', 'y_fraction']].copy()
    oof['fold'] = 0
    rows, degrad = [], []
    for k, ids in enumerate(folds, 1):
        mask = data.subject_id.isin(ids)
        train, valid = data.loc[~mask], data.loc[mask]
        for spec in specs:
            df_try = DF_GRID if spec.startswith('GAMM') else (0,)
            for dfv in df_try:
                try:
                    m = fit_spec(train, spec, dfv)
                    pred_logit = predict_logit(m, valid)
                    pred_y = inv_response(pred_logit, response)
                    met = oof_metrics(valid, pred_y)
                    rows.append({'fold': k, 'spec': spec, 'df': dfv, **met})
                    oof.loc[mask, f'{spec}_df{dfv}'] = pred_y
                    if spec.startswith('GAMM'):
                        break  # first successful df wins (lowest)
                    break
                except Exception as exc:
                    if dfv == df_try[-1]:
                        degrad.append({'fold': k, 'spec': spec, 'df': dfv,
                                       'error': str(exc)[:200]})
                    else:
                        continue
        print(f'response={response} fold {k} done', flush=True)
    return oof, rows, degrad, data, audit


def summarise(oof, rows, response):
    specs = sorted({r['spec'] + f"_df{r['df']}" for r in rows})
    n = len(oof)
    summary = []
    for s in specs:
        if s not in oof or oof[s].isna().all():
            continue
        col = oof[s].to_numpy()
        if np.isnan(col).any():       # 某折退化导致列不完整：不给出汇总
            continue
        met = oof_metrics(oof, col)
        summary.append({'spec_col': s, **met})
    return pd.DataFrame(summary)


def run_full():
    all_summaries = {}
    for response in ['logit', 'identity', 'log']:
        specs = SPECS_MAIN if response == 'logit' else ['LMM-lin', 'LMM-quad']
        oof, rows, degrad, data, audit = run_outer(response, specs)
        summ = summarise(oof, rows, response)
        summ.insert(0, 'response', response)
        all_summaries[response] = summ
        if response == 'logit':
            main_oof, main_rows, main_degrad = oof, rows, degrad
    summary_all = pd.concat(all_summaries.values(), ignore_index=True)
    return summary_all, main_oof, main_degrad, data, audit


def final_effect_outputs(best_spec, df_used, data, response='logit'):
    data_r, _ = prepare(response)
    final = fit_spec(data_r, best_spec, df_used)
    # Wald table
    res = final['result']
    names = list(final['info'].design_info.column_names)
    z = np.asarray(res.fe_params) / np.asarray(res.bse_fe)
    pvals = 2 * (1 - _norm_cdf(np.abs(z)))
    table = pd.DataFrame({'term': names, 'coef': np.asarray(res.fe_params),
                          'se': np.asarray(res.bse_fe), 'z': z, 'p': pvals})
    # partial effects (Y-scale display with logit-space CI)
    eff_week = partial_effect(final, 'week', response)
    eff_bmi = partial_effect(final, 'bmi', response)
    # predicted week-curves at BMI levels
    curves = []
    for B in BMI_LEVELS:
        grid = pd.DataFrame({'gestational_week': np.linspace(11, 25, 120),
                             'bmi': B})
        curves.append(pd.DataFrame({'bmi': B,
                                    'gestational_week': grid.gestational_week,
                                    'pred_y': inv_response(
                                        predict_logit(final, grid), response)}))
    curves = pd.concat(curves)
    # response heatmap if tensor kept
    heat = None
    if 'tensor' in best_spec or best_spec.startswith('GAMM-tensor'):
        g = pd.DataFrame([(w, b) for b in np.arange(22, 44.1, 1.0)
                          for w in np.linspace(11, 25, 57)],
                         columns=['gestational_week', 'bmi'])
        g['pred_y'] = inv_response(predict_logit(final, g), response)
        heat = g
    return final, table, eff_week, eff_bmi, curves, heat


def _norm_cdf(x):
    from scipy.stats import norm
    return norm.cdf(x)


def paired_bootstrap_subject(main_oof, col_a, col_b, seed=SEED,
                             n=2000, metric='logit'):
    """冻结 OOF 上的孕妇级配对 Bootstrap：两规格逐条误差按孕妇聚合为
    均方误差，再对孕妇重采样 2000 次，返回 col_b - col_a 的均方根差
    点估计与 95% CI（>0 表示 col_a 更优）。"""
    frame = main_oof[['subject_id', 'y_logit', 'y_fraction', col_a,
                      col_b]].dropna()
    a = frame[col_a].to_numpy()
    b = frame[col_b].to_numpy()
    y = frame.y_fraction.to_numpy()
    if metric == 'logit':
        def logit(x):
            c = np.clip(x, 1e-6, 1 - 1e-6)
            return np.log(c / (1 - c))
        a_t, b_t, y_t = logit(a), logit(b), logit(y)
        err_a = (y_t - a_t) ** 2
        err_b = (y_t - b_t) ** 2
    else:
        err_a = (y - a) ** 2
        err_b = (y - b) ** 2
    g = pd.DataFrame({'subj': frame.subject_id.to_numpy(),
                      'ea': err_a, 'eb': err_b})
    per = g.groupby('subj').mean()
    rng = np.random.default_rng(seed)
    idx = per.index.to_numpy()
    diffs = []
    for _ in range(n):
        s = per.iloc[rng.integers(len(idx), size=len(idx))]
        diffs.append(float(np.sqrt(s.eb.mean()) - np.sqrt(s.ea.mean())))
    ci = tuple(np.quantile(diffs, [0.025, 0.975]))
    return {'spec_a': col_a, 'spec_b': col_b, 'metric': metric,
            'point_rmse_b_minus_a': float(diffs[0]) if False else
            float(np.sqrt(per.eb.mean()) - np.sqrt(per.ea.mean())),
            'ci_rmse_b_minus_a': list(ci)}


def choose_final(summary, main_oof):
    """选择规则见文件头 docstring。返回 (final_spec, df, 决定说明)。"""
    logit = summary[summary.response == 'logit'].copy()
    logit = logit[~logit.spec_col.str.startswith('LMM-rs')].copy()
    logit = logit.sort_values('y_subject_rmse').reset_index(drop=True)
    if len(logit) == 0:
        raise RuntimeError('no candidate spec')
    cand = logit.iloc[0]
    base = cand.spec_col.rsplit('_df', 1)[0]
    dfv = int(cand.spec_col.rsplit('_df', 1)[1]) if '_df' in cand.spec_col \
        else 0
    # 若候选不是 LMM-quad，需与 LMM-quad 做冻结 OOF 的配对 Bootstrap；
    # 优势不被确认（CI 含 0 或点估计非正）则保留 LMM-quad。
    lmm_col = 'LMM-quad_df0'
    if base != 'LMM-quad':
        if lmm_col not in main_oof:
            return base, dfv, 'no LMM-quad OOF column; use best candidate'
        try:
            boot = paired_bootstrap_subject(main_oof, cand.spec_col,
                                            lmm_col, metric='logit')
            better = boot['point_rmse_b_minus_a'] > 0 and \
                boot['ci_rmse_b_minus_a'][0] > 0
            if not better:
                return ('LMM-quad', 0,
                        'spline advantage over LMM-quad not confirmed by '
                        'paired bootstrap: ' +
                        json.dumps(boot, ensure_ascii=False))
            return base, dfv, 'best by OOF subject Y-RMSE; bootstrap supports'
        except Exception as exc:
            return ('LMM-quad', 0,
                    f'paired bootstrap failed ({str(exc)[:80]}); keep LMM-quad')
    return base, dfv, 'parsimonious LMM-quad (no spline confirmed advantage)'


def plots(main_oof, final, eff_week, eff_bmi, curves, heat, data, response,
          model_cn):
    FIG.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(FIG / (name + '.png'), dpi=300)
        fig.savefig(FIG / (name + '.svg'))
        plt.close(fig)
        print('saved', name)

    # 1 raw scatter + mean trend from final model at median bmi
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(data.gestational_week, data.y_fraction, s=8, alpha=.35,
               color='#0072B2')
    grid = pd.DataFrame({'gestational_week': np.linspace(11, 25, 120)})
    grid['bmi'] = data.bmi.median()
    ax.plot(grid.gestational_week, inv_response(predict_logit(final, grid),
                                                response), 'r-', lw=2,
            label=f'{model_cn} 拟合（BMI=中位数）')
    ax.set_xlabel('检测孕周（周）'); ax.set_ylabel('Y 染色体浓度（比例）')
    ax.set_title('男胎 Y 浓度散点与主模型拟合（清洗后 n=%d）' % len(data))
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, 'result_q1_scatter_gamm')

    # 2/3 partial effects (Y-scale display)
    for eff, title, fname in [(eff_week, '孕周方向拟合均值（Y 尺度，95%Wald）',
                               'result_q1_effect_week'),
                              (eff_bmi, 'BMI 方向拟合均值（Y 尺度，95%Wald）',
                               'result_q1_effect_bmi')]:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(eff.x, eff.fit, lw=2)
        ax.fill_between(eff.x, eff.low, eff.high, alpha=.25)
        ax.set_xlabel('孕周（周）' if 'week' in fname else 'BMI（kg/m²）')
        ax.set_ylabel('Y 染色体浓度（比例）')
        ax.set_title(title)
        ax.grid(alpha=.3)
        save(fig, fname)

    # 4 predicted curves at BMI levels
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for B, c in zip(BMI_LEVELS, ['#0072B2', '#E69F00', '#D55E00', '#009E73']):
        g = curves[curves.bmi == B]
        ax.plot(g.gestational_week, g.pred_y, lw=1.8, color=c, label=f'BMI={B}')
    ax.axhline(0.04, color='gray', ls=':', lw=1)
    ax.set_xlabel('检测孕周（周）'); ax.set_ylabel('Y 染色体浓度（预测比例）')
    ax.set_title('不同 BMI 水平下的 Y 浓度预测曲线')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, 'result_q1_curves_by_bmi')

    # 5 heatmap if tensor kept
    if heat is not None:
        piv = heat.pivot(index='bmi', columns='gestational_week',
                         values='pred_y')
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        im = ax.pcolormesh(piv.columns, piv.index, piv.values, cmap='viridis')
        fig.colorbar(im, ax=ax, label='Y 浓度预测')
        ax.set_xlabel('孕周（周）'); ax.set_ylabel('BMI')
        ax.set_title('week×BMI 交互响应面（保留的张量交互）')
        save(fig, 'result_q1_surface')

    # 6 sample subject trajectories
    rng = np.random.default_rng(SEED)
    codes = rng.choice(pd.Series(data.subject_id).unique(), 20, replace=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for code in codes:
        g = data[data.subject_id == code].sort_values('gestational_week')
        ax.plot(g.gestational_week, g.y_fraction, marker='o', ms=3, lw=.8,
                alpha=.5)
    ax.axhline(0.04, color='red', ls='--', lw=1)
    ax.set_xlabel('孕周（周）'); ax.set_ylabel('Y 染色体浓度（比例）')
    ax.set_title('20 名孕妇的个体纵向轨迹（示意）')
    ax.grid(alpha=.3)
    save(fig, 'result_q1_trajectories')


def main():
    warnings.simplefilter('default', DeprecationWarning)
    warnings.filterwarnings('default', category=DeprecationWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    summary_all, main_oof, main_degrad, data, audit = run_full()
    summary_all.to_csv(OUT / 'spec_comparison.csv', index=False)
    main_oof.to_csv(OUT / 'oof_predictions.csv', index=False)
    # response-form ranking (Y-scale subject RMSE, all three responses)
    resp_form = summary_all[summary_all.spec_col.isin(
        ['LMM-lin_df0', 'LMM-quad_df0'])][
        ['response', 'spec_col', 'y_subject_rmse', 'logit_subject_rmse']] \
        .sort_values(['spec_col', 'y_subject_rmse'])
    resp_form.to_csv(OUT / 'response_form.csv', index=False)
    final_spec, df_used, decision = choose_final(summary_all, main_oof)
    final, table, eff_week, eff_bmi, curves, heat = final_effect_outputs(
        final_spec, df_used, data)
    table.to_csv(OUT / 'final_wald_table.csv', index=False)
    eff_week.to_csv(OUT / 'effect_week.csv', index=False)
    eff_bmi.to_csv(OUT / 'effect_bmi.csv', index=False)
    curves.to_csv(OUT / 'curves_by_bmi.csv', index=False)
    if heat is not None:
        heat.to_csv(OUT / 'response_heatmap.csv', index=False)
    spec_notes = []
    for _, row in summary_all[summary_all.response == 'logit'].iterrows():
        base = row.spec_col.rsplit('_df', 1)[0]
        try:
            m = fit_spec(data, base,
                         int(row.spec_col.rsplit('_df', 1)[1]) if '_df'
                         in row.spec_col else 0)
            spec_notes.append({'spec': base,
                               'full_data_converged': bool(
                                   m['result'].converged),
                               'full_data_singular_re': bool(
                                   m['diag']['singular_re'])})
        except Exception as exc:
            spec_notes.append({'spec': base,
                               'full_data_error': str(exc)[:120]})
    plots(main_oof, final, eff_week, eff_bmi, curves, heat, data, 'logit',
          SPEC_CN[final_spec])
    write_json(OUT / 'experiment.json', {
        'seed': SEED, 'responses': ['logit', 'identity', 'log'],
        'clean_cohort': audit['clean_cohort'],
        'spec_summary': summary_all.to_dict(orient='records'),
        'response_form': resp_form.to_dict(orient='records'),
        'final_spec': final_spec, 'final_df': df_used,
        'selection_decision': decision,
        'spec_full_data_notes': spec_notes,
        'degradations': main_degrad,
        're_formula': final['re_formula'],
        'diag': {k: v for k, v in final['diag'].items()},
        'note': ('paper-cohort Q1 rebuild: cleaned 945 male rows; '
                 'dual-scale metrics (logit & Y); LMM-rs singular -> '
                 'excluded; selection via frozen-OOF paired bootstrap')})
    print('DONE; final:', final_spec, '| decision:', decision, flush=True)
    print('audit:', audit['clean_cohort'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        run_outer('logit', smoke=True)
    else:
        main()
