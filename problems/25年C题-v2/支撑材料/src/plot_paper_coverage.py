"""Nonredundant raw/validation/decision evidence missing from the paper set."""
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from q23_explore import ROOT, load
sys.path.insert(0, str(ROOT))
from utils.plot_style import apply_publication_style, export_figure

FIG = ROOT/'outputs/figures/paper_coverage'


def main():
    data, _ = load()
    apply_publication_style(language='zh')
    first = data.drop_duplicates('subject_id')
    fig, ax = plt.subplots(figsize=(6., 3.4), layout='constrained')
    ax.hist(data.groupby('subject_id').size(), bins=np.arange(.5, 10.6, 1), color='#0072B2')
    ax.set(xlabel='每名孕妇检测记录数（次）', ylabel='孕妇人数（人）', title='重复检测结构')
    export_figure(fig, FIG/'raw_q1_repeat_counts'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6., 3.4), layout='constrained')
    ax.scatter(data.gestational_week, data.baseline_bmi, s=7, alpha=.25, color='#0072B2')
    ax.set(xlabel='观测孕周（周）', ylabel='首次观测 BMI', title='孕周与 BMI 覆盖')
    export_figure(fig, FIG/'raw_q2_observation_support'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6., 3.4), layout='constrained')
    ax.scatter(first.baseline_age, first.baseline_height, s=12, alpha=.5, color='#0072B2')
    ax.set(xlabel='首次观测年龄（岁）', ylabel='身高（cm）', title='多因素协变量覆盖')
    export_figure(fig, FIG/'raw_q3_covariates'); plt.close(fig)

    report = json.loads((ROOT/'outputs/q23_exploration/experiment.json').read_text(encoding='utf-8'))
    names = ['q3_basic', 'q3_full']
    means = np.array([report['paired_subject_brier'][name]['mean'] for name in names])
    ci = np.array([report['paired_subject_brier'][name]['ci95'] for name in names])
    fig, ax = plt.subplots(figsize=(6., 3.2), layout='constrained')
    ax.errorbar(means, [0, 1], xerr=np.vstack([means-ci[:, 0], ci[:, 1]-means]), fmt='o', capsize=4, color='#0072B2')
    ax.axvline(0, ls='--', color='.5')
    ax.set(yticks=[0, 1], yticklabels=['年龄、身高', '加入孕产与受孕方式'],
           xlabel='主体 Brier 差（多因素 − BMI 模型）', title='多因素预测增益检验')
    export_figure(fig, FIG/'process_q3_paired_comparison'); plt.close(fig)

    policies = pd.read_csv(ROOT/'outputs/q23_threshold_sensitivity/conditional_policies.csv')
    fig, ax = plt.subplots(figsize=(6., 3.4), layout='constrained')
    for model, color, style, label in [('q2', '#0072B2', '-', 'BMI 模型'),
                                      ('q3_basic', '#D55E00', '--', '多因素模型')]:
        row = policies.loc[(policies.model == model) & (policies.threshold == .04) &
                           (policies.scenario == 'medium_delay') & (policies.k == 4)].iloc[0]
        cuts, weeks = json.loads(row.merged_cutpoints), json.loads(row.merged_weeks)
        edges = [float(first.baseline_bmi.min()), *cuts, float(first.baseline_bmi.max())]
        ax.stairs(weeks, edges, baseline=None, color=color, linestyle=style, label=label, linewidth=1.8)
    ax.set(xlabel='首次观测 BMI', ylabel='条件选中孕周（周）', title='中延迟情景策略')
    ax.legend(loc='lower right')
    export_figure(fig, FIG/'result_q3_conditional_policy'); plt.close(fig)


if __name__ == '__main__':
    main()
