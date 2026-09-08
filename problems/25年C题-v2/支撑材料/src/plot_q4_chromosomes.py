"""Q4 evidence: positive subject counts, calibration and paired loss intervals."""
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from q4_chromosomes import ROOT, OUT, TARGETS
sys.path.insert(0, str(ROOT))
from utils.plot_style import apply_publication_style, export_figure

FIG = ROOT/'outputs/figures/q4_chromosomes'


def main():
    apply_publication_style(language='zh')
    report = json.loads((OUT/'experiment.json').read_text(encoding='utf-8'))
    labels = ['T13', 'T18', 'T21', '任一异常']
    fig, ax = plt.subplots(figsize=(6., 3.2), layout='constrained')
    counts = [report['audit']['per_label'][t]['positive_subjects'] for t in TARGETS]
    ax.barh(labels, counts, color='#0072B2')
    for i, n in enumerate(counts):
        ax.text(n+.7, i, str(n), va='center')
    ax.set(xlabel='阳性孕妇人数（人）', xlim=(0, max(counts)+8), title='各标签阳性样本')
    export_figure(fig, FIG/'raw_q4_positive_subjects'); plt.close(fig)

    cal = pd.read_csv(OUT/'calibration.csv').query("target == 'a_any'")
    fig, ax = plt.subplots(figsize=(6., 3.4), layout='constrained')
    ax.plot([0, 1], [0, 1], '--', color='.5', label='理想校准')
    ax.scatter(cal.predicted, cal.observed, color='#0072B2', s=35, label='验证分箱')
    for row in cal.itertuples():
        ax.annotate(f'n={row.rows}', (row.predicted, row.observed), xytext=(6, 7),
                    textcoords='offset points', fontsize=8)
    ax.set(xlabel='分箱平均预测概率', ylabel='分箱实际阳性比例',
           xlim=(0, 1), ylim=(0, 1), title='总体标签概率校准')
    ax.legend(loc='lower right')
    export_figure(fig, FIG/'process_q4_calibration'); plt.close(fig)

    stats = pd.read_csv(OUT/'comparison.csv').set_index('target').loc[TARGETS]
    fig, ax = plt.subplots(figsize=(6., 3.4), layout='constrained')
    x = stats.subject_delta_brier.to_numpy()
    ax.errorbar(x, np.arange(4), xerr=np.vstack([x-stats.delta_ci_low, stats.delta_ci_high-x]),
                fmt='o', color='#0072B2', capsize=4)
    ax.axvline(0, color='.5', ls='--')
    ax.set(yticks=np.arange(4), yticklabels=labels, xlabel='主体 Brier 差（模型 − 基准）',
           title='各标签误差变化')
    export_figure(fig, FIG/'result_q4_brier_intervals'); plt.close(fig)


if __name__ == '__main__':
    main()
