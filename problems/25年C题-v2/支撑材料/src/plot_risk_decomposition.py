"""Q2 risk decomposition: FailureRisk, DelayRisk, TotalRisk curves by BMI band.
Generates the figure the user specifically requested to validate that
't*=12 weeks for all BMI' is not an artifact of scale mismatch."""
from pathlib import Path
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q2_decision'
FIG = ROOT / 'outputs' / 'figures' / 'q2_decision'
FIG.mkdir(parents=True, exist_ok=True)


def late_risk(w):
    if w <= 12: return 0.0
    if w <= 28: return (w - 12) / 16.0
    return 1.0 + (w - 28) / 8.0


def main():
    surface = pd.read_csv(OUT / 'probability_surface.csv')
    bands = [(20.7, 32.1, '低 BMI'), (32.1, 34.7, '中 BMI'), (34.7, 47.0, '高 BMI')]
    colors = {'FailureRisk': '#D55E00', 'DelayRisk': '#0072B2', 'TotalRisk': '#009E73'}

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), sharey=True)
    for ax, (lo, hi, label) in zip(axes, bands):
        sub = surface[(surface.bmi >= lo) & (surface.bmi < hi)]
        mean_p = sub.groupby('gestational_week').probability.mean().reset_index()
        mean_p.columns = ['week', 'p']
        mean_p['fail'] = 1 - mean_p.p
        mean_p['delay'] = [late_risk(w) for w in mean_p.week]
        mean_p['total'] = mean_p.fail + mean_p.delay
        for col, lbl, c in [('fail', 'FailureRisk 1−p', '#D55E00'),
                            ('delay', 'DelayRisk L(t)', '#0072B2'),
                            ('total', 'TotalRisk R(t)', '#009E73')]:
            ax.plot(mean_p.week, mean_p[col], lw=1.8, label=lbl, color=c)
        # Mark minimum of total risk
        imin = mean_p.total.idxmin()
        ax.axvline(mean_p.loc[imin, 'week'], color='red', ls=':', lw=1)
        ax.annotate(f"t*={mean_p.loc[imin,'week']:.1f}",
                    (mean_p.loc[imin, 'week'], mean_p.loc[imin, 'total']),
                    fontsize=8, color='red', xytext=(5, 10),
                    textcoords='offset points')
        ax.set_title(f'{label} [{lo:.1f}, {hi:.1f})', fontsize=10)
        ax.set_xlabel('孕周（周）')
        ax.legend(fontsize=7)
        ax.grid(alpha=.3)
    axes[0].set_ylabel('风险')
    fig.suptitle('Q2 风险函数分解：FailureRisk / DelayRisk / TotalRisk（GAMM-Binomial 概率模型）', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / 'result_q2_risk_decomposition.png', dpi=300)
    fig.savefig(FIG / 'result_q2_risk_decomposition.svg')
    plt.close(fig)
    print('saved risk decomposition figure')

    # Also output the numerical data
    all_rows = []
    for lo, hi, label in bands:
        sub = surface[(surface.bmi >= lo) & (surface.bmi < hi)]
        mean_p = sub.groupby('gestational_week').probability.mean().reset_index()
        mean_p.columns = ['week', 'p']
        mean_p['fail'] = 1 - mean_p.p
        mean_p['delay'] = [late_risk(w) for w in mean_p.week]
        mean_p['total'] = mean_p.fail + mean_p.delay
        mean_p['band'] = label
        all_rows.append(mean_p)
    pd.concat(all_rows).to_csv(OUT / 'risk_decomposition.csv', index=False)
    print('saved risk_decomposition.csv')


if __name__ == '__main__':
    main()
