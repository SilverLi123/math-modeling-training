"""Q4 figures: ROC, PR curve and calibration for pooled outer-OOF output."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve,
                             average_precision_score, precision_recall_curve)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q4_model'
FIG = ROOT / 'outputs' / 'figures' / 'q4_model'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / (name + '.png'), dpi=300)
    fig.savefig(FIG / (name + '.svg'))
    plt.close(fig)
    print('saved', name)


def main():
    oof = pd.read_csv(OUT / 'oof_predictions.csv')
    y = oof.y.to_numpy()
    p = oof.p_elasticnet.to_numpy()
    z = oof.z_abs.to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, lw=1.8, label=f'Elastic-Net (AUC={roc_auc_score(y, p):.3f})')
    fpr_z, tpr_z, _ = roc_curve(y, z)
    axes[0].plot(fpr_z, tpr_z, lw=1.5, ls='--', color='gray',
                 label=f'|Z|max baseline (AUC={roc_auc_score(y, z):.3f})')
    axes[0].plot([0, 1], [0, 1], 'k:', lw=.8)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title('女胎 A_any 判定 ROC（外层 OOF）')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].plot(rec, prec, lw=1.8,
                 label=f'Elastic-Net (PR-AUC={average_precision_score(y, p):.3f})')
    axes[1].axhline(y.mean(), color='gray', ls='--', lw=1,
                    label=f'阳性率 {y.mean():.3f}')
    axes[1].set_xlabel('召回率'); axes[1].set_ylabel('精确率')
    axes[1].set_title('A_any 判定 PR 曲线（外层 OOF）')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    save(fig, 'result_q4_roc_pr')

    cal = pd.read_csv(OUT / 'calibration.csv')
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.plot([0, 1], [0, 1], 'k:', lw=.9, label='理想校准')
    ax.plot(cal.mean_prediction, cal.observed_rate, 'o-', ms=5, lw=1.4,
            label='Elastic-Net')
    for _, r in cal.iterrows():
        ax.annotate(f"n={int(r['rows'])}", (r.mean_prediction, r.observed_rate),
                    fontsize=7, xytext=(4, 5), textcoords='offset points')
    ax.set_xlabel('预测概率'); ax.set_ylabel('实际阳性率')
    ax.set_title('A_any 概率校准（固定分箱，记录级描述）')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, 'process_q4_calibration')


if __name__ == '__main__':
    main()
