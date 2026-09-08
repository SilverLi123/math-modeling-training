"""Exploratory Q1 figures at fixed physical size; no clinical inference."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.plot_style import apply_publication_style, export_figure


def main():
    apply_publication_style(language='zh')
    source=ROOT/'outputs'/'q1_comparison'
    out=ROOT/'outputs'/'figures'/'q1_comparison'
    p=pd.read_csv(source/'oof_predictions.csv')
    f=pd.read_csv(source/'fold_metrics.csv')
    e=pd.read_csv(source/'effects.csv')
    # Contract: paired points show stability across identical held-out groups;
    # curves show conditional-on-basis Wald intervals; scatter/QQ diagnose OOF error.
    fig,ax=plt.subplots(figsize=(6.3,3.8),layout='constrained')
    for fold,g in f.groupby('fold'):
        g=g.set_index('model')
        ax.plot([0,1],[g.loc['LMM','subject_rmse'],g.loc['Spline','subject_rmse']],
                marker='o',color='#6B7280',alpha=.65)
        ax.annotate(f'折 {int(fold)}',(1,g.loc['Spline','subject_rmse']),xytext=(8,0),textcoords='offset points',fontsize=7)
    ax.set(xticks=[0,1],xticklabels=['LMM','加性样条混合模型'],xlim=(-.25,1.3),
           ylabel='孕妇等权 RMSE（logit 尺度）',title='相同留出孕妇上的误差对比')
    export_figure(fig,out/'result_q1_comparison'); plt.close(fig)
    for variable,label in [('gestational_week','孕周（周）'),('bmi','BMI（kg/m²）')]:
        g=e[e.variable==variable]
        fig,ax=plt.subplots(figsize=(6.3,3.8),layout='constrained')
        ax.fill_between(g.x,g.low,g.high,color='#0072B2',alpha=.18,label='逐点 95% Wald 区间')
        ax.plot(g.x,g.fit,color='#0072B2',label='固定效应预测')
        ax.scatter(p[variable],np.repeat(0.02,len(p)),marker='|',color='#6B7280',alpha=.15,
                   transform=ax.get_xaxis_transform(),s=9)
        ax.set(xlabel=label,ylabel='预测 logit(Y 浓度)',title='加性样条效应切片')
        ax.legend(loc='best')
        export_figure(fig,out/f'result_q1_effect_{variable}'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained')
    residual=p.y_logit-p.Spline
    axes[0].scatter(p.Spline,residual,s=5,alpha=.3,color='#0072B2')
    axes[0].axhline(0,color='#222222',linestyle='--',linewidth=.8)
    axes[0].set(xlabel='OOF 固定效应预测',ylabel='OOF 残差（logit）',title='预测残差')
    (q,ordered),(slope,intercept,_) = stats.probplot(residual)
    axes[1].scatter(q,ordered,s=5,alpha=.3,color='#0072B2')
    axes[1].plot(q,intercept+slope*q,color='#222222',linestyle='--')
    axes[1].set(xlabel='标准正态理论分位数',ylabel='OOF 残差分位数',title='残差 Q–Q 图')
    export_figure(fig,out/'process_q1_oof_diagnostics'); plt.close(fig)


if __name__=='__main__':
    main()
