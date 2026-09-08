"""Q2 exploratory calibration and marginal-probability figures."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from utils.plot_style import apply_publication_style, export_figure


def main():
    apply_publication_style(language='zh')
    source=ROOT/'outputs'/'q2_probability'
    out=ROOT/'outputs'/'figures'/'q2_probability'
    cal=pd.read_csv(source/'calibration.csv')
    time=pd.read_csv(source/'time_calibration.csv')
    surface=pd.read_csv(source/'surface.csv')
    # Calibration contract: held-out probabilities, fixed bins, counts in CSV;
    # descriptive points, no row-independent confidence intervals.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained')
    for name,label,color,marker in [('mixed_logistic','混合 Logistic','#0072B2','o'),
                                    ('training_rate','训练集达标率','#D55E00','s')]:
        g=cal[cal.model==name]
        axes[0].scatter(g.mean_prediction,g.observed_rate,s=20,color=color,marker=marker,label=label)
    axes[0].plot([0,1],[0,1],color='#222222',linestyle='--',linewidth=.8)
    axes[0].set(xlim=(0,1.02),ylim=(0,1.02),xlabel='分箱平均 OOF 概率',ylabel='分箱实际达标率',title='样本外概率校准')
    axes[0].legend(loc='upper left')
    x=np.arange(len(time))
    axes[1].scatter(x-.12,time.observed_rate,color='#222222',marker='x',label='实际达标率',s=22)
    axes[1].scatter(x+.12,time.mixed_logistic,color='#0072B2',marker='o',label='平均 OOF 概率',s=22)
    axes[1].set(xticks=x,xticklabels=[f'{int(a)}–<{int(b)}' for a,b in zip(time.week_low,time.week_high)],
                ylim=(0,1.02),xlabel='观测孕周分组（周）',ylabel='达标比例 / 概率',title='按孕周检查校准')
    axes[1].legend(loc='lower right')
    export_figure(fig,out/'process_q2_calibration');plt.close(fig)
    probability=surface.pivot(index='baseline_bmi',columns='gestational_week',values='probability')
    support=surface.pivot(index='baseline_bmi',columns='gestational_week',values='local_subjects')
    valid=surface.pivot(index='baseline_bmi',columns='gestational_week',values='supported')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained')
    cmap=plt.get_cmap('cividis').copy();cmap.set_bad('#E5E7EB')
    mesh=axes[0].pcolormesh(probability.columns,probability.index,np.ma.masked_where(~valid.to_numpy(),probability.to_numpy()),
                            cmap=cmap,vmin=0,vmax=1,shading='nearest')
    bar=fig.colorbar(mesh,ax=axes[0],label='边际达标概率')
    bar.solids.set_rasterized(False)
    mesh=axes[1].pcolormesh(support.columns,support.index,support.to_numpy(),cmap='cividis',shading='nearest')
    bar=fig.colorbar(mesh,ax=axes[1],label='局部独立孕妇数')
    bar.solids.set_rasterized(False)
    for ax,title in zip(axes,['观测支持内的概率','局部数据覆盖']):
        ax.set(xlabel='候选孕周（周）',ylabel='首条 BMI（kg/m²）',title=title)
    export_figure(fig,out/'result_q2_probability_surface');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,3.8),layout='constrained')
    for bmi,color,style in [(28,'#0072B2','-'),(32,'#D55E00','--'),(36,'#009E73',':')]:
        g=surface[surface.baseline_bmi==bmi]
        # Break curves where local support is insufficient rather than silently
        # connecting across unsupported sections.
        y=g.probability.where(g.supported,np.nan)
        ax.plot(g.gestational_week,y,color=color,linestyle=style,label=f'BMI {bmi}')
    ax.axhline(.9,color='#6B7280',linestyle='--',linewidth=.7,label='90% 情景')
    ax.axhline(.95,color='#222222',linestyle=':',linewidth=.7,label='95% 情景')
    ax.set(xlabel='候选孕周（周）',ylabel='边际达标概率',ylim=(0,1),title='固定 BMI 的概率切片')
    ax.legend(loc='lower right')
    export_figure(fig,out/'result_q2_probability_slices');plt.close(fig)


if __name__=='__main__':
    main()
