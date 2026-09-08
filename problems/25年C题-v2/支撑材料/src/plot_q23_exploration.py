from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.plot_style import apply_publication_style, export_figure

SOURCE = ROOT / 'outputs' / 'q23_exploration'
FIGURES = ROOT / 'outputs' / 'figures' / 'q23_exploration'


def main():
    apply_publication_style(language='zh')
    metrics = pd.read_csv(SOURCE / 'comparison.csv')
    policies = pd.read_csv(SOURCE / 'policies.csv')

    names = ['q2', 'q3_basic', 'q3_full']
    labels = ['Q2', 'Q3 basic', 'Q3 full']
    fig, ax = plt.subplots(figsize=(6.5, 3.5), layout='constrained')
    x = np.arange(len(metrics)); width = .36
    ax.bar(x-width/2, metrics.record_brier, width, color='#0072B2', label='Record Brier')
    ax.bar(x+width/2, metrics.subject_brier, width, color='#D55E00', label='Subject Brier')
    ax.set(xticks=x, xticklabels=labels, ylabel='OOF Brier', title='Q2 vs Q3 probability')
    ax.legend(loc='upper left')
    export_figure(fig, FIGURES / 'process_q23_probability_metrics'); plt.close(fig)

    held = policies.dropna(subset=['heldout_expected_risk'])
    q2 = held[held.model == 'q2'].groupby(['scenario', 'k']).heldout_expected_risk.mean().unstack()
    fig, ax = plt.subplots(figsize=(6.8, 3.6), layout='constrained')
    x = np.arange(len(q2.index)); width = .25
    for offset, k in zip((-.25, 0, .25), q2.columns):
        ax.bar(x+offset, q2[k], width, label=f'{k} groups')
    ax.set(xticks=x, xticklabels=q2.index, ylabel='Model expected risk', title='Policy risk sensitivity')
    ax.legend(ncol=3)
    export_figure(fig, FIGURES / 'result_q23_policy_risk'); plt.close(fig)

    from q23_explore import load
    data, _ = load()
    low = policies[(policies.model == 'q2') & (policies.scenario == 'low_delay') & (policies.k == 4)]
    fig, ax = plt.subplots(figsize=(6.8, 3.6), layout='constrained')
    bmi_min, bmi_max = data.baseline_bmi.min(), data.baseline_bmi.max()
    for row in low.itertuples():
        cuts = json.loads(row.cutpoints); weeks = json.loads(row.weeks)
        edges = [bmi_min] + cuts + [bmi_max]
        for index, week in enumerate(weeks):
            ax.step([edges[index], edges[index+1]], [week, week], where='post', linewidth=2, alpha=.8)
    ax.set(xlabel='Baseline BMI', ylabel='Selected week', title='Fold policy stability')
    export_figure(fig, FIGURES / 'result_q23_low_delay_stability'); plt.close(fig)


if __name__ == '__main__':
    main()
