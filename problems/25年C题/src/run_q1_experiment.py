"""Reproduce Q1 comparison, figures and a hash-bound manifest."""
import sys
from q1_spline_comparison import ROOT, SEED, run
from plot_q1_comparison import main as plot
from q1_lmm_baseline import write_json
sys.path.insert(0,str(ROOT))
from utils.repro_manifest import build_manifest

if __name__=='__main__':
    run()
    plot()
    inputs=[ROOT/'附件.xlsx', ROOT/'docs'/'analysis'/'题目分析报告.md', ROOT/'docs'/'analysis'/'术语表格.md',ROOT/'requirements.txt']
    inputs+=list((ROOT/'src').glob('*q1*.py'))+list((ROOT/'utils').glob('*.py'))
    inputs+=list((ROOT/'outputs'/'q1_comparison').glob('*.csv'))
    inputs+=[ROOT/'outputs'/'q1_comparison'/'experiment.json']
    inputs+=list((ROOT/'outputs'/'figures'/'q1_comparison').rglob('*.png'))
    inputs+=list((ROOT/'outputs'/'figures'/'q1_comparison').glob('*.svg'))
    manifest=build_manifest(inputs,SEED,{'outer_folds':5,'inner_folds':3,'candidate_df':[3,4],
        'bootstrap_replicates':2000,'response':'logit Y','scope':'Q1 small comparison; not whole-problem P2'},
        r'.\.venv\Scripts\python.exe -W error src\run_q1_experiment.py',
        ['numpy','pandas','patsy','statsmodels','scipy','matplotlib','openpyxl','pyparsing'])
    write_json(ROOT/'results'/'q1_comparison_manifest.json',manifest)
