"""Single entry point: Q2 probability validation, plots and provenance."""
import sys
from q2_probability import ROOT, OUT, run, DEFAULT_SEED
from plot_q2_probability import main as plot
from q1_lmm_baseline import write_json
sys.path.insert(0,str(ROOT))
from utils.repro_manifest import build_manifest


if __name__=='__main__':
    run()
    plot()
    files=[ROOT/'附件.xlsx',ROOT/'题目分析报告.md',ROOT/'术语表格.md',ROOT/'requirements.txt',
           ROOT/'src'/'q1_lmm_baseline.py',ROOT/'utils'/'plot_style.py',ROOT/'utils'/'repro_manifest.py']
    files+=list((ROOT/'src').glob('*q2*.py'))
    files+=list(OUT.glob('*.csv'))+[OUT/'experiment.json',OUT/'README.md']
    figures=ROOT/'outputs'/'figures'/'q2_probability'
    files+=list(figures.rglob('*.png'))+list(figures.glob('*.svg'))
    manifest=build_manifest(files,DEFAULT_SEED,{'features':['gestational_week','baseline_bmi'],
        'threshold':.04,'outer_folds':5,'random_effect':'normal intercept','prediction':'marginal',
        'quadrature_orders':[80,160,320,640],'bootstrap_replicates':2000,
        'scope':'Q2 probability validation only, no decision optimization'},
        r'.\.venv\Scripts\python.exe -W error src\run_q2_experiment.py',
        ['numpy','pandas','scipy','statsmodels','openpyxl','matplotlib','pyparsing'])
    write_json(ROOT/'results'/'q2_probability_manifest.json',manifest)
