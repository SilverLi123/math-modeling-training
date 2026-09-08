"""Single entry point for fixed Q2/Q3 exploration and provenance."""
from q1_lmm_baseline import DEFAULT_SEED, write_json
from q23_explore import ROOT, OUT, run
from plot_q23_exploration import main as plot
import sys
sys.path.insert(0, str(ROOT))
from utils.repro_manifest import build_manifest

if __name__ == '__main__':
    run(); plot()
    files = [ROOT / '附件.xlsx', ROOT / '题目分析报告.md', ROOT / '术语表格.md',
             ROOT / 'requirements.txt', ROOT / 'src' / 'q1_lmm_baseline.py',
             ROOT / 'src' / 'q2_probability.py', ROOT / 'src' / 'q2_nonlinear_calibration.py',
             ROOT / 'src' / 'q23_explore.py', ROOT / 'src' / 'test_q23_explore.py',
             ROOT / 'src' / 'plot_q23_exploration.py', ROOT / 'src' / 'run_q23_exploration.py',
             ROOT / 'utils' / 'plot_style.py', ROOT / 'utils' / 'repro_manifest.py',
             OUT / 'README.md', OUT / 'experiment.json']
    files += list(OUT.glob('*.csv'))
    figures = ROOT / 'outputs' / 'figures' / 'q23_exploration'
    files += list(figures.glob('*.png')) + list(figures.glob('*.svg'))
    manifest = build_manifest(
        files, DEFAULT_SEED,
        {'probability_specs': ['q2', 'q3_basic', 'q3_full'],
         'outer_folds': 5, 'random_effect': 'normal intercept',
         'policy_groups': [1, 3, 4], 'minimum_group_subjects': 25,
         'support_subjects_within_weeks': 1, 'minimum_support_subjects': 10,
         'scenarios': {'step_only': [0, .1], 'low_delay': [.005, .1],
                       'medium_delay': [.02, .1], 'high_delay': [.05, .1]},
         'scope': 'fixed-specification exploration; no final clinical policy'},
        r'.\.venv\Scripts\python.exe -W error src\run_q23_exploration.py',
        ['numpy', 'pandas', 'scipy', 'statsmodels', 'patsy', 'openpyxl',
         'matplotlib', 'pyparsing'])
    write_json(ROOT / 'results' / 'q23_exploration_manifest.json', manifest)
