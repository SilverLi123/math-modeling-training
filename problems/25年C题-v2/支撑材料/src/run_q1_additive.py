"""Single entry point: Q1 additive-linear control and provenance."""
import sys
from pathlib import Path

from q1_additive_linear import ROOT, OUT, SEED, run
from q1_lmm_baseline import write_json

sys.path.insert(0, str(ROOT))
from utils.repro_manifest import build_manifest  # noqa: E402


if __name__ == '__main__':
    if '--skip-run' not in sys.argv:
        run()
    files = [ROOT / '附件.xlsx', ROOT / '题目分析报告.md', ROOT / '术语表格.md',
             ROOT / 'requirements.txt',
             ROOT / 'src' / 'q1_lmm_baseline.py',
             ROOT / 'src' / 'q1_spline_comparison.py',
             ROOT / 'src' / 'q1_additive_linear.py',
             ROOT / 'src' / 'run_q1_additive.py',
             ROOT / 'utils' / 'plot_style.py', ROOT / 'utils' / 'repro_manifest.py']
    files += list(OUT.glob('*.csv')) + [OUT / 'experiment.json', OUT / 'README.md']
    manifest = build_manifest(
        files, SEED,
        {'outer_folds': 5, 'inner_folds': 3, 'candidate_df': [3, 4],
         'bootstrap_replicates': 2000, 'response': 'logit Y',
         'scope': 'Q1 additive-linear control: isolate interaction removal '
                  'vs nonlinearity'},
        r'.\.venv\Scripts\python.exe -W error src\run_q1_additive.py',
        ['numpy', 'pandas', 'patsy', 'statsmodels', 'scipy', 'matplotlib',
         'openpyxl', 'pyparsing'])
    write_json(ROOT / 'results' / 'q1_additive_manifest.json', manifest)
