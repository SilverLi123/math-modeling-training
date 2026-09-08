"""Single entry point: Q4 Elastic-Net main experiment and provenance."""
import sys
from pathlib import Path

from q4_model import ROOT, OUT, C_GRID, L1_RATIO, SEED, run
from q1_lmm_baseline import write_json

sys.path.insert(0, str(ROOT))
from utils.repro_manifest import build_manifest  # noqa: E402


if __name__ == '__main__':
    if '--skip-run' not in sys.argv:
        run()
    files = [ROOT / '附件.xlsx', ROOT / '题目分析报告.md', ROOT / '术语表格.md',
             ROOT / 'requirements.txt', ROOT / 'src' / 'q1_lmm_baseline.py',
             ROOT / 'src' / 'q4_label_audit.py', ROOT / 'src' / 'q4_model.py',
             ROOT / 'src' / 'run_q4_model.py',
             ROOT / 'utils' / 'repro_manifest.py']
    files += list((ROOT / 'outputs' / 'q4_label_audit').glob('*.csv'))
    files += list(OUT.glob('*.csv')) + [OUT / 'experiment.json', OUT / 'README.md']
    files += [ROOT / 'src' / 'plot_q4_model.py']
    files += list((ROOT / 'outputs' / 'figures' / 'q4_model').glob('*.png'))
    files += list((ROOT / 'outputs' / 'figures' / 'q4_model').glob('*.svg'))
    manifest = build_manifest(
        files, SEED,
        {'threshold': .04, 'l1_ratio': L1_RATIO, 'C_grid': C_GRID,
         'outer': 'StratifiedGroupKFold(5) by subject',
         'inner': '3 subject-grouped folds; C by PR-AUC; tau by F2',
         'scope': 'Q4 A_any main model, nested threshold, no per-type CV',
         'optimizer': 'custom deterministic FISTA (intercept unpenalized)'},
        r'.\.venv\Scripts\python.exe -W error src\run_q4_model.py',
        ['numpy', 'pandas', 'scipy', 'statsmodels', 'openpyxl',
         'scikit-learn', 'matplotlib', 'pyparsing'])
    write_json(ROOT / 'results' / 'q4_model_manifest.json', manifest)
