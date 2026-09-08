"""Single entry point: Q2 prior-covariate candidate experiment and provenance."""
import sys
from pathlib import Path
from q2_covariates import ROOT, OUT, SPECS, BOOT_REPLICATES, run
from q1_lmm_baseline import write_json

sys.path.insert(0, str(ROOT))
from utils.repro_manifest import build_manifest  # noqa: E402


if __name__ == '__main__':
    run()
    files = [ROOT / '附件.xlsx', ROOT / '题目分析报告.md', ROOT / '术语表格.md',
             ROOT / 'requirements.txt', ROOT / 'src' / 'q1_lmm_baseline.py',
             ROOT / 'src' / 'q2_probability.py', ROOT / 'src' / 'q2_covariates.py',
             ROOT / 'src' / 'run_q2_covariates.py',
             ROOT / 'utils' / 'plot_style.py', ROOT / 'utils' / 'repro_manifest.py']
    files += list(OUT.glob('*.csv')) + [OUT / 'experiment.json', OUT / 'README.md']
    manifest = build_manifest(
        files, 20250904,
        {'features_by_spec': SPECS, 'threshold': .04, 'outer_folds': 5,
         'random_effect': 'normal intercept', 'prediction': 'marginal',
         'quadrature_orders': [80, 160, 320, 640],
         'bootstrap_replicates': BOOT_REPLICATES,
         'scope': 'Q2 candidate screening of prior-known covariates only; '
                  'no decision optimization'},
        r'.\.venv\Scripts\python.exe -W error src\run_q2_covariates.py',
        ['numpy', 'pandas', 'scipy', 'statsmodels', 'openpyxl', 'matplotlib',
         'pyparsing'])
    write_json(ROOT / 'results' / 'q2_covariates_manifest.json', manifest)
