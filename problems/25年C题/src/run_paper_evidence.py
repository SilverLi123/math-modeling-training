"""Recompute the complete numerical evidence used by the four-question paper."""
import subprocess
import sys
from pathlib import Path
from q1_lmm_baseline import write_json, DEFAULT_SEED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.repro_manifest import build_manifest

STEPS = [
    ['q1_lmm_baseline.py'], ['run_q1_experiment.py'], ['run_q2_experiment.py'],
    ['run_q2_refinement.py'], ['q23_explore.py'], ['q23_policy_summary.py'],
    ['q23_threshold_sensitivity.py'], ['q23_measurement_sensitivity.py', '--repeats', '20'],
    ['summarize_measurement_sensitivity.py'], ['q4_chromosomes.py'],
    ['plot_q4_chromosomes.py'], ['plot_paper_coverage.py'], ['stage_paper_figures.py']]


def main():
    for index, (name, *args) in enumerate(STEPS, 1):
        print(f'[{index}/{len(STEPS)}] {name}', flush=True)
        subprocess.run([sys.executable, '-W', 'error', str(ROOT/'src'/name), *args], cwd=ROOT, check=True)
    subprocess.run([sys.executable, '-W', 'error', '-m', 'unittest', 'discover', '-s', 'src',
                    '-p', 'test_*.py'], cwd=ROOT, check=True)
    files = [ROOT/'附件.xlsx', ROOT/'C题.pdf', ROOT/'requirements.txt']
    files += list((ROOT/'src').glob('*.py'))+list((ROOT/'utils').glob('*.py'))
    for name in ['q1_comparison', 'q2_probability', 'q2_refinement', 'q23_exploration',
                 'q23_threshold_sensitivity', 'q23_measurement_sensitivity', 'q4_chromosomes',
                 'metrics', 'tables']:
        files += [p for p in (ROOT/'outputs'/name).glob('*') if p.suffix in ('.csv', '.json') and 'smoke' not in p.name]
    files += list((ROOT/'paper/figures').glob('*.png'))+list((ROOT/'paper/figures').glob('*.svg'))
    manifest = build_manifest(files, DEFAULT_SEED,
        {'q1_q23_outer_folds': 5, 'q4_outer_folds': 3, 'measurement_bootstrap_repeats': 20,
         'scope': 'Four-question numerical evidence; document rendering is a separate build'},
        r'.\.venv\Scripts\python.exe -W error src\run_paper_evidence.py',
        ['numpy', 'scipy', 'pandas', 'statsmodels', 'scikit-learn', 'patsy', 'openpyxl', 'matplotlib', 'pyparsing'])
    write_json(ROOT/'results/复现清单.json', manifest)
    print('All numerical evidence rebuilt and tests passed', flush=True)


if __name__ == '__main__':
    main()
