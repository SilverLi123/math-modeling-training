"""Single entry point for Q4 outputs, figures and reproducibility manifest."""
import sys
from q1_lmm_baseline import DEFAULT_SEED, write_json
from q4_abnormality_baseline import ROOT, OUT, run
from plot_q4_abnormality import main as plot
sys.path.insert(0, str(ROOT))
from utils.repro_manifest import build_manifest

if __name__ == "__main__":
    run(); plot()
    files = [ROOT / "附件.xlsx", ROOT / "docs" / "analysis" / "题目分析报告.md", ROOT / "docs" / "analysis" / "术语表格.md",
             ROOT / "requirements.txt", ROOT / "src" / "q4_abnormality_baseline.py",
             ROOT / "src" / "test_q4_abnormality.py", ROOT / "src" / "plot_q4_abnormality.py",
             ROOT / "src" / "run_q4_experiment.py", ROOT / "utils" / "plot_style.py",
             ROOT / "utils" / "repro_manifest.py"]
    files += list(OUT.glob("*.csv")) + list(OUT.glob("*.json")) + [OUT / "README.md"]
    figures = ROOT / "outputs" / "figures" / "q4_abnormality"
    files += list(figures.rglob("*.png")) + list(figures.glob("*.svg"))
    manifest = build_manifest(files, DEFAULT_SEED, {"question": "q4", "target": "attachment AB screening label",
        "outer_folds": 5, "inner_folds": 3, "elasticnet_l1_ratio": .5, "solver": "saga", "tol": .01,
        "scope": "OOF discrimination of attachment labels; not clinical diagnosis"},
        r".\.venv\Scripts\python.exe -W error src\run_q4_experiment.py",
        ["numpy", "pandas", "openpyxl", "scikit-learn", "matplotlib", "pyparsing"])
    write_json(ROOT / "results" / "q4_manifest.json", manifest)
