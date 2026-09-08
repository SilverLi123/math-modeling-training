"""Single entry point: Q2 refinement, figures and provenance manifest."""
import sys

from q1_lmm_baseline import write_json
from q2_nonlinear_calibration import DEFAULT_SEED, OUT, ROOT, run
from plot_q2_refinement import main as plot

sys.path.insert(0, str(ROOT))
from utils.repro_manifest import build_manifest


if __name__ == "__main__":
    run()
    plot()
    files = [
        ROOT / "附件.xlsx", ROOT / "docs" / "analysis" / "题目分析报告.md", ROOT / "docs" / "analysis" / "术语表格.md",
        ROOT / "requirements.txt", ROOT / "src" / "q1_lmm_baseline.py",
        ROOT / "utils" / "plot_style.py", ROOT / "utils" / "repro_manifest.py",
        ROOT / "src" / "q2_probability.py", ROOT / "src" / "q2_nonlinear_calibration.py",
        ROOT / "src" / "plot_q2_refinement.py", ROOT / "src" / "test_q2_refinement.py",
        ROOT / "src" / "run_q2_refinement.py",
    ]
    files += list(OUT.glob("*.csv")) + [OUT / "experiment.json", OUT / "README.md"]
    figure_dir = ROOT / "outputs" / "figures" / "q2_refinement"
    files += list(figure_dir.rglob("*.png")) + list(figure_dir.glob("*.svg"))
    manifest = build_manifest(
        files, DEFAULT_SEED,
        {
            "features": ["gestational_week", "baseline_bmi"],
            "threshold": 0.04, "outer_folds": 5, "inner_calibration_folds": 3,
            "specifications": ["linear", "week_spline_df3"],
            "calibration": "subject-weighted intercept/slope from inner OOF only",
            "bootstrap_replicates": 2000,
            "early_subset": "all records from subjects first observed by week 12",
            "scope": "Q2 refinement and decision gate; no DP optimization",
        },
        r".\.venv\Scripts\python.exe -W error src\run_q2_refinement.py",
        ["numpy", "pandas", "scipy", "statsmodels", "patsy", "openpyxl",
         "matplotlib", "pyparsing"],
    )
    write_json(ROOT / "results" / "q2_refinement_manifest.json", manifest)
