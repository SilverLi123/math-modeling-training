"""Figures for Q2 nonlinearity, calibration and early-subset sensitivity."""
from pathlib import Path
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.plot_style import apply_publication_style, export_figure

SOURCE = ROOT / "outputs" / "q2_refinement"
FIGURES = ROOT / "outputs" / "figures" / "q2_refinement"
LABELS = {"linear_raw": "线性：未校准", "linear_calibrated": "线性：内层校准",
          "week_spline_df3_raw": "孕周样条：未校准",
          "week_spline_df3_calibrated": "孕周样条：内层校准"}
COLORS = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00"]


def main():
    apply_publication_style(language="zh")
    comparison = pd.read_csv(SOURCE / "comparison.csv").set_index("model")
    calibration = pd.read_csv(SOURCE / "calibration.csv")
    sensitivity = pd.read_csv(SOURCE / "early_sensitivity.csv")
    report = json.loads((SOURCE / "experiment.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    for index, name in enumerate(LABELS):
        axes[0].scatter(index, comparison.loc[name, "subject_brier"], color=COLORS[index], s=28)
        axes[1].scatter(index, comparison.loc[name, "subject_log_loss"], color=COLORS[index], s=28)
    labels = [LABELS[name].replace("：", "\n") for name in LABELS]
    axes[0].set(xticks=range(4), xticklabels=labels, ylabel="孕妇等权 Brier",
                title="样本外概率误差")
    axes[1].set(xticks=range(4), xticklabels=labels, ylabel="孕妇等权 Log-loss",
                title="样本外对数损失")
    export_figure(fig, FIGURES / "result_q2_refinement_metrics"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 3.8), layout="constrained")
    for index, name in enumerate(LABELS):
        group = calibration[calibration.model == name]
        ax.plot(group.mean_prediction, group.observed_rate, marker="o",
                color=COLORS[index], label=LABELS[name])
    ax.plot([0, 1], [0, 1], color="#222222", linestyle="--", linewidth=0.8)
    ax.set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="分箱平均 OOF 概率",
           ylabel="分箱实际达标率", title="非线性与校准后的可靠性")
    ax.legend(loc="upper left")
    export_figure(fig, FIGURES / "process_q2_refinement_calibration"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained", sharey=True)
    for axis, spec, title in zip(axes, ["linear", "week_spline_df3"],
                                 ["线性混合模型", "孕周样条混合模型"]):
        for bmi, color, style in [(28, "#0072B2", "-"), (32, "#D55E00", "--"),
                                  (36, "#009E73", ":")]:
            group = sensitivity[sensitivity.baseline_bmi == bmi]
            axis.plot(group.gestational_week, group[f"{spec}_full"], color=color,
                      linestyle=style)
            axis.plot(group.gestational_week, group[f"{spec}_early69"], color=color,
                      linestyle=style, alpha=0.35, linewidth=3)
        axis.set(xlabel="孕周（周）", title=title)
    axes[0].set_ylabel("边际达标概率")
    handles = [
        Line2D([0], [0], color=color, linestyle=style, label=f"BMI {bmi}")
        for bmi, color, style in [(28, "#0072B2", "-"), (32, "#D55E00", "--"),
                                  (36, "#009E73", ":")]
    ] + [
        Line2D([0], [0], color="#444444", linewidth=1.2, label="全样本"),
        Line2D([0], [0], color="#777777", alpha=0.35, linewidth=3,
               label="早期69人"),
    ]
    axes[1].legend(handles=handles, loc="lower right")
    export_figure(fig, FIGURES / "result_q2_early_sensitivity"); plt.close(fig)


if __name__ == "__main__":
    main()
