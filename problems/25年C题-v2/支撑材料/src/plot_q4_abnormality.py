"""Q4 figures: attachment labels, OOF PR curve and performance comparison."""
from pathlib import Path
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.plot_style import apply_publication_style, export_figure

SOURCE = ROOT / "outputs" / "q4_abnormality"
FIG = ROOT / "outputs" / "figures" / "q4_abnormality"


def main():
    apply_publication_style(language="zh")
    audit = json.loads((SOURCE / "label_audit.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(SOURCE / "oof_predictions.csv")
    comparison = pd.read_csv(SOURCE / "comparison.csv").set_index("model")

    combo = audit["label_combinations"]
    names, counts = list(combo.keys()), list(combo.values())
    fig, ax = plt.subplots(figsize=(5.5, 3.4), layout="constrained")
    y = np.arange(len(names))
    ax.barh(y, counts, color=["#999999"] + ["#D55E00"] * (len(names) - 1))
    ax.set(yticks=y, yticklabels=["无 AB 标签" if x == "<empty>" else x for x in names],
           xlabel="检测记录数", title="女胎附件的 AB 标签组成")
    ax.invert_yaxis()
    for yy, count in zip(y, counts): ax.text(count + 4, yy, str(count), va="center", fontsize=8)
    export_figure(fig, FIG / "raw_q4_ab_label_composition"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
    precision, recall, _ = precision_recall_curve(oof.a_any, oof.elasticnet_probability)
    ax.plot(recall, precision, color="#0072B2", linewidth=1.8,
            label=f"Elastic-Net（PR-AUC={comparison.loc['elasticnet_logistic','pr_auc']:.3f}）")
    ax.axhline(oof.a_any.mean(), color="#555555", linestyle="--", linewidth=1,
               label=f"阳性率={oof.a_any.mean():.3f}")
    ax.set(xlabel="召回率", ylabel="精确率", xlim=(0, 1), ylim=(0, 1), title="孕妇分组 OOF PR")
    ax.legend(loc="upper right")
    export_figure(fig, FIG / "process_q4_oof_pr"); plt.close(fig)

    metrics = ["pr_auc", "mcc", "sensitivity", "specificity"]
    labels = ["PR-AUC", "MCC", "灵敏度", "特异度"]
    fig, ax = plt.subplots(figsize=(6.5, 3.5), layout="constrained")
    x = np.arange(len(metrics)); width = .34
    ax.bar(x - width/2, [comparison.loc['max_Z_rule', m] for m in metrics], width, color="#999999", label="最大 Z 值规则")
    ax.bar(x + width/2, [comparison.loc['elasticnet_logistic', m] for m in metrics], width, color="#0072B2", label="Elastic-Net Logistic")
    ax.set(xticks=x, xticklabels=labels, ylim=(-.1, 1), ylabel="OOF 指标值", title="总体 AB 标签的样本外判别表现")
    ax.legend(loc="upper right")
    export_figure(fig, FIG / "result_q4_model_comparison"); plt.close(fig)


if __name__ == "__main__": main()
