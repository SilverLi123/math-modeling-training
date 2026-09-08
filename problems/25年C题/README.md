# 2025 CUMCM C：NIPT 的时点选择与胎儿异常判定

## Problem

- Competition: 全国大学生数学建模竞赛（CUMCM）
- Year: 2025
- Problem: C
- Problem type: 纵向统计、阈值决策、分组优化、异常分类
- Source: `C题.pdf`

## Status

已完成 [Q2/Q3 固定规格与策略敏感性探索](outputs/q23_exploration/README.md)：Q3 检测前协变量未稳定改善达标概率；BMI 分组解对延迟权重高度敏感，不能作为最终临床策略。

已完成 [Q4 女胎 AB 标签判别](outputs/q4_abnormality/README.md)。Elastic-Net 的孕妇分组 OOF PR-AUC 为 0.3763，但结果只针对附件筛查标签而非临床诊断；运行 `.\.venv\Scripts\python.exe -W error src\run_q4_experiment.py` 重现。

已完成 [Q2 非线性、嵌套校准与早期样本敏感性](outputs/q2_refinement/README.md)。孕周样条的孕妇等权 Brier 改善区间跨零，内层校准未改善孕妇等权 Brier，早期 69 人敏感性显示约 4.6%–5.1% 的平均绝对概率变化；决策门未通过，仍不进入 BMI 分组和动态规划。运行 `.\.venv\Scripts\python.exe -W error src\run_q2_refinement.py` 可复现。

已完成 [Q2 概率验证与图表](outputs/q2_probability/README.md)：运行 `.\.venv\Scripts\python.exe -W error src\run_q2_experiment.py` 重现。混合 Logistic 相比固定达标率只小幅改善，校准仍有偏差，暂不进入最佳时点优化。下方旧阶段记录按各次实验时间理解。

已新增 [Q1 样条对比结果与图表](outputs/q1_comparison/README.md)。运行 ` .\.venv\Scripts\python.exe -W error src\run_q1_experiment.py` 可重现本轮全部成果。

`Modeling — Q1 LMM baseline implemented`

## Data

原始附件为 `附件.xlsx`，包含男胎 1082 条检测记录/267 名孕妇和女胎 605 条检测记录/147 名孕妇。独立抽样单位是孕妇；同一孕妇的全部记录必须位于同一训练或验证折。原始文件只读，派生数据由代码写入 `data/processed/`。

## Current methods

- Q1 baseline：随机截距线性混合效应模型（LMM）。
- 响应：`logit(Y染色体浓度)`。
- 固定效应：中心化孕周、中心化 BMI 及交互。
- 验证：固定种子、按孕妇分组的 5 折外推验证。

GAMM、Q2–Q4、正式图表和最终论文尚未实现。

## Reproduction

```powershell
D:\anaconda\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -W error -m unittest src.test_q1_pipeline -v
.\.venv\Scripts\python.exe -W error src\q1_lmm_baseline.py
```

机器可读复现信息记录在 `results/repro_manifest.json`：其中绑定原始附件、模型文档、代码、测试和全部 P1 输出的 SHA-256，并记录运行环境、参数、测试命令与退出状态。`.venv` 被 Git 忽略。

## Current baseline result

2026-09-07 已修订模型合同。上述复现清单绑定上一版报告，是历史运行快照；当前文档哈希与旧快照不同。代码及数值输出未因本次文档修改而变化，新实验运行时需重新生成清单。

模型在完整数据上收敛，无拟合警告；孕妇内相关系数约为 0.728，表明重复测量结构很强。对未见孕妇的 5 折固定效应预测在 logit 尺度上 RMSE 为 0.514、R² 为 −0.030，说明线性孕周和 BMI 不足以解释新孕妇差异，GAMM 或增加有依据的事前协变量需要通过公平验证证明改进。

## Most valuable review points

- 不把 1082 行当作 1082 个独立样本。
- 训练/验证的预处理和分组都以孕妇代码为边界。
- 训练内条件拟合明显好于新孕妇外推，不能把随机效应吸收的个体差异误报成泛化能力。
- 当前结果只是 baseline，不是问题 1 的最终结论。

## Lessons Learned

重复测量模型必须同时报告条件拟合与面向新个体的固定效应外推；二者差距大时，应优先检查可观测个体特征是否不足，而不是只增加黑箱复杂度。

## Knowledge-base export

- Long-term knowledge repository updated: No
- 当前仅为单题 baseline，尚不足以提升为长期规律。
