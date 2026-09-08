# 2025 CUMCM C：NIPT 时点选择与胎儿异常判定

本目录保存 2025 年高教社杯全国大学生数学建模竞赛 C 题的完整研究过程、复现结果和论文交付物。当前主线模型已经完成；`25年C题-v2` 是独立的另一版提交包，不在本目录内混合维护。

## 目录导航

| 目录/文件 | 内容 |
|---|---|
| `C题.pdf` | 官方题面（只读） |
| `附件.xlsx` | 原始男胎、女胎数据（只读） |
| `docs/analysis/` | 题目分析、数据分析、问题拆解、术语表 |
| `docs/experiments/` | 建模日志、阶段结果与复盘记录 |
| `docs/workflow/` | 协作、复现和论文使用指南 |
| `data/processed/` | 由代码生成的清洗与派生数据 |
| `src/` | 模型、绘图、复现入口和测试 |
| `outputs/` | 各问题输出的 CSV、JSON、图表和指标 |
| `results/` | 复现清单、哈希和审查记录 |
| `paper/` | Markdown 论文、图表、验证记录和 LaTeX 稿件 |
| `完整论文.docx` | Word 论文终稿 |
| `论文支撑材料.zip` | 代码与结果归档 |

## 模型主线

- Q1：logit(Y) 随机截距线性混合模型，样条模型作为对照。
- Q2：随机截距二项达标概率、随机效应积分、连续 BMI 动态规划分组及时点风险。
- Q3：年龄、身高及孕产代理变量的固定规格扩展，按孕妇主体留出作配对比较。
- Q4：女胎 T13/T18/T21 及任一异常标记的主体隔离正则化 Logistic；同时报告 MCC 与 90% 灵敏度约束阈值。

所有验证以孕妇为分组单位，预处理和阈值选择只在训练主体内完成。BMI 切点、4% 阈值及 L/M/H 风险情景均是当前附件和决策设定下的条件结果，不应直接解释为临床边界。

## 复现入口

在目录根部安装 `requirements.txt` 后，可按问题运行：

```powershell
.\.venv\Scripts\python.exe -W error src\run_q1_experiment.py
.\.venv\Scripts\python.exe -W error src\run_q2_experiment.py
.\.venv\Scripts\python.exe -W error src\run_q2_refinement.py
.\.venv\Scripts\python.exe -W error src\run_q23_exploration.py
.\.venv\Scripts\python.exe -W error src\run_q4_experiment.py
```

论文重建、图表来源和最终检查见 `paper/`；2026 LaTeX 版本从 `paper/latex_2026/论文.tex` 编译。

## 数据与文件规则

原始题面和附件只读，不覆盖。代码、Markdown、LaTeX、小型 CSV/JSON、图表和最终论文可以提交；Python 缓存、虚拟环境、临时渲染目录和大型中间文件不提交。若新增实验，必须记录输入、随机种子、主体划分、模型规格、验证指标和失败结果。
