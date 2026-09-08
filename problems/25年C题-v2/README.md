# 2025 高教社杯全国大学生数学建模竞赛 C 题 —— 提交包（v2·融合终稿）

本目录为**正式提交材料**：论文正文与全部结论可复现、可审计；所有随机实验固定种子
`20250904`，输入数据 SHA-256 与运行环境记录见 `outputs/*/experiment.json`、
`results/` 复现清单与论文附录文件表。

## 目录结构

```
25年C题-v2/
├── 论文.pdf            # 正式论文（融合终稿扩写版，22 页，与 工程归档/paper/latex_final/论文.pdf 同源同 hash）
├── 论文.docx           # Word 版（由 论文.tex 经 pandoc 转换，正文同源；版式以 PDF 为准）
├── README.md           # 本说明
├── 提交前自查说明.md   # 提交前清单（可删除）
├── 支撑材料/           # 建模工程镜像（可运行、可复现）—— 建议上传的内容
│   ├── 附件.xlsx            # 竞赛原始数据（只读）
│   ├── requirements.txt     # Python 依赖（含版本）
│   ├── src/                 # 全部源码（唯一可运行副本，勿在别处另改）
│   ├── utils/               # 复现清单生成器等工具
│   ├── outputs/             # 各问题主结果（json/csv），论文数字均出自此处
│   │   ├── metrics/         #   Q1 基线指标、数据审计
│   │   ├── q1_comparison/   #   Q1：混合模型 vs 样条对照（主体留出）
│   │   ├── q2_probability/  #   Q2：边际达标概率模型与校准
│   │   ├── q23_measurement_sensitivity/  # Q2/Q3：测量波动尺度与 160 次扰动拟合
│   │   ├── q4_model/        #   Q4：Elastic-Net 主结果（AUC/PR/Brier/系数）
│   │   └── tables/ …        # 论文表/图来源
│   ├── results/             # 复现清单（各实验 manifest；复现清单.json）
│   ├── figures/             # 论文插图副本（PNG+SVG，与论文中图一致）
│   ├── docs/                # 题目、数据审查、建模日志与结果记录
│   └── 求解/mainline_code/  # 论文附录文件表所引脚本目录（与 src/ 内容一致）
└── 工程归档/           # 完整工程（仅供追溯/复跑，不需上传）
    ├── paper/latex_final/   # 论文 LaTeX 源（xelatex 两遍可复现 PDF）
    ├── data/  outputs/  results/  src/  utils/
    ├── 附件.xlsx  C题.pdf  等
    └── 旧代提交包_20260908/ # 早期代提交包备份（论文/支撑材料旧版）
```

> 上传提示：`论文.pdf`（如需 docx 一并上传则加 `论文.docx`）与 `支撑材料/`
> 即可构成提交；`工程归档/` 与 `提交前自查说明.md` 不属于上传内容。
> 若组委会要求“队伍编号_论文.pdf”等命名，请在上传前重命名，不要改动内部内容。

## 快速复现

在 `支撑材料/` 根目录安装依赖后，按问题顺序运行（唯一入口见论文附录文件表）：

```powershell
# Q1（logit 尺度随机截距混合模型 + 样条对照）
python src/q1_lmm_baseline.py
python src/q1_spline_comparison.py

# Q2（边际达标概率 → 条件风险/BMI 分组）
python src/q2_probability.py

# Q2/Q3（同次抽血测量波动尺度、主体重采样与噪声敏感性，含 160 次重拟合，耗时较长）
python src/q23_measurement_sensitivity.py

# Q4（女胎异常 Elastic-Net 主模型，嵌套主体分组验证）
python src/q4_model.py
python src/q4_coef_export.py
```

> 全部随机过程固定种子 `20250904`；`src/` 内脚本以题目工程根（本 `支撑材料/`）
> 为工作目录解析 `附件.xlsx` 与输出路径，不要移动单个文件运行。
> 论文引用的是已提交的 `outputs/` 结果，重跑应得到一致数字（同种子同数据）。

## 论文 → 结果 对照速查（评委提问口径）

| 论文结论 | 依据文件 |
|---|---|
| Q1：ICC≈0.7283；主体留出 RMSE≈0.514、R²≈−0.030；条件 R²≈0.789 | outputs/metrics/q1_lmm_metrics.json |
| Q1：样条对照记录误差略低、主体误差差异很小 | outputs/q1_comparison/experiment.json、comparison.csv |
| Q2：混合模型主体 Brier≈0.1116（vs 常数率 0.1131），校准偏差有限 | outputs/q2_probability/experiment.json、calibration.csv |
| Q2：M 情景三组条件时点 11周 / 11周6天 / 12周6天（200/25/42 人）即论文表 tab:q2policy；入口 src/q2_probability.py（绑定阈值与风险情景） | outputs/q2_probability/（概率/校准）+ 论文正文表；策略在主体重采样下的分布见 outputs/q23_measurement_sensitivity/policies*.csv |
| Q2/Q3：σ̂_m=0.1237（logit），2000 次主体 Bootstrap 区间 [0.0817,0.1677] | outputs/q23_measurement_sensitivity/scale.json |
| Q3：多因素扩展无稳定增益（q3_basic 差值 +0.00032，95% 区间跨零） | outputs/q23_exploration/experiment.json、README.md |
| Q4：AUC 0.808（95%CI [0.756,0.859]）、PR-AUC 0.460、Brier 0.160；嵌套阈值中位 0.491 → 敏感度 68.7% / 特异度 77.3% | outputs/q4_model/experiment.json、coefficients_top6.json |

完整逐项映射以论文正文与 `工程归档/paper/latex_final/论文.tex` 附录文件表为准。

## Word 版说明

`论文.docx` 由 `工程归档/paper/latex_final/论文.tex`（融合终稿唯一 LaTeX 源）
经 pandoc 自动转换生成：摘要、关键词、表格、12 幅插图与正文全部保留，公式以
Word 原生 OMML 呈现；**版式、分页以 `论文.pdf` 为准**。若组委会只收 PDF，
可不上传 docx。

## 运行环境

Python 3.13；依赖见 `requirements.txt`（numpy/pandas/scipy/statsmodels/patsy/
matplotlib/openpyxl/scikit-learn==1.9.0/pyparsing）。全部随机过程固定种子
`20250904`；输入 `附件.xlsx` 的 SHA-256 记录于各 `experiment.json` 与复现清单。
