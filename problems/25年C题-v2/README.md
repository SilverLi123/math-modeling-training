# 2025 高教社杯全国大学生数学建模竞赛 C 题 —— 提交包说明

本目录为正式提交材料。论文正文与全部结论可复现、可审计；所有随机实验固定种子
`20250904`，输入数据 SHA-256 与运行环境记录见各 `outputs/*/experiment.json` 与
`results/` 清单。

## 目录结构

```
提交_25年C题/
├── 论文.pdf            # 正式论文（排版终稿）
├── 论文.docx           # Word 版（与 PDF 同源）
└── 支撑材料/           # 建模工程镜像（可运行、可复现）
    ├── 附件.xlsx            # 竞赛原始数据（只读）
    ├── requirements.txt     # Python 依赖（含版本）
    ├── src/                 # 全部源码（唯一可运行副本，勿在别处另改）
    ├── utils/               # 复现清单生成器等工具
    ├── outputs/             # 各问题主结果（json/csv），论文数字均出自此处
    │   ├── q1_lmm_gamm/     #   Q1：规格比较、双尺度 OOF、Wald、效应
    │   ├── q2_gamm_binomial/  # Q2：达标概率模型指标
    │   ├── q2_decision/     #   Q2：覆盖率、风险情景、t90 分段、误差扰动、KM 对照
    │   ├── q2_dp_tool/      #   有序 DP 工具示例
    │   ├── q3_multifactor/  # Q3：多因素 OOF 比较与 Bootstrap
    │   ├── q4_model/        # Q4：Elastic-Net 主结果与折系数
    │   ├── q4_label_audit/  # Q4：女胎标签审计
    │   ├── metrics/  tables/ #   数据审计与 Q1 早期基线
    │   └── (figures/ 由脚本在复现时重新生成)
    ├── results/             # 复现清单（各实验 manifest；复现清单.json 为早期 A 版）
    │                        #   —— A 版清单仅作溯源，其对应旧脚本未随包附上；
    │                        #      本文正文全部数字以 outputs/*/experiment.json 为准
    ├── figures/             # 论文插图副本（PNG+SVG，便于快速查看）
    ├── docs/                # 题目分析/数据审查/建模日志/结果记录
    └── 求解/mainline_code/  # 论文附录文件表所引脚本目录（与 src/ 内容一致）
```

## 快速复现

在支撑材料根目录安装依赖后，按问题顺序运行（均为唯一入口）：

```powershell
# Q1（清洗 945 条口径，logit/Y 双尺度指标，主模型 LMM-quad）
python src/q1_lmm_gamm.py

# Q2：概率模型 → 覆盖率/风险/t90 分段
python src/q2_gamm_binomial.py        # GAMM-Binomial 主模型
python src/q2_decision.py             # 决策层（覆盖率、风险、DP）
python src/q2_km_comparison.py        # KM vs GAMM 公平对照
python src/q2_error_sensitivity.py --banding   # t90 有序分段
python src/q2_error_sensitivity.py --mc        # 附加扰动敏感性（较慢，含重拟合）

# Q3（多因素，无增益结论）
python src/q3_multifactor_probability.py

# Q4（女胎判定）
python src/q4_model.py
python src/q4_coef_export.py
python src/q4_label_audit.py
```

> `--mc` 会对 Q2/Q3 做各 20 次扰动重拟合，耗时较长；论文引用的是已提交的
> `outputs/q2_decision/error_sensitivity.json`，重跑应得到一致结果（种子固定）。

## 论文 → 结果 对照速查（评委提问口径）

| 论文结论 | 依据文件 |
|---|---|
| Q1 主模型 LMM-quad；OOF Y主体 RMSE≈0.034、logit≈0.506；ICC≈0.744 | outputs/q1_lmm_gamm/spec_comparison.csv、final_wald_table.csv |
| Q2 覆盖率 85/90/95% → 19.9/22.1/23.1 周（44/53 可达） | outputs/q2_decision/coverage_thresholds.csv |
| Q2 三档分段 <32.75 / 32.75–35.25 / ≥35.25（约20/21/22周，K=3 肘点） | outputs/q2_decision/t90_banding.json |
| Q2 附加扰动（1.8/3.5pp）→ 22.1→23.1/24.4 周 | outputs/q2_decision/error_sensitivity.json |
| KM 对照 16.4/16.1/20.9 vs GAMM 21.2/20.8/23.2 | outputs/q2_decision/km_comparison.csv |
| Q3 多因素无增益（A 0.1130 vs 基线0.1116；B 持平） | outputs/q3_multifactor/experiment.json |
| Q4 AUC 0.808、PR-AUC 0.460、Brier 0.160；阈值 0.491/0.30 来源 | outputs/q4_model/experiment.json、coefficients_top6.json |

## 运行环境

Python 3.13；依赖见 `requirements.txt`（numpy/pandas/scipy/statsmodels/patsy/
matplotlib/openpyxl/scikit-learn==1.9.0/pyparsing）。全部随机过程固定种子
`20250904`；`src/` 内脚本以题目工程根（本 `支撑材料/`）为工作目录解析
`附件.xlsx` 与输出路径，不要移动单个文件运行。
