# Q4 主模型：A_any 直接 Elastic-Net（FISTA）——嵌套阈值主体分层 CV

女胎 605 行 / 147 人（dropna 后 604 行）；A_any 阳性记录 67、**独立阳性孕妇 44 人**（见 q4_label_audit）。独立单位是孕妇：同一孕妇全部记录同折（StratifiedGroupKFold 5 折，按主体分层）。

## 模型与验证

- 16 个特征：13/18/21/X 染色体 Z 值、X 染色体浓度、总体与 13/18/21 GC、比对/重复/过滤比例、log10 读段数（原始/唯一）、BMI、年龄（记录级可观测）。
- 弹性网逻辑回归（l1_ratio=0.5，C 网格 [0.02…1.0]），**自研确定性 FISTA**（近端梯度，截距不惩罚，权重按类平衡；规避 sklearn saga 收敛/弃用告警）。
- 内层：每个外层训练部分内按主体 3 折选 C（内层 PR-AUC 最大）与操作阈值 τ（内层 OOF 上 F2 最大）；外层测试折完全不参与选择（**嵌套阈值**）。
- 不确定性：冻结外层 OOF 的主体重采样 Bootstrap 2000 次（不含重拟合/模型选择调整）。

## 外层 OOF 汇总结果

| 指标 | 数值 |
|---|---:|
| AUC | 0.808（主体 Bootstrap 95%CI [0.756, 0.859]） |
| PR-AUC | 0.460 |
| Brier | 0.160（95%CI [0.145, 0.176]） |
| 嵌套 τ（各折自选，合并中位） | 0.491 → 敏感度 68.7%、特异度 77.3%、PPV 27.4%、NPV 95.2%、F1 0.391 |

Z 值连续评分基线（对照，非概率）：AUC 0.479、PR-AUC 0.110 —— 与既往结论一致（固定 |Z|≥3 在本数据上不可靠）。

## 与 2025 基线论文（`2025/`，A 版）对照

| 口径 | 基线论文（类加权 LR，GroupKFold5） | 本项目 Q4（Elastic-Net，嵌套阈值） |
|---|---|---|
| 外层 CV AUC | 0.805（约登阈值 0.526 为训练集性能） | **0.808（阈值在折内选择，样本外报告）** |
| CV 召回（阈值口径） | 0.687（固定 0.5） | 0.687（每折自选 τ，中位 0.491） |
| PR-AUC / Brier / 校准 | 未报告 | 已报告（0.460 / 0.160 / calibration.csv） |

主要增益不在 AUC，而在**评估口径**：阈值选择被移入内层折，所有混淆指标均为样本外；并补齐 PR-AUC、Brier 与校准表。

## 文件与复现

- `oof_predictions.csv`（逐记录 OOF 概率/τ/折）、`fold_metrics.csv`、`calibration.csv`（固定概率箱描述性校准）、`experiment.json`（C/τ 选择与 Bootstrap）。
- 图：`outputs/figures/q4_model/result_q4_roc_pr.png/svg`（ROC+PR）、`process_q4_calibration.png/svg`（校准，记录级描述）。
- 复现：`.\.venv\Scripts\python.exe -W error src\q4_model.py`；绘图 `src\plot_q4_model.py`；清单入口 `src\run_q4_model.py` → `results/q4_model_manifest.json`。

## 边界

- 记录级标签由无分隔符字符串正则解析（T13/T18/T21）；类型级信息见 q4_label_audit，**未**做 T13/T21 独立 CV 模型（阳性主体仅 5~8 人）。
- Bootstrap 为冻结 OOF 重采样，不含重拟合；校准为描述性分箱，无独立区间。
- 单数据（附件）无外部验证；临床部署前需独立队列。
