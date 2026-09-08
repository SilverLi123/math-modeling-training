# Results — 2025-CUMCM-C

## Q3 固定协变量与策略敏感性探索

Q3 basic 与 Q3 full 的孕妇等权 OOF Brier 分别为 0.111920、0.113840，均不优于 Q2 的 0.111597；Bootstrap 区间跨零且点估计变差。固定 Q2 模型做四种延迟情景下的连续 BMI 分组 DP：低延迟情景出现中低 BMI 约12.857周、高 BMI 25周的分隔，但切点五折间不稳定；中等/高延迟几乎全选 11–13 周，分组收益很小。这些是模型依赖敏感性，不是临床推荐。详见 [Q2/Q3 探索](outputs/q23_exploration/README.md) 与 results/q23_exploration_manifest.json。
## Q4 女胎 AB 标签判别已验证

附件“染色体的非整倍体”标签的总体阳性为 67/605 条记录，涉及44/147名孕妇。五折孕妇 OOF 中，最大 Z 值规则 PR-AUC 为0.109894、MCC为−0.016974；Elastic-Net Logistic 的 PR-AUC 为0.376300、ROC-AUC为0.764551、Brier为0.191870。该结论仅描述对附件 AB 筛查标签的判别能力，不能视为临床诊断效能。详见 [Q4 输出](outputs/q4_abnormality/README.md) 与 `results/q4_manifest.json`。

## Q2 改进实验未通过决策门

在相同五折孕妇划分中，线性未校准、线性内层校准、孕周样条未校准、孕周样条内层校准的孕妇等权 Brier 分别为 0.111597、0.112780、0.111262、0.112308。样条未校准相对线性的差值 95% 配对 Bootstrap 区间为 [−0.001346, 0.000645]，跨零；两种校准也未稳定改善。首条观测≤12周的69人子集相对全样本曲线平均绝对概率差约 0.046–0.051。因此保留结果但不进入 BMI 分组/DP。详见 [Q2 改进结果](outputs/q2_refinement/README.md) 与 `results/q2_refinement_manifest.json`。

## Q2 概率 baseline 已验证

混合 Logistic 孕妇等权 OOF Brier 为 0.111597，训练孕妇平均达标率对照为 0.113145；配对差 95% Bootstrap 区间 [−0.006331, 0.002851] 跨零。预测 90%–95% 箱实际达标率 87.2%，说明高概率校准仍需改进，不能据此给出可靠的推荐检测周。详见 [Q2 结果与三张图](outputs/q2_probability/README.md)，复现清单为 `results/q2_probability_manifest.json`。风险函数/DP 尚未执行。

## 2026-09-07 Q1 小规模对比

已完成相同五折孕妇划分下的嵌套验证：LMM OOF RMSE 0.514063，加性自然样条混合模型 0.507167；孕妇等权 RMSE 分别为 0.506284、0.500832。配对 OOF 误差 Bootstrap 差值区间跨零，尚不足以确认稳定优势。详见 [实验结果与四张图](outputs/q1_comparison/README.md)。新结果清单为 `results/q1_comparison_manifest.json`。以下保留原 LMM 记录；“未生成图表”等旧阶段说明仅指最初 baseline。

## Verified results

当前仅记录问题 1 的 LMM baseline：

- 记录数：1082；独立孕妇：267。
- 完整数据拟合收敛，无 warning、NaN 或无穷结果。
- 孕妇随机截距方差：0.1871；残差方差：0.06979；ICC：0.7283。
- 固定效应设计矩阵条件数：13.20，未达到病态阈值 `10^7`。
- 孕周固定效应估计 0.04701，p 值约 `4.57×10^-80`。
- BMI 固定效应估计 −0.02446，p 值约 0.00229。
- 孕周×BMI 交互估计 0.001203，p 值约 0.0968；在 5% 水平下不显著。

以上系数位于 `logit(Y浓度)` 尺度，并在中心化协变量下解释；不能直接当成百分点变化。

## Baseline comparison

- 训练内固定效应 R²：−0.0254。
- 训练内包含随机截距的条件 R²：0.7890。
- 按孕妇分组的 5 折新个体外推：RMSE 0.5141、MAE 0.4232、R² −0.0302（均为 logit 尺度）。

条件拟合与新个体外推的巨大差距说明随机截距吸收了大量个体差异，当前固定效应不足以预测新孕妇。这是引入 GAMM 或额外事前协变量的依据，不是最终性能结论。

## Validation and uncertainty

5 个外层折均收敛且无拟合警告；折内 R² 范围约 −0.116 至 0.062。正式置信区间、Bootstrap 与 GAMM 对比尚未执行。

## Sensitivity and failure cases

当前未运行 3.5%/4.5% 阈值敏感性、随机斜率、GAMM 或测量误差模拟；这些属于后续阶段。

## Figure and table index

- `outputs/tables/q1_lmm_coefficients.csv`
- `outputs/tables/q1_lmm_predictions.csv`（含每条记录的受试者级验证折号、OOF 固定效应预测及训练内预测）
- `outputs/metrics/data_audit.json`
- `outputs/metrics/q1_lmm_metrics.json`
- `results/repro_manifest.json`（输入、代码、测试、参数、环境与输出哈希）

正式图表尚未生成；P1 通过前不进入正式出图。（以上为早期 Q1 baseline 历史状态；后续实验以对应 outputs 子目录为准。）

## 2026-09-08 — Q2/Q3 策略解释复核

误差分析后续已完成：18组重复测序来自14名孕妇，logit单次误差标准差0.123745，主体Bootstrap尺度区间[0.081746, 0.167701]。20轮主体Bootstrap × 4噪声尺度 × 2模型共160拟合0失败，输出640策略，见outputs/q23_measurement_sensitivity。0额外噪声的Q2低延迟情景平均推荐孕周仍在12.857–23.818间变化，说明样本组成已带来明显不确定性；该范围不是置信区间。

Q4 后续：已完成 `outputs/q4_chromosomes` 的四目标固定 Ridge Logistic 验证。T13/T18/T21/任一异常 AP 分别 0.1967/0.2924/0.0302/0.3823；记录 Brier 分别 0.03317/0.06252/0.02123/0.08419。不能混用记录级指标与主体等权区间；T21 无可靠区分证据，总体 MCC 阈值灵敏度仅 17.9%，不构成临床筛查方案。

后续补充：`outputs/q23_threshold_sensitivity/` 已完成六次完整数据拟合与 72 条条件策略。Q2 低延迟代价四组方案合并后，3.5% 阈值为 1 组；4% 的有效切点 33.8471、人数 212/55；4.5% 的切点 29.0781、人数 38/229。后两项时点均为 12 周 6 天与 25 周。该差异证明当前策略对阈值敏感，不是新的 OOF 评价；测量误差扰动仍未完成。

本轮从冻结的 `outputs/q23_exploration/policies.csv` 派生 `policies_collapsed.csv` 和 `policy_summary.csv`，未重新拟合模型。四组 Q2 策略在五折中合并相邻相同时点后，高延迟情景均为 1 组，低延迟均为 2 组，中延迟为 2–4 组，仅阶跃代价为 1–2 组。按验证孕妇数加权的模型期望风险分别为 0.226480、0.196422、0.225467、0.152928；这不是观测反事实风险，不可据此证明实际临床获益。

每种 Q2 情景共含 2 名 BMI 超出对应训练范围的验证孕妇，属于外推。首条观测 BMI 是代理：早于首条观测的推荐时点不能被描述为已完成前瞻性验证。11/25 周边界解、外部代价权重、阈值与测量误差敏感性仍需进一步检验。

验证：`python -W error -m unittest src/test_q23_explore.py -v` 的 4 项测试通过；其中 DP 与 36 个穷举组合一致，合并前后组分配等价。派生汇总在严格警告模式下运行成功。历史复现清单保持原样，新派生表由 `policy_summary_provenance.json` 绑定原始策略表和脚本哈希。
