# Results — 2025-CUMCM-C

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

正式图表尚未生成；P1 通过前不进入正式出图。

## 2026-09-07 — Q2 事前协变量候选（Rejected）
在相同五折中给 Q2 基线（week+bmi）追加首条记录年龄与受孕方式：孕妇等权 Brier 0.111597→0.112466（+age）→0.113049（+conception）；配对 Bootstrap 95%CI 分别 [−0.00020,0.00190] 与 [0.00015,0.00302]。无稳定改进且受孕标志显著更差，已记录为失败实验；基线维持，DP 仍未解锁。详见 [Q2 协变量结果](outputs/q2_covariates/README.md) 与 esults/q2_covariates_manifest.json。


## 2026-09-07 — Q4 标签盘点（A_any 44 阳性主体；类型重叠与折叠可行性已记录）
见 outputs/q4_label_audit/audit.json。


## 2026-09-07 — Q4 A_any Elastic-Net（嵌套阈值）
外层 OOF AUC 0.808（95%CI [0.756,0.859]）、PR-AUC 0.460、Brier 0.160；嵌套 τ 下敏感度 68.7%/特异度 77.3%/PPV 27.4%/NPV 95.2%。对照 2025 基线：AUC 相当、口径更严（阈值折内选择）。详见 outputs/q4_model/README.md 与 results/q4_model_manifest.json。


## 2026-09-07 — Q1 加性线性对照（交互移除无影响，非线性小幅增益方向一致但未过严格显著）
详见 outputs/q1_additive/README.md 与 results/q1_additive_manifest.json。


## 2026-09-08 — Q2 KM vs GAMM 公平对照（P1 支撑）
- 新增 src/q2_km_comparison.py：同队列（男胎1082/267）同 BMI 分带（<32.1 / 32.1–34.7 / ≥34.7）计算 KM 90%（首次观测到达标，删失）并与 GAMM 90% 覆盖带内均值对照：KM 16.4/16.1/20.9 周 vs GAMM 21.2/20.8/23.2 周。方向一致、量纲不同，正文按此改写（不再断言 KM 系统性偏晚于概率模型）。
- 输出 outputs/q2_decision/km_comparison.{csv,json}。
