# Modeling Log — 2025-CUMCM-C

## 2026-09-07 — Q2 非线性、嵌套校准与早期敏感性

- 状态：实验完成，候选改进未通过决策门；失败结果保留。
- 在固定五折孕妇外层验证中比较线性孕周与 df=3 自然样条；校准器只从外层训练集的三折孕妇 OOF 预测拟合，验证标签未参与。
- 样条未校准的孕妇等权 Brier 为 0.111262，相对线性 0.111597 的配对区间跨零；校准后的孕妇等权 Brier 反而更高。
- 首条观测≤12周的69人子集与全样本曲线平均绝对概率差约 0.046–0.051，说明样本进入时点敏感性不可忽略。
- 入口 `src/run_q2_refinement.py`；输出位于 `outputs/q2_refinement/` 和 `outputs/figures/q2_refinement/`。不运行风险函数、BMI 分组或 DP。

## Q2 — 混合 Logistic 概率验证

- 状态：可复现 baseline，未接受为最终决策概率模型。
- 只用孕周、首次 BMI 代理；随机截距正态分布，GH 积分似然与新个体边际概率，多初值并加倍阶数核对数值精度。
- 与 Q1 相同五折孕妇划分；对照为训练记录达标率与训练孕妇平均达标率。
- 孕妇等权 Brier 0.111597，对照 0.113145；配对差区间跨零。时间分层与高概率段校准均有偏差。
- 入口 `src/run_q2_experiment.py`；输出 `outputs/q2_probability/` 与 `outputs/figures/q2_probability/`。
- 保留未通过决策适用性检验的结果；后续检查非线性和训练折内校准，风险/DP 暂未运行。

## 2026-09-07 — Q1 嵌套样条对比已执行

- 外层五折与 baseline 一致，内层三折选择 df=3/4，训练折内中心化和样条拟合。
- 样条 RMSE 0.507167，baseline 0.514063；孕妇等权配对差区间跨零，保留为候选，不宣称稳定改善。
- 输出 `outputs/q1_comparison/` 与 `outputs/figures/q1_comparison/`；唯一入口 `src/run_q1_experiment.py`。
- 绘图首次因 matplotlib 与 pyparsing 新版弃用警告在严格模式失败，固定 pyparsing 3.2.3 后解决；rug 改为散点集合以通过布局设计审计，未改变数据。
- 这是 Q1 小规模实验，Q2/Q4 尚未执行，完整重拟合不确定性和最终模型推断待后续。

## 2026-09-07 — 收缩模型合同（未运行新实验）

保留既有 LMM 结果。下一轮 Q1 先加性样条随机截距；Q2/Q3 先二项混合模型并积分未知随机效应，再复用带外层策略验证的 DP。18 组重复测序降为额外扰动敏感性，Q4 先核对标签独立阳性人数。详见修订后的题目分析报告。旧复现清单绑定旧合同，是历史快照；本次不更改数值输出或伪装为重新运行。

失败实验不得删除；使用 `Rejected` 状态并记录失败证据。

## 2026-09-06 — Q1 LMM baseline

- Question: Q1
- Status: Accepted as baseline; not a final model
- Hypothesis: 孕周、BMI 及二者线性交互可以解释部分 Y 浓度变化；孕妇随机截距处理重复测量。
- Baseline: `logit(Y) ~ gestational_week_c * bmi_c + (1 | subject_id)`
- Data split: 固定种子 `20250904`，按孕妇分组的 5 折验证；未见孕妇预测时随机效应设为 0。
- Results: 完整模型收敛且无警告；ICC 约 0.728。训练内条件 R² 约 0.789，但受试者级外推 R² 约 −0.030。
- Reproduction command: `.\.venv\Scripts\python.exe src\q1_lmm_baseline.py`
- Interpretation: 个体差异远大于当前固定效应解释力；不能用训练内条件拟合评价新孕妇泛化。
- Next step: 在相同分组折上实现 GAMM，并检查平滑非线性或有依据的事前协变量能否降低外推误差。
## 2026-09-07 — Q2 事前协变量候选实验（Rejected）
- 状态：Rejected（决策门未通过，失败证据保留）。
- 在相同五折孕妇外层验证中给基线（孕周+首次BMI随机截距二项混合）追加首条记录年龄与非自然受孕标志。
- 孕妇等权 Brier：week_bmi 0.111597；+age 0.112466；+age+conception 0.113049（Log-loss 0.4372 明显恶化）。配对 Bootstrap 95%CI（相对基线）：+age [−0.00020,0.00190]；+conception [0.00015,0.00302]。
- 结论：任意追加可观测协变量未获支持；基线 week_bmi 维持；风险/DP 仍未解锁。入口 src/q2_covariates.py、src/run_q2_covariates.py；输出 outputs/q2_covariates/ 与 results/q2_covariates_manifest.json。
- 下一步候选：Q1 同折惩罚/加性平滑+有据协变量对照，或独立推进 Q4 Elastic-Net（不依赖 Q2 门）。

## 2026-09-07 — Q4 标签盘点（合同第一步）
- 女胎 605 行/147 人；A_any 阳性记录 67、独立阳性孕妇 44（负 103）。类型（主体级）：T18-only 19、T13-only 5、T21-only 6、T18+T21 1、T13+T21 3、T13+T18 8、三型 2；跨行多型主体 14。类型标注为无分隔符字符串（如 T13T18），用正则 T13|T18|T21 解析，顺序无关。
- 含义：记录级阳性数会夸大独立样本数；T13/T21 独立阳性仅约 10 个量级，逐型独立建模需低维特征并报告不确定性；主性能展示用 A_any 直接模型。
- 入口 src/q4_label_audit.py；输出 outputs/q4_label_audit/（audit.json、label_counts.csv、subject_labels.csv）。

## 2026-09-07 — Q4 主模型（A_any Elastic-Net，嵌套阈值；可复现 baseline 候选）
- 外层按主体分层 5 折、内层 3 折选 C（PR-AUC）与 τ（F2）；阈值选择完全在训练折内。弹性网用自研 FISTA 拟合（确定性）。
- 外层 OOF：AUC 0.808（主体 Bootstrap 95%CI [0.756,0.859]）、PR-AUC 0.460、Brier 0.160（[0.145,0.176]）；嵌套 τ 中位 0.491：敏感度 68.7%、特异度 77.3%、PPV 27.4%、NPV 95.2%。Z 连续评分基线 AUC 0.479。
- 与 2025 基线论文对照：AUC 相当（0.808 vs 0.805），增益在评估口径（阈值折内选择+PR-AUC/Brier/校准报告）。
- 入口 src/q4_model.py / src/run_q4_model.py；输出 outputs/q4_model/、results/q4_model_manifest.json。

## 2026-09-07 — Q1 加性线性对照实验
- 同五折/种子插入 AddLin（仅移除 week×bmi 交互）：AddLin−LMM 孕妇等权 RMSE 差 95%CI [−0.0010,0.0032]（跨零）；Spline−AddLin [−0.0150,0.0001]。
- 解读：交互移除无影响；样条小幅增益来自非线性（证据偏弱）。LMM 维持报告基线。
- 入口 src/q1_additive_linear.py、run_q1_additive.py；输出 outputs/q1_additive/、results/q1_additive_manifest.json。

## 2026-09-07 — 有序 DP 分组×时点工具（纯算法，门控内）
- 实现 src/q2_dp_tool.py（O(n^2*K*m) DP；BMI 有序组、孕周非降、w_late 情景、可选覆盖率约束；infeasible 如实返回）+ 单测 5/5 通过 + 合成示例 outputs/q2_dp_tool/example.json（w_late=0.5/1/2 计划表）。
- 门控：不发布未过门模型的推荐时点；仅在合成数据演示。
- Q4 类型级不确定性补入 outputs/q4_label_audit/README.md（Wilson 95%CI，n=147）。
- 技术遗留记录：惩罚 P-spline GAMM 公平对照（含重拟合 Bootstrap 效应推断）未实施，因需自研惩罚基/更大工作且当前样条候选增益弱且不显著，列为后续候选。


## 2026-09-08 — Q1 主线重建（P2，含早期缺陷记录）
- 缺陷：原 q1_lmm_gamm 的 summarise 路径对 Y 尺度列二次求逆（RMSE 0.44 为假值）；LMM-rs 随机斜率全数据/分折奇异（斜率方差→0，cov_re 在边界）仍被选为主模型，导致 Wald 表与效应 CI 失控（±1e7）。
- 修正：双尺度指标分开报告（logit & Y，记录/主体）；规格选择排除奇异模型；主模型＝LMM-quad（随机截距，二次+交互），与 GAMM-add 的 OOF 差异经冻结配对 Bootstrap（logit，2000次）跨零未确认，按简约原则保留 LMM-quad。
- 口径：问题一按正文清洗为 945 条（剔非整倍体126＋孕周>25 的11）；Q2/Q3 仍用全部男胎 1082（267 人），两类口径已在 假设4/总体分析/5.1 注明。
- 结果（945/267，种子20250904）：Y 尺度主体 RMSE≈0.0337（LMM-quad）、GAMM-add≈0.0332；logit 主体≈0.506/0.500；ICC≈0.744；resid sd(logit)≈0.255。Wald/效应/图已重出。
- 入口 src/q1_lmm_gamm.py；输出 outputs/q1_lmm_gamm/、figures/q1_lmm_gamm/。
