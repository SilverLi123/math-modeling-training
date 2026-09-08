# 四问论文主张—证据大纲（内部写作记录）

目标交付：中文完整Word论文，标题拟为《考虑重复检测与观测不确定性的NIPT时点选择和异常判定》。按2025国赛C题作为历史训练研究，不冒充临床建议或真实参赛提交。数值以统一复现结果为准。正文覆盖四问，附录给完整可运行源码和支撑文件列表。

## 摘要拟用结果

- 男胎1082记录、267孕妇；女胎605记录、147孕妇。
- Q1随机截距ICC=0.7283；孕周正相关、BMI负相关；条件拟合与新孕妇预测必须区分。
- Q2/Q3概率模型按孕妇五折；两种固定多因素规格未显示稳定Brier增益，不能推论所有多因素均无效。
- 4%阈值、中延迟代价下完整数据条件分组给明确区间和时间；其他权重、阈值及额外噪声的变动作为局限与稳健性结果。
- Q4总体标签AP=0.3823、Brier=0.08419；T21 AP=0.0302，不能宣称各染色体均有效。

## 正文结构和精确证据位置

| 章节 | 核心内容与公式 | 数值证据 | 图 |
|---|---|---|---|
| 1 问题重述 | 四问、可知信息与输出合同 | C题.pdf；题目分析报告.md | 不额外画装饰图 |
| 2 数据与假设 | 主体/抽血/测序层级，周+天/7，logit(Y)；原始低浓度保留 | outputs/metrics/data_audit.json；outputs/q4_chromosomes/experiment.json | raw_q1_repeat_counts；raw_q2_observation_support |
| 3 Q1关系模型 | logit(Yij)=β0+βt tc+βB Bc+βtB tcBc+bi+εij；随机截距；低维加性样条对照 | outputs/tables/q1_lmm_coefficients.csv；outputs/q1_comparison/comparison.csv、experiment.json | result_q1_effect_gestational_week；result_q1_effect_bmi；result_q1_comparison；process_q1_oof_diagnostics |
| 4 Q2达标与分组 | P(Aij=1|bi)=expit(Xβ+bi)，对bi分布积分；可加组风险，DP递推与支持约束 | outputs/q2_probability/experiment.json；outputs/q23_threshold_sensitivity/conditional_policies.csv | result_q2_probability_slices；process_q2_calibration；raw_q2_observation_support |
| 5 Q3多因素 | 首次年龄/身高、受孕方式与孕产代理；固定规格比较；仍按BMI分组 | outputs/q23_exploration/comparison.csv、experiment.json；outputs/q23_threshold_sensitivity/conditional_policies.csv | raw_q3_covariates；process_q3_paired_comparison；result_q3_conditional_policy |
| 6 误差与敏感性 | 同抽血组内logit方差；14主体尺度Bootstrap；267主体重采样后额外噪声重拟合 | outputs/q23_measurement_sensitivity/scale.json、summary.csv、experiment.json；阈值条件表 | 主要用表，避免重复画同一类策略曲线 |
| 7 Q4异常标签 | 固定L2 Logistic；3折主体隔离；内层MCC阈值；AP、Brier与混淆矩阵 | outputs/q4_chromosomes/comparison.csv、oof.csv、calibration.csv、experiment.json | raw_q4_positive_subjects；process_q4_calibration；result_q4_brier_intervals |
| 8 评价与结论 | 数据泄漏防范、观测支持、非因果、标签真值与稀有阳性；四问逐项答案 | 上述输出一致性 | 不引入未验证结论 |
| 附录 | 符号、源代码与支撑文件列表、AI辅助研究说明 | src、requirements.txt；适用格式及AI规则记录 | 不计入正文图数 |

## Q2/Q3 必须写清的决策口径

主分析明确选用中延迟代价作为演示性外部情景：R=1−p+0.02(t−11)+0.1I(t≥13)，不是临床估计权重。搜索11至25周，按天；同BMI不拆分，每原始组至少25孕妇，推荐点±1周至少10名训练孕妇有观测。给4组DP解再合并相邻完全同日组；同日的原切点没有实际意义。以原始全样本范围作为适用域，不将区间无限外推。

完整条件解与外层留出概率验证分开；训练模型对留出协变量自评的期望风险不称为真实干预效果。首条BMI在更早预约时未必可用，仅代理。13周阶跃及11/25周网格边界解释清楚；20轮模拟范围不称为95%置信区间。Q4每目标独立分折，不能拼成已验证联合分类器。

### 中延迟情景的明确条件答案

从conditional_policies.csv筛选threshold=.04、scenario=medium_delay、k=4。下列区间限制在样本首次BMI范围[20.703125,46.875]；内部切点左闭右开，最末组含上端点。数值为复现精度，正文可保留两位但边界计算沿用原值。

| 模型 | BMI切点 | 三组时点 | 三组人数 |
|---|---|---|---|
| Q2 | 33.35885589395795；34.411814519222446 | 11周；11周6天；12周6天 | 200；25；42 |
| Q3_basic | 33.411452741852145；35.00033464 | 11周；12周2天；12周6天 | 203；31；33 |

前者作为BMI条件方案，后者作为年龄/身高条件扩展方案，两者都不是经反事实验证的临床最优。全样本方案人数不用于声称全体未来孕妇分布相同。

## 图表与篇幅规划

拟正文选择10–12幅非重复证据图，至少覆盖四问；余图作为支撑。正文按约20页质量目标规划，无目录、摘要一页、A4、边距至少2.5cm；不把2026的30页上限误称2025规则。附录代码页数单列，不能把代码字数计入正文15000字词质量目标。最终以实际渲染页数、正文计数和来源核验决定合规性，不以填充文字凑指标。

## 当前未过的交付门禁

整题统一复现正在执行；独立P2及本大纲W1尚未通过；尚未开始长篇正文或生成完整Word。文献双引擎记录位于paper/reference_search，引用前仍须对应原始出版内容核对。不得把本大纲当作完整论文交付。
