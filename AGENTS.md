# Codex Rules for the Training Repository

## Scope

本仓库保存完整训练项目。不同题目必须放在 `problems/YYYY-COMPETITION-PROBLEM/` 的独立目录中。禁止使用 branch 代表不同题目；branch 只用于成员或任务协作。

## Starting a new problem

1. 从 `problems/TEMPLATE/` 创建新目录，并立即登记到 `problems/index.md`。
2. 在查阅历史解法前，独立完成：题目拆解 -> 数据审查 -> baseline 设计。
3. 历史经验只能作为候选，不得先搜旧题后直接套模型。
4. 把每个子问映射到明确输入、输出、约束、评价指标和交付物。

## Data review

必须检查并记录：

- 数据规模与真正的独立样本单位；
- 缺失、重复、异常和单位；
- ID、时间顺序、空间结构和重复测量；
- 标签、未来信息、主体重叠和预处理造成的数据泄漏；
- 原始数据是否保持只读，以及 processed 数据能否由代码重建。

不得默认一行就是一个独立样本。按个体、时间、地点或批次产生的数据，验证切分必须与真实推广目标一致。

## Modeling workflow

遵循：简单模型 -> baseline -> 验证 -> 有针对性的改进 -> 公平比较 -> Reviewer 审查。

- 新复杂度必须对应 baseline 的明确不足。
- 优化模型必须独立检查可行性；随机算法报告多随机种子和计算预算。
- 统计/机器学习模型必须防止泄漏，并报告不确定性和失败案例。
- 物理/数值模型必须检查量纲、边界、守恒及数值收敛。
- 不得虚构数据、结果、指标、参考文献或最优性证明。

## Required project records

持续维护：

- `problem_analysis.md`：题意、子问关系、假设、变量和交付物；
- `data_analysis.md`：数据字典、质量问题、独立单位和泄漏风险；
- `modeling_log.md`：baseline、每次实验、参数、结果和失败原因；
- `results.md`：已核验结果、图表索引、不确定性和最终结论。

失败实验不得删除。可以标为 rejected，并说明失败证据及其对下一步的影响。

## Files and reproducibility

- `problem/` 保存题面与规则；`data/raw/` 保持只读；清洗结果进入 `data/processed/`。
- 代码进入 `src/`，探索进入 `notebooks/`，结果进入 `outputs/`，论文进入 `paper/`。
- 代码不得依赖个人绝对路径；从当前文件位置解析项目根目录。
- 必要的小型数据、图表、Markdown、LaTeX 与最终论文可以提交。
- 可再生成的缓存、中间文件、超大模型和无关二进制不得提交。

## Completion and knowledge export

完成一道题后：

1. 在 README 填写最终模型、最终结果和 `Lessons Learned`；
2. 运行 Reviewer 棹查，确认结论与验证证据匹配；
3. 把真正跨题有价值的经验整理为独立候选 lesson，写明来源、适用/不适用条件、baseline、验证、失败模式和证据等级；
4. 若同机存在可访问的 `math-modeling-knowledge`，更新其 lesson/card/index；否则生成 `knowledge_export.md`；
5. 不因知识库同步而延误当前训练或比赛交付。

## Git collaboration

- 从最新 `main` 创建 feature branch，通过 PR 合并。
- PR 聚焦一个成员职责或一个清晰任务，避免跨题混杂修改。
- 合并前复现受影响的结果，检查数据和最终论文没有被误删。
- 不强推共享分支，不覆盖其他成员未提交的工作。
