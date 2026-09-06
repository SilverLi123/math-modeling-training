# Collaboration Workflow

## Branch and PR

1. 从更新后的 `main` 创建按职责命名的 feature branch。
2. 在对应问题目录内工作，并持续更新四个研究记录文件。
3. 提交前运行受影响的代码，记录命令、环境和关键输出。
4. PR 描述说明：改了什么、为什么、如何验证、是否改变论文结论。
5. Reviewer 从数据泄漏、baseline、公平比较、可重复性和论证边界五方面检查。

同一成员同时承担多个任务时，也应拆成短生命周期分支。不同赛题始终用目录隔离，不创建永久的“一题一分支”。

## Conflict reduction

- 数据成员主要维护 `data_analysis.md`、处理脚本和数据字典。
- 建模成员主要维护 `src/`、`modeling_log.md` 与指标。
- 论文成员根据已核验的 `results.md` 写作，不手工改造计算结果。
- Reviewer 尽量通过独立检查和小修 PR 提交意见，避免大范围格式化。
