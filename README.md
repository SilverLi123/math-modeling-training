# Mathematical Modeling Training

这是团队进行历年题、模拟题和专项训练的主仓库。每道题在 `problems/` 下拥有独立目录，完整保存题面、数据、代码、实验、图表、论文与复盘；不同题目不使用 Git branch 隔离。

## Repository roles

- 本仓库：保存每道训练题的完整研究过程。
- `math-modeling-knowledge`：保存跨题可迁移的长期经验，不接收完整赛题项目。
- 正式比赛仓库：保存当届比赛工作，不应被训练实验污染。

## 推荐工具

| 用途 | 推荐工具 | 主要作用 |
|---|---|---|
| 查文献 / 文献综述 | **Elicit** | 搜论文、筛选论文、提取方法、数据和结论，适合快速做文献调研 |
| 画流程图 / 技术路线图 | **Next AI Draw.io** | 用自然语言生成可编辑的流程图、算法图、技术路线图、模型流程图 |
| 画科研框架图 / Graphical Abstract | **BioRender AI** | 生成科研论文风格的框架图、机制图、研究总览图，科研图标资源丰富 |

- 找论文、做文献综述：优先使用 **Elicit**。
- 画算法流程、建模路线、技术路线：优先使用 **Next AI Draw.io**。
- 画论文中的科研框架图、机制图、Graphical Abstract：优先使用 **BioRender AI**。

使用这些工具时注意：

- Elicit 找到的论文必须回到原论文核对；
- AI 生成的流程图必须人工检查逻辑；
- BioRender AI 生成的科研图不能表达未经数据或模型支持的结论；
- 最终进入论文的文献、图表和结论都必须可追溯。

## Start a new problem

推荐 ID 为 `YYYY-COMPETITION-PROBLEM`，例如 `2025-CUMCM-C` 或 `2023-MCM-C`。

```powershell
./scripts/new_problem.ps1 -Id 2025-CUMCM-C
```

脚本会复制 `problems/TEMPLATE/`，替换模板中的 ID，并提醒更新 [`problems/index.md`](problems/index.md)。新题开始后，先独立完成题目拆解和数据审查，再检索历史经验。

## Collaboration

```text
main
  ↑
 PR
  ↑
feature branch
```

`main` 保持相对稳定。每位成员或每项工作使用短期 feature branch，例如：

- `silver-modeling`
- `member-data`
- `member-paper`
- `reviewer`

分支表示“谁在做什么”，不表示“哪一道题”。同一道题的工作在同一问题目录内通过 PR 合并。提交时避免无关格式化，PR 说明应列出影响的题目、数据或结论，以及复现实验的方法。

## What to commit

应提交：代码、小型必要数据、关键处理数据、最终图表、指标摘要、Markdown、LaTeX 和最终论文。不要提交 Python cache、IDE 文件、LaTeX 临时文件、可重新生成的大量中间输出或超大模型。

大型二进制文件即使尚未触及托管平台限制，也会明显影响 clone、历史体积和 PR。对于大型原始数据、视频、模型权重或批量中间结果：

1. 优先记录公开下载地址、校验和与获取脚本；
2. 需要版本化时使用 Git LFS 或团队对象存储；
3. 在题目 README 中记录恢复步骤，不用空占位文件冒充数据。

详细规则见 [`AGENTS.md`](AGENTS.md)，协作检查见 [`docs/collaboration.md`](docs/collaboration.md)。

## Knowledge sync

完成一道题后，在题目 README 中写 `Lessons Learned`。真正跨题可复用的经验按长期知识库的 lesson 模板整理，并更新同机的 `math-modeling-knowledge`。若知识库不可访问，则在题目目录生成 `knowledge_export.md`，以后再同步。知识整理不得阻塞当前赛题推进。
