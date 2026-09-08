# 2026 LaTeX 论文稿

本目录以 `D:\mathmodeling\2026latex模板` 的 2026 格式类为基础。最终稿的唯一入口是 `论文.tex`，其中保留了既有模型、数据口径、验证方式和已复现的数值结果，仅按竞赛模板重写论文结构与排版。

在本目录运行以下命令可生成 PDF：

```powershell
xelatex -interaction=nonstopmode -halt-on-error 论文.tex
xelatex -interaction=nonstopmode -halt-on-error 论文.tex
```

图像来自上级目录 `paper/figures/` 的已复现 PNG 副本；模型和实验代码仍位于赛题根目录的 `src/` 与 `outputs/`。模板中其余分章节 `.tex` 文件为原始模板参考，不参与 `论文.tex` 编译。
