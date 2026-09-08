# Q1 加性线性对照实验（分离交互移除与非线性贡献）

目的：早前样条对照（`outputs/q1_comparison`）将“移除 week×bmi 交互”与“加入非线性平滑”混在一起；本实验在**相同五折孕妇划分与种子**下插入加性线性规格作控制：
- LMM：`1 + week_c * bmi_c`（交互，接受基线）；
- AddLin：`1 + week_c + bmi_c`（加性线性，仅移除交互）；
- Spline：`1 + cr(week_c) + cr(bmi_c)`（df=3/4 由内层折选择，复用 q1_spline_comparison 实现）。

## 结果（外层 OOF，logit 尺度）

| 规格 | 行级 RMSE | 行级 R² | 孕妇等权 RMSE |
|---|---:|---:|---:|
| LMM | 0.514063 | −0.03023 | 0.506284 |
| AddLin | 0.514338 | −0.03133 | 0.507156 |
| Spline | 0.507167 | −0.00278 | 0.500832 |

冻结 OOF 孕妇配对 Bootstrap（2000 次，95% CI，不含重拟合）：
- AddLin−LMM 孕妇等权 RMSE 差：[−0.0010, 0.0032]（跨零、均值略正）→ **移除交互几乎无影响**；
- Spline−AddLin：[−0.0150, +0.0001]（区间基本位于负侧、上端仅微超零）→ 样条相对增益主要来自**非线性**而非交互移除；置信证据偏弱（未达严格显著）。

## 结论

“交互项无用”得到支持（AddLin≈LMM）；“平滑非线性带来小幅改进”方向与先前一致但未强到跨过严格显著门槛。两者均为冻结 OOF 比较，未含模型选择与重拟合不确定性。LMM 继续作为报告基线，Spline 保持解释性候选。

## 文件与复现

- `oof_predictions.csv`、`fold_metrics.csv`、`comparison.csv`、`experiment.json`（含选择与 Bootstrap CI）。
- 复现：`.\.venv\Scripts\python.exe -W error src\q1_additive_linear.py`；清单入口 `src\run_q1_additive.py` → `results/q1_additive_manifest.json`。
