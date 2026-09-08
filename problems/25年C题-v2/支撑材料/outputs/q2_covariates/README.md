# Q2 事前协变量候选实验（age / conception）— Rejected

在固定五折孕妇外层验证中，给已接受基线模型（孕周 + 首次记录 BMI 的随机截距二项混合，GH 积分边际概率）追加**首条记录即已知**的静态协变量，检查其能否稳定改善新孕妇的样本外概率预测。

## 设计

- 沿用 4% 阈值、固定种子 `20250904`、按孕妇分组的五折外层；任何孕妇不跨折。
- 规格：
  - `week_bmi`：基线（复算，与 `outputs/q2_probability` 一致）；
  - `week_bmi_age`：+ 首条记录年龄（岁）；
  - `week_bmi_age_conception`：+ 非自然受孕标志（IUI/IVF=1，仅首条记录；其余为自然受孕=0）。
- 协变量一律取该孕妇排序后的首条记录（日期→孕周→样本号），不使用任何未来信息；每折训练集内中心化/标准化。
- 若某折训练集内二值协变量类别不足（nunique<2）则该规格该折标记 `infeasible_binary_fold`（本次 5 折全部可用，无此情形）。
- 逐折 GH 拟合采用与基线相同的多节点（80/160/320/640）加倍核对收敛；预测为积分随机效应后的边际概率。
- Bootstrap：对冻结的外层 OOF 损失按孕妇等权配对重采样 2000 次（不含重拟合不确定性），差值为相对 `week_bmi`。

## 样本外结果

| 规格 | 孕妇等权 Brier | 孕妇等权 Log-loss | 记录级 Brier |
|---|---:|---:|---:|
| week_bmi（基线） | 0.111597 | 0.383614 | 0.115400 |
| week_bmi_age | 0.112466 | 0.385993 | 0.116152 |
| week_bmi_age_conception | 0.113049 | 0.437150 | 0.116841 |

相对基线的孕妇等权 Brier 差（95% 配对 Bootstrap 区间）：

- `+age`：均值 +0.000869，区间 [−0.000203, 0.001900]（跨零，略差）；
- `+age+conception`：均值 +0.001452，区间 [0.000148, 0.003022]（整体显著更差；Log-loss 大幅上升，提示极稀类别的折内不稳定性）。

## 结论

**Rejected（决策门未通过）**：两个协变量扩展均未稳定改善样本外概率；非自然受孕标志（男胎中 IUI/IVF 合计仅 17 人）带来明显更差的对数损失。基线 `week_bmi` 维持为当前 Q2 候选；本次结果**不**支持“任意追加可观测协变量即可改进”，也不否定其他有据协变量或更好的类别编码（如仅 IVF 哑变量、收缩/分层），后续如再评估需重新做公平折叠比较。风险函数/BMI 分组/DP 仍未到启动条件。

## 文件

- `oof_predictions.csv`：逐记录外层 OOF（各规格概率列）。
- `fold_metrics.csv`：5 折×3 规格指标与状态（15 行 ok）。
- `comparison.csv`：全样本汇总指标。
- `calibration.csv`：固定概率箱描述性校准（与基线同箱边）。
- `experiment.json`：拟合诊断、收敛历史与 Bootstrap 摘要。
- `results/q2_covariates_manifest.json`：输入/代码/输出绑定（由 `run_q2_covariates.py` 生成）。

## 复现

```powershell
.\.venv\Scripts\python.exe -W error src\q2_covariates.py --smoke
.\.venv\Scripts\python.exe -W error src\q2_covariates.py
.\.venv\Scripts\python.exe -W error src\run_q2_covariates.py
```
