# Data Analysis — 2025-CUMCM-C

## Q2 静态代理与验证口径

每名孕妇按检测日期、孕周、样本序号排序，首次 BMI 映射回其所有观测，模型仅用此值与孕周。首次观测不晚于 12 周者为 69 人；不能认定其余人在更早孕周时 BMI 与首次记录一致。标签为观测 Y≥0.04，1082 条中 937 条达标。逐行代理源样本号、首次孕周、验证折与预测保存在 `outputs/q2_probability/oof_predictions.csv`；未使用测序后质控量或本人随机效应估计。校准箱保留记录数和独立孕妇数，避免将分箱记录数当独立样本量。

Q2 改进实验另用这 69 名早期进入者的全部纵向记录重拟合，并与全样本模型在同一 BMI/孕周网格比较。该子集由观测进入时间定义，可能存在选择偏倚；敏感性差异不能解释为因果效应，也不能把 69 人当作全体首次检测人群的代表样本。

## Data inventory and dictionary

- 原始只读文件：`附件.xlsx`，SHA-256 `14827156218BD4F7E4F16DB4AA6D9F757C6648379E038AE6C6B58383648614AF`。
- 工作表：`男胎检测数据`、`女胎检测数据`，均为 31 列。
- Q1 当前使用男胎表的孕妇代码、检测孕周、BMI 和 Y 染色体浓度；其他人口学与测序质量字段保留供后续扩展。

## Scale and independent sampling unit

- 男胎：1082 条记录、267 名孕妇；每人 1–8 条，中位数 4 条。
- 女胎：605 条记录、147 名孕妇；每人 1–9 条，中位数 4 条。
- 独立抽样单位为孕妇，不是检测行。

## Missing values

- Q1 LMM 的孕妇代码、检测孕周、BMI、Y 浓度均无缺失，未静默删行。
- 男胎末次月经有 12 个缺失；当前直接使用附件的检测孕周，不由末次月经反推。
- 完整字段缺失计数由 `outputs/metrics/data_audit.json` 生成。

## Duplicates and IDs

- 样本序号唯一。
- 有 18 组同一孕妇、同抽血次数、同日期、同孕周的重复检测，不能按普通重复行删除。
- 重复检测可能反映同次抽血的重复测序，是后续估计测量误差的重要依据。

## Outliers and units

- 孕周范围 11.00–29.00 周。
- BMI 范围 20.70–46.88 kg/m²。
- Y 浓度范围约 0.0100–0.2342，代码中按比例而非百分数建模。
- 937/1082 条男胎记录达到或超过 4% 主阈值。

## Time and repeated-measure structure

检测孕周由 `周w+天` 转换为连续周；检测日期的 Excel 日期对象和 `YYYYMMDD` 整数统一为 ISO 日期。LMM 使用孕妇随机截距，交叉验证按孕妇分组。

## Leakage risks

- 禁止同一孕妇跨训练/验证折。
- 后续初次时点决策不得使用检测完成后才获得的 GC、读段数、比对率等质量指标。
- 后续 BMI 决策使用首条 BMI 作为预约 BMI 的静态代理，不用未来 BMI 回填。

## Raw-to-processed lineage

```text
附件.xlsx / 男胎检测数据
  → src/q1_lmm_baseline.py（字段验证、孕周与日期标准化、中心化）
  → data/processed/male_q1_model.csv
  → outputs/tables/q1_lmm_coefficients.csv
  → outputs/tables/q1_lmm_predictions.csv
  → outputs/metrics/q1_lmm_metrics.json
```

预测表逐行保留受试者级交叉验证折号和 OOF 固定效应预测，便于从汇总指标回溯到原始记录；同一孕妇的所有记录始终位于同一个验证折。
