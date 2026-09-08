r"""2025 CUMCM C题：问题1数据审查与线性混合效应基线。

唯一入口从项目根目录执行：
    .\.venv\Scripts\python.exe src\q1_lmm_baseline.py

原始 Excel 只读；所有派生文件写入 data/processed 和 outputs。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


SHEET_NAME = "男胎检测数据"
EXPECTED_ROWS = 1082
EXPECTED_COLUMNS = 31
DEFAULT_SEED = 20250904
FORMULA = "y_logit ~ gestational_week_c * bmi_c"

SOURCE_COLUMNS = {
    "序号": "sample_id",
    "孕妇代码": "subject_id",
    "年龄": "age_years",
    "身高": "height_cm",
    "体重": "weight_kg",
    "末次月经": "last_menstrual_period",
    "IVF妊娠": "conception_type",
    "检测日期": "test_date",
    "检测抽血次数": "draw_number",
    "检测孕周": "gestational_age_raw",
    "孕妇BMI": "bmi",
    "原始读段数": "raw_reads",
    "在参考基因组上比对的比例": "mapped_ratio",
    "重复读段的比例": "duplicate_ratio",
    "唯一比对的读段数": "unique_reads",
    "GC含量": "gc_content",
    "Y染色体浓度": "y_fraction",
    "被过滤掉读段数的比例": "filtered_ratio",
}

NUMERIC_COLUMNS = [
    "sample_id",
    "age_years",
    "height_cm",
    "weight_kg",
    "draw_number",
    "bmi",
    "raw_reads",
    "mapped_ratio",
    "duplicate_ratio",
    "unique_reads",
    "gc_content",
    "y_fraction",
    "filtered_ratio",
]

MODEL_COLUMNS = ["subject_id", "gestational_week", "bmi", "y_fraction"]


@dataclass(frozen=True)
class RunPaths:
    project_root: Path
    input_xlsx: Path
    processed_csv: Path
    audit_json: Path
    coefficients_csv: Path
    predictions_csv: Path
    metrics_json: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gestational_week(value: Any) -> float:
    """将 11w+6、23w 等孕周字符串转换为连续周。"""
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)w(?:\+(\d+))?", text)
    if not match:
        raise ValueError(f"无法解析检测孕周: {value!r}")
    weeks = int(match.group(1))
    days = int(match.group(2) or 0)
    if not 0 <= days <= 6:
        raise ValueError(f"孕周天数必须在0到6之间: {value!r}")
    return weeks + days / 7.0


def normalize_excel_date(value: Any) -> str | None:
    """统一 Excel 日期对象、YYYYMMDD 整数和常见字符串日期。"""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, (int, np.integer)):
        text = f"{int(value):08d}"
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    if isinstance(value, float) and value.is_integer():
        text = f"{int(value):08d}"
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    parsed = pd.to_datetime(str(value).strip(), errors="raise")
    return parsed.date().isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_and_prepare(input_xlsx: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取男胎表、验证数据契约并生成问题1建模表。"""
    if not input_xlsx.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_xlsx}")

    raw = pd.read_excel(input_xlsx, sheet_name=SHEET_NAME, header=0, engine="openpyxl")
    raw.columns = [str(column).strip() for column in raw.columns]

    if raw.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(
            f"男胎工作表尺寸异常: 实际 {raw.shape}, "
            f"期望 ({EXPECTED_ROWS}, {EXPECTED_COLUMNS})"
        )

    missing_source = sorted(set(SOURCE_COLUMNS) - set(raw.columns))
    if missing_source:
        raise ValueError(f"男胎工作表缺少字段: {missing_source}")

    data = raw[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS).copy()
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["subject_id"] = data["subject_id"].astype("string").str.strip()
    data["gestational_week"] = data["gestational_age_raw"].map(parse_gestational_week)
    data["test_date_iso"] = data["test_date"].map(normalize_excel_date)

    invalid_ids = int(data["subject_id"].isna().sum() + (data["subject_id"] == "").sum())
    if invalid_ids:
        raise ValueError(f"孕妇代码存在 {invalid_ids} 个空值")
    if not data["sample_id"].is_unique:
        raise ValueError("样本序号不是唯一键")

    if not data["gestational_week"].between(10, 30).all():
        raise ValueError("检测孕周超出题面与附件的合理范围 10–30 周")
    if not data["bmi"].between(10, 60).all():
        raise ValueError("BMI 超出预设审查范围 10–60 kg/m²")
    if not data["y_fraction"].between(0, 1, inclusive="neither").all():
        raise ValueError("Y 染色体浓度必须严格位于 (0,1)")

    missing_by_column = {column: int(raw[column].isna().sum()) for column in raw.columns}
    model_missing = {column: int(data[column].isna().sum()) for column in MODEL_COLUMNS}
    if any(model_missing.values()):
        raise ValueError(f"LMM核心字段存在缺失，未进行静默删行: {model_missing}")

    occasion_columns = [
        "subject_id",
        "draw_number",
        "test_date_iso",
        "gestational_age_raw",
    ]
    occasion_sizes = data.groupby(occasion_columns, dropna=False).size()
    repeated_occasions = occasion_sizes[occasion_sizes > 1]

    subject_counts = data.groupby("subject_id").size()
    y_reached = data["y_fraction"] >= 0.04

    data["y_logit"] = np.log(data["y_fraction"] / (1.0 - data["y_fraction"]))
    data["gestational_week_c"] = data["gestational_week"] - data["gestational_week"].mean()
    data["bmi_c"] = data["bmi"] - data["bmi"].mean()
    data["week_bmi_interaction"] = data["gestational_week_c"] * data["bmi_c"]

    audit = {
        "input_file": str(input_xlsx.resolve()),
        "input_sha256": sha256_file(input_xlsx),
        "sheet": SHEET_NAME,
        "rows": int(len(data)),
        "columns_in_source": int(raw.shape[1]),
        "first_sample_id": int(data["sample_id"].iloc[0]),
        "last_sample_id": int(data["sample_id"].iloc[-1]),
        "unique_subjects": int(data["subject_id"].nunique()),
        "records_per_subject": {
            "min": int(subject_counts.min()),
            "median": float(subject_counts.median()),
            "max": int(subject_counts.max()),
        },
        "gestational_week": {
            "min": float(data["gestational_week"].min()),
            "max": float(data["gestational_week"].max()),
        },
        "bmi": {"min": float(data["bmi"].min()), "max": float(data["bmi"].max())},
        "y_fraction": {
            "min": float(data["y_fraction"].min()),
            "max": float(data["y_fraction"].max()),
            "records_at_or_above_4pct": int(y_reached.sum()),
            "records_below_4pct": int((~y_reached).sum()),
        },
        "same_occasion_repeat_groups": int(len(repeated_occasions)),
        "same_occasion_extra_records": int((repeated_occasions - 1).sum()),
        "missing_by_source_column": missing_by_column,
        "model_field_missing": model_missing,
        "rows_removed_for_q1_lmm": 0,
    }

    output_columns = [
        "sample_id",
        "subject_id",
        "age_years",
        "height_cm",
        "weight_kg",
        "conception_type",
        "test_date_iso",
        "draw_number",
        "gestational_age_raw",
        "gestational_week",
        "bmi",
        "y_fraction",
        "y_logit",
        "gestational_week_c",
        "bmi_c",
        "week_bmi_interaction",
        "raw_reads",
        "mapped_ratio",
        "duplicate_ratio",
        "unique_reads",
        "gc_content",
        "filtered_ratio",
    ]
    return data[output_columns].copy(), audit


def fit_random_intercept_lmm(data: pd.DataFrame):
    """拟合随机截距 LMM；失败时保留原始异常。"""
    model = smf.mixedlm(FORMULA, data=data, groups=data["subject_id"], re_formula="1")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = model.fit(reml=False, method="lbfgs", maxiter=1000, disp=False)
    warning_messages = [str(item.message) for item in captured]
    return result, warning_messages


def predict_fixed_effects(result, data: pd.DataFrame) -> np.ndarray:
    """对未见孕妇只用总体固定效应预测。"""
    beta = result.fe_params
    return (
        beta["Intercept"]
        + beta["gestational_week_c"] * data["gestational_week_c"].to_numpy()
        + beta["bmi_c"] * data["bmi_c"].to_numpy()
        + beta["gestational_week_c:bmi_c"]
        * data["gestational_week_c"].to_numpy()
        * data["bmi_c"].to_numpy()
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else float("nan")
    return {"rmse": math.sqrt(mse), "mae": mae, "r2": r2}


def subject_group_folds(subject_ids: pd.Series, n_splits: int, seed: int) -> list[set[str]]:
    subjects = np.array(sorted(subject_ids.astype(str).unique()))
    if n_splits < 2 or n_splits > len(subjects):
        raise ValueError("交叉验证折数必须在2与独立孕妇数之间")
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    return [set(chunk.tolist()) for chunk in np.array_split(subjects, n_splits)]


def grouped_cross_validate(
    data: pd.DataFrame, n_splits: int, seed: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    folds = subject_group_folds(data["subject_id"], n_splits, seed)
    predictions = np.full(len(data), np.nan, dtype=float)
    fold_ids = np.full(len(data), -1, dtype=int)
    fold_rows: list[dict[str, Any]] = []

    for fold_index, validation_subjects in enumerate(folds, start=1):
        is_validation = data["subject_id"].astype(str).isin(validation_subjects)
        train = data.loc[~is_validation].copy()
        validation = data.loc[is_validation].copy()
        result, warning_messages = fit_random_intercept_lmm(train)
        fold_prediction = predict_fixed_effects(result, validation)
        validation_indices = np.flatnonzero(is_validation.to_numpy())
        predictions[validation_indices] = fold_prediction
        fold_ids[validation_indices] = fold_index
        metrics = regression_metrics(validation["y_logit"].to_numpy(), fold_prediction)
        fold_rows.append(
            {
                "fold": fold_index,
                "train_subjects": int(train["subject_id"].nunique()),
                "validation_subjects": int(validation["subject_id"].nunique()),
                "validation_rows": int(len(validation)),
                "converged": bool(result.converged),
                "warnings": warning_messages,
                **metrics,
            }
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError("分组交叉验证未生成完整的有限预测")
    if (fold_ids < 1).any():
        raise RuntimeError("分组交叉验证未给每条记录分配验证折")

    overall = regression_metrics(data["y_logit"].to_numpy(), predictions)
    summary = {
        "strategy": "deterministic shuffled subject-level folds",
        "n_splits": n_splits,
        "seed": seed,
        "prediction_for_unseen_subject": "fixed effects only; random intercept set to zero",
        "overall": overall,
        "folds": fold_rows,
        "all_folds_converged": all(row["converged"] for row in fold_rows),
    }
    return summary, predictions, fold_ids


def coefficient_table(result) -> pd.DataFrame:
    names = list(result.fe_params.index)
    confidence = result.conf_int().loc[names]
    return pd.DataFrame(
        {
            "term": names,
            "estimate": result.fe_params.loc[names].to_numpy(),
            "std_error": result.bse_fe.loc[names].to_numpy(),
            "z_value": result.fe_params.loc[names].to_numpy() / result.bse_fe.loc[names].to_numpy(),
            "p_value": result.pvalues.loc[names].to_numpy(),
            "ci_95_low": confidence.iloc[:, 0].to_numpy(),
            "ci_95_high": confidence.iloc[:, 1].to_numpy(),
        }
    )


def run(paths: RunPaths, seed: int, cv_folds: int) -> dict[str, Any]:
    data, audit = load_and_prepare(paths.input_xlsx)

    for directory in {
        paths.processed_csv.parent,
        paths.audit_json.parent,
        paths.coefficients_csv.parent,
        paths.predictions_csv.parent,
        paths.metrics_json.parent,
    }:
        directory.mkdir(parents=True, exist_ok=True)

    data.to_csv(paths.processed_csv, index=False, encoding="utf-8-sig")
    write_json(paths.audit_json, audit)

    result, fit_warnings = fit_random_intercept_lmm(data)
    if not result.converged:
        raise RuntimeError("完整数据 LMM 未收敛，停止生成正式结果")

    fixed_prediction = predict_fixed_effects(result, data)
    conditional_prediction = np.asarray(result.fittedvalues, dtype=float)
    if not np.isfinite(fixed_prediction).all() or not np.isfinite(conditional_prediction).all():
        raise RuntimeError("LMM 产生 NaN 或无穷预测")

    design_condition_number = float(np.linalg.cond(result.model.exog))
    if design_condition_number > 1e7:
        raise RuntimeError(f"固定效应设计矩阵病态: condition={design_condition_number:.3e}")

    coefficients = coefficient_table(result)
    coefficients.to_csv(paths.coefficients_csv, index=False, encoding="utf-8-sig")

    cross_validation, oof_prediction, cv_fold = grouped_cross_validate(data, cv_folds, seed)

    predictions = data[["sample_id", "subject_id", "gestational_week", "bmi", "y_fraction", "y_logit"]].copy()
    predictions["cv_fold"] = cv_fold
    predictions["oof_predicted_logit_fixed"] = oof_prediction
    predictions["oof_predicted_y_fixed"] = 1.0 / (1.0 + np.exp(-oof_prediction))
    predictions["predicted_logit_fixed"] = fixed_prediction
    predictions["predicted_logit_conditional"] = conditional_prediction
    predictions["predicted_y_fixed"] = 1.0 / (1.0 + np.exp(-fixed_prediction))
    predictions["predicted_y_conditional"] = 1.0 / (1.0 + np.exp(-conditional_prediction))
    predictions.to_csv(paths.predictions_csv, index=False, encoding="utf-8-sig")

    random_intercept_variance = float(result.cov_re.iloc[0, 0])
    residual_variance = float(result.scale)
    icc = random_intercept_variance / (random_intercept_variance + residual_variance)

    metrics = {
        "question": "q1",
        "model": "random-intercept linear mixed-effects baseline",
        "formula": FORMULA,
        "response": "logit(Y chromosome fraction)",
        "units": {
            "gestational_week": "weeks",
            "bmi": "kg/m^2",
            "y_fraction": "proportion in (0,1)",
        },
        "seed": seed,
        "n_records": int(len(data)),
        "n_subjects": int(data["subject_id"].nunique()),
        "converged": bool(result.converged),
        "fit_warnings": fit_warnings,
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "random_intercept_variance": random_intercept_variance,
        "residual_variance": residual_variance,
        "intraclass_correlation": float(icc),
        "fixed_design_condition_number": design_condition_number,
        "training_fixed_effects_only": regression_metrics(
            data["y_logit"].to_numpy(), fixed_prediction
        ),
        "training_conditional": regression_metrics(
            data["y_logit"].to_numpy(), conditional_prediction
        ),
        "subject_group_cross_validation": cross_validation,
        "output_files": {
            "processed_data": str(paths.processed_csv.relative_to(paths.project_root)),
            "data_audit": str(paths.audit_json.relative_to(paths.project_root)),
            "coefficients": str(paths.coefficients_csv.relative_to(paths.project_root)),
            "predictions": str(paths.predictions_csv.relative_to(paths.project_root)),
        },
    }
    write_json(paths.metrics_json, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="问题1 LMM 基线与数据审查")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cv-folds", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    input_xlsx = (args.input or project_root / "附件.xlsx").resolve()
    paths = RunPaths(
        project_root=project_root,
        input_xlsx=input_xlsx,
        processed_csv=project_root / "data" / "processed" / "male_q1_model.csv",
        audit_json=project_root / "outputs" / "metrics" / "data_audit.json",
        coefficients_csv=project_root / "outputs" / "tables" / "q1_lmm_coefficients.csv",
        predictions_csv=project_root / "outputs" / "tables" / "q1_lmm_predictions.csv",
        metrics_json=project_root / "outputs" / "metrics" / "q1_lmm_metrics.json",
    )
    metrics = run(paths, args.seed, args.cv_folds)
    summary = {
        "status": "ok",
        "records": metrics["n_records"],
        "subjects": metrics["n_subjects"],
        "converged": metrics["converged"],
        "cv_rmse_logit": metrics["subject_group_cross_validation"]["overall"]["rmse"],
        "cv_r2_logit": metrics["subject_group_cross_validation"]["overall"]["r2"],
        "metrics_file": str(paths.metrics_json),
    }
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
