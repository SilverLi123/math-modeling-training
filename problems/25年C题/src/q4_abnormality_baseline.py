"""Q4: grouped, leakage-safe baseline for the female-fetus AB labels.

The target is the spreadsheet column ``染色体的非整倍体``.  It is an
attachment-provided screening label, not a confirmed clinical diagnosis.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, matthews_corrcoef,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from q1_lmm_baseline import DEFAULT_SEED, parse_gestational_week, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "q4_abnormality"
SHEET = "女胎检测数据"
EXPECTED_SHAPE = (605, 31)
LABEL_COLUMN = "染色体的非整倍体"
SOURCE = {
    "序号": "sample_id", "孕妇代码": "subject_id", "检测孕周": "gestational_age_raw",
    "孕妇BMI": "bmi", "年龄": "age_years", "13号染色体的Z值": "z13",
    "18号染色体的Z值": "z18", "21号染色体的Z值": "z21",
    "X染色体的Z值": "zx", "X染色体浓度": "x_fraction", "GC含量": "gc_content",
    "原始读段数": "raw_reads", "在参考基因组上比对的比例": "mapped_ratio",
    "重复读段的比例": "duplicate_ratio", "唯一比对的读段数": "unique_reads",
    "被过滤掉读段数的比例": "filtered_ratio", LABEL_COLUMN: "ab_label",
}
FEATURES = ["z13", "z18", "z21", "zx", "x_fraction", "gc_content", "raw_reads",
            "mapped_ratio", "duplicate_ratio", "unique_reads", "filtered_ratio",
            "gestational_week", "bmi", "age_years"]


def load_data() -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(ROOT / "附件.xlsx", sheet_name=SHEET, header=0, engine="openpyxl")
    raw.columns = [str(c).strip() for c in raw.columns]
    if raw.shape != EXPECTED_SHAPE:
        raise ValueError(f"女胎表尺寸异常：{raw.shape}，预期为 {EXPECTED_SHAPE}")
    missing = set(SOURCE) - set(raw.columns)
    if missing:
        raise ValueError(f"女胎表缺少字段：{sorted(missing)}")
    data = raw[list(SOURCE)].rename(columns=SOURCE).copy()
    data["subject_id"] = data.subject_id.astype("string").str.strip()
    if data.subject_id.isna().any() or (data.subject_id == "").any() or not data.sample_id.is_unique:
        raise ValueError("女胎孕妇代码为空或样本序号不唯一")
    data["gestational_week"] = data.gestational_age_raw.map(parse_gestational_week)
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    label = data.ab_label.fillna("").astype(str).str.strip()
    for chromosome in ("13", "18", "21"):
        data[f"a{chromosome}"] = label.str.contains(f"T{chromosome}", regex=False).astype(int)
    data["a_any"] = (data[["a13", "a18", "a21"]].sum(axis=1) > 0).astype(int)
    if not np.array_equal(data.a_any.to_numpy(), (label != "").to_numpy()):
        raise ValueError("AB 标签解析与非空标签不一致")
    subject_any = data.groupby("subject_id").a_any.max()
    audit = {
        "sheet": SHEET, "rows": int(len(data)), "subjects": int(data.subject_id.nunique()),
        "label_source": LABEL_COLUMN, "nonempty_ab_rows": int(data.a_any.sum()),
        "positive_subjects_any": int(subject_any.sum()),
        "per_label": {name: {"positive_rows": int(data[name].sum()),
                              "positive_subjects": int(data.groupby("subject_id")[name].max().sum())}
                      for name in ["a13", "a18", "a21", "a_any"]},
        "label_combinations": {str(k): int(v) for k, v in label.replace("", "<empty>").value_counts().items()},
        "feature_missing": {col: int(data[col].isna().sum()) for col in FEATURES},
    }
    return data, audit


def folds(data: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(splitter.split(data, data.a_any, groups=data.subject_id))


def pipeline(c: float) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", LogisticRegression(C=c, l1_ratio=0.5,
                        class_weight="balanced", solver="saga", max_iter=10000,
                        tol=1e-2, random_state=DEFAULT_SEED))])


def select_c(train: pd.DataFrame, seed: int) -> float:
    candidates = (0.03, 0.1, 0.3, 1.0)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = []
    for c in candidates:
        values = []
        for left, right in splitter.split(train, train.a_any, groups=train.subject_id):
            model = pipeline(c).fit(train.iloc[left][FEATURES], train.iloc[left].a_any)
            probability = model.predict_proba(train.iloc[right][FEATURES])[:, 1]
            values.append(average_precision_score(train.iloc[right].a_any, probability))
        scores.append(float(np.mean(values)))
    return float(candidates[int(np.argmax(scores))])


def threshold_from_training_scores(y: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 19), score])
    mcc = [matthews_corrcoef(y, score >= value) for value in candidates]
    return float(candidates[int(np.argmax(mcc))])


def metrics(y: np.ndarray, probability: np.ndarray, threshold: np.ndarray, include_brier: bool = True) -> dict:
    prediction = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    result = {"pr_auc": float(average_precision_score(y, probability)),
            "roc_auc": float(roc_auc_score(y, probability)),
            "mcc": float(matthews_corrcoef(y, prediction)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else math.nan,
            "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
            "threshold_median": float(np.median(threshold)), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    if include_brier:
        result["brier"] = float(brier_score_loss(y, probability))
    return result


def run(smoke: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = load_data()
    write_json(OUT / "label_audit.json", audit)
    split = folds(data, DEFAULT_SEED)
    if smoke:
        left, right = split[0]
        model = pipeline(0.3).fit(data.iloc[left][FEATURES], data.iloc[left].a_any)
        probability = model.predict_proba(data.iloc[right][FEATURES])[:, 1]
        result = {"status": "ok", "rows": len(data), "subjects": int(data.subject_id.nunique()),
                  "positive_rows": int(data.a_any.sum()), "fold0_probability_range": [float(probability.min()), float(probability.max())]}
        write_json(OUT / "smoke.json", result)
        print(json.dumps(result, ensure_ascii=False)); return result

    output = data[["sample_id", "subject_id", "a13", "a18", "a21", "a_any"]].copy()
    output[["fold", "z_rule_probability", "elasticnet_probability", "z_rule_threshold", "elasticnet_threshold"]] = np.nan
    for fold, (left, right) in enumerate(split, 1):
        train, valid = data.iloc[left], data.iloc[right]
        # Z-rule score is the largest target-chromosome Z value; its threshold is selected only in training.
        z_train = train[["z13", "z18", "z21"]].max(axis=1).to_numpy()
        z_threshold = threshold_from_training_scores(train.a_any.to_numpy(), z_train)
        c = select_c(train, DEFAULT_SEED + fold)
        model = pipeline(c).fit(train[FEATURES], train.a_any)
        elastic_score = model.predict_proba(valid[FEATURES])[:, 1]
        # Nested OOF predictions are used only to choose the outer-fold classification threshold.
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=DEFAULT_SEED + 100 * fold)
        inner_score = np.full(len(train), np.nan)
        for inner_left, inner_right in inner.split(train, train.a_any, groups=train.subject_id):
            inner_model = pipeline(c).fit(train.iloc[inner_left][FEATURES], train.iloc[inner_left].a_any)
            inner_score[inner_right] = inner_model.predict_proba(train.iloc[inner_right][FEATURES])[:, 1]
        e_threshold = threshold_from_training_scores(train.a_any.to_numpy(), inner_score)
        output.loc[valid.index, "fold"] = fold
        output.loc[valid.index, "z_rule_probability"] = z_train.mean() * 0 + valid[["z13", "z18", "z21"]].max(axis=1).to_numpy()
        output.loc[valid.index, "elasticnet_probability"] = elastic_score
        output.loc[valid.index, "z_rule_threshold"] = z_threshold
        output.loc[valid.index, "elasticnet_threshold"] = e_threshold
    if output.isna().any().any() or output.fold.astype(int).nunique() != 5:
        raise RuntimeError("Q4 OOF 预测不完整")
    # Z scores are decision scores rather than probabilities: Brier is intentionally not reported for this rule.
    summary = [{"model": "max_Z_rule", **metrics(data.a_any.to_numpy(), output.z_rule_probability.to_numpy(), output.z_rule_threshold.to_numpy(), include_brier=False)},
               {"model": "elasticnet_logistic", **metrics(data.a_any.to_numpy(), output.elasticnet_probability.to_numpy(), output.elasticnet_threshold.to_numpy())}]
    output.to_csv(OUT / "oof_predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "comparison.csv", index=False)
    report = {"seed": DEFAULT_SEED, "outer_folds": 5, "inner_folds": 3, "target": "attachment AB screening label", "audit": audit, "comparison": summary,
              "warning": "AB is not a clinical gold-standard label; Z-rule score is uncalibrated and has no Brier score."}
    write_json(OUT / "experiment.json", report)
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
