"""Nested validation of Q2 week nonlinearity and train-only calibration."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize
from scipy.special import expit, logit

from q1_lmm_baseline import DEFAULT_SEED, subject_group_folds, write_json
from q2_probability import ROOT, prepare, normal_rule, objective, metrics

OUT = ROOT / "outputs" / "q2_refinement"
SPECS = ("linear", "week_spline_df3")


def _fit_integrated(x: np.ndarray, y: np.ndarray, subject_id: pd.Series) -> dict:
    groups = pd.factorize(subject_id, sort=True)[0]
    prevalence = float(y.mean())
    starts = [np.r_[np.log(prevalence / (1 - prevalence)), np.zeros(x.shape[1] - 1), np.log(s)]
              for s in (0.5, 1.5, 3.0)]
    history = []
    for count in (80, 160, 320, 640):
        nodes, weights = normal_rule(count)
        candidates = []
        for start in starts:
            result = minimize(
                objective, start, args=(x, y, groups, nodes, weights), jac=True,
                method="L-BFGS-B", bounds=[(-30, 30)] * x.shape[1] + [(-7, 3)],
                options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-6},
            )
            history.append({"nodes": count, "success": bool(result.success),
                            "nll": float(result.fun),
                            "gradient_max": float(np.max(np.abs(result.jac)))})
            if result.success and np.max(np.abs(result.jac)) < 0.002:
                candidates.append(result)
        if not candidates:
            raise RuntimeError(f"No converged GLMM fit: {history}")
        best = min(candidates, key=lambda result: result.fun)
        check_nodes, check_weights = normal_rule(count * 2)
        check_nll, check_gradient = objective(best.x, x, y, groups, check_nodes, check_weights)
        sigma = float(np.exp(best.x[-1]))
        eta = x @ best.x[:-1]
        p0 = expit(eta[:, None] + sigma * nodes) @ np.exp(weights)
        p1 = expit(eta[:, None] + sigma * check_nodes) @ np.exp(check_weights)
        if (abs(check_nll - best.fun) < 1e-4
                and np.max(np.abs(p0 - p1)) < 1e-6
                and np.max(np.abs(check_gradient)) < 0.003):
            return {"theta": best.x.tolist(), "sigma": sigma, "nodes": count * 2,
                    "nll": float(check_nll), "gradient_max": float(np.max(np.abs(check_gradient))),
                    "quadrature_nll_difference": float(abs(check_nll - best.fun)),
                    "quadrature_probability_difference": float(np.max(np.abs(p0 - p1))),
                    "history": history}
        starts = [best.x, best.x + np.r_[np.zeros(x.shape[1]), 0.1]]
    raise RuntimeError(f"Quadrature did not stabilize: {history}")


def make_design(train: pd.DataFrame, spec: str):
    frame = train.copy()
    bmi_center = float(frame.baseline_bmi.mean())
    bmi_scale = float(frame.baseline_bmi.std(ddof=0))
    week_center = float(frame.gestational_week.mean())
    week_scale = float(frame.gestational_week.std(ddof=0))
    frame["bmi_z"] = (frame.baseline_bmi - bmi_center) / bmi_scale
    frame["week_z"] = (frame.gestational_week - week_center) / week_scale
    formula = "1 + week_z + bmi_z" if spec == "linear" else (
        "1 + cr(gestational_week, df=3, constraints='center') + bmi_z"
    )
    design = patsy.dmatrix(formula, frame, return_type="dataframe")
    if np.linalg.matrix_rank(np.asarray(design)) != design.shape[1]:
        raise RuntimeError(f"Rank-deficient design for {spec}")
    transform = {"spec": spec, "bmi_center": bmi_center, "bmi_scale": bmi_scale,
                 "week_center": week_center, "week_scale": week_scale,
                 "design_info": design.design_info}
    return np.asarray(design), transform


def apply_design(frame: pd.DataFrame, transform: dict) -> np.ndarray:
    frame = frame.copy()
    frame["bmi_z"] = (frame.baseline_bmi - transform["bmi_center"]) / transform["bmi_scale"]
    frame["week_z"] = (frame.gestational_week - transform["week_center"]) / transform["week_scale"]
    return np.asarray(patsy.build_design_matrices([transform["design_info"]], frame)[0])


def fit_model(train: pd.DataFrame, spec: str) -> dict:
    x, transform = make_design(train, spec)
    model = _fit_integrated(x, train.attained.to_numpy(dtype=float), train.subject_id)
    model["spec"] = spec
    model["transform"] = transform
    return model


def predict_model(model: dict, frame: pd.DataFrame) -> np.ndarray:
    x = apply_design(frame, model["transform"])
    nodes, weights = normal_rule(model["nodes"])
    eta = x @ np.asarray(model["theta"][:-1])
    probability = expit(eta[:, None] + model["sigma"] * nodes) @ np.exp(weights)
    if not np.isfinite(probability).all() or not np.all((probability > 0) & (probability < 1)):
        raise RuntimeError("Invalid probability")
    return probability


def fit_calibrator(y: np.ndarray, probability: np.ndarray, subject_id: pd.Series) -> dict:
    score = logit(np.clip(probability, 1e-8, 1 - 1e-8))
    counts = subject_id.map(subject_id.value_counts()).to_numpy(dtype=float)
    weights = 1.0 / counts
    weights *= len(weights) / weights.sum()

    def loss(theta):
        eta = theta[0] + theta[1] * score
        value = np.sum(weights * (np.logaddexp(0, eta) - y * eta))
        error = expit(eta) - y
        gradient = np.array([np.sum(weights * error), np.sum(weights * error * score)])
        return float(value), gradient

    result = minimize(loss, [0.0, 1.0], jac=True, method="L-BFGS-B",
                      bounds=[(-10, 10), (0, 5)], options={"ftol": 1e-12, "gtol": 1e-8})
    bounds = [(-10.0, 10.0), (0.0, 5.0)]
    projected = np.asarray(result.jac, dtype=float).copy()
    boundary = []
    for index, (lower, upper) in enumerate(bounds):
        at_lower = result.x[index] <= lower + 1e-7
        at_upper = result.x[index] >= upper - 1e-7
        if (at_lower and projected[index] > 0) or (at_upper and projected[index] < 0):
            projected[index] = 0.0
        if at_lower or at_upper:
            boundary.append("intercept" if index == 0 else "slope")
    if not result.success or np.max(np.abs(projected)) > 1e-4:
        raise RuntimeError(f"Calibration failed: {result.message}, gradient={result.jac}")
    return {"intercept": float(result.x[0]), "slope": float(result.x[1]),
            "gradient_max": float(np.max(np.abs(result.jac))),
            "projected_gradient_max": float(np.max(np.abs(projected))),
            "boundary_parameters": boundary,
            "boundary_warning": bool(boundary)}


def apply_calibrator(calibrator: dict, probability: np.ndarray) -> np.ndarray:
    return expit(calibrator["intercept"] + calibrator["slope"]
                 * logit(np.clip(probability, 1e-8, 1 - 1e-8)))


def inner_oof(train: pd.DataFrame, spec: str, seed: int) -> np.ndarray:
    prediction = np.full(len(train), np.nan)
    for validation_subjects in subject_group_folds(train.subject_id, 3, seed):
        mask = train.subject_id.isin(validation_subjects).to_numpy()
        prediction[mask] = predict_model(fit_model(train.loc[~mask], spec), train.loc[mask])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete inner OOF")
    return prediction


def serializable_model(model: dict) -> dict:
    return {key: value for key, value in model.items() if key != "transform"} | {
        "transform": {key: value for key, value in model["transform"].items()
                      if key != "design_info"},
        "design_columns": model["transform"]["design_info"].column_names,
    }


def run(smoke: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    data, _ = prepare()
    outer_folds = subject_group_folds(data.subject_id, 5, DEFAULT_SEED)
    if smoke:
        mask = data.subject_id.isin(outer_folds[0])
        train, valid = data.loc[~mask], data.loc[mask]
        result = {}
        for spec in SPECS:
            model = fit_model(train, spec)
            result[spec] = {"fit": serializable_model(model),
                            "metrics": metrics(valid, predict_model(model, valid))}
        write_json(OUT / "smoke.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return

    output = data[["sample_id", "subject_id", "gestational_week", "baseline_bmi",
                   "first_observed_week", "attained"]].copy()
    fit_records = []
    for fold, validation_subjects in enumerate(outer_folds, start=1):
        mask = data.subject_id.isin(validation_subjects)
        train, valid = data.loc[~mask], data.loc[mask]
        output.loc[mask, "fold"] = fold
        for spec in SPECS:
            inner_prediction = inner_oof(train, spec, DEFAULT_SEED + 100 * fold)
            calibrator = fit_calibrator(train.attained.to_numpy(), inner_prediction, train.subject_id)
            model = fit_model(train, spec)
            raw = predict_model(model, valid)
            calibrated = apply_calibrator(calibrator, raw)
            output.loc[mask, f"{spec}_raw"] = raw
            output.loc[mask, f"{spec}_calibrated"] = calibrated
            fit_records.append({"fold": fold, "spec": spec, "calibrator": calibrator,
                                "fit": serializable_model(model)})
        print(f"Q2 refinement fold {fold} complete", flush=True)
    if output.isna().any().any():
        raise RuntimeError("Incomplete outer OOF output")
    output.to_csv(OUT / "oof_predictions.csv", index=False)

    names = [f"{spec}_{suffix}" for spec in SPECS for suffix in ("raw", "calibrated")]
    summary = [{"model": name, **metrics(data, output[name].to_numpy())} for name in names]
    pd.DataFrame(summary).to_csv(OUT / "comparison.csv", index=False)

    calibration_rows = []
    calibration_edges = np.array([0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    for name in names:
        bins = np.clip(np.searchsorted(calibration_edges, output[name], side="right") - 1,
                       0, len(calibration_edges) - 2)
        for bin_index in range(len(calibration_edges) - 1):
            mask = bins == bin_index
            if not mask.any():
                continue
            group = output.loc[mask]
            calibration_rows.append({"model": name,
                                     "bin_low": calibration_edges[bin_index],
                                     "bin_high": calibration_edges[bin_index + 1],
                                     "rows": int(len(group)),
                                     "subjects": int(group.subject_id.nunique()),
                                     "mean_prediction": float(group[name].mean()),
                                     "observed_rate": float(group.attained.mean())})
    pd.DataFrame(calibration_rows).to_csv(OUT / "calibration.csv", index=False)

    subject_losses = {}
    for name in names:
        squared = (data.attained.to_numpy() - output[name].to_numpy()) ** 2
        subject_losses[name] = pd.Series(squared).groupby(data.subject_id.to_numpy()).mean()
    rng = np.random.default_rng(DEFAULT_SEED)
    bootstrap = {}
    baseline = subject_losses["linear_raw"].to_numpy()
    for name in names[1:]:
        delta = subject_losses[name].to_numpy() - baseline
        draws = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(2000)]
        bootstrap[name] = {"mean_difference": float(delta.mean()),
                           "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}

    grid = pd.DataFrame([(week, bmi) for bmi in (28.0, 32.0, 36.0)
                         for week in np.arange(77, 141) / 7],
                        columns=["gestational_week", "baseline_bmi"])
    early = data[data.first_observed_week <= 12].copy()
    sensitivity = grid.copy()
    full_models = {}
    for spec in SPECS:
        full_model = fit_model(data, spec)
        early_model = fit_model(early, spec)
        sensitivity[f"{spec}_full"] = predict_model(full_model, grid)
        sensitivity[f"{spec}_early69"] = predict_model(early_model, grid)
        full_models[spec] = {"full": serializable_model(full_model),
                             "early69": serializable_model(early_model)}
    sensitivity.to_csv(OUT / "early_sensitivity.csv", index=False)
    sensitivity_summary = {}
    for spec in SPECS:
        delta = sensitivity[f"{spec}_early69"] - sensitivity[f"{spec}_full"]
        sensitivity_summary[spec] = {"mean_absolute_difference": float(np.mean(np.abs(delta))),
                                     "maximum_absolute_difference": float(np.max(np.abs(delta)))}

    report = {"seed": DEFAULT_SEED, "outer_folds": 5, "inner_calibration_folds": 3,
              "specifications": list(SPECS), "comparison": summary, "bootstrap": bootstrap,
              "bootstrap_note": "2000 subject resamples of frozen outer-OOF Brier differences; no refit",
              "fit_records": fit_records, "full_models": full_models,
              "early_subjects": int(early.subject_id.nunique()),
              "early_sensitivity": sensitivity_summary,
              "calibration": "subject-weighted intercept and nonnegative slope fitted only to inner OOF",
              "decision_gate": "requires stable Brier/log-loss and calibration improvement before DP"}
    write_json(OUT / "experiment.json", report)
    print(json.dumps({"comparison": summary, "bootstrap": bootstrap,
                      "early_sensitivity": sensitivity_summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args().smoke)
