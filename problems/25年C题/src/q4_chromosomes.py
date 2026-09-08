"""Fixed ridge Logistic, separate AB labels, subject-held-out validation."""
import argparse
import hashlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from q4_abnormality_baseline import ROOT, FEATURES, load_data, threshold_from_training_scores, metrics
from q1_lmm_baseline import DEFAULT_SEED, write_json

OUT = ROOT/'outputs/q4_chromosomes'
TARGETS = ['a13', 'a18', 'a21', 'a_any']


def model():
    # Fixed shrinkage; no class weights, so fitted probabilities retain prevalence.
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=.1, solver='lbfgs', l1_ratio=0.,
                                            max_iter=5000, tol=1e-9))


def split(data, target, seed):
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                 .split(data, data[target], data.subject_id))
    for left, right in folds:
        if set(data.iloc[left].subject_id) & set(data.iloc[right].subject_id):
            raise ValueError('Subject leakage')
        if data.iloc[left][target].nunique() != 2 or data.iloc[right][target].nunique() != 2:
            raise ValueError('Each fold must contain both classes')
    return folds


def run(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = load_data()
    summaries, predictions, fits, calibrations = [], [], [], []
    for target in TARGETS[:1] if smoke else TARGETS:
        oof = data[['sample_id', 'subject_id', target]].rename(columns={target: 'y'}).copy()
        for fold, (left, right) in enumerate(split(data, target, DEFAULT_SEED), 1):
            train, valid = data.iloc[left], data.iloc[right]
            inner_p = np.full(len(train), np.nan)
            for il, ir in split(train, target, DEFAULT_SEED+fold):
                fitted = model().fit(train.iloc[il][FEATURES], train.iloc[il][target])
                inner_p[ir] = fitted.predict_proba(train.iloc[ir][FEATURES])[:, 1]
            threshold = threshold_from_training_scores(train[target].to_numpy(), inner_p)
            fitted = model().fit(train[FEATURES], train[target])
            oof.loc[valid.index, 'probability'] = fitted.predict_proba(valid[FEATURES])[:, 1]
            oof.loc[valid.index, 'baseline_probability'] = train[target].mean()
            oof.loc[valid.index, 'threshold'] = threshold
            oof.loc[valid.index, 'fold'] = fold
            final = fitted.steps[-1][1]
            fits.append({'target': target, 'fold': fold, 'threshold': threshold,
                         'train_subjects': sorted(train.subject_id.unique().tolist()),
                         'valid_subjects': sorted(valid.subject_id.unique().tolist()),
                         'imputer': fitted.steps[0][1].statistics_.tolist(),
                         'center': fitted.steps[1][1].mean_.tolist(),
                         'scale': fitted.steps[1][1].scale_.tolist(),
                         'coef': final.coef_.tolist(), 'intercept': final.intercept_.tolist(),
                         'iterations': final.n_iter_.tolist()})
        if oof.isna().any().any():
            raise ValueError('Missing OOF predictions')
        oof['target'] = target
        oof['predicted_label'] = (oof.probability >= oof.threshold).astype(int)
        result = metrics(oof.y.to_numpy(), oof.probability.to_numpy(), oof.threshold.to_numpy())
        result['average_precision'] = result.pop('pr_auc')
        result['baseline_brier'] = float(np.mean((oof.y-oof.baseline_probability)**2))
        losses = pd.DataFrame({'subject': oof.subject_id, 'loss': (oof.y-oof.probability)**2,
                               'delta': (oof.y-oof.probability)**2-(oof.y-oof.baseline_probability)**2})
        subject = losses.groupby('subject')[['loss', 'delta']].mean()
        rng = np.random.default_rng(DEFAULT_SEED)
        draws = subject.delta.to_numpy()[rng.integers(len(subject), size=(2000, len(subject)))].mean(1)
        result.update(subject_brier=float(subject.loss.mean()),
                      subject_delta_brier=float(subject.delta.mean()),
                      delta_ci_low=float(np.quantile(draws, .025)),
                      delta_ci_high=float(np.quantile(draws, .975)))
        summaries.append({'target': target, **result})
        predictions.append(oof)
        for lower, upper in zip([0, .05, .1, .2, .4], [.05, .1, .2, .4, 1.00001]):
            part = oof.loc[(oof.probability >= lower) & (oof.probability < upper)]
            if len(part):
                calibrations.append({'target': target, 'lower': lower, 'upper': min(upper, 1.),
                                     'rows': len(part), 'subjects': part.subject_id.nunique(),
                                     'predicted': part.probability.mean(), 'observed': part.y.mean()})
        print(f'{target}: AP={result["average_precision"]:.4f} Brier={result["brier"]:.4f}', flush=True)
    prefix = 'smoke_' if smoke else ''
    pd.DataFrame(summaries).to_csv(OUT/f'{prefix}comparison.csv', index=False)
    pd.concat(predictions).to_csv(OUT/f'{prefix}oof.csv', index=False)
    pd.DataFrame(calibrations).to_csv(OUT/f'{prefix}calibration.csv', index=False)
    write_json(OUT/f'{prefix}experiment.json', {'audit': audit, 'fits': fits, 'features': FEATURES,
        'C': .1, 'seed': DEFAULT_SEED, 'folds': 3, 'inner_folds': 3,
        'input_sha256': hashlib.sha256((ROOT/'附件.xlsx').read_bytes()).hexdigest(),
        'warning': 'Attachment AB label replication, not clinical diagnosis. Fixed ridge C=.1, '
                   'unweighted likelihood, no probability recalibration. Inner subject OOF chooses MCC threshold. '
                   '2000 frozen subject-loss bootstraps, no refit. Targets use separate grouped splits.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args().smoke)
