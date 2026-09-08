"""Subject-bootstrap with extra logit-scale noise; not latent error correction."""
import argparse
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from q23_explore import ROOT, load, fit, strategy
from q1_lmm_baseline import DEFAULT_SEED, write_json

OUT = ROOT/'outputs/q23_measurement_sensitivity'


def error_scale(data):
    rows = []
    for keys, group in data.groupby(['subject_id', 'test_date_iso', 'draw_number', 'gestational_week']):
        if len(group) > 1:
            z = logit(group.y_fraction.to_numpy())
            rows.append({'subject_id': keys[0], 'n': len(z),
                         'sse': float(np.sum((z-z.mean())**2)), 'df': len(z)-1})
    groups = pd.DataFrame(rows)
    if groups.empty:
        raise ValueError('No technical replicates')
    # Within-draw variance estimates one measurement variance, not difference variance.
    sigma = float(np.sqrt(groups.sse.sum()/groups.df.sum()))
    return groups, sigma


def resample(data, rng):
    ids = data.subject_id.unique()
    pieces = []
    for j, sid in enumerate(rng.choice(ids, len(ids), replace=True)):
        group = data.loc[data.subject_id == sid].copy()
        group['subject_id'] = f'bootstrap_{j}'
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def run(repeats=20):
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = load()
    groups, sigma = error_scale(data)
    groups.to_csv(OUT/'repeat_groups.csv', index=False)
    by_subject = groups.groupby('subject_id')[['sse', 'df']].sum()
    rng = np.random.default_rng(DEFAULT_SEED)
    draws = rng.integers(len(by_subject), size=(2000, len(by_subject)))
    sigma_draws = np.sqrt(by_subject.sse.to_numpy()[draws].sum(1)/by_subject.df.to_numpy()[draws].sum(1))
    write_json(OUT/'scale.json', {'sigma_logit': sigma, 'ci95': np.quantile(sigma_draws, [.025, .975]).tolist(),
                                'groups': len(groups), 'subjects': len(by_subject),
                                'assumption': 'Independent equal-variance technical errors within draw; '
                                              '2000 subject bootstraps of replicate groups, exploratory only.'})
    results, diagnostics, failures = [], [], []
    for repeat in range(repeats):
        # Same bootstrap subjects and standard noise across scales/specifications.
        rng = np.random.default_rng(DEFAULT_SEED+1000+repeat)
        sample = resample(data, rng)
        noise = rng.normal(size=len(sample))
        for scale in (0., .5, 1., 1.5):
            frame = sample.copy()
            frame['attained'] = (expit(logit(frame.y_fraction.to_numpy())+scale*sigma*noise) >= .04).astype(int)
            for spec in ('q2', 'q3_basic'):
                try:
                    fitted = fit(frame, spec)
                    diagnostics.append({'repeat': repeat, 'scale': scale, 'model': spec, 'fit': fitted})
                    for row in strategy(fitted, frame, frame, 0, spec):
                        if row['k'] != 4:
                            continue
                        for key in ('heldout_expected_risk', 'heldout_subjects', 'heldout_outside_bmi_range'):
                            row.pop(key, None)
                        results.append(row | {'repeat': repeat, 'scale': scale,
                                             'status': 'bootstrap_conditional' if row['status'] != 'infeasible' else 'infeasible'})
                except (RuntimeError, ValueError) as exc:
                    failures.append({'repeat': repeat, 'scale': scale, 'model': spec, 'error': str(exc)})
        pd.DataFrame(results).to_csv(OUT/'policies.csv', index=False)
        write_json(OUT/'experiment.json', {'repeats_requested': repeats, 'repeats_finished': repeat+1,
                   'sigma_logit': sigma, 'seed': DEFAULT_SEED, 'fits': diagnostics, 'failures': failures,
                   'input_sha256': audit['input_sha256'],
                   'scope': 'Cluster bootstrap refits plus extra independent measurement noise. '
                            'Existing observed noise is not removed. Scale zero isolates bootstrap variability. '
                            'Scenario frequencies are Monte Carlo summaries, not clinical confidence intervals.'})
        print(f'bootstrap {repeat+1}/{repeats}, failed fits {len(failures)}', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} fits failed; inspect experiment.json before interpreting frequencies')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    run(args.repeats)
