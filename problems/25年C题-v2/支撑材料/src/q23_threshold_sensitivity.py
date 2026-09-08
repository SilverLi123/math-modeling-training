"""Full-data conditional policy sensitivity; not an OOF evaluation."""
import hashlib
import json
import pandas as pd
from q23_explore import ROOT, load, fit, strategy
from q23_policy_summary import collapse
from q1_lmm_baseline import write_json

OUT = ROOT/'outputs'/'q23_threshold_sensitivity'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = load()
    fits, policies = [], []
    for threshold in (.035, .04, .045):
        frame = data.copy()
        frame['attained'] = (frame.y_fraction >= threshold).astype(int)
        for spec in ('q2', 'q3_basic'):
            model = fit(frame, spec)
            fits.append({'threshold': threshold, 'spec': spec, 'fit': model,
                         'positive_records': int(frame.attained.sum())})
            for row in strategy(model, frame, frame, 0, spec):
                # This is a full-data conditional fit, not a heldout population.
                row.pop('heldout_expected_risk', None)
                row.pop('heldout_subjects', None)
                row.pop('heldout_outside_bmi_range', None)
                row['status'] = 'full_data_conditional' if row['status'] != 'infeasible' else 'infeasible'
                row['threshold'] = threshold
                if row['status'] != 'infeasible':
                    cuts, weeks, counts = collapse(json.loads(row['cutpoints']),
                                                  json.loads(row['weeks']), json.loads(row['counts']))
                    row.update(effective_groups=len(weeks), merged_cutpoints=json.dumps(cuts),
                               merged_weeks=json.dumps(weeks), merged_counts=json.dumps(counts))
                policies.append(row)
            # Save each completed model to preserve work if a later fit fails.
            write_json(OUT/'fits.json', {'fits': fits, 'input_sha256': audit['input_sha256']})
            pd.DataFrame(policies).to_csv(OUT/'conditional_policies.csv', index=False)
            print(f'threshold={threshold} spec={spec} complete', flush=True)
    inputs = [ROOT/'附件.xlsx', ROOT/'src/q23_threshold_sensitivity.py',
              ROOT/'src/q23_explore.py', ROOT/'src/q23_policy_summary.py',
              ROOT/'src/q2_probability.py', ROOT/'src/q2_nonlinear_calibration.py',
              ROOT/'src/q1_lmm_baseline.py']
    write_json(OUT/'provenance.json', {
        'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'command': r'.\.venv\Scripts\python.exe -W error src\q23_threshold_sensitivity.py',
        'scope': 'Full-data conditional refits at 3.5%, 4%, 4.5%. No OOF validation, '
                 'no clinical weights estimated, no additive measurement-error simulation. '
                 'Only within-model scenario sensitivity; cross-model self-risk not a performance comparison.'})


if __name__ == '__main__':
    main()
