"""Summarize frozen policies without refitting or claiming causal validation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from q23_explore import OUT
from q1_lmm_baseline import write_json


def collapse(cuts, weeks, counts):
    """Remove only adjacent identical-time boundaries; preserve assignments."""
    if len(cuts) != len(weeks)-1 or len(counts) != len(weeks):
        raise ValueError('Inconsistent policy dimensions')
    merged_cuts, merged_weeks, merged_counts = [], [], []
    for i, (week, count) in enumerate(zip(weeks, counts)):
        if merged_weeks and week == merged_weeks[-1]:
            merged_counts[-1] += count
        else:
            if i:
                merged_cuts.append(cuts[i-1])
            merged_weeks.append(week)
            merged_counts.append(count)
    return merged_cuts, merged_weeks, merged_counts


def main():
    source = OUT/'policies.csv'
    policies = pd.read_csv(source)
    rows = []
    for row in policies.to_dict('records'):
        if row['status'] == 'infeasible':
            continue
        cuts, weeks, counts = collapse(json.loads(row['cutpoints']),
                                      json.loads(row['weeks']), json.loads(row['counts']))
        rows.append(row | {'effective_groups': len(weeks),
                           'merged_cutpoints': json.dumps(cuts),
                           'merged_weeks': json.dumps(weeks),
                           'merged_counts': json.dumps(counts),
                           'boundary_groups': sum(w in (11., 25.) for w in weeks)})
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'policies_collapsed.csv', index=False)
    summary = []
    for (model, scenario, k), group in result.groupby(['model', 'scenario', 'k']):
        summary.append({'model': model, 'scenario': scenario, 'requested_groups': int(k),
                        'effective_groups_min': int(group.effective_groups.min()),
                        'effective_groups_max': int(group.effective_groups.max()),
                        'subject_weighted_model_risk': float(np.average(
                            group.heldout_expected_risk, weights=group.heldout_subjects)),
                        'outside_training_bmi_subjects': int(group.heldout_outside_bmi_range.sum())})
    pd.DataFrame(summary).to_csv(OUT/'policy_summary.csv', index=False)
    write_json(OUT/'policy_summary_provenance.json', {
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'command': r'.\.venv\Scripts\python.exe -W error src\q23_policy_summary.py',
        'scope': 'Frozen policy postprocessing only. Equal-time adjacent groups merged exactly; '
                 'risk weighted by heldout subjects, not independent counterfactual validation. '
                 'Out-of-range BMI assignments remain extrapolations, not validated recommendations.'})
    print(pd.DataFrame(summary).query("model == 'q2' and requested_groups == 4").to_string(index=False))


if __name__ == '__main__':
    main()
