import json
import numpy as np
import pandas as pd
from q23_measurement_sensitivity import OUT
from q23_policy_summary import collapse


def main():
    data = pd.read_csv(OUT/'policies.csv')
    experiment = json.loads((OUT/'experiment.json').read_text(encoding='utf-8'))
    if experiment['failures'] or experiment['repeats_finished'] != experiment['repeats_requested']:
        raise ValueError('Incomplete simulations')
    records = []
    for row in data.to_dict('records'):
        if row['status'] == 'infeasible':
            raise ValueError('Infeasible policy: report separately before summarizing')
        cuts, weeks, counts = collapse(json.loads(row['cutpoints']), json.loads(row['weeks']), json.loads(row['counts']))
        records.append(row | {'effective_groups': len(weeks), 'merged_cuts': json.dumps(cuts),
                              'merged_weeks': json.dumps(weeks), 'merged_counts': json.dumps(counts),
                              'fraction_at_25': sum(c for c, w in zip(counts, weeks) if w == 25.)/sum(counts),
                              'mean_week': float(np.average(weeks, weights=counts))})
    result = pd.DataFrame(records)
    result.to_csv(OUT/'policies_collapsed.csv', index=False)
    summary = []
    for (model, scale, scenario), part in result.groupby(['model', 'scale', 'scenario']):
        summary.append({'model': model, 'scale': scale, 'scenario': scenario, 'repeats': len(part),
                        'groups_min': int(part.effective_groups.min()), 'groups_max': int(part.effective_groups.max()),
                        'mean_week_median': part.mean_week.median(),
                        'mean_week_min': part.mean_week.min(), 'mean_week_max': part.mean_week.max(),
                        'fraction_at_25_median': part.fraction_at_25.median()})
    table = pd.DataFrame(summary)
    table.to_csv(OUT/'summary.csv', index=False)
    print(table.query("scenario == 'low_delay'").to_string(index=False))


if __name__ == '__main__':
    main()
