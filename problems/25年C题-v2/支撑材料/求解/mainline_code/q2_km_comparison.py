"""P1 support: fair KM-vs-GAMM 90% comparison on the SAME male cohort.

Q2 GAMM-Binomial and its coverage thresholds use the full male cohort
(1082 rows / 267 subjects via load_and_prepare). The historical KM baseline
in the 2025 A-version paper was computed on a cleaned male subset with greedy
SSE bands [20.7,32.1],[32.1,34.7],[34.7,46.9] and reported 90% first-
observed weeks 16.6/16.6/22.1.

This script recomputes the Kaplan-Meier 90% quantile of the first-observed
达标 time on the SAME full male cohort and the SAME BMI bands, and aligns it
with the GAMM-Binomial 90% coverage threshold averaged over BMI grid levels
inside each band (from outputs/q2_decision/coverage_thresholds.csv), so that
the comparison in the paper is fair (same cohort & bands, and clearly states
that the two statistics answer different questions: a per-subject event-time
quantile vs a per-draw marginal-probability threshold).

Outputs -> outputs/q2_decision/km_comparison.csv
"""
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from q1_lmm_baseline import load_and_prepare, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q2_decision'
BANDS = [(20.703, 32.1, 'BMI<32.1'),
         (32.1, 34.7, '32.1<=BMI<34.7'),
         (34.7, 46.9, 'BMI>=34.7')]


def km_first_observed_90(data, lo, hi):
    """Product-limit 90% quantile of time-to-FIRST-OBSERVED 达标 per subject;
    never-达标 subjects are right-censored at their last observed week."""
    events = []
    subj = data[data.baseline_bmi.between(lo, hi)]
    for sid, g in subj.groupby('subject_id'):
        g = g.sort_values('gestational_week')
        hit = g[g.attended == 1]
        if len(hit):
            events.append(('e', float(hit.gestational_week.iloc[0])))
        else:
            events.append(('c', float(g.gestational_week.max())))
    by = defaultdict(list)
    for typ, tt in events:
        by[tt].append(typ)
    atrisk = len(events)
    surv = 1.0
    found = None
    for tt in sorted(by):
        n_e = by[tt].count('e')
        if n_e > 0 and atrisk > 0:
            surv *= 1 - n_e / atrisk
            if 1 - surv >= 0.90 and found is None:
                found = tt
        atrisk -= len(by[tt])
    return found, len(events), sum(1 for typ, _ in events if typ == 'e')


def main():
    data, _ = load_and_prepare(ROOT / '附件.xlsx')
    first = (data.sort_values(['subject_id', 'gestational_week'])
             .drop_duplicates('subject_id').set_index('subject_id'))
    data['baseline_bmi'] = data.subject_id.map(first.bmi)
    data['attended'] = (data.y_fraction >= 0.04).astype(int)

    cov = pd.read_csv(OUT / 'coverage_thresholds.csv')
    rows, notes = [], []
    for lo, hi, nm in BANDS:
        km90, nsubj, n_ev = km_first_observed_90(data, lo, hi)
        cq = cov[(cov.target == 0.90) & cov.bmi.between(lo, hi)]
        ok = cq[cq.achieved]
        gamm_mean = float(ok.week.mean()) if len(ok) else np.nan
        rows.append({'band': nm, 'bmi_lo': lo, 'bmi_hi': hi,
                     'subjects': int(nsubj), 'observed': int(n_ev),
                     'km90_first_observed_week': float(km90)
                     if km90 is not None else np.nan,
                     'gamm90_coverage_mean_week': gamm_mean,
                     'gamm90_bmi_levels_achieved': int(len(ok))})
        notes.append({'band': nm, 'n_subjects': int(nsubj),
                      'km90_week': float(km90) if km90 else None,
                      'gamm_mean90_week': gamm_mean})
    pd.DataFrame(rows).to_csv(OUT / 'km_comparison.csv', index=False)
    payload = {
        'cohort': 'full male load_and_prepare cohort (1082 rows / 267 subjects)',
        'bands': BANDS,
        'km_definition': ('KM 90% = product-limit 90% quantile of time to '
                          'first OBSERVED Y>=4% per subject; never-observed '
                          'right-censored at last observed week'),
        'gamm_definition': ('GAMM-Binomial 90% coverage week = earliest week '
                            'with marginal p(t,bmi)>=0.9, averaged over BMI '
                            'grid levels in the band'),
        'caveat': ('The two statistics answer different questions (per-subject '
                   'event-time quantile vs per-draw marginal-probability '
                   'threshold); they are aligned on the same cohort and BMI '
                   'bands but are not interchangeable.'),
        'rows': notes,
    }
    write_json(OUT / 'km_comparison.json', payload)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
