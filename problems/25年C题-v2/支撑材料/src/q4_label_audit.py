"""Q4 label audit: positive records/subjects per aneuploidy type, overlaps,
and fold-feasibility report for the female-fetus sheet.

Contract step (see 题目分析报告.md Q4): before modelling, count per-label
positive records, independent positive SUBJECTS, label overlap and per-fold
class coverage; do not silently reduce chromosome-type information.
Reads the raw attachment read-only; writes only under outputs/.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'q4_label_audit'
INPUT = ROOT / '附件.xlsx'
LABEL_TYPES = ['T13', 'T18', 'T21']


def load_female():
    df = pd.read_excel(INPUT, sheet_name='女胎检测数据')
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if str(c).startswith('Unnamed')])
    df = df.rename(columns={'孕妇代码': 'subject_id', '孕妇年龄': 'age_years',
                            '孕妇身高': 'height_cm', '孕妇体重': 'weight_kg',
                            '孕妇BMI': 'bmi'})
    raw = df['染色体的非整倍体']
    parsed = raw.map(lambda s: tuple(sorted(re.findall(r'T13|T18|T21', str(s))))
                     if pd.notna(s) and str(s).strip() else ())
    df['aneuploid_types'] = parsed
    df['any_positive'] = parsed.map(bool).astype(int)
    df['row_id'] = np.arange(len(df))
    return df


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_female()
    subject_set = df.subject_id.nunique()
    row_counts = {}
    subject_counts = {}
    for t in LABEL_TYPES:
        hit = df.aneuploid_types.map(lambda types: t in types)
        row_counts[t] = int(hit.sum())
        subject_counts[t] = int(df.loc[hit, 'subject_id'].nunique())
    row_counts['A_any'] = int(df.any_positive.sum())
    subject_counts['A_any'] = int(df.loc[df.any_positive == 1,
                                         'subject_id'].nunique())
    # Subject-level type membership (any positive row per subject per type).
    subj = df[['subject_id']].drop_duplicates().set_index('subject_id')
    subj['any'] = df.groupby('subject_id').any_positive.max()
    for t in LABEL_TYPES:
        hit = df.aneuploid_types.map(lambda types: t in types)
        subj[t] = hit.groupby(df.subject_id).max().astype(int)
    subj = subj.reset_index()
    combo = subj.groupby(['T13', 'T18', 'T21']).size()
    overlap = {f"T13={int(a)};T18={int(b)};T21={int(c)}": int(n)
               for (a, b, c), n in combo.items() if n}
    overlap_positive = {k: v for k, v in overlap.items() if '1' in k}
    # Per-subject label variety across rows (same subject can carry one type
    # on one draw and another on a later draw).
    variety = df.groupby('subject_id').aneuploid_types.agg(
        lambda s: len(set().union(*[set(x) for x in s if x])))
    summary = {
        'rows': int(len(df)),
        'subjects': int(subject_set),
        'positive_rows': int(df.any_positive.sum()),
        'positive_subjects': subject_counts['A_any'],
        'negative_subjects': int(subject_set - subject_counts['A_any']),
        'per_type_row_counts': row_counts,
        'per_type_positive_subjects': subject_counts,
        'subject_type_overlap': overlap,
        'subject_type_overlap_positive_combos': overlap_positive,
        'subjects_with_multi_type_across_rows': int((variety > 1).sum()),
        'labels_parsed': ('type tokens extracted with regex T13|T18|T21 from '
                          'concatenated strings such as T13T18; order-insensitive'),
        'splitting_rule': ('subjects are the independent unit; folds must keep '
                           'all rows of a subject together'),
        'fold_feasibility_note': ('with 5 subject-grouped folds the positive '
                                  'subject pool must cover every fold; exact '
                                  'fold split happens at model stage with '
                                  'stratified group folds'),
        'fold_feasibility_check': {
            'n_splits': 5,
            'min_positive_subjects_per_fold': 1,
            'possible': subject_counts['A_any'] >= 5,
        },
    }
    (OUT / 'audit.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    counts = pd.DataFrame([
        {'label': k, 'positive_records': row_counts[k],
         'positive_subjects': subject_counts[k]}
        for k in ['A_any'] + LABEL_TYPES])
    counts.to_csv(OUT / 'label_counts.csv', index=False)
    subj[['subject_id', 'any'] + LABEL_TYPES].to_csv(
        OUT / 'subject_labels.csv', index=False)
    return summary


if __name__ == '__main__':
    s = audit()
    print(json.dumps(s, ensure_ascii=False, indent=1)[:1800], flush=True)
