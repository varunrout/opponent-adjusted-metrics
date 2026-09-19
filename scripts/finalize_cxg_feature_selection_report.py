from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from google.cloud import bigquery

PROJECT='oam-varun-260819'
DATASET='oam_analysis'
RUN_ID='cxg-analysis-20260820T201934Z'
OUT=Path('audit_outputs/cxg_analysis')/RUN_ID/'cxg_final_feature_selection_report.md'
client=bigquery.Client(project=PROJECT)

def rows(sql):
    return [dict(r.items()) for r in client.query(sql, location='europe-west2').result()]

manifest=rows(f"""
SELECT feature_family,column_name,data_type,non_null_count,null_pct,point_biserial_corr,abs_signal,max_abs_corr,preliminary_decision
FROM `{PROJECT}.{DATASET}.cxg_feature_decision_manifest_v1`
ORDER BY feature_family,
  CASE preliminary_decision WHEN 'control' THEN 0 WHEN 'candidate_keep' THEN 1 WHEN 'candidate_review' THEN 2 WHEN 'redundant_review' THEN 3 WHEN 'watchlist_sparse' THEN 4 ELSE 5 END,
  abs_signal DESC, non_null_count DESC, column_name
""")
interactions=rows(f"""
SELECT feature_family,left_column,right_column,left_bin,right_bin,row_count,goal_count,goal_rate,lift_vs_baseline
FROM `{PROJECT}.{DATASET}.cxg_pair_interaction_target_v1`
WHERE row_count >= 100 AND lift_vs_baseline IS NOT NULL
ORDER BY lift_vs_baseline DESC, row_count DESC
LIMIT 60
""")
bins=rows(f"""
SELECT feature_family,column_name,bin_label,row_count,goal_count,goal_rate,lift_vs_baseline
FROM `{PROJECT}.{DATASET}.cxg_binned_target_response_v1`
WHERE row_count >= 50 AND lift_vs_baseline IS NOT NULL
ORDER BY lift_vs_baseline DESC, row_count DESC
LIMIT 80
""")
redundant=rows(f"""
SELECT left_family,right_family,left_column,right_column,pearson_corr,pair_non_null_count,redundancy_band
FROM `{PROJECT}.{DATASET}.cxg_redundancy_pairs_v1`
ORDER BY abs_corr DESC, pair_non_null_count DESC
""")

by_family=defaultdict(list)
for r in manifest:
    by_family[r['feature_family']].append(r)

reason_map={
 'candidate_keep':'Selected: strongest target signal with acceptable support and not dominated by redundancy.',
 'candidate_review':'Shortlist: useful signal, but needs model-stage confirmation or redundancy/coverage handling.',
 'redundant_review':'Not first-pass selected: overlaps strongly with another feature; keep only if it wins in model validation.',
 'watchlist_sparse':'Not first-pass selected: signal is too sparse/coverage-limited for primary training.',
 'weak_signal_review':'Not first-pass selected: weak or unstable standalone signal in this analysis pass.',
 'control':'Control only: identity/target/base field, not a contextual candidate.',
}

lines=[]
lines.append('# CxG Final Feature Selection Report')
lines.append('')
lines.append(f'Generated: {datetime.now(timezone.utc).isoformat()}')
lines.append(f'Run: `{RUN_ID}`')
lines.append('')
lines.append('This report converts the findings-grade analysis tables into a practical first-pass feature selection. It is not model training: final promotion still requires fold-safe model validation. Selection here is based on target signal, support, redundancy, sparsity and bivariate interaction evidence.')
lines.append('')
lines.append('## Decision Rules')
lines.append('')
lines.append('- Selected: `candidate_keep`. Use in the first serious model/features pass.')
lines.append('- Shortlist: `candidate_review`. Keep available for controlled experiments, but do not force all into the first compact model.')
lines.append('- Rejected for first pass: `redundant_review`, `watchlist_sparse`, `weak_signal_review`. These remain auditable, but are not first-pass selected.')
lines.append('- Control: `base_identity_target`. Used for joins, QA, stratification or baselines, not selected as contextual signal.')
lines.append('')

for fam in ['base_identity_target','event_context','defensive_360','goalkeeper_360','line_shape_360']:
    fr=by_family.get(fam,[])
    if not fr: continue
    lines.append(f'## {fam}')
    lines.append('')
    counts=defaultdict(int)
    for r in fr: counts[r['preliminary_decision']]+=1
    lines.append('Decision counts: ' + ', '.join(f'{k}={v}' for k,v in sorted(counts.items())))
    lines.append('')
    for bucket,title in [('candidate_keep','Selected'),('candidate_review','Shortlist / Review'),('redundant_review','Rejected First Pass - Redundant'),('watchlist_sparse','Rejected First Pass - Sparse'),('weak_signal_review','Rejected First Pass - Weak'),('control','Control')]:
        items=[r for r in fr if r['preliminary_decision']==bucket]
        if not items: continue
        lines.append(f'### {title}')
        lines.append('')
        lines.append('| Feature | Signal | Non-null | Null % | Reason |')
        lines.append('|---|---:|---:|---:|---|')
        for r in items:
            sig=r['point_biserial_corr']
            sig_s='' if sig is None else f"{sig:.3f}"
            null_pct=100*float(r['null_pct']) if r['null_pct'] is not None else 0
            lines.append(f"| `{r['column_name']}` | {sig_s} | {r['non_null_count']} | {null_pct:.1f}% | {reason_map[bucket]} |")
        lines.append('')
    top_int=[r for r in interactions if r['feature_family']==fam][:5]
    if top_int:
        lines.append('### Top Supporting Bivariate Interactions')
        lines.append('')
        lines.append('| Pair | Bins | Rows | Goal rate | Lift |')
        lines.append('|---|---|---:|---:|---:|')
        for r in top_int:
            lines.append(f"| `{r['left_column']}` + `{r['right_column']}` | {r['left_bin']} / {r['right_bin']} | {r['row_count']} | {r['goal_rate']:.3f} | {r['lift_vs_baseline']:.2f} |")
        lines.append('')
    top_bins=[r for r in bins if r['feature_family']==fam][:5]
    if top_bins:
        lines.append('### Top One-Feature Target Bins')
        lines.append('')
        lines.append('| Feature | Bin | Rows | Goal rate | Lift |')
        lines.append('|---|---|---:|---:|---:|')
        for r in top_bins:
            lines.append(f"| `{r['column_name']}` | {r['bin_label']} | {r['row_count']} | {r['goal_rate']:.3f} | {r['lift_vs_baseline']:.2f} |")
        lines.append('')

lines.append('## Redundancy Notes')
lines.append('')
lines.append('| Family pair | Features | r | Overlap | Band |')
lines.append('|---|---|---:|---:|---|')
for r in redundant[:25]:
    lines.append(f"| {r['left_family']} / {r['right_family']} | `{r['left_column']}` <> `{r['right_column']}` | {r['pearson_corr']:.3f} | {r['pair_non_null_count']} | {r['redundancy_band']} |")
lines.append('')
lines.append('## Completion Status')
lines.append('')
lines.append('Analysis phase is complete for feature review: univariate, bivariate/pair interactions, within-family correlation, cross-family correlation, redundancy, chart packs and final first-pass selection have all been produced. Full model-stage multivariate validation remains a later modelling task, not part of this analysis-only phase.')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text('\n'.join(lines), encoding='utf-8')
print(OUT)
