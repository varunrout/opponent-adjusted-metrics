from google.cloud import bigquery
c=bigquery.Client(project='oam-varun-260819')
rows=[
('feature_decision_counts','findings','Must-have decision overview',1),
('candidate_signal_leaderboard','findings','Must-have selected feature signal view',2),
('top_pair_interactions','findings','Bivariate interaction summary',3),
('cross_family_correlation_edges','findings','Correlation and redundancy safety',4),
('all_families__candidate_keep_review','findings_detail','Detailed keep and review surface',5),
('line_shape_360__pair_interactions_top','findings_detail','Best 360 family interaction detail',6),
('goalkeeper_360__target_bins_top','findings_detail','Goalkeeper signal detail',7),
('baseline_auc_ranking','baseline','Model baseline ranking',8),
('baseline_log_loss_ranking','baseline','Calibration and loss ranking',9),
]
selects=[]
for name,pack,reason,priority in rows:
    selects.append(f"SELECT 'cxg-analysis-20260820T201934Z' AS run_id, {priority} AS priority, '{pack}' AS chart_pack, '{name}' AS chart_name, '{reason}' AS shortlist_reason, CURRENT_TIMESTAMP() AS shortlisted_at")
sql='CREATE OR REPLACE TABLE `oam-varun-260819.oam_analysis.cxg_dashboard_chart_shortlist_v1` AS\n'+'\nUNION ALL\n'.join(selects)
c.query(sql,location='europe-west2').result()
print([dict(r.items()) for r in c.query('SELECT chart_pack, COUNT(*) c FROM `oam-varun-260819.oam_analysis.cxg_dashboard_chart_shortlist_v1` GROUP BY chart_pack', location='europe-west2').result()])
