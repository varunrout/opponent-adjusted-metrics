"""Get examples for xA metric explanation."""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from opponent_adjusted.features.cxa.xa_baseline import compute_xa_baseline
from opponent_adjusted.features.cxa.xa_plus_passes import compute_xa_plus_passes

passes = pd.read_parquet('feature_store/cxa/pass_sequences.parquet')
passes_baseline, _ = compute_xa_baseline(passes)
passes_plus, _ = compute_xa_plus_passes(passes)

assists = passes_baseline[passes_baseline['is_assist']].copy()
assists = assists.merge(passes_plus[['pass_id', 'xa_plus']], on='pass_id', how='left')

print("EXAMPLE 1: High xA Baseline + High xA+")
ex1 = assists[(assists['xa_baseline'] > 0.05) & (assists['xa_plus'] > 0.9)].iloc[0]
print("Passer:", ex1['passer_name'])
print("Team:", ex1['team_name'])
print("Pass Type:", ex1['pass_type'])
print("xA Baseline:", round(ex1['xa_baseline'], 4))
print("xA+ Pass:", round(ex1['xa_plus'], 4))

print()
print("EXAMPLE 2: High xA Baseline + Low xA+")
ex2 = assists[(assists['xa_baseline'] > 0.05) & (assists['xa_plus'] < 0.45)].iloc[0]
print("Passer:", ex2['passer_name'])
print("Team:", ex2['team_name'])
print("Pass Type:", ex2['pass_type'])
print("xA Baseline:", round(ex2['xa_baseline'], 4))
print("xA+ Pass:", round(ex2['xa_plus'], 4))

print()
print("EXAMPLE 3: Low xA Baseline + High xA+")
ex3 = assists[(assists['xa_baseline'] < 0.01) & (assists['xa_plus'] > 0.95)].iloc[0]
print("Passer:", ex3['passer_name'])
print("Team:", ex3['team_name'])
print("Pass Type:", ex3['pass_type'])
print("xA Baseline:", round(ex3['xa_baseline'], 4))
print("xA+ Pass:", round(ex3['xa_plus'], 4))

print()
print("=" * 70)
print("SEQUENCE BREAKDOWN FOR EXAMPLE 2 (showing shared credit):")
print("=" * 70)
seq_id = ex2['sequence_id']
seq_passes = passes_plus[passes_plus['sequence_id'] == seq_id].sort_values('minute')
print("Passes in this goal sequence:")
for idx, row in seq_passes.iterrows():
    marker = " <-- ASSIST" if row['is_assist'] else ""
    print("  ", row['passer_name'], "-> credit:", round(row['xa_plus'], 3), marker)
print("Total credit distributed:", round(seq_passes['xa_plus'].sum(), 1))
