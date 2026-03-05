# cXA Phase 0 — Goal Population Alignment

This report aligns goal/assist populations between pass-only sequences and action sequences.

## Counts

| Dataset | Goals |
|---------|-------|
| sequences.parquet | **369** |
| action_sequences.parquet | **439** |
| Overlap (common `shot_id`) | **360** |

## Alignment Status

⚠️ **NOT ALIGNED** — Goal populations differ.

For fair comparisons, either:
1. Run comparisons on the **overlap set only** (360 goals)
2. Fix the action sequence builder to include missing goals (recommended)

## Missing Goals (in sequences but NOT in actions)

Count: **9**

| shot_id |
|---------|
| 700 |
| 1340 |
| 1752 |
| 3083 |
| 4089 |
| 4215 |
| 4984 |
| 5621 |
| 5777 |

## Extra Goals (in actions but NOT in sequences)

Count: **79**

These are likely goals with no passes in buildup (solo runs, direct shots after winning ball).