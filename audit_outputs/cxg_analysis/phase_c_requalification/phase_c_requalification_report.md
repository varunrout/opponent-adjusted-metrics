# Statistical Qualification for Phase C Rolling-Window Features (BOTH tracks)

Scoped addition, not a full pool re-run — only rows involving the 6 new Phase C features
(`defensive_action_rate_15m/30m/45m/60m`, `territorial_dominance_last_15m`,
`cross_match_defensive_rate`) were touched; every existing untouched result was left alone,
same discipline as every prior requalification round.

## Step 0 — confirmation

Re-queried `oam_features.cxg_training_matrix_v1` live (`data_version`/`silver_schema_version`
pinned): event-wide n=15,737, `territorial_dominance_last_15m` null=16, `cross_match_
defensive_rate` null=906 (5.76%); CxG+ n=3,960, null=3 / 273 (6.89%); all four
`defensive_action_rate_*m` windows 0 null in both tracks. **All match Phase C's report
exactly — no discrepancies, script proceeded.**

Baseline pool before this task, confirmed live from `cxg_bivariate_candidate_pool_v1`: CxG
event-wide = 5 features (`shot_x_sb`, `last_action_interval_s`, `last_box_entry_to_shot_s`,
`regain_height_speed_interaction`, `first_box_entry_to_shot_s`); CxG+ = 20 (16 continuous +
4 categorical). This is the first requalification round to touch the CxG event-wide track's
pool since v1 — every prior round (Phase A/B/v2) was CxG+-only, since 360-derived features
by definition never applied to the event-wide track.

## Step 1 — Univariate (both tracks)

471 pre-existing rows untouched. Added 36 rows (6 features × 2 tracks × 3 splits) to
`cxg_split_univariate_v1`, `run_id=cxg-phase-c-requal-20260822T224248Z`. Per the project's
non-negotiable principle, weak/no univariate signal never disqualified a feature — used
only as an input to the redundancy tie-break below (min(|point-biserial r| across
train/val/test)).

`abs_signal` for the 4 defensive-action-rate windows sits in the ~0.045–0.066 range in both
tracks (weak but nonzero, consistent with a defensive-context feature not a direct shot
predictor); `cross_match_defensive_rate` and `territorial_dominance_last_15m` are similarly
weak on their own. None of this gated inclusion.

## Step 2 — Correlation / redundancy screen (both tracks, train-split, r≥0.85)

**1. The 4 `defensive_action_rate_{15,30,45,60}m` windows against each other** — confirmed
severely redundant, both tracks, exactly as the task anticipated ("nested/overlapping
trailing windows on the same underlying event stream"):

| pair | r_train (cxg_event) | r_train (cxg_plus) |
|---|---|---|
| 15m × 30m | 0.9622 | 0.9612 |
| 15m × 45m | 0.9546 | 0.9522 |
| 15m × 60m | 0.9544 | 0.9518 |
| 30m × 45m | 0.9929 | 0.9923 |
| 30m × 60m | 0.9926 | 0.9917 |
| 45m × 60m | 0.9998 | 0.9996 |

All 6 pairs exceed the 0.85 threshold in both tracks. Tie-break applied: kept whichever
window had the higher min(|point-biserial r| across train/val/test) at each pairwise
comparison — **`defensive_action_rate_30m` survives every comparison in both tracks**
(min-abs-r ≈ 0.0485 event-wide / 0.0660 CxG+, marginally higher than its neighbours at
every step: 15m loses to 30m twice directly and would also lose to 45m/60m; 45m and 60m
both lose to 30m directly). **Final: 15m, 45m, 60m dropped; only 30m kept**, both tracks —
not "keep all 4 by default."

**2. `territorial_dominance_last_15m` vs. `territorial_dominance_last_5m`** (due-diligence
sanity check, `last_5m` was never a formal pool member on either track): r_train = 0.7624
(cxg_event, n=10,881), r_train = 0.7889 (cxg_plus, n=2,779). **Below the 0.85 threshold in
both tracks** — correlated (same underlying construction, overlapping windows) but not
redundant; the 15-minute window genuinely captures different information from the frozen
5-minute one at the margin. Both features stay in their respective tables (5m is frozen/not
a pool candidate; 15m is now a qualified pool candidate).

**3. `cross_match_defensive_rate` vs. the within-match rate features** — checked against
all 4 windows in both tracks; max |r_train| = 0.1618 (event-wide, vs. 45m) / 0.1454 (CxG+,
vs. `territorial_dominance_last_15m`). **Genuinely complementary, not redundant** — a team's
cross-match defensive-activity history and its within-match rate leading up to THIS shot are
only weakly related, confirming the task's "could be genuinely complementary" hypothesis
over the "consistent style" alternative.

**No existing pool member was found redundant with any new feature in either track** — all
new-vs-existing correlations were well under 0.85 (max observed: `cross_match_defensive_
rate` vs `visible_goal_angle_proxy`-adjacent features, all |r| < 0.2). `dropped_existing = []`
for both tracks — nothing in the pre-existing 5 (CxG event) or 20 (CxG+) pool members was
disturbed.

**Final pool (both tracks add exactly 3 new survivors):** `defensive_action_rate_30m`,
`territorial_dominance_last_15m`, `cross_match_defensive_rate`.

- CxG event-wide: 5 → **8** (`cxg_bivariate_candidate_pool_v1` confirmed live: 8 rows).
- CxG+: 20 → **23** (confirmed live: 23 rows).

## Step 3 — PCA re-run (both tracks)

Full per-track rewrite (diagnostic-only, pool-composition-dependent by construction, same
convention as every prior round) over each track's enlarged, redundancy-trimmed numeric
pool. Median-impute + StandardScaler fit on train, PCA fit on train.

**CxG event-wide — first PCA re-run since this track's pool was this small (previously 5
features → now 8).** Genuinely different shape from CxG+, not a mirror: 8 components for 8
features (no dimensionality collapse — all 3 survivors carry independent variance), 80%
cumulative variance needs 6 of 8 components (comparatively flat scree — this small pool has
no dominant single factor). PC1 top loadings: `last_box_entry_to_shot_s` (0.593),
`first_box_entry_to_shot_s` (0.549), `shot_x_sb` (−0.325), `territorial_dominance_last_15m`
(0.291), `defensive_action_rate_30m` (0.246) — PC1 reads as a "shot-buildup proximity to
goal" axis, with the two new rolling-window features contributing meaningfully but not
dominantly.

**CxG+ — 19 components for 19 numeric features (categoricals excluded from PCA, unchanged
convention), 80% cumulative variance needs 10 of 19.** PC1 top loadings essentially
unchanged from the pre-Phase-C shape (`shot_corridor_occlusion` 0.394, `defenders_between_
ball_and_goal` 0.392, `estimated_goalface_occlusion` 0.352, `defensive_reset_index` −0.342,
`gk_distance_to_shooter` 0.341) — none of the 3 new features crack the PC1 top-5, meaning
Phase C's signal is genuinely orthogonal-ish to the existing 360-geometry-dominated first
component rather than reinforcing it.

## Step 4 — Bivariate Tier 1 re-run (both tracks)

New pairs only (new×new + new×existing-pool, BH-FDR recalibrated over the combined family
per pair, matching the `materialize_cxg_v2_21st_feature_requalification.py` precedent —
existing rows' stored `p_fdr` values are left as-is, not rewritten, per that same precedent).

**CxG event-wide (18 new pairs tested) — first genuine bivariate look this track has ever
had** (its pool was too small, 5 features / 10 pairs, for a prior round to be worth running
separately): **one new confirmed Tier 1 interaction** —
`defensive_action_rate_30m × territorial_dominance_last_15m`, `p_fdr = 0.00732`, validated
on the validation split. This is notable specifically because it does NOT mirror CxG+'s
result set (which found nothing new) — the event-wide track's first-ever confirmed
interaction is between two Phase C features themselves (a team facing high defensive
pressure recently AND controlling territory recently jointly modulate goal probability more
than either alone), not a carry-over of a CxG+-style interaction. Reported as found, not
assumed.

**CxG+ (63 new pairs tested) — zero new confirmed interactions.** None of the 3 CxG+ survivors
crossed `p_fdr < 0.10` against the existing 20-feature pool or against each other.

**Existing confirmed CxG+ interactions — explicitly re-checked, all 6 STILL_CONFIRMED,
unaffected by the enlarged pool:**

| pair | p_fdr | validated |
|---|---|---|
| `defensive_profile_cluster × visible_goal_angle_proxy` | 1.67e-10 | true |
| `defensive_profile_cluster × gk_distance_to_shooter` | 0.01225 | true |
| `pre_shot_receiver_space × visible_goal_angle_proxy` | 0.02405 | true |
| `defensive_line_depth × pre_shot_receiver_space` | 0.03842 | true |
| `defensive_profile_cluster × nearest_defender_zone_displacement` | 0.02405 | true |
| `nearest_defender_gap × visible_goal_angle_proxy` | 0.02405 | true |

None of their member features were dropped in Step 2 (`dropped_existing = []`), and their
rows were never touched by this task's scoped deletes (only rows referencing the 6 new
features, or a dropped-existing feature, were deleted — neither condition applied here).

Tiers 2/3/4 were not touched (no delete issued against those rows).

## Final enlarged candidate pool per track

**CxG event-wide (8):** `shot_x_sb`, `last_action_interval_s`, `last_box_entry_to_shot_s`,
`regain_height_speed_interaction`, `first_box_entry_to_shot_s`, `defensive_action_rate_30m`,
`territorial_dominance_last_15m`, `cross_match_defensive_rate`.

**CxG+ (23):** the existing 16 continuous (`last_action_interval_s`,
`defenders_between_ball_and_goal`, `defensive_reset_index`, `nearest_defender_distance_
delta`, `pre_shot_receiver_space`, `gk_distance_to_shooter`, `defensive_line_depth`,
`defensive_width`, `estimated_goalface_occlusion`, `goal_mouth_defender_count`, `max_goal_
exposure`, `min_defensive_compactness_sequence`, `shot_corridor_occlusion`, `visible_goal_
angle_proxy`, `nearest_defender_zone_displacement`, `nearest_defender_gap`) + 4 categorical
(`defensive_profile_cluster`, `nearest_defender_role`, `second_nearest_defender_role`,
`nearest_defender_style_archetype`) + the 3 new survivors (`defensive_action_rate_30m`,
`territorial_dominance_last_15m`, `cross_match_defensive_rate`).

## Row-count reconciliation

- `cxg_split_univariate_v1`: 471 → 507 (+36 = 6 features × 2 tracks × 3 splits, confirmed live).
- `cxg_feature_correlation_v1`: 135 → 291 (+156 new-pair rows, confirmed live: cxg_event
  15 new×new + 30 new×existing-5 = 45; cxg_plus 15 new×new + 96 new×existing-16-continuous
  = 111; categoricals correctly excluded from the Pearson screen per established convention,
  so cxg_plus's correlation-stage "existing" pool is the 16 continuous members, not all 20).
  Existing rows untouched (verified: 0 rows deleted from the pre-existing set, since
  `dropped_existing` was empty on both tracks).
- `cxg_bivariate_candidate_pool_v1`: cxg_event 5→8, cxg_plus 20→23 (confirmed live).
- `cxg_pca_components_v1` / `cxg_pca_loadings_v1`: full per-track rewrite (8 components for
  cxg_event, 19 for cxg_plus).
- `cxg_bivariate_interaction_v1` (tier=1, confirmed live): cxg_event 10→28 (+18 new pairs),
  cxg_plus 210→273 (+63 new pairs, no duplicate pairs introduced — verified); tier 2/3/4 rows
  (cxg_plus only) and all 6 prior-confirmed rows left untouched.

## Chart coverage

Chart registry copy-forwarded from `cxg-analysis-20260822T165227Z` to
`cxg-analysis-phase-c-requal-20260822T224713Z` (45 rows, delete-then-insert by run_id, never
`CREATE OR REPLACE`) via `scripts/materialize_cxg_phase_c_chart_registry.py`. No new chart
type/row definitions were needed — `feature_correlation_heatmap`, `pca_scree`, and
`bivariate_significance_grid` already existed for both `cxg_event` and `cxg_plus` (reading
directly from the now-updated `cxg_feature_correlation_v1` / `cxg_pca_components_v1` /
`cxg_bivariate_interaction_v1`), so a plain copy-forward + re-render was sufficient. All 45
charts rendered locally (`audit_outputs/cxg_analysis/cxg-analysis-phase-c-requal-
20260822T224713Z/rendered_charts/`) and uploaded to GCS
(`gs://oam-varun-260819-artifacts/analysis/cxg/cxg-analysis-phase-c-requal-20260822T224713Z/
rendered_charts/`), including the first meaningful-content renders of `cxg_event_
correlation_heatmap.{html,png}` and `cxg_event_pca_scree.{html,png}`.

## Tests + regression check

Full suite: **310 passed**, 0 regressions (baseline confirmed at 310 after Phase C before
starting this task; this task added no new pure-logic code requiring new unit tests — it
consumes Phase C's already-tested feature-computation code and only orchestrates BigQuery
analysis-table writes).

## Files

- `scripts/materialize_cxg_phase_c_requalification.py` — univariate/correlation/PCA/Tier1
  bivariate, both tracks.
- `scripts/materialize_cxg_phase_c_chart_registry.py` — chart-registry copy-forward.
- `audit_outputs/cxg_analysis/phase_c_requalification/run_summary.json` — raw run output.
- `audit_outputs/cxg_analysis/phase_c_requalification/phase_c_requalification_report.md` —
  this report.
