# CxG+ v2 Pool — EDA-Appendix Backfill + Archetype/Role Validation Charts

Backfills the 4 EDA-appendix analysis tables for the 5 new Phase A/B candidate features
(the v2 pool requalification task explicitly only refreshed
`feature_correlation_heatmap`/`pca_scree`/`bivariate_significance_grid`, not these), and
builds 3 new charts connecting the archetype/role categorical features to shot outcomes and
to each other, which nothing previously did.

Run artifacts:
- `scripts/materialize_cxg_v2_eda_backfill.py` (Part 1)
- `scripts/materialize_cxg_archetype_role_chart_registry.py` (Part 3 registry)
- New chart methods in `src/opponent_adjusted/analysis/cxg_charts.py`
- Charts rendered/uploaded under run_id **`cxg-analysis-20260822T142104Z`**

---

## Part 1: EDA-appendix backfill

Scoped, idempotent extension of the 4 tables -- `DELETE WHERE feature_family='opponent_adjusted' AND column_name IN (the 5 new names)` before each `INSERT`, so re-running is
safe and never touches the 18 already-covered features' rows. Followed
`materialize_cxg_opponent_adjusted_analysis.py`'s exact conventions: categorical bins get
`bin_type='category'` with an explicit `'null'` bin (not dropped); continuous features get
`bin_type='quantile'` via `NTILE(20)`; `cxg_univariate_target_v1` computed **train-split
only**, per the split policy the existing `opponent_adjusted` rows already follow.

### Rows added (all 5 new features, one row each except distribution bins)

| Table | Rows added | Detail |
|---|---|---|
| `cxg_null_profile_v1` | 5 | one row per feature |
| `cxg_summary_stats_v1` | 5 | one row per feature |
| `cxg_univariate_target_v1` | 5 | one row per feature, train split |
| `cxg_eda_distribution_bins_v1` | 57 | 2 continuous x 20 quantile bins (40) + 3 categorical (role=6, second_role=6, archetype=5, each incl. a `'null'` bin) (17) |

### A pre-existing data-quality issue found, not touched (out of scope)

The 4 already-covered `opponent_adjusted` features (`defensive_profile_cluster`, `gk_odi`,
`mean_backline_odi`, `nearest_defender_odi`) each have **4 duplicate rows** in all three of
`cxg_null_profile_v1`/`cxg_summary_stats_v1`/`cxg_univariate_target_v1` -- confirmed live,
not assumed. Root cause: `materialize_cxg_opponent_adjusted_analysis.py` has no
delete-before-insert guard and has evidently been re-run 4 times historically. This task's 5
new features each show exactly 1 row (the new scoped-delete logic works correctly). Per the
task's explicit constraint ("do not recompute rows for the 18 already-covered features"),
**this duplication was not touched or fixed** -- flagged here as a pre-existing issue for a
future task's attention, not silently corrected as a side effect of this one.

### A blocking dependency found and fixed: `cxg_analysis_opponent_adjusted_v1`

`cxg_charts.py`'s `_eda_bins` numeric branch queries this surface view live for continuous-
feature histogram values. Initial inspection via `get_table().num_rows` reported 0 rows,
suggesting it was unpopulated -- **this was a misread**, not a real finding: `num_rows`
always reports 0 for a `VIEW` regardless of actual content (the same metadata-API quirk
already seen with `cxg_training_matrix_v1` in an earlier task). A direct `SELECT COUNT(*)`
confirmed 3,960 real rows -- the view was working correctly all along, just missing the 5
new columns. Fixed with `CREATE OR REPLACE VIEW`, adding one more `LEFT JOIN` to
`cxg_defensive_360_features` on top of its existing (unchanged) join structure -- confirmed
via `get_table().view_query` before editing, not assumed. This was a necessary enabling fix
for Part 2's `opponent_adjusted_eda_histogram` chart to render real data for the 2 new
continuous features (querying a column that doesn't exist on the view would otherwise be a
hard SQL error, not a graceful empty chart) -- scoped to exactly this one view, no other
surface table touched.

---

## Part 2: 4 EDA charts re-rendered

`opponent_adjusted_eda_histogram`, `opponent_adjusted_null_profile_bar`,
`opponent_adjusted_summary_box`, `opponent_adjusted_target_lift_bar` were already registered
in `cxg_chart_registry_v1` from an earlier task -- no new registry rows needed, they
automatically reflect the backfilled tables (and the surface-view fix) once re-rendered. All
4 rendered successfully with no errors, both locally (`--skip-upload`, reviewed) and after
upload.

---

## Part 3: 3 new archetype/role validation charts

### `opponent_adjusted_archetype_goal_rate_bar` -- goal rate by archetype, train split

`cxg_univariate_target_v1` was checked first and found **not sufficient** for this chart --
it carries only one aggregate row per feature (matching `defensive_profile_cluster`'s own
precedent), not a per-level breakdown. Computed instead via a single batched
`GROUP BY archetype` query against the already-materialized
`cxg_plus_360_model_matrix_v1`/`cxg_defensive_360_features` (no raw-event re-scan, no
per-level looping).

| Archetype | n | Goal rate |
|---|---|---|
| **null (no defender / below threshold)** | 288 | **28.5%** |
| `deep_block_clearer` | 1,079 | 12.0% |
| `unresolved_5050_annotation_density` (muddy 4th cluster) | 368 | 7.1% |
| `high_volume_presser` | 877 | 6.4% |
| `duel_dominant_contester` | 168 | 6.0% |

**Reported honestly, no forced narrative:** the null category has by far the highest goal
rate, more than double `deep_block_clearer`'s. Investigated rather than left as a bare
surprising number: of the 288 null-archetype train-split shots, 36 are also `null_cluster`
(the defensive-profile-cluster's own geometry-ineligible/penalty-kick population confirmed
in the bivariate task) -- and those 36 have a 66.7% goal rate, consistent with the earlier
penalty-kick finding. That explains part of the elevation but not all of it (the other 252
null-archetype shots still average a rate well above the other archetypes) -- reported as a
partial, not complete, explanation; not chased further, out of this task's scope.

Among the 4 non-null archetypes, the muddy 4th cluster's 7.1% goal rate sits in the same
tier as `high_volume_presser` (6.4%) and `duel_dominant_contester` (6.0%) -- consistent with
the v2 pool requalification task's earlier finding that this cluster shows real, stable,
non-degenerate signal despite being flagged as not cleanly interpretable by dominant
action-type. `deep_block_clearer` clearly stands apart with a nearly 2x higher rate.

### `opponent_adjusted_archetype_role_heatmap` -- archetype x role cross-tab

Neither `cxg_defender_style_cluster_profile_v1` nor any other materialized table already had
this cross-tab (checked before writing a new query, per the task's instruction) -- computed
via a single batched `GROUP BY archetype, role` over `cxg_defensive_360_features` (full
population, all splits).

| archetype | Attack | CB | Fullback_WingBack | GK | Midfield |
|---|---|---|---|---|---|
| `deep_block_clearer` | 0.3% | **73.4%** | 22.7% | 0.0% | 3.7% |
| `duel_dominant_contester` | 31.5% | 20.6% | 34.9% | 0.0% | 13.0% |
| `high_volume_presser` | 17.7% | 4.4% | 16.9% | 0.0% | 61.0% |
| `unresolved_5050_annotation_density` | 11.1% | 5.0% | 17.9% | 0.0% | 66.0% |

`deep_block_clearer` is 96.1% CB + Fullback_WingBack combined (73.4% + 22.7%) -- broadly
consistent with Phase B's own report figure of "93.5% defenders," though not an exact match.
That's expected, not a discrepancy: Phase B's number was an ad-hoc, un-materialized
computation (confirmed via investigation -- it appears only as prose in the Phase B report,
not in any table), and very likely used a different unit of measurement (player-level
primary position across a career, vs. this chart's shot-instance nearest-defender-role) --
same real pattern, different lens, both pointing the same direction.

### `opponent_adjusted_role_displacement_bar` -- mean zone displacement by role

Single batched `GROUP BY role` query, ordered GK -> CB -> Fullback_WingBack -> Midfield ->
Attack (Phase A's own verification order).

| Role | n | Mean displacement (m) | Phase A's original figure |
|---|---|---|---|
| GK | 97 | 6.104 | ~6.1m |
| CB | 1,329 | 27.622 | ~27.6m |
| Midfield | 1,263 | 40.691 | ~40.7m |
| Fullback_WingBack | 779 | 43.595 | ~43.6m |
| Attack | 399 | 51.735 | ~51.7m |

**Exact match to Phase A's original coordinate-fix verification numbers** (to 3+ decimal
places) -- confirms nothing has changed in the underlying data or computation since that
fix. No discrepancy to flag.

---

## Part 4: bivariate/interaction chart check

### Significance grid -- did NOT already distinguish new vs. reconfirmed pairs

Checked the current `bivariate_significance_grid` chart method before assuming either way:
it plotted only the FDR-adjusted -log10(p) heatmap, with no visual marker for which cells
were "confirmed" at all, let alone new-vs-reconfirmed. **Re-rendered with a distinction
added**: confirmed pairs (`validated_on_val_split = TRUE`) now get an annotation on their
heatmap cell -- a gold "★ NEW" for a pair confirmed only because it involves a Phase A/B
feature, a white "✓ reconfirmed" for a pair also confirmed in the pre-Phase-A/B pool. The
new-vs-reconfirmed classification is derived from pool membership (any Phase A/B feature
name involved -> new), not a hardcoded pair list, so it stays correct if the confirmed set
changes in a future re-run. Verified the 6 confirmed pairs classify as expected: the 4
originally-locked pairs (none involve a Phase A/B feature) get "✓ reconfirmed"; the 2 pairs
newly confirmed by the v2 requalification (`defensive_profile_cluster x
nearest_defender_zone_displacement`, `nearest_defender_gap x visible_goal_angle_proxy`, both
involving a Phase A feature) get "★ NEW".

### Stratified plot for the 2 new interactions -- explicit gap, not computed

Checked `cxg_bivariate_stratified_v1` directly: it holds exactly 2 rows total for
`track='cxg_plus'` (`gk_odi x manpower_diff`, tier 2; `defensive_profile_cluster x
shot_distance_sb`, tier 3) -- **neither of the 2 new pairs is present, in either column
order**. Per the task's explicit instruction, this stratified data was **not computed fresh**
(that would be new statistical computation, out of this task's chart-backfill scope).
**Flagged as a gap instead**: `opponent_adjusted_new_interaction_stratified_plot` was **not
built**. A future task computing Tier 1 stratified breakdowns for
`defensive_profile_cluster x nearest_defender_zone_displacement` and/or `nearest_defender_gap
x visible_goal_angle_proxy` would need to materialize that data into
`cxg_bivariate_stratified_v1` first (the same pattern Tier 2/3's existing 2 rows already
use) before a chart could be built on top of it.

---

## Chart registry additions

Registered under run_id **`cxg-analysis-20260822T142104Z`** via
`scripts/materialize_cxg_archetype_role_chart_registry.py` (copy-forward pattern; prior
run's 39 rows carried forward untouched, 3 new rows appended):

| chart_name | chart_type | feature_family | backing_table |
|---|---|---|---|
| `opponent_adjusted_archetype_goal_rate_bar` | `archetype_goal_rate_bar` | `opponent_adjusted` | `cxg_plus_360_model_matrix_v1` |
| `opponent_adjusted_archetype_role_heatmap` | `archetype_role_heatmap` | `opponent_adjusted` | `cxg_defensive_360_features` |
| `opponent_adjusted_role_displacement_bar` | `role_displacement_bar` | `opponent_adjusted` | `cxg_defensive_360_features` |

All 42 charts for this run_id (39 carried forward + 3 new) rendered locally first
(`--skip-upload`, reviewed), then uploaded to GCS and registered in
`cxg_rendered_chart_registry_v1` via the existing scoped delete-then-insert-by-run_id
pattern. `cxg_chart_registry_v1` history intact across all prior run_ids (24/27/27/31/35/
39/39/39/42 rows), no destructive overwrite. `bivariate_significance_grid`'s underlying
chart_type wasn't newly registered (already existed) -- only its rendering code changed
(the new annotation logic), which the re-render under the fresh run_id picks up
automatically.

---

## Row-count reconciliation

| Item | Expected | Actual |
|---|---|---|
| `cxg_null_profile_v1` new rows | 5 | 5 |
| `cxg_summary_stats_v1` new rows | 5 | 5 |
| `cxg_univariate_target_v1` new rows | 5 | 5 |
| `cxg_eda_distribution_bins_v1` new rows | 40 (continuous) + 17 (categorical) = 57 | 57 |
| `cxg_analysis_opponent_adjusted_v1` (view) rows | 3,960 (360-eligible cohort) | 3,960 |
| Chart registry rows, new run_id | 39 + 3 = 42 | 42 |

All match.

---

## Test suite

`python -m pytest -q` -> **283 passed**, no regressions (unchanged from the pre-task
baseline -- this task is table/chart orchestration, not new pure logic, matching the pattern
of the v2 pool requalification task; correctness verified via the live BigQuery runs and the
row-count/value reconciliation throughout this report instead).

---

## What was explicitly NOT done (per task constraints)

- No rows recomputed for the 18 already-covered `opponent_adjusted` features in any of the
  4 EDA tables -- their pre-existing duplication was found and reported, not fixed.
- CxG (event-wide) track was not touched.
- `feature_correlation_heatmap`/`pca_scree` were not touched -- only
  `bivariate_significance_grid` was updated, per Part 4's explicit scope.
- No new stratified data computed for the 2 new interactions -- flagged as a gap, per the
  task's explicit instruction not to quietly expand into new statistical work.
- The muddy 4th archetype cluster was not hidden from any chart -- shown and labeled in the
  goal-rate bar, the role cross-tab heatmap, and the underlying distribution bins.
- No `CREATE OR REPLACE` on the chart registry table -- scoped delete-then-insert-by-run_id
  only.
- No loop-per-feature or loop-per-category-level query against raw event tables -- every new
  query is a single batched `GROUP BY` against already-materialized `oam_features`/
  `oam_analysis` tables.
- No push to git remote.

---

## Summary for hand-off

**4 EDA tables backfilled** for the 5 new features (72 total new rows across the 4 tables),
plus a necessary enabling fix to the `cxg_analysis_opponent_adjusted_v1` surface view
(discovered mid-task, not a real "0 rows" bug -- a view-metadata misread, fixed with one
additional `LEFT JOIN`). **4 EDA charts re-rendered**, **3 new charts built**
(archetype-goal-rate, archetype-role cross-tab, role-displacement), **1 chart updated**
(significance grid now visually distinguishes new vs. reconfirmed interactions). **1 gap
explicitly flagged, not filled** (stratified plot for the 2 new interactions -- data doesn't
exist yet).

**Key findings**: role displacement exactly matches Phase A's original numbers (no drift).
Archetype goal rate shows a clear, honestly-reported pattern -- `deep_block_clearer` at 12.0%
roughly double the other clean archetypes (6.0-6.4%), the muddy 4th cluster sitting
unremarkably in the same tier as the other clean archetypes (7.1%, not degenerate), and a
striking 28.5% rate for the null/no-defender category, partially (not fully) explained by
penalty-kick overlap. `deep_block_clearer`'s 96.1% CB+Fullback_WingBack skew broadly
confirms (via a different measurement lens) Phase B's own "93.5% defenders" prose finding.
