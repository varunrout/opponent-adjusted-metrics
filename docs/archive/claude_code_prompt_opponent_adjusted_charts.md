# Task: Chart Coverage for the `opponent_adjusted` Family (ODI + Defensive Profile Clusters)

## Context

The `opponent_adjusted` family (4 features: `nearest_defender_odi`, `mean_backline_odi`, `gk_odi`, `defensive_profile_cluster`) is registered in the governed taxonomy and has rows in `cxg_feature_inventory_v1`, `cxg_null_profile_v1`, `cxg_summary_stats_v1`, `cxg_eda_distribution_bins_v1`, `cxg_univariate_target_v1`, and `cxg_split_univariate_v1` (see `audit_outputs/cxg_analysis/opponent_adjusted_extension/opponent_adjusted_analysis_extension_report.md`). None of this has been charted yet — a repo audit found zero chart artifacts for this family; all 178 existing PNGs under `audit_outputs/cxg_analysis/cxg-analysis-20260820T201934Z/` are from the prior 20 Aug run, covering only the pre-existing families.

Two renderers exist and need to be extended, not replaced:
- `src/opponent_adjusted/analysis/cxg_charts.py` (`CxGChartRenderer`) — family-level overview charts (null profile, normalized summary ranges, EDA histogram/category, target-lift), driven by `cxg_chart_registry_v1` as the source of *what* to render.
- `scripts/render_cxg_feature_eda_appendix.py` (`AppendixRenderer`) — per-feature EDA appendix (one histogram/category chart per feature), driven directly by `cxg_feature_inventory_v1` (already includes the 4 new features via its `column_role = 'feature'` filter — confirm this, don't assume).

**The blocker in both:** `_surface_for_family()` (identical logic in both files) maps `feature_family` to either `cxg_analysis_360_v1` or `cxg_analysis_event_v1` for raw per-row values (needed for histograms/scatter, not for the null-profile/summary/target-lift charts which read pre-aggregated tables). Neither surface has `nearest_defender_odi`, `mean_backline_odi`, `gk_odi`, or `defensive_profile_cluster` as columns — those live in `oam_analysis.cxg_odi_features_v1` and `oam_analysis.cxg_defensive_profile_clusters_v1`, keyed on `event_id`. `opponent_adjusted` also doesn't end in `_360`, so today it would silently route to the wrong surface (`cxg_analysis_event_v1`) and either error on missing columns or (worse) silently return empty.

**Decision (locked): create a new read-only BigQuery view, not a schema change.** Do not add columns to `cxg_analysis_360_v1` — that table is a materialized surface other locked family analyses already depend on, and touching it re-opens scope this task doesn't need. Instead:

## Build

**1. New view: `oam_analysis.cxg_analysis_opponent_adjusted_v1`.**
`LEFT JOIN` of `cxg_analysis_360_v1` (or whatever subset of its columns are actually needed — at minimum `event_id`, `is_goal`, and anything the pitch/scatter chart types might want) with `cxg_odi_features_v1` and `cxg_defensive_profile_clusters_v1`, both joined on `event_id`. This must be a genuine `CREATE VIEW` (or `CREATE OR REPLACE VIEW`) — read-only, no materialization, no data duplication — so it always reflects current state of the three underlying tables. Row population: the 360-eligible cohort (3,960), matching every other `_360` family surface. Confirm row count reconciles to 3,960 after creating it.

**2. Extend `_surface_for_family()` in both `cxg_charts.py` and `render_cxg_feature_eda_appendix.py`.**
Add the `opponent_adjusted` → `cxg_analysis_opponent_adjusted_v1` mapping explicitly (don't rely on a naming-convention fallback like the `_360` suffix check — add an explicit family-to-surface dict/mapping so this is unambiguous and doesn't silently mis-route if a future family name doesn't match a suffix pattern either). Keep both files' logic consistent with each other — they should not diverge on how they resolve a family to its surface.

**3. Register `opponent_adjusted` chart rows in `cxg_chart_registry_v1`.**
This table is what `cxg_charts.py`'s `_load_chart_registry()` reads to know what to render — confirm it currently has zero `opponent_adjusted` rows (expected, per the audit), then insert the standard chart set every other family gets: `null_profile_bar`, `summary_box`, `eda_histogram`, `target_lift_bar` (matching the existing `{family}_{chart_type}` naming convention visible in the other 178 files, e.g. `defensive_360_null_profile_bar`). Do **not** add a `pitch_heatmap`/`pitch_scatter`/`correlation_heatmap` chart type for this family — those chart types have hardcoded family-specific logic (`goalkeeper_360` vs. default in `_pitch_scatter`/`_write_pitch_png`) that doesn't apply here, and correlation/bivariate charts are explicitly out of scope per the standing split-policy scope boundary.

**4. Run both renderers for `opponent_adjusted` only if practical** (check both scripts' CLI args — `cxg_charts.py` doesn't appear to support filtering by family, only by `--run-id`/`--limit`; if there's no clean way to render only the new family without re-rendering everything else, render the full set for a fresh run_id rather than guessing at a partial-render hack, and just confirm via the manifest/registry counts that all pre-existing family charts are unchanged in content, just re-rendered). Use a new `run_id` for this render pass (do not reuse `cxg-analysis-20260820T201934Z` — that's the frozen prior run referenced throughout existing reports; check what `run_id` convention/format has been used and follow it, e.g. a fresh UTC timestamp).

**5. Validation.**
- Confirm 4 new PNG+HTML pairs exist for `opponent_adjusted` under the appendix output (one per feature) and 4 new PNG+HTML pairs under the family-overview output (null_profile_bar, summary_box, eda_histogram, target_lift_bar).
- Confirm the family-overview `eda_histogram` chart for `opponent_adjusted` correctly renders a mix: 3 numeric features (ODI) via histogram, `defensive_profile_cluster` needs to be handled as categorical if `_eda_bins`'s logic picks it up — trace through whether the existing numeric-vs-categorical branching in `_eda_bins`/`_write_distribution_png` handles a 4-category INT64-stored-but-categorical field correctly (it may need the categorical branch triggered explicitly rather than falling into the numeric branch just because the column is INT64-typed — check `cxg_summary_stats_v1`'s `distinct_count` for `defensive_profile_cluster`, which should be 4, low enough that the existing `distinct_count > 2` numeric-detection heuristic might misclassify it; verify and report which path it actually took).
- Confirm no existing chart for any other family changed in a way that isn't just "re-rendered with fresh timestamp" — i.e. don't let this task accidentally alter existing family chart logic.
- Write a short validation note into `audit_outputs/cxg_analysis/opponent_adjusted_extension/` (extend the existing report or add a companion file — your call) confirming the above.

## What NOT to do

- Do not modify `cxg_analysis_360_v1` or `cxg_analysis_event_v1` schemas.
- Do not add pitch/correlation chart types for this family.
- Do not build or reactivate any bivariate/multivariate/model-spec output — this is charts-only, on top of already-existing univariate-and-earlier analysis tables.
- Do not touch Phase 1/2 ODI/defprofile code/tables, frozen S/E1-E12 feature code, `three_sixty_context.py`, or Silver/oam_core.
- Do not silently let the numeric/categorical detection heuristic misclassify `defensive_profile_cluster` — verify and fix if needed, don't just accept whatever the existing heuristic happens to do.

## Deliverables checklist

- [ ] `oam_analysis.cxg_analysis_opponent_adjusted_v1` view, row count confirmed = 3,960
- [ ] `_surface_for_family()` extended consistently in both renderer files, explicit mapping not suffix-based fallback
- [ ] `cxg_chart_registry_v1` rows added for `opponent_adjusted` (4 chart types, no pitch/correlation)
- [ ] Both renderers run successfully for a fresh `run_id`
- [ ] 4 per-feature EDA appendix chart pairs + 4 family-overview chart pairs confirmed to exist (HTML+PNG each)
- [ ] `defensive_profile_cluster` numeric-vs-categorical rendering path verified and correct
- [ ] Validation note confirming no unintended change to other families' chart logic
- [ ] No schema changes to existing analysis surfaces, no bivariate work, no Phase 1/2/Silver code touched

Report back with a summary and the local file paths to the new charts when complete.
