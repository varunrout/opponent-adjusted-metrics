# Player-Season CxG/CxA Quadrant Scatter v1 (Hard gate 2)

Date: 2026-09-21
Closes Hard gate 2 (`docs/dashboard_design_spec_v2.md` section 9): "`oam_serving`
must be populated with player-level CxG/CxA values before: Track B (build-your-own
quadrant scatter, originally scoped for Analysis) can be built at all." Investigated
first, per this task's own instruction, before writing any materialization SQL or
frontend code.

## 1. What "player-level CxG/CxA values" means here, and why

The design spec's component inventory (section 10) gives the entire spec for this
chart: "Two metrics, axes crossed at league median." No metric names, no fixed pair
-- "build-your-own" (its own words, Hard gate 2) implies a user-selected metric pair,
not one hardcoded chart, consistent with Analysis being "a real-time research tool"
(section 5) rather than a pre-rendered Explore-zone card.

**Checked for an existing player-aggregate pattern to reuse rather than invent.**
`bigquery_store.py`'s `list_player_seasons` (the only existing player-season
aggregate) computes `shots`, `goals`, `total_xg` from `oam_core.shots`/`events`
directly -- **StatsBomb xG only, all splits, no CxG or CxA involved at all.** There
is no existing player-level CxG or CxA rollup anywhere in this codebase to extend;
this is genuinely new.

**Metric definitions actually used, and why:** mean CxG per shot (event and plus
tracks) and mean CxA per chance-creating pass (event and plus tracks) -- four
metrics total, any two selectable as X/Y. These are not invented from scratch: they
reuse the exact same per-shot `v3_predicted_prob` (CxG) and per-pass
`cxa_combined_score` (CxA) values already computed and already public elsewhere in
this project (the Models page, the CxA detail page, the per-shot display in
[`cxa_pass_detail_v1.md`](cxa_pass_detail_v1.md)) -- just aggregated to player-season
grain by a plain `AVG()`, not a new metric definition invented for this chart alone.
Total/count companions (`_total`, `_total_xg`, `_goals`, `_n_shots`/
`_n_passes_created`) are materialized alongside the means so a future chart or
caller isn't limited to means only, at negligible extra cost (same source rows,
same GROUP BY).

## 2. Test split only, and why

**CxG's own public numbers are test-split only, confirmed live in
`cxg_coverage.py`:** `BigQueryCxgCoverageStore` hardcodes `COVERAGE_SPLIT = "test"`
and every query filters `WHERE split = 'test'`, even though the underlying
`oam_ml.cxg_{track}_v3_predictions` tables carry all three splits. CxA's own
coverage reads (`cxa_models.py`) do the exact same thing, explicitly called out as
"decision 4" in that module's docstring. **This rollup matches that established
discipline exactly** -- the materialized table itself carries a `split` column for
all three splits (never baking the filter into the table, same reasoning both
precedents already state), but the read API (`GET /v1/analysis/quadrant-scatter`)
only ever queries `split='test'`.

## 3. Minimum sample size: flagged, not filtered

**Checked the one existing low-n precedent in this codebase (CxA+'s "Experimental"
badge)** before inventing a new rule. It is a **static, hand-authored caption
string** ("2,830 total chance-creating passes, 419 in test, zero Premier League
rows") plus a `Badge status="experimental"` -- not a dynamic per-item n-threshold
rule. There is no existing automated low-n gate anywhere in this project to reuse.

**Decision for this chart, made explicitly (no precedent to fall back on): flag,
don't filter.** A quadrant scatter is a scatter of *individual player-seasons* --
most chance-creating-pass counts are tiny (test-split CxA+ coverage is 419 passes
total, spread across hundreds of players), so a hard per-point filter would empty
the chart, not clean it up. Every point plots regardless of its `n`; points with
`min(n_x, n_y) < 5` render at reduced opacity, and every point's exact `n` for both
axes is always in its hover tooltip -- never hidden, never silently dropped. `5` is
this component's own new threshold, chosen as a round, conservative number with no
existing precedent to match; a future task could tune it.

## 4. Materialized table: `oam_serving.player_season_cxg_cxa_v1`

Naming matches the existing `oam_serving.cxa_{track}_combined_v1` convention
(`{subject}_v1`). **Grain: one row per `(player_id, competition_id, season_id,
split)`.** Built by `scripts/materialize_player_season_cxg_cxa_v1.py`, pure SQL
aggregation (no ML refit needed -- unlike the CxA combined scorer, every input here
is already a frozen, already-scored prediction table):

- `cxg_{event,plus}_n_shots` / `_mean` / `_total` / `_total_xg` / `_goals` -- rolled
  up from `oam_ml.cxg_{event,plus}_v3_predictions` joined to `oam_core.shots` (for
  player/competition/season identity -- the predictions table itself carries neither).
- `cxa_{event,plus}_n_passes_created` / `_mean` / `_total` -- rolled up from
  `oam_serving.cxa_{event,plus}_combined_v1` **restricted to
  `cxa_combined_score IS NOT NULL`** (chance-creating passes only -- a pass that
  never created a chance has no CxA value to average), joined to `oam_core.events`
  for the **passer's** identity (events, not shots -- a pass is not a shot).

**Null vs zero, enforced by construction:** `*_n_shots`/`*_n_passes_created` are
always a real integer (`COALESCE(..., 0)`), `*_mean`/`*_total`/`*_total_xg` are
left as raw `NULL` whenever their `_n` is 0 (they simply don't appear in a `LEFT
JOIN`'d subquery that has no group for that key) -- never coalesced to 0. A
player-season absent from all four metrics for a split isn't written at all (the
`keys` CTE is a `UNION DISTINCT` over only the four metrics' own real groups).

**A critical join-safety issue found and handled, not assumed away:**
`bigquery_store.py`'s own comment documents a real, previously-shipped incident: `
oam_core.shots`/`events` each carry 3 full lineage-versioned copies of every row
(one per `silver_schema_version`), and every query joining against them must filter
to the single active version (`statsbomb_silver_v1_2`) or silently triple-count.
**Checked live, not assumed, that this filter alone is sufficient** (a single
`data_version` value exists within that `silver_schema_version`, for both `shots`
and `events` -- confirmed via a direct `COUNT(DISTINCT data_version)` query,
returned `1` for both). The materialization script re-runs this exact check itself
before writing anything, and refuses to write if it ever stops holding.

## 5. Verification, before writing this doc

Ran the full rollup as a read-only `SELECT` against live BigQuery, checked before
running the actual `CREATE OR REPLACE TABLE`:

| Check | Result |
|---|---|
| `shots`/`events` distinct `data_version` within `statsbomb_silver_v1_2` | 1 / 1 (join-safety filter confirmed sufficient) |
| Rollup test-split player-season rows | 822 |
| `cxg_event`: rollup sum of `n_shots` vs source `COUNTIF(split='test')` | 2427 vs 2427 (exact match) |
| `cxg_plus`: same | 590 vs 590 (exact match) |
| `cxa_event`: rollup sum of `n_passes_created` vs source chance-creating test count | 1736 vs 1736 (exact match) |
| `cxa_plus`: same | 419 vs 419 (exact match, matches the CxA+ Experimental caption's own stated "419 in test") |
| Rows dropped to `player_id IS NULL` (spot-checked for `cxg_event`) | 0 |

All four exact matches confirm no join exploded or dropped rows beyond the
(zero, confirmed) null-player case. Table written: **3,635 total rows** (all three
splits), **822 test-split player-season rows**.

Live sanity check of the read path itself (`BigQueryQuadrantScatterStore` against
the real table, not just mocked unit tests): returned **822 rows** (matching the
materialization script's own count exactly), null-vs-zero preserved correctly on a
sampled row (`cxg_plus_n_shots=0` with `cxg_plus_mean=None`, not `0.0`), and
**455 of the 822 rows have both `cxg_event_mean` and `cxa_event_mean` covered** --
the pairing the frontend's own default X/Y selection uses, confirmed non-trivial.

Backend unit tests: `tests/api/test_quadrant_scatter.py`, 3/3 passing --
split-filter + null-vs-zero preservation, competition/season filter application,
per-scope caching. Full backend suite: 462 passed (pre-existing, unrelated
collection error in `tests/features/cxg/test_opponent_adjusted_family.py`
excluded, same finding as every prior CxA task).

## 6. API and frontend

**`GET /v1/analysis/quadrant-scatter`** (admin-gated, added to the existing
`routers/analysis.py` -- see section 7 on why Analysis is still the right home),
optional `competition_id`/`season_id` filters, always `split='test'`. Returns every
player-season row for the requested scope; the frontend picks which two of its
several metrics to plot, not the API.

**`web/components/analysis/QuadrantScatter.tsx`** -- hand-rolled inline SVG, same
pattern as every other chart in this project (`XgTimeline.tsx`, no charting library
pulled in). Dashed cross-hair lines at the league median of whichever two metrics
are selected (never a fixed 0/0 origin). A player-season plots only when **both**
selected metrics are non-null for it (absent, not shown at a fabricated zero, per
`PlayerSeasonQuadrantRow`'s own null-vs-zero contract) -- rows covered on only one
axis are simply left off that particular pairing.

Wired into `web/app/analysis/page.tsx` as a fourth tab ("Quadrant scatter"),
alongside the existing Feature browser / Rendered charts / Model results tabs --
confirming Analysis (not Explore) is still the right home, per the design spec's
own "originally scoped for Analysis" wording (Hard gate 2) and the component
inventory listing it as admin-tab chart, not an Explore-zone card. Two `<select>`
dropdowns (X axis, Y axis) drive the "build-your-own" pairing; defaults to
CxG (event) x CxA (event), the two full-coverage tracks.

## 7. What's still deferred

- **A dynamic n-threshold, tunable by the viewer** -- the `5`-shot/-pass dimming
  threshold is a fixed constant in `QuadrantScatter.tsx`, not a UI control. Adding
  one would be a small, low-risk follow-up if the flat default proves wrong in
  practice.
- **Competition/season filter controls on the Analysis tab UI** -- the API supports
  `competition_id`/`season_id` filters, but the panel doesn't yet expose them (it
  always requests the unfiltered, all-competitions/all-seasons scope). Every
  player-season is still its own row/dot regardless (a player who appears in two
  seasons gets two dots), so this is a missing convenience filter, not a data gap.
- **CxG+/CxA+ tracks' near-zero coverage** in this chart follows through
  mechanically from their own upstream coverage limits (documented at length in
  `cxa_dashboard_models_page_v1.md`/`cxa_detail_page_v1.md`) -- not a new caveat
  this task introduces, just inherited.
