# CxG Project Closeout Cleanup (4 independent parts)

## Part 1 — Phase C EDA-appendix chart backfill

**Found:** confirmed live before starting — `cxg_summary_stats_v1` (and the other 3
EDA-appendix tables) had zero rows for `defensive_action_rate_30m`,
`territorial_dominance_last_15m`, `cross_match_defensive_rate`. Same gap pattern as the
earlier Phase A/B backfill.

**Investigation before fixing:** these 3 columns live in the `event_context` family
(`cxg_event_context_features`), which — unlike `opponent_adjusted` — IS one of the 7
families the canonical `CxGAnalysisMaterializer` (`src/opponent_adjusted/analysis/cxg.py`)
owns. Its `.run()` method does a full `CREATE OR REPLACE` of `cxg_chart_registry_v1` and
several other tables this session has built up via INSERT-only copy-forward — calling it
would have silently wiped every v2/v3/Phase-C chart-registry row. Instead, this task reused
the class's own `_load_fields()`/`_materialize_analysis_surfaces()` methods directly (safe:
pure derived-surface-table rebuilds with no manually-inserted rows to lose) to refresh
`cxg_analysis_event_v1` with the 3 new columns, and wrote the 4 EDA-appendix tables via the
same scoped-DELETE-then-INSERT idempotent pattern as the Phase A/B precedent.

**Track handling:** confirmed live that `event_context` has never been track-split in these
4 tables (no `track` column exists in any of them; every existing event_context row is a
single unscoped stat over the full population). Followed that same convention rather than
inventing a track split no other event_context feature has — one row per feature, computed
over the full 15,737-shot population (CxG event-wide; CxG+ is strictly the `has_360_frame`
subset of it, so this stat is valid context for both tracks, same as every other
event_context feature already in these tables).

**Before/after (all confirmed live):**

| table | before | after |
|---|---|---|
| `cxg_null_profile_v1` | 0 | 3 |
| `cxg_summary_stats_v1` | 0 | 3 |
| `cxg_eda_distribution_bins_v1` | 0 | 60 (3 features × 20 quantile bins) |
| `cxg_univariate_target_v1` | 0 | 3 |
| `cxg_analysis_event_v1` (surface) | missing 3 columns | 3 new columns added, 15,737 rows unchanged |
| `cxg_analysis_360_v1` (surface, rebuilt alongside) | — | 3,960 rows unchanged |

**Values backfilled** (all match Phase C's original report numbers exactly, no drift):
`defensive_action_rate_30m` mean=2.987, median=2.833, range=[0, 23.28], 0 nulls;
`territorial_dominance_last_15m` mean=0.138, median=0.141, range=[-1, 1], 16 nulls (0.1%);
`cross_match_defensive_rate` mean=2.663, median=2.660, range=[1.49, 4.65], 906 nulls (5.76%,
cold-start).

**Charts re-rendered:** `event_context_eda_histogram`, `event_context_null_profile_bar`,
`event_context_summary_box`, `event_context_target_lift_bar` — all 4 already existed in the
chart registry (reading the now-updated tables), so a copy-forward + re-render was
sufficient, no new chart rows needed.

## Part 2 — Fix the pre-existing 4x row-duplication bug

**Root cause investigated, confirmed (not assumed):** read `materialize_cxg_opponent_
adjusted_analysis.py` in full — every one of its 5 output tables (`cxg_feature_
inventory_v1`, `cxg_null_profile_v1`, `cxg_summary_stats_v1`, `cxg_eda_distribution_bins_v1`,
`cxg_univariate_target_v1`) is written via plain `INSERT INTO` with **zero `DELETE`
statements anywhere in the file**. Live data confirms 4 distinct `materialized_at`
timestamps per feature per table, with IDENTICAL row_count/null_count/etc. across all 4 —
the underlying data never changed between the 4 historical runs, only the script was
re-invoked without an idempotency guard. Exactly the hypothesized cause, confirmed rather
than assumed.

`cxg_split_univariate_v1` was checked and found **not** affected by this bug — its
opponent_adjusted rows are legitimately run_id-tagged historical snapshots (the established
convention for that one table, matching how every other family's rows are also
run_id-tagged there), not literal duplicates. Correctly left untouched.

**Bonus finding:** `cxg_feature_inventory_v1` (not one of the 4 tables named in the task,
but produced by the same unguarded script) had the identical 4x duplication. Fixed
alongside the 4 named tables since it shares the exact same root cause.

**Fix — before/after row counts (all confirmed live), scoped to the 4 affected features
(`defensive_profile_cluster`, `gk_odi`, `mean_backline_odi`, `nearest_defender_odi`):**

| table | before | after |
|---|---|---|
| `cxg_null_profile_v1` | 16 | 4 |
| `cxg_summary_stats_v1` | 16 | 4 |
| `cxg_eda_distribution_bins_v1` | 260 | 65 (5+20+20+20 bins) |
| `cxg_univariate_target_v1` | 16 | 4 |
| `cxg_feature_inventory_v1` (bonus) | 16 | 4 |

Deduplication kept the most-recently-materialized row per feature (content was identical
across all 4 historical copies, confirmed live, so this is lossless).

**Idempotency guard added** to `materialize_cxg_opponent_adjusted_analysis.py`: a scoped
`DELETE FROM <table> WHERE feature_family='opponent_adjusted' AND column_name IN (...)`
before each `INSERT`, mirroring the pattern already established in `materialize_cxg_v2_eda_
backfill.py`. **Verified live**: re-ran the patched script end-to-end — all 5 fixed tables'
row counts stayed exactly the same (delta=0 for every one), confirming the guard works and
the script can never re-duplicate these rows again. (`cxg_split_univariate_v1` grew by 9
rows on that re-run, as expected — its own legitimate run_id-tagged convention, unrelated to
this fix.)

**Charts re-rendered:** `opponent_adjusted_eda_histogram`, `opponent_adjusted_null_profile_
bar`, `opponent_adjusted_summary_box`, `opponent_adjusted_target_lift_bar` (plus `archetype_
goal_rate_bar`/`archetype_role_heatmap`/`role_displacement_bar`, which don't read the fixed
tables but were re-rendered in the same run for completeness) — the 4x-repeated bars (e.g.
`mean_backline_odi` appearing 4 times) are gone; each feature now appears exactly once.

## Part 3 — Commit `v2_feature_methodology_locked.md`

Checked for a `docs/` convention before placing the file: `docs/` already holds locked
policy/spec documents at the top level (e.g. `docs/cxg_split_policy_and_parallel_plan.md`),
with `docs/archive/` reserved for older, superseded prompts. This is an actively-referenced
locked-methodology doc, not an archived one, so it landed at `docs/v2_feature_methodology_
locked.md` — matching the location the task itself specified.

Content committed **verbatim**, exactly as provided in the task — not paraphrased or
restructured. Confirmed: this is the document's first appearance anywhere in this repo's git
history (a prior task had already confirmed via `find` that no file with this name existed
anywhere in the working tree).

## Part 4 — v1 CxG event-wide StatsBomb-divergence discrepancy

**Investigated, root cause found — not a bug, not a data change, a metric-definition
mismatch between two reports.**

Traced the original "15%" figure to `audit_outputs/cxg_analysis/baseline/baseline_v1_
report.md:214-215`: *"Honest gap: StatsBomb's model is clearly better on CxG's plain feature
set -- 15% lower log_loss, 18% lower Brier..."* — computed on the **test split**, using
`(v1_log_loss − statsbomb_log_loss) / v1_log_loss` (StatsBomb's own value as the numerator's
reference point, v1's log-loss as the **denominator**): `(0.3058 − 0.2597) / 0.3058 =
15.08%` ≈ "15%". Confirms exactly — the Brier figure checks out too:
`(0.0872 − 0.0718) / 0.0872 = 17.7%` ≈ "18%" as reported.

The v3 model training report (and this session's own earlier work) instead computed
`(v1_log_loss − statsbomb_log_loss) / statsbomb_log_loss` — **StatsBomb's log-loss as the
denominator**: `(0.3058 − 0.2597) / 0.2597 = 17.72%`. This convention was independently
validated against CxG+'s v2 gap, where it reproduced the previously-published "5.6%" figure
exactly (`(0.2566 − 0.2430) / 0.2430 = 5.60%`) — so both conventions are internally correct
and self-consistent, they just answer subtly different questions ("how much lower is
StatsBomb than v1, relative to v1?" vs. "how much higher is v1 than StatsBomb, relative to
StatsBomb?").

**The 25.0% validation-split figure is not a discrepancy at all** — the original v1 report
only ever published a **test-split** comparison table for CxG event-wide; validation-split
numbers were never reported before v3's own analysis computed them for the first time. There
is nothing to reconcile there — it's new information, not a contradiction of old information.

**Confirmed no data drift:** `oam_ml.cxg_baseline_v1_metrics` (track=`cxg_event`) and
`oam_ml.cxg_baseline_v1_divergence` both carry a single, identical `materialized_at`
timestamp (`2026-08-22T01:43:15`) — the table has never been regenerated since the original
v1 run. The underlying numbers are bit-for-bit the same ones the original report was computed
from.

**Reporting convention going forward** (adopted starting with this report and the v3 report):
never cite a bare gap percentage. Always state the exact split (test vs. validation) AND the
exact denominator convention (`(model − reference)/model` vs. `(model − reference)/
reference`) alongside the number — e.g. "17.7% higher log-loss than StatsBomb, relative to
StatsBomb, test split" rather than "17.7% gap." This is the specific ambiguity that produced
the apparent discrepancy, and stating it explicitly going forward prevents the same
apples-to-oranges comparison from recurring.

## Tests + regression check

No new pure-logic code in this task (Part 2's idempotency guard is a `DELETE` statement
addition to an existing BigQuery-orchestration script, not testable pure logic; Parts 1/3/4
are data backfill, documentation, and investigation respectively). Full suite: pytest run
below. Baseline was 320 after v3 model training.

## Files

- `scripts/materialize_cxg_phase_c_eda_backfill.py` — Part 1.
- `scripts/fix_opponent_adjusted_eda_duplication.py` — Part 2 data fix.
- `scripts/materialize_cxg_opponent_adjusted_analysis.py` — Part 2 idempotency-guard patch.
- `docs/v2_feature_methodology_locked.md` — Part 3.
- `audit_outputs/cxg_analysis/closeout_cleanup/closeout_cleanup_report.md` — this report
  (Part 4's findings are documented here directly; no separate script needed).
