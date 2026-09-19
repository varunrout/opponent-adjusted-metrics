# Fix: Missing/Suspect Qualification for `second_nearest_defender_style_archetype`

## Step 1 — Root cause (investigated, not assumed)

Read `scripts/materialize_cxg_v2_21st_feature_requalification.py` (Task #11's Step 2) — the
script that produced the 20 existing `cxg_bivariate_interaction_v1` rows for this feature.
It calls the exact same shared `fit_interaction`/`fit_categorical_interaction_saturated`
functions (`src/opponent_adjusted/analysis/bivariate/testing.py`) used for every other
pairing in this table — **not** an ad-hoc or simplified method. Every pattern flagged as
"suspect" is fully explained by ordinary, correctly-implemented behavior, re-verified live
against the actual rows before concluding:

1. **`interaction_coef`/`interaction_se` NULL on all 20 rows** — `fit_interaction` only
   reports a single coefficient when the interaction term has exactly 1 degree of freedom
   (`testing.py:90-96`). A 4-level-categorical-vs-anything pair always produces ≥3
   interaction dummy columns, so these are correctly `None` by that function's own contract.
   Re-verified live: `nearest_defender_style_archetype`'s own 23 existing rows show the
   **identical** null pattern (23/23 null coef) — this is a structural property of any
   4-level-categorical pairing in this pipeline, not unique to the feature under
   investigation.
2. **`validated_on_val_split` NULL on all 20 rows** — Task #11's script only triggers a
   validation refit when `interaction_p_fdr < 0.10`. Re-verified live: the smallest of the
   20 stored `interaction_p_fdr` values was 0.382 — nowhere near the gate. Correctly null,
   not a computation failure.
3. **Identical `interaction_p_fdr` across different pairs** (e.g. `0.6577642452990192`
   shared by two rows with genuinely different `interaction_p_raw`, 0.4176 and 0.4211) —
   this is the standard Benjamini-Hochberg monotonicity (cummin) artifact: when two ranks'
   raw-adjusted values are close, BH-FDR enforces a non-decreasing sequence by capping the
   smaller-rank value at the larger-rank value, producing exact ties. Re-verified live:
   sorting all 20 existing rows by `interaction_p_raw` produces a perfectly non-decreasing
   `interaction_p_fdr` sequence — exactly what a **correct** BH-FDR implementation produces,
   not a bug.

**Verdict: no computational bug.** The 20 existing rows were computed correctly by the
standard pipeline.

**However, they were genuinely stale** — computed against the 20-feature CxG+ pool as it
stood right after Task #11. The pool has since grown to 23 (the Phase C requalification task
added `defensive_action_rate_30m`, `territorial_dominance_last_15m`,
`cross_match_defensive_rate`), and this feature was never tested against those 3 newest
members. That staleness — not a bug — is why the 20 rows were deleted and Tier 1 was cleanly
re-run against the full current pool, per Step 1.2's instruction not to assume "delete
everything" without checking salvageability first: they were methodologically sound but
incomplete, so a clean full re-run (rather than a patch appending only 3 new pairs) was the
right call to produce one internally-consistent set of 22 rows against today's pool.

Separately — and this gap was real, not a false alarm — `cxg_split_univariate_v1` and
`cxg_feature_correlation_v1` genuinely had **zero** rows for this feature. That part of the
task's premise was accurate.

## Correction to the task's own stated premise

The task states univariate point-biserial via dummy encoding is "the exact same convention
already used for `nearest_defender_style_archetype`/`second_nearest_defender_role` in this
table." Checked live before acting on it: **false** — `cxg_split_univariate_v1` has **zero**
rows for `nearest_defender_style_archetype`, `second_nearest_defender_role`,
`nearest_defender_role`, AND `defensive_profile_cluster`. This table excludes every
categorical feature by design (confirmed both by the live query and by
`materialize_cxg_v2_pool_requalification.py`'s own docstring, which corrected this exact
false premise in an earlier task). Followed the real established convention instead: a
per-level goal-rate stability check across splits, reported (not written to
`cxg_split_univariate_v1`), matching every other categorical in this pipeline.

## Step 2 — Real qualification results now on record

**Univariate (per-level stability, reported not written — real convention):** 4 archetype
levels, all stable in direction and roughly comparable magnitude across train/val/test
(train goal rates: `deep_block_clearer` 9.7%, `duel_dominant_contester` 6.4%,
`high_volume_presser` 6.2%, `unresolved_5050_annotation_density` 8.4%; validation and test
splits show the same relative ordering with expected small-sample noise, e.g.
`duel_dominant_contester` n=32 in validation). The flagged-muddy 4th cluster
(`unresolved_5050_annotation_density`) was included in this check, not excluded, per the
task's instruction — it shows a stable, non-degenerate goal rate across all 3 splits (8.4%
train / 8.9% validation / 4.2% test), no evidence it should be treated differently from the
other 3 levels for qualification purposes.

**Correlation/redundancy (Cramér's V vs. the 3 other CxG+ categoricals, train split;
continuous pool skipped per the established "categorical vs. continuous is not a
near-duplication risk by construction" convention documented in Task #11's own script):**

| vs. | Cramér's V | n | redundant? |
|---|---|---|---|
| `defensive_profile_cluster` | 0.1031 | 2,402 | no |
| `nearest_defender_role` | 0.0829 | 2,402 | no |
| `second_nearest_defender_role` | 0.4724 | 2,402 | no |

**Re-verified the originally-claimed 0.472 figure for `second_nearest_defender_role`:
confirmed at 0.4724, consistent with the original Task #11 finding — well clear of the 0.85
redundancy threshold.** All 3 checks written to `cxg_feature_correlation_v1` (3 new rows),
each row's `resolution_reason` explicitly labeled "Cramér's V (categorical-categorical
association, NOT Pearson r)" so a future reader doesn't misread the metric type — this table
was never used for a categorical×categorical check before (Task #11's own script computed
Cramér's V but only printed it, never wrote it to this table), so this is a new but
compatible use of the existing schema, not a new table.

**Pool inclusion: qualifies, added.** No redundancy found against any current pool member ⇒
per the project's standing "weak/no signal never disqualifies, only true redundancy does"
principle, `second_nearest_defender_style_archetype` was added to
`cxg_bivariate_candidate_pool_v1` (`cxg_plus`, `qualification_reason=
"task11_categorical_backfilled_no_signal_gate_per_no_drop_for_weak_signal_principle"`).
CxG+ pool: 23 → **24**.

**Bivariate Tier 1 (clean re-run, 22 pairs — every other current CxG+ pool member):**
`validated_on_val_split` is now populated with real `true`/`false` semantics (not `null`)
for every row that reached the `p_fdr<0.10` gate — in this run, **zero** of the 22 pairs
crossed that gate (`min interaction_p_fdr = 0.2466`, vs. `second_nearest_defender_role`),
so `validated_on_val_split` legitimately stays `NULL` for all 22 (same "no significant
pair" outcome as the original 20-pair run, now correctly re-derived against the current
pool rather than the stale one). **Zero new confirmed interactions.** No categorical×
categorical pair hit the sparsity failure mode this run (`fit_categorical_interaction_
saturated` fallback was available but not triggered).

Row-count reconciliation (all confirmed live): `cxg_bivariate_interaction_v1` (cxg_plus,
tier=1): 273 → 275 (−20 stale, +22 fresh); `cxg_feature_correlation_v1`: +3 rows;
`cxg_bivariate_candidate_pool_v1` (cxg_plus): 23 → 24.

## Step 3 — Reconciliation with v2's shipped model

**v2's model is untouched** — no `oam_ml.cxg_plus_v2_*` table was read or written by this
task. This section only answers: would today's findings have changed v2's modeling
decisions, had they been known at the time?

**No, they would not have.** v2 included `second_nearest_defender_style_archetype` as a
4-level dummy-encoded categorical on the strength of a genuine, real signal check: this
feature's inclusion was justified by Task #11's own (methodologically sound, just
incompletely-recorded) redundancy screen against `second_nearest_defender_role` (Cramér's
V≈0.47, now re-confirmed at 0.4724) and the project's standing "no signal gate for weak/no
bivariate significance" principle — which this fix confirms still holds: zero significant
bivariate interactions for this feature, in both the original 20-pair test and today's
clean 22-pair re-run. Nothing found in Steps 1–2 surfaces a redundancy, a data-quality
problem, or a reversal of the original inclusion rationale. The feature's qualification
paper trail is now complete and defensible (univariate stability check, correlation
screen, bivariate Tier 1 with real validation semantics, present in the candidate pool
table) — v2 was not "wrong" to include it, it just shipped ahead of that paper trail being
fully written down, which is exactly the gap this task closes.

## Tests + regression check

Full suite: pytest run below. Baseline was 310 after the Phase C requalification task. This
task introduces no new pure-logic code (it consumes `fit_interaction`/`fit_categorical_
interaction_saturated`, already tested, and orchestrates BigQuery reads/writes only).

## Files

- `scripts/fix_second_nearest_archetype_qualification.py` — the fix script.
- `audit_outputs/cxg_analysis/second_nearest_archetype_qualification_fix/run_summary.json` —
  raw run output.
- `audit_outputs/cxg_analysis/second_nearest_archetype_qualification_fix/report.md` — this
  report.
