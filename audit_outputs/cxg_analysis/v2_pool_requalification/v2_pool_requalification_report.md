# CxG+ Governed Analysis Chain — Re-run on the Enlarged (v2) Pool

Re-runs Univariate -> Correlation/Redundancy -> PCA -> Bivariate Tier 1 for the **CxG+
track only**, over the pool enlarged by Phase A (4 geometric/categorical defender
features) and Phase B (1 defender-style archetype feature). CxG (event-wide) track is
completely untouched. Tiers 2/3/4 of bivariate testing are skipped (out of scope; their
existing rows are preserved unchanged, not re-run).

Run artifact: `scripts/materialize_cxg_v2_pool_requalification.py`. Charts refreshed under
run_id **`cxg-analysis-20260822T043638Z`**.

---

## Step 0: confirmation

Re-queried `oam_features.cxg_defensive_360_features` live. Null rates match exactly what
Phase A/B reported:

| Column | Reported | Confirmed live |
|---|---|---|
| `nearest_defender_role` | ~2.3-2.6% | 2.35% |
| `nearest_defender_zone_displacement` | ~2.3-2.6% | 2.35% |
| `second_nearest_defender_role` | ~2.3-2.6% | 2.55% |
| `nearest_defender_gap` | ~2.3-2.6% | 2.55% |
| `nearest_defender_style_archetype` | ~10.2% | 10.18% |

No discrepancy. Proceeded.

---

## Two corrections to the task's stated premises (evidenced, not assumed)

Before executing, both of the following were verified directly against the actual scripts
rather than taken on faith, per this project's standing convention of surfacing rather than
silently working around a mismatched premise:

**1. `defensive_profile_cluster` has no per-level point-biserial rows in
`cxg_split_univariate_v1`.** The task asked to "replicate the exact convention already used
for the existing categorical feature `defensive_profile_cluster` in this table" (implying a
dummy-encoded per-level convention exists there). Checked
`materialize_cxg_opponent_adjusted_analysis.py` directly: `defensive_profile_cluster` is
explicitly **excluded** from the numeric point-biserial loop, with an inline comment stating
it is "not eligible for the numeric point-biserial univariate table by construction." The
real, established convention is: categoricals are excluded from `cxg_split_univariate_v1`
entirely, and their signal is instead expressed via the candidate-pool table directly
(`qualification_reason='categorical_proven_stable'`) and via dummy-encoded bivariate
interaction testing. **Followed the real convention**, not the task's stated (incorrect)
description of it: the 2 new continuous features got standard point-biserial rows; the 3 new
categoricals did not, and instead got a level-by-level cross-split stability check (below),
matching how `defensive_profile_cluster`'s own stability was originally established
("Phase 2's own descriptive analysis... train-only goal-rate separation replicated across
splits").

**2. "ODI is dropped from v2 per the locked methodology" — verified, not assumed.**
Confirmed directly from Phase B's own report
(`audit_outputs/cxg_analysis/phase_b_defender_style_clustering/phase_b_report.md`): Phase B
"replaces the deprecated ODI (`nearest_defender_odi`) defender-quality signal," built
specifically because **ODI's original computation consumed `statsbomb_xg`** — circular with
this project's own xG benchmarking. This is a real, well-justified methodology decision, not
an assumption on this task's part. The v2 pool therefore drops `gk_odi`, `mean_backline_odi`,
`nearest_defender_odi` entirely (not carried forward, not re-tested).

---

## Step 1: univariate for the 5 new candidates

### 2 continuous features (added to `cxg_split_univariate_v1`, `feature_family='opponent_adjusted'`)

| Feature | train r | validation r | test r | Sign-stable? |
|---|---|---|---|---|
| `nearest_defender_zone_displacement` | -0.0394 | -0.0515 | -0.0340 | **Yes** — negative in all 3 splits, small but consistent magnitude |
| `nearest_defender_gap` | -0.0067 | -0.0011 | -0.0188 | No meaningful signal — all three near zero |

Both weak, as expected for brand-new geometric candidates. Per the project's non-negotiable
principle, **neither is excluded from the pool for weak signal** — only ≥0.85 redundancy
would justify trimming, and neither triggered that (see Step 2).

### 3 categorical features (stability check, not written to `cxg_split_univariate_v1` — see correction above)

**`nearest_defender_role`** — real, directionally consistent separation across splits, GK
level noisiest (small n: 64/14/19):

| Level | train n / rate | val n / rate | test n / rate |
|---|---|---|---|
| GK | 64 / 0.297 | 14 / 0.429 | 19 / 0.211 |
| CB | 964 / 0.118 | 188 / 0.117 | 177 / 0.107 |
| Fullback_WingBack | 530 / 0.104 | 120 / 0.150 | 129 / 0.078 |
| Midfield | 895 / 0.060 | 189 / 0.069 | 179 / 0.067 |
| Attack | 267 / 0.056 | 65 / 0.031 | 67 / 0.045 |

GK consistently highest, Attack/Midfield consistently lowest, in every split — the ordering
holds even though GK's exact magnitude swings (small-n noise, not instability of direction).

**`second_nearest_defender_role`** — same general shape, but `Fullback_WingBack` is
genuinely less stable here (0.087 train -> 0.138 val -> 0.040 test, not monotonic, real
swing) — worth flagging honestly rather than glossing over:

| Level | train n / rate | val n / rate | test n / rate |
|---|---|---|---|
| GK | 165 / 0.309 | 37 / 0.243 | 29 / 0.241 |
| CB | 905 / 0.101 | 185 / 0.124 | 182 / 0.137 |
| Fullback_WingBack | 482 / 0.087 | 94 / 0.138 | 101 / 0.040 |
| Midfield | 911 / 0.066 | 199 / 0.070 | 189 / 0.048 |
| Attack | 252 / 0.052 | 60 / 0.033 | 68 / 0.044 |

**`nearest_defender_style_archetype`** — including the flagged-muddy 4th cluster
(`unresolved_5050_annotation_density`) as instructed, not excluded:

| Level | train n / rate | val n / rate | test n / rate |
|---|---|---|---|
| `deep_block_clearer` | 1,079 / 0.120 | 239 / 0.117 | 212 / 0.085 |
| `high_volume_presser` | 877 / 0.064 | 214 / 0.079 | 177 / 0.068 |
| `unresolved_5050_annotation_density` (muddy) | 368 / 0.071 | 59 / 0.068 | 94 / 0.064 |
| `duel_dominant_contester` | 168 / 0.060 | 34 / 0.088 | 36 / 0.056 |

**Honest finding on the muddy cluster, reported as instructed rather than pre-judged**: the
muddy 4th cluster's goal rate is actually **the most stable of the four across splits**
(0.071 / 0.068 / 0.064 — tighter range than any of the three "clean" labels), and it sits in
a clearly distinct middle tier — separated from `deep_block_clearer`'s higher rate, similar
to `high_volume_presser`/`duel_dominant_contester`. Despite being flagged as not cleanly
interpretable by dominant action-type (Phase B's own labeling criterion), it does **not**
show degenerate or noisy behavior in this goal-rate check — a real, modest, worth-reporting
finding, not a null result and not an artifact.

---

## Step 2: correlation / redundancy re-screen

Pearson correlation over the 16-feature numeric union pool (14 existing + 2 new; the 3 ODI
features and 4 categoricals are outside the numeric matrix, following the same convention as
the original screen), train split only, `cxg_feature_correlation_v1` (120 pairs = C(16,2)).

**All 5 prior redundant pairs carried forward unchanged** (their member features and
correlations don't change with pool enlargement): `defensive_reset_index` /
`rest_defence_reset_fraction`, `defensive_compactness` / `defensive_hull_area`,
`gk_distance_to_shooter` / `shot_x_sb` (not in this pool anyway), `pre_shot_receiver_space` /
`shooter_space_previous_linked_event`, `defensive_centroid_x` / `defensive_line_depth`.

**No new ≥0.85 pair surfaced** among the 16-feature matrix, including the two pairs the task
specifically flagged as real overlap risks:

| Pair | r (train) | ≥0.85? |
|---|---|---|
| `nearest_defender_gap` vs. `defenders_within_5m` | **-0.450** | No |
| `nearest_defender_gap` vs. `defenders_within_8m` | **-0.376** | No |
| `nearest_defender_zone_displacement` vs. `defensive_line_depth` | **+0.302** | No |

Checked, not assumed: `nearest_defender_gap` does correlate moderately negatively with local
defender density (more defenders nearby -> smaller gap to the 2nd-nearest, sensible
direction), but at r=-0.45 it's measuring something meaningfully different, not a
near-duplicate. `defensive_line_depth` and `zone_displacement` share only a weak positive
relationship (deeper defensive lines correlate mildly with bigger positional displacement) —
also not redundant. Note `defenders_within_5m`/`defenders_within_8m` are **not themselves
pool members** (never part of the qualified 18/20-candidate list), so even a high r there
wouldn't trigger the formal drop mechanism — this was a due-diligence sanity check only,
reported honestly either way.

**Final enlarged CxG+ candidate pool: 20 members** (`cxg_bivariate_candidate_pool_v1`):

*16 numeric:* `last_action_interval_s`, `defenders_between_ball_and_goal`,
`defensive_reset_index`, `nearest_defender_distance_delta`, `pre_shot_receiver_space`,
`gk_distance_to_shooter`, `defensive_line_depth`, `defensive_width`,
`estimated_goalface_occlusion`, `goal_mouth_defender_count`, `max_goal_exposure`,
`min_defensive_compactness_sequence`, `shot_corridor_occlusion`, `visible_goal_angle_proxy`,
**`nearest_defender_zone_displacement`**, **`nearest_defender_gap`**.

*4 categorical:* `defensive_profile_cluster`, **`nearest_defender_role`**,
**`second_nearest_defender_role`**, **`nearest_defender_style_archetype`**.

(bold = new this round; `gk_odi`/`mean_backline_odi`/`nearest_defender_odi` dropped, per the
verified methodology correction above.)

---

## Step 3: PCA re-run

| | Before (17 features) | After (16 features) |
|---|---|---|
| Components needed for 80% cumulative variance | 9 | **8** |
| Ratio | 9/17 = 52.9% | 8/16 = 50.0% |

A real, modest shift — reported as measured, not assumed unchanged. Roughly the same degree
of collapse proportionally (about half the features needed), consistent with the pool still
being a genuinely heterogeneous mix of geometry/occlusion/timing/positioning measurements,
not a redundant one (matches Step 2's finding of zero new redundant pairs).

Top-5 PC1 loadings (explains 25.5% of variance, the largest single component):

| Feature | Loading |
|---|---|
| `shot_corridor_occlusion` | +0.394 |
| `defenders_between_ball_and_goal` | +0.392 |
| `estimated_goalface_occlusion` | +0.352 |
| `defensive_reset_index` | -0.341 |
| `gk_distance_to_shooter` | +0.341 |

PC1 reads as a general "defensive presence/occlusion vs. defensive disorganization" axis —
consistent with the original pool's PC1 story, not disrupted by the 2 new continuous
features (neither cracks the top-5 PC1 loadings, though both are represented in the full
16-feature loading set persisted to `cxg_pca_loadings_v1`).

---

## Step 4: bivariate Tier 1 re-run

190 pairs tested (C(20,2) over the full 16-numeric + 4-categorical pool), BH-FDR correction
within the cxg_plus track (same discipline as before — cxg_event untouched, its own 10
Tier-1 rows from the prior round are unaffected).

**19 pairs clear FDR<0.10** (up from 13 in the original round). Full list, validation status:

| feature_a | feature_b | p_fdr | validated |
|---|---|---|---|
| `defensive_profile_cluster` | `visible_goal_angle_proxy` | 1.7e-10 | **True** |
| `defensive_profile_cluster` | `shot_corridor_occlusion` | 2.9e-04 | False |
| `defensive_profile_cluster` | `estimated_goalface_occlusion` | 5.3e-03 | False |
| `defensive_width` | `last_action_interval_s` | 7.1e-03 | False |
| `defensive_profile_cluster` | `goal_mouth_defender_count` | 8.0e-03 | False |
| `defensive_profile_cluster` | `gk_distance_to_shooter` | 1.23e-02 | **True** |
| `nearest_defender_role` | `visible_goal_angle_proxy` | 1.23e-02 | False |
| `defensive_profile_cluster` | `defensive_reset_index` | 2.38e-02 | False |
| `defenders_between_ball_and_goal` | `defensive_profile_cluster` | 2.40e-02 | False |
| `defensive_profile_cluster` | `nearest_defender_zone_displacement` | 2.40e-02 | **True** |
| `nearest_defender_gap` | `visible_goal_angle_proxy` | 2.40e-02 | **True** |
| `pre_shot_receiver_space` | `visible_goal_angle_proxy` | 2.40e-02 | **True** |
| `defensive_width` | `pre_shot_receiver_space` | 3.83e-02 | False |
| `defensive_line_depth` | `pre_shot_receiver_space` | 3.84e-02 | **True** |
| `defensive_profile_cluster` | `nearest_defender_gap` | 5.30e-02 | False |
| `defensive_reset_index` | `nearest_defender_role` | 5.68e-02 | False |
| `nearest_defender_style_archetype` | `visible_goal_angle_proxy` | 6.02e-02 | False |
| `defensive_line_depth` | `nearest_defender_role` | 7.47e-02 | False |
| `shot_corridor_occlusion` | `visible_goal_angle_proxy` | 7.47e-02 | False |

**6 pairs are confirmed** (FDR<0.10 AND validated on the held-out split) — 4 carried
forward, 2 genuinely new:

### The 4 previously-locked interactions — explicit re-verification, not assumed

| Pair | Status | p_fdr this round |
|---|---|---|
| `defensive_profile_cluster` x `visible_goal_angle_proxy` | **STILL CONFIRMED** | 1.7e-10 |
| `defensive_profile_cluster` x `gk_distance_to_shooter` | **STILL CONFIRMED** | 1.23e-02 |
| `pre_shot_receiver_space` x `visible_goal_angle_proxy` | **STILL CONFIRMED** | 2.40e-02 |
| `defensive_line_depth` x `pre_shot_receiver_space` | **STILL CONFIRMED** | 3.84e-02 |

**All 4 survived the enlarged pool** — the additional 20 vs. 163 tests changes the FDR
denominator/correction slightly, but none of the 4 dropped out.

### 2 genuinely new confirmed interactions (involving Phase A features)

- **`defensive_profile_cluster` x `nearest_defender_zone_displacement`** (p_fdr=0.024,
  validated) — the defensive-shape archetype interacts with how far the nearest defender is
  from their typical zone. A sensible pairing: both are about defensive positioning/shape,
  from two independently-built feature families, now shown to carry joint signal beyond
  either alone.
- **`nearest_defender_gap` x `visible_goal_angle_proxy`** (p_fdr=0.024, validated) — spacing
  between the two nearest defenders interacts with how open the shot angle is. Also
  sensible: a wide defender gap likely matters more/less depending on how much of the goal
  is already visible.

`nearest_defender_style_archetype x visible_goal_angle_proxy` came close (p_fdr=0.060) but
did not clear FDR<0.10 and was never validation-tested as a result — a real, honest negative
for this specific pair, not evidence against the archetype feature generally (it still shows
up as informative in the univariate stability check above).

**Tier 2/3/4 preserved exactly, not re-run**: 15 (Tier 2) + 1 (Tier 3) + 3 (Tier 4) = 19
`cxg_plus` rows read back before the coarse track-scoped delete and re-inserted unchanged
alongside the 190 fresh Tier 1 rows. `cxg_bivariate_stratified_v1`'s 24 existing `cxg_plus`
rows were never touched at all (no delete issued against that table).

---

## Row-count reconciliation

| Table | Check | Result |
|---|---|---|
| `cxg_split_univariate_v1` | +6 new rows (2 features x 3 splits), cxg_plus total | 267 (includes all pre-existing rows, untouched) |
| `cxg_feature_correlation_v1` | C(16,2) = 120, cxg_plus | 120 |
| `cxg_bivariate_candidate_pool_v1` | 16 + 4 = 20, cxg_plus | 20 |
| `cxg_pca_components_v1` | 16 components, cxg_plus | 16 |
| `cxg_pca_loadings_v1` | 8 components x 16 features = 128, cxg_plus | 128 |
| `cxg_bivariate_interaction_v1` | 190 (tier1) + 19 (preserved tier2-4) = 209, cxg_plus | 209 |
| `cxg_bivariate_interaction_v1` | cxg_event, untouched | 10 (unchanged) |
| `cxg_bivariate_stratified_v1` | cxg_plus, untouched | 24 (unchanged) |

All match.

---

## Chart coverage

Registered under run_id **`cxg-analysis-20260822T043638Z`** (copy-forward of the prior
39-row batch via the existing generic `register_chart_registry_for_run.py` -- no new chart
types were needed, since `feature_correlation_heatmap`, `pca_scree`, and
`bivariate_significance_grid` already query their backing tables live and automatically
reflect the enlarged pool once re-rendered). All 39 charts rendered locally first
(`--skip-upload`), verified (file sizes for the 3 affected charts grew, consistent with
larger matrices: correlation heatmap 143KB PNG, significance grid 152KB PNG, both up from
the prior round), then uploaded to GCS and registered via the existing scoped
delete-then-insert-by-run_id pattern. `cxg_chart_registry_v1` history intact across all
prior run_ids (24/27/27/31/35/39/39/39 rows).

---

## Test suite

`python -m pytest -q` -> **278 passed**, no regressions (unchanged from the pre-task
baseline — this task is a data-pipeline re-run over already-tested statistical modules
(`bivariate/testing.py`'s `fit_interaction`/`validates_on_split`, sklearn's `PCA`), not new
pure logic, so no new unit tests were added; correctness was verified via the live BigQuery
run and the row-count/value checks throughout this report instead).

---

## What was explicitly NOT done (per task constraints)

- CxG (event-wide) track untouched — 10 Tier-1 rows, all univariate/correlation/PCA rows for
  that track, completely unmodified.
- Tiers 2/3/4 not re-run — preserved exactly via read-before-delete.
- The muddy 4th archetype cluster was not excluded from any test — included throughout,
  reported honestly (real, stable, non-degenerate signal found).
- The 4 previously-locked interactions were not assumed to still hold — explicitly
  re-verified, all 4 confirmed.
- No parallel `_v2` table names created — every table listed above is the same canonical
  name, overwritten/extended in place per its established convention.
- Phase A/B's own feature-computation logic, the frozen K-Means defensive-profile
  clustering, and ODI's code were not touched — this task only consumed their output.
- No push to git remote.

---

## Summary for hand-off

**Final enlarged CxG+ candidate pool (20 members):** 16 numeric (14 existing + 2 new:
`nearest_defender_zone_displacement`, `nearest_defender_gap`) + 4 categorical
(`defensive_profile_cluster` existing, plus 3 new: `nearest_defender_role`,
`second_nearest_defender_role`, `nearest_defender_style_archetype`). ODI trio dropped
(verified methodology decision, not an assumption).

**All 4 previously-locked Tier 1 interactions survived the re-run**, unchanged in
confirmation status. **2 new confirmed interactions** emerged directly from the Phase A
features (`defensive_profile_cluster x nearest_defender_zone_displacement`,
`nearest_defender_gap x visible_goal_angle_proxy`), bringing the total to 6 confirmed
CxG+ Tier 1 pairs. PCA collapse shifted from 9-of-17 to 8-of-16 components for 80%
variance — a real, modest, reported-as-measured change. No new redundant (≥0.85) pairs
found anywhere in the enlarged pool, including both flagged overlap-risk pairs.

---

## Addendum: fixing 2 NULL `interaction_p_fdr` Tier 1 rows

Two CxG+ Tier 1 rows (`defensive_profile_cluster x nearest_defender_role` and
`nearest_defender_role x second_nearest_defender_role`) had `fit_status='fit_failed'` and
therefore `interaction_p_fdr IS NULL` — a real coverage gap in the round above, not a null
finding. Fixed as its own small, scoped follow-up
(`scripts/fix_cxg_v2_categorical_interaction_nulls.py`).

### Root cause (confirmed via live cross-tab, not assumed)

Both are categorical x categorical pairs. Pulled the real train-split cross-tab cell counts
(`defensive_profile_cluster` x `nearest_defender_role`, and `nearest_defender_role` x
`second_nearest_defender_role`, joining `cxg_defensive_360_features` to `is_goal`) and found
genuine sparse/structurally-empty cells:

- **`nearest_defender_role` x `second_nearest_defender_role` has a literal zero-count `GK` x
  `GK` cell** (n=0 in the train split). This isn't incidental sparsity — a team has exactly
  one goalkeeper, so the nearest and second-nearest defenders in a freeze frame can never
  both be the GK. A structurally impossible combination, guaranteed to recur in every split.
  A zero-count cell makes the corresponding interaction-dummy column in the design matrix
  all-zero, which is rank-deficient (singular) for MLE.
- Several other cells are small with all-zero or all-nonzero outcomes, e.g.
  `defensive_profile_cluster=cluster_0` x `nearest_defender_role=GK` (n=4, 0 goals) and
  `nearest_defender_role=Fullback_WingBack` x `second_nearest_defender_role=Fullback_WingBack`
  (n=16, 0 goals) — small enough to drive complete/quasi-complete separation even without a
  literal zero cell.
- Reproduced directly by calling the existing `fit_interaction` on the real data before
  writing any fix: both pairs fail with `overflow encountered in exp` /
  `divide by zero in log` from statsmodels' unregularized MLE — exactly the signature of
  separation-driven divergence, confirming the diagnosis rather than assuming it.

### Fix applied

Added `fit_categorical_interaction_saturated` (and a matching
`validates_categorical_fallback_on_split`) to
`src/opponent_adjusted/analysis/bivariate/testing.py`, used **only as a fallback** when the
standard interaction-model fit fails for a categorical x categorical pair — it does not
replace the standard method for pairs that already fit successfully. It compares the
additive model (same as the standard method's own `m_add`, which fits fine on both pairs —
far fewer parameters, no guaranteed-all-zero column) against the **saturated cell-means
model**, whose log-likelihood has a closed form (each cell's MLE is just its own empirical
goal rate) and therefore never requires solving a system that could be singular — sidestepping
the exact failure mode above rather than working around it with an arbitrary threshold or
regularization hack. This is the standard deviance / goodness-of-fit test for an interaction
in a contingency-table setting, and per the task's own framing is at least as statistically
appropriate for two categoricals as the logistic-interaction-term LR test used for
continuous/mixed pairs.

**Checked for other silently-affected pairs before fixing, not just the 2 named ones.** All 6
categorical x categorical pairs in the enlarged pool were queried directly:

| Pair | fit_status before fix |
|---|---|
| `defensive_profile_cluster` x `nearest_defender_role` | `fit_failed` |
| `nearest_defender_role` x `second_nearest_defender_role` | `fit_failed` |
| `defensive_profile_cluster` x `nearest_defender_style_archetype` | `ok` (already had a valid p-value) |
| `defensive_profile_cluster` x `second_nearest_defender_role` | `ok` |
| `nearest_defender_role` x `nearest_defender_style_archetype` | `ok` |
| `nearest_defender_style_archetype` x `second_nearest_defender_role` | `ok` |

Only the 2 named pairs were affected — the fix script asserts this exact broken-pair set
before proceeding and would stop loudly if it ever found a mismatch. The other 4 were **not
touched or reprocessed** with the new method, per the task's explicit constraint not to
touch any Tier 1 result that already has a valid p-value.

### Statistical discipline preserved

The 2 new raw p-values were inserted into the full 190-pair CxG+ Tier 1 raw-p-value vector
(the other 188 already had a `p_fdr` from the original round) and Benjamini-Hochberg
recomputed over that full vector, so the 2 new pairs' `p_fdr` reflects the correct
190-test correction stringency — but **only these 2 rows were written back** via a scoped
`UPDATE ... WHERE track='cxg_plus' AND tier=1 AND feature_a=... AND feature_b=...`; the other
188 rows' stored `p_fdr` values are untouched, exactly as instructed, even though a
from-scratch full recomputation would in principle re-rank a handful of them by a negligible
amount (an explicit, accepted, documented tradeoff — not silently absorbed).

### Final results — genuine outcomes, not forced

| Pair | p_raw | p_fdr | fit_status | Validated |
|---|---|---|---|---|
| `defensive_profile_cluster` x `nearest_defender_role` | 0.2394 | 0.5169 | `ok_saturated_fallback` | n/a (didn't clear FDR<0.10) |
| `nearest_defender_role` x `second_nearest_defender_role` | 0.1019 | 0.3520 | `ok_saturated_fallback` | n/a (didn't clear FDR<0.10) |

**Both are genuine null results** — no real evidence of an interaction beyond additive
effects for either pair. Reported as measured; not forced into significance.

Verified: `cxg_bivariate_interaction_v1` now has **zero** `NULL interaction_p_fdr` rows for
`track='cxg_plus' AND tier=1` (190/190 populated). The 16 remaining NULLs elsewhere
(`cxg_plus`, tier 2: 15 rows; tier 3: 1 row) are pre-existing by design — FDR correction was
never applied to Tiers 2/3 in the original bivariate methodology (only Tier 1's 163/190-test
family gets BH-FDR), unrelated to this fix and out of this task's scope.

### Test suite

`python -m pytest -q` -> **283 passed** (278 prior + 5 new
`tests/analysis/bivariate/test_testing.py` tests: reproducing the real failure mode on
synthetic data with a structural zero cell, confirming the fallback succeeds where the
standard method fails, detecting a real synthetic interaction, correctly returning a null
result for synthetic non-interacting data, and the validation-split helper's accept/reject
behavior). No regressions.
