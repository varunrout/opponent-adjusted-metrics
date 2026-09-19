# CxG+ v2 Model — Ridge Logistic Regression, 21-Feature Pool + Confirmed Interactions

CxG+ track only (v2 does not apply to CxG event-wide -- no 360 features exist there). v1's
`oam_ml.cxg_baseline_v1_*` tables are untouched and remain the frozen comparison point; v2
writes new, additive `oam_ml.cxg_plus_v2_*` tables.

Run artifacts:
- `scripts/materialize_cxg_second_nearest_defender_style_archetype.py` (Step 1 / Task #11)
- `scripts/materialize_cxg_v2_21st_feature_requalification.py` (Step 2)
- `src/opponent_adjusted/analysis/v2model/modeling.py` + `scripts/materialize_cxg_v2_model_training.py` (Step 3)
- Charts under run_id **`cxg-analysis-20260822T165227Z`**

---

## Step 1 (Task #11): `second_nearest_defender_style_archetype`

A lookup/join task, not a new clustering fit, per the spec -- confirmed and followed
exactly: reused the existing k=4 archetype labels straight from
`cxg_defender_style_clusters_v1` (`player_id -> style_archetype`), no refit. The only new
work was resolving the SECOND-nearest defender's identity per shot -- built by copying
`cxg_defensive_involvement_v1`'s own `INVOLVEMENT_QUERY` (`shot_freeze_frame_players`,
`teammate=FALSE`, same Euclidean distance, same ordinal tie-break) and changing
`QUALIFY ROW_NUMBER() ... = 1` to `= 2`. Null handling reused
`defstyle.shot_join.resolve_shot_archetype` completely unchanged (same function, same
null-reason vocabulary), just called with the second-nearest player_id.

**Coverage** (3,960 CxG+ shots): 3,433 assigned (86.7%), 527 null --
`no_freeze_frame_defender_resolved`: 101 (exactly matches Phase A's own "fewer than 2
defenders identified" count, a strong independent consistency check), `defender_not_in_cluster_table`: 67,
`below_min_action_threshold`: 359. Landed additively on `cxg_defensive_360_features`
(`ALTER TABLE ADD COLUMN IF NOT EXISTS` + scoped staging-table `UPDATE`, never
`CREATE OR REPLACE`), mirroring Phase B's own `merge_gold_column` exactly.

---

## Step 2: 21st-feature requalification

**Univariate stability** (per-split goal rate, `deep_block_clearer` consistently highest in
all 3 splits -- 0.097/0.116/0.116 train/val/test -- directionally stable, same shape as the
first-nearest archetype's own pattern).

**Redundancy screen vs. all 20 other pool members.** The 16 continuous features are not a
near-duplication risk for a categorical by construction (documented, not computed). The 4
other categoricals got Cramer's V (the categorical analogue of |r|, since Pearson
correlation isn't defined for two categoricals -- there was no existing categorical-vs-
categorical redundancy precedent to instead replicate):

| vs. | Cramer's V | Redundant (>=0.85)? |
|---|---|---|
| **`second_nearest_defender_role`** (the flagged overlap-risk pair) | **0.472** | **No** |
| `defensive_profile_cluster` | 0.103 | No |
| `nearest_defender_role` | 0.083 | No |
| `nearest_defender_style_archetype` | 0.064 | No |

The flagged pair shows real, moderate association (expected -- both describe the same
second-nearest defender, just role vs. style) but nowhere near the 0.85 near-duplication
threshold. **Qualification outcome: the 21st feature is INCLUDED** -- clears the redundancy
screen cleanly, and per the standing "no drop for weak signal" principle, its Tier 1 results
(below) don't change that.

**Tier 1 re-run**: 20 new pairs (this feature x every other of the 20 pool members), BH-FDR
recomputed over the full 210-pair family for correct calibration, only these 20 rows
written. **Zero fit failures** (no repeat of the earlier categorical-sparsity issue), **zero
pairs confirmed** (FDR<0.10 and validated) -- a genuine null result for this feature's own
interactions, reported as found, not forced.

---

## Step 3: model

### Missing-category handling

Reused v1's exact convention (`baseline/modeling.py`): cast each categorical to `str` before
computing dummy levels, so a missing value becomes the literal category `"nan"` and gets its
own explicit dummy indicator (not row-dropped, not silently absorbed into the reference
level). **A real, environment-specific bug was caught and fixed while replicating this**:
under this environment's Arrow-backed pandas string dtype, plain `.astype(str)` on a column
with `NaN`/`None` leaves those entries as an actual `float('nan')` object instead of
converting them to the string `"nan"` -- confirmed by direct inspection, then fixed with an
explicit `series.map(lambda v: "nan" if pd.isna(v) else str(v))` that's independent of
pandas' internal string backend. v1 never hit this in practice (its one categorical,
`defensive_profile_cluster`, is pre-mapped to string labels including an explicit
`"null_cluster"` before reaching the design matrix, so genuine `NaN` never reaches this code
path there) -- v1 is unaffected and was not touched; flagged here for anyone extending v1's
pattern to a genuinely-nullable categorical in the future.

A second real bug was caught by this module's own tests before touching real data: the
first draft of `_categorical_dummies` didn't drop a reference level, so a k-level
categorical's dummies summed to 1 everywhere -- perfectly collinear with the intercept. Both
bugs are structural/collinearity failure modes, not v2-methodology decisions; fixed and
covered by `tests/analysis/v2model/test_modeling.py` (8 tests) before any real fit was run.

### Ridge vs. plain logistic -- not a close call

Both fit at the winning (archetype-encoding, gap-transform) combination, per the task's
explicit instruction to report honestly rather than force ridge to look better. **The plain
(unpenalized) logistic fit could not be fit at all** -- `numpy.linalg.LinAlgError: Singular
matrix` inside statsmodels' Hessian inversion. With ~54 design columns (21 features'
dummies/continuous terms + 6 interaction-term expansions) against 2,780 train rows, and the
task's own documented rationale (v2's pool has correlated-but-not-redundant structure,
confirmed at the correlation-screen stage), the unregularized fit is numerically singular.
**This directly confirms the task's stated rationale for choosing ridge -- not merely that
ridge scored better, but that plain logistic is not a usable alternative on this pool at
all.** Final model: **ridge**, by necessity as well as by validation performance.

### Grid search (validation log-loss decides everything, nothing defaulted)

20 ridge fits: {archetype encoding: levels vs. collapsed} x {gap transform: raw vs. log1p}
x {C: 0.01, 0.1, 1.0, 10.0, 100.0}.

| Decision | Options compared | Validation-supported choice |
|---|---|---|
| `nearest_defender_style_archetype` / `second_nearest_defender_style_archetype` levels | all 4 levels vs. collapsed `is_deep_block_clearer` binary | **All 4 levels** (val log_loss 0.29602 vs. 0.29684 best-collapsed) |
| `nearest_defender_gap` transform | raw vs. log1p | **log1p** (val log_loss 0.29602 vs. 0.29668 best-raw) |
| Ridge `C` | 0.01 / 0.1 / 1.0 / 10.0 / 100.0 | **C=0.1** (val log_loss 0.29602, the minimum across the whole 20-fit grid) |

**All 4 levels beating the collapsed binary is not a foregone conclusion** -- the
goal-rate-by-archetype chart from the prior task showed `deep_block_clearer` clearly
separated from the other 3, which cluster tightly together, so a plausible hypothesis was
that the fine-grained distinction among those 3 wouldn't earn its keep. The data says
otherwise: keeping all 4 levels wins on held-out validation, even if narrowly. Reported as
the evidence-supported decision, not the pre-registered guess.

### `nearest_defender_zone_displacement` bimodality -- checked, not assumed away

Investigated directly before deciding whether a separate binning transform was needed. The
confirmed `defensive_profile_cluster x nearest_defender_zone_displacement` interaction was
checked for whether cluster membership explains the marginal distribution's two humps
(around 27 and 40 units): **it does not, cleanly**. Per-cluster means are 34.8-40.5 across
all 4 non-null clusters (`null_cluster` alone is lower, 27.4, but is only 56 of 3,867
non-null rows -- far too small to produce the two large humps at ~920 and ~933 shots each).
`nearest_defender_role`'s means (GK 6.1 / CB 27.6 / Midfield 40.7 / Fullback_WingBack 43.6 /
Attack 51.7) line up with the two humps far better -- but `nearest_defender_role x
nearest_defender_zone_displacement` was tested in the original Tier 1 run and did **not**
clear FDR (`p_fdr=0.203`, confirmed live, not assumed). Per "do not include any unconfirmed
interaction as a model term," this was **not** added, and no separate binning transform was
added either (that would require its own governed qualification, out of this task's scope).
**Honestly flagged as a known limitation**: the confirmed cluster interaction does not fully
explain the bimodality; a future task could test `nearest_defender_role x
nearest_defender_zone_displacement` (or a role-based binning) through the proper governed
pipeline if this gap is worth closing.

---

## Metrics (test split, n=590)

| Model | log_loss | Brier | ROC-AUC |
|---|---|---|---|
| v1 | 0.26902 | 0.07801 | 0.82922 |
| **v2** | **0.25657** | **0.07164** | **0.83024** |
| statsbomb_xg | 0.24295 | 0.06654 | 0.84756 |

(validation split, n=590, for completeness: v1 log_loss=0.30000/brier=0.08771/auc=0.78217;
v2 log_loss=0.29602/brier=0.08780/auc=0.79458; statsbomb_xg
log_loss=0.29501/brier=0.08679/auc=0.79524.)

### v1 vs. v2 -- does the enlarged pool actually pay off?

| Metric | v1 -> v2 (test) | Relative change |
|---|---|---|
| log_loss | 0.26902 -> 0.25657 | **-4.6%** (better) |
| Brier | 0.07801 -> 0.07164 | **-8.2%** (better) |
| ROC-AUC | 0.82922 -> 0.83024 | +0.001 (essentially flat) |

**Real but modest.** log_loss and Brier both improve meaningfully; AUC barely moves. Reported
honestly rather than oversold: Phase A/B's engineering effort produced a genuine, if not
dramatic, improvement in probability calibration/accuracy, with almost no change in ranking
ability (AUC). One caveat noted rather than hidden: on the **validation** split specifically,
v2's Brier (0.08780) is a hair worse than v1's (0.08771) -- a ~0.0001 difference, most
plausibly noise at n=590, but not selectively omitted just because the test-split story is
cleaner.

### StatsBomb xG divergence -- does v2 narrow the gap v1 found?

| Metric | v1 gap vs. statsbomb (test) | v2 gap vs. statsbomb (test) |
|---|---|---|
| log_loss | **10.7%** higher | **5.6%** higher |
| Brier | **17.2%** higher | **7.7%** higher |
| ROC-AUC | 0.0183 lower | 0.0173 lower |

**v2 roughly halves the log_loss and Brier gaps to StatsBomb's own model**, while the AUC
gap barely moves. Overall correlation between predicted probability and `statsbomb_xg`
(test split, both with a StatsBomb value, n=590): **r=0.878**, up from v1's r=0.739 -- a
substantial increase, consistent with the narrower gap. Overall mean divergence
(`v2_predicted_prob - statsbomb_xg`): **-0.0021**, essentially at parity on average (v1's was
also small, -0.0028) -- the improvement shows up in per-shot calibration/log-loss, not in a
systematic overall bias shift.

**Divergence by `defensive_profile_cluster`** (test split): all 5 strata now sit within
-0.019 to +0.003 of zero -- notably tighter than v1's corresponding breakdown. The
`null_cluster` (penalty-kick-dominated) stratum in particular improved from **+0.0585** in
v1 to **-0.0188** in v2 -- a real, substantial reduction in exactly the stratum v1's own
report flagged as the largest divergence outlier.

**Divergence by `nearest_defender_style_archetype`** (replacing v1's ODI-tercile breakdown
-- ODI is dropped from v2 entirely per the locked methodology, so an ODI-tercile
stratification of a model that doesn't use ODI wouldn't describe anything v2 actually does;
`nearest_defender_style_archetype` is the natural v2-relevant analogue, the feature that
replaced ODI's role): `deep_block_clearer` shows the largest divergence at +0.014 (v2
slightly overpredicts relative to StatsBomb for this group); the other 3 archetypes are
within +/-0.001 of zero.

---

## Coefficient interpretability

Ridge coefficients only -- **no classical std_error/p_value for the ridge fit** (penalization
deliberately introduces bias exactly where the standard asymptotic variance formula stops
applying; reported as `NULL` with the reason documented, not approximated). Where a specific
term's significance matters, this report cites that term's own properly-powered Tier 1 test
instead (a 2-6-parameter model, not this 54-parameter one, so classical inference is valid
there).

### `nearest_defender_zone_displacement` -- sign check requested by the task

| Stage | Value | Sign |
|---|---|---|
| Original univariate point-biserial (train/val/test) | -0.039 / -0.052 / -0.034 | negative, stable |
| v2 ridge main-effect coefficient (log1p-scaled predictors) | -0.0268 | **negative -- holds** |
| Interaction with `defensive_profile_cluster` (per level, relative to `cluster_0` reference) | cluster_1: +0.215, cluster_2: -0.092, cluster_3: -0.212, null_cluster: -0.212 | varies by cluster, as expected for a real interaction |

**Does not flip or vanish** -- the main effect stays negative from univariate through the
full 21-feature ridge fit, controlling for everything else and its own confirmed
interaction. The cluster-level interaction coefficients vary in both magnitude and sign
across clusters, which is exactly the qualitative behavior a genuine (not spurious)
interaction should show. No classical p-value available for the main effect in this fit
(ridge); the interaction itself already carries its own confirmed significance
(`p_fdr=0.024`, validated) from its original 2-feature Tier 1 test.

### `nearest_defender_gap` (log1p-transformed)

Main effect +0.134, interaction with `visible_goal_angle_proxy` +0.066 -- both positive.
Notable evolution worth reporting honestly: this feature's own univariate signal was
essentially zero and sign-unstable (train -0.007 / val -0.001 / test -0.019) -- yet once
combined with its confirmed interaction partner in the full model, it shows a modest,
consistently-signed positive contribution. Consistent with the non-negotiable
"never drop for weak univariate signal" principle actually paying off here: this feature
would have been wrongly excluded by a signal-strength gate.

### `nearest_defender_style_archetype` -- levels vs. reference (`deep_block_clearer`, dropped as reference)

| Level | Coefficient (relative to `deep_block_clearer`) |
|---|---|
| `duel_dominant_contester` | -0.310 |
| `unresolved_5050_annotation_density` (muddy 4th cluster) | -0.165 |
| `high_volume_presser` | -0.152 |
| `nan` (no defender / below threshold) | +0.218 |

Every non-reference archetype level is negative relative to `deep_block_clearer` --
consistent with the univariate goal-rate chart (`deep_block_clearer` ~12% vs. 6-7% for the
other 3). The muddy 4th cluster sits in the middle of the other two "clean" archetypes, not
as an outlier -- consistent with the earlier finding that it carries real, non-degenerate
signal despite its interpretability flag. The `nan` (no resolvable defender / below
clustering threshold) dummy is strongly positive, consistent with that group's elevated raw
goal rate (dominated by penalty kicks, per the earlier archetype-chart investigation).

---

## Deliverables and row-count reconciliation

| Table | Rows | Check |
|---|---|---|
| `oam_ml.cxg_plus_v2_predictions` | 3,960 | train (2,780) + val (590) + test (590) |
| `oam_ml.cxg_plus_v2_metrics` | 4 | 2 splits (val, test) x 2 models (v2, statsbomb_xg) |
| `oam_ml.cxg_plus_v2_coefficients` | 54 | const + continuous + categorical dummies + interaction-term expansions |
| `oam_ml.cxg_plus_v2_divergence` | 9 | 5 `defensive_profile_cluster` levels + 4 `nearest_defender_style_archetype` levels |

`oam_ml.cxg_baseline_v1_*` tables: unmodified, row counts unchanged (verified, not
assumed).

---

## Chart coverage

Registered under run_id **`cxg-analysis-20260822T165227Z`** (42 carried forward + 3 new =
45), via `scripts/materialize_cxg_v2_model_chart_registry.py` (copy-forward pattern, never
`CREATE OR REPLACE`). Reused the existing `baseline_calibration`/`baseline_divergence_bar`
chart TYPES per the task's explicit instruction -- generalized (not duplicated) in
`cxg_charts.py` to detect a `<track>_v2` feature_family and read from the v2 tables/columns
instead of v1's, with the v1 code path completely unchanged for `<track>_baseline` charts.

| chart_name | chart_type | Note |
|---|---|---|
| `cxg_plus_v2_calibration` | `baseline_calibration` | Overlays dumb-baseline, v1, and v2 on one calibration curve (test split) -- richer than a v2-only chart, made possible since both prediction tables are already at hand |
| `cxg_plus_v2_divergence_cluster` | `baseline_divergence_bar` | By `defensive_profile_cluster`, directly comparable to v1's own chart |
| `cxg_plus_v2_divergence_archetype` | `baseline_divergence_bar` | By `nearest_defender_style_archetype`, replacing v1's ODI-tercile chart for the reason given above |

All 45 charts rendered locally first (`--skip-upload`, reviewed), then uploaded to GCS and
registered via the existing scoped delete-then-insert-by-run_id pattern.
`cxg_chart_registry_v1` history intact across all prior run_ids (24 through 45 rows across
the full session).

---

## Test suite

`python -m pytest -q` -> **291 passed** (283 prior baseline + 8 new
`tests/analysis/v2model/test_modeling.py` tests). No regressions. The 8 new tests caught 2
real bugs before any real data was fit: the Arrow-backed-pandas missing-category string
quirk, and a missing reference-level drop that caused perfect collinearity with the
intercept -- both described above, both fixed pre-training.

---

## What was explicitly NOT done (per task constraints)

- `oam_ml.cxg_baseline_v1_*` tables were not touched -- v1 stays frozen and comparable,
  confirmed via row-count check.
- No unconfirmed interaction was added as a model term -- `nearest_defender_role x
  nearest_defender_zone_displacement` was checked and explicitly excluded despite being
  a plausible explanation for the zone-displacement bimodality, because it never cleared
  Tier 1 FDR.
- `nearest_defender_style_archetype`'s levels were not silently collapsed or silently kept
  -- both variants were fit, validation decided, the decision and evidence are reported
  above.
- No new train/val/test split was invented -- reused `cxg_plus_360_model_matrix_v1`'s
  existing `split` column throughout (train=2,780/val=590/test=590, unchanged).
- `statsbomb_xg`, `player_id`, `team_id` were not used as model inputs -- `statsbomb_xg` is
  read only for the benchmark comparison and divergence tables; `player_id` is used only as
  an internal join key when resolving the second-nearest defender's archetype, never exposed
  as an output/feature column.
- CxG (event-wide) track was not touched.
- No push to git remote.

---

## Summary for hand-off

**Final v2 model**: ridge logistic regression (C=0.1), 21-feature pool (16 continuous incl.
log1p-transformed `nearest_defender_gap`, 5 categorical incl. all 4
`nearest_defender_style_archetype` levels, not collapsed), 6 confirmed Tier 1 interaction
terms. Plain (unpenalized) logistic is not fittable on this pool at all (singular design
matrix) -- ridge is necessary, not just preferred.

**v2 beats v1** on test-split log_loss (-4.6%) and Brier (-8.2%), with AUC essentially
unchanged -- a real, modest improvement, honestly not a dramatic one. **v2 roughly halves
v1's gap to StatsBomb's own xG model** on both log_loss (10.7% -> 5.6%) and Brier (17.2% ->
7.7%), with the `null_cluster`/penalty-kick divergence outlier shrinking from +0.059 to
-0.019. Phase A/B's feature-engineering effort produced a real, worth-having improvement.
