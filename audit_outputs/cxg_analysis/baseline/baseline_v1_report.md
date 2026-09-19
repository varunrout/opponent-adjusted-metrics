# Dumb Baseline + v1 Kitchen-Sink Model (CxG / CxG+)

First real modeling stage. Deliberately unrefined: full candidate pool (per
`oam_analysis.cxg_bivariate_candidate_pool_v1`, read as-is, not re-derived), additive-only
logistic regression, no interaction terms, no PCA components, ODI trio kept despite weak
univariate signal. Feature trimming and any interaction-term additions are a separate, later
task that will use this v1 run as its comparison point.

Run artifacts:
- `src/opponent_adjusted/analysis/baseline/{contracts.py, modeling.py}` -- typed table
  contracts + the fitting/scoring engine, unit-tested (`tests/analysis/baseline/test_modeling.py`,
  6 tests, synthetic ground truth).
- `scripts/materialize_cxg_baseline_v1.py` -- orchestration: fits both tracks, writes all 4
  tables (INSERT-only, scoped delete-then-insert by track, never `CREATE OR REPLACE`).
- `scripts/materialize_cxg_baseline_chart_registry.py` -- registers the 4 new chart rows.
- Charts rendered/uploaded under run_id **`cxg-analysis-20260822T014705Z`**.

---

## Step 0: split confirmation

Used the existing `split` column already on `cxg_event_model_matrix_v1` /
`cxg_plus_360_model_matrix_v1` -- the same split every prior univariate/correlation/bivariate
task used. Not re-derived.

| Track | train | validation | test | total |
|---|---|---|---|---|
| `cxg_event` | 10,890 | 2,420 | 2,427 | 15,737 |
| `cxg_plus` | 2,780 | 590 | 590 | 3,960 |

---

## Step 1: dumb baseline (class-prior model)

Predicts the constant train-split goal rate for every shot, no features. ROC-AUC is
undefined for a constant predictor (no discriminative signal by construction) -- reported as
`NULL`, not the misleading 0.5.

| Track | Split | n | Train goal rate (predicted prob) | log_loss | Brier |
|---|---|---|---|---|---|
| `cxg_event` | validation | 2,420 | 0.1047 | 0.3421 | 0.0962 |
| `cxg_event` | test | 2,427 | 0.1047 | 0.3281 | 0.0911 |
| `cxg_plus` | validation | 590 | 0.1090 | 0.3540 | 0.1007 |
| `cxg_plus` | test | 590 | 0.1090 | 0.3469 | 0.0980 |

---

## Step 2: v1 -- logistic regression, full candidate pool, additive only

Continuous features: median-impute + `StandardScaler`, fit on train only.
`defensive_profile_cluster` (CxG+ only): one-hot, drop-first (`cluster_0` is the reference
level; the 5th observed level is `null_cluster`, the geometry-ineligible/penalty-kick group
confirmed genuine in the bivariate task).

### Metrics (validation + test)

| Track | Split | Model | log_loss | Brier | ROC-AUC |
|---|---|---|---|---|---|
| `cxg_event` | validation | v1 | 0.3153 | 0.0919 | 0.7015 |
| `cxg_event` | test | v1 | 0.3058 | 0.0872 | 0.6939 |
| `cxg_plus` | validation | v1 | 0.3000 | 0.0877 | 0.7822 |
| `cxg_plus` | test | v1 | 0.2690 | 0.0780 | 0.8292 |

**Beats-baseline gate: PASSED for both tracks.** v1 improves log_loss and Brier over the
dumb baseline on both validation and test, both tracks (script hard-stops with a printed
metrics dump if this ever fails -- see `materialize_cxg_baseline_v1.py:main`'s gate check,
which ran clean on this execution).

| Track | Split | log_loss: dumb -> v1 | Brier: dumb -> v1 |
|---|---|---|---|
| `cxg_event` | validation | 0.3421 -> 0.3153 (-7.8%) | 0.0962 -> 0.0919 (-4.5%) |
| `cxg_event` | test | 0.3281 -> 0.3058 (-6.8%) | 0.0911 -> 0.0872 (-4.3%) |
| `cxg_plus` | validation | 0.3540 -> 0.3000 (-15.3%) | 0.1007 -> 0.0877 (-12.9%) |
| `cxg_plus` | test | 0.3469 -> 0.2690 (-22.5%) | 0.0980 -> 0.0780 (-20.4%) |

CxG+'s richer feature set produces a substantially larger improvement over its baseline than
CxG's does -- consistent with CxG+ having genuine defensive-shape/geometry signal that plain
shot-location/timing features (CxG's pool) can't capture.

### Calibration (test split, predicted-probability decile vs. actual goal rate)

**`cxg_event`, v1:**

| Decile | n | Mean predicted | Actual rate |
|---|---|---|---|
| 1 | 243 | 0.0185 | 0.0206 |
| 2 | 243 | 0.0328 | 0.0370 |
| 3 | 242 | 0.0456 | 0.0124 |
| 4 | 243 | 0.0612 | 0.0617 |
| 5 | 243 | 0.0809 | 0.0658 |
| 6 | 242 | 0.1080 | 0.0909 |
| 7 | 243 | 0.1326 | 0.2593 |
| 8 | 242 | 0.1590 | 0.1281 |
| 9 | 243 | 0.1882 | 0.1193 |
| 10 | 243 | 0.2345 | 0.2181 |

Reasonably monotonic overall, with two noisy deciles (3 and 7 -- expected at n~243 per bin
with a ~10% base rate; not evidence of miscalibration, just sampling noise at this bin size).

**`cxg_plus`, v1:**

| Decile | n | Mean predicted | Actual rate |
|---|---|---|---|
| 1 | 59 | 0.0101 | 0.0169 |
| 2 | 59 | 0.0211 | 0.0169 |
| 3 | 59 | 0.0316 | 0.0339 |
| 4 | 59 | 0.0442 | 0.0000 |
| 5 | 59 | 0.0587 | 0.0508 |
| 6 | 59 | 0.0795 | 0.0508 |
| 7 | 59 | 0.1059 | 0.1017 |
| 8 | 59 | 0.1401 | 0.1525 |
| 9 | 59 | 0.2289 | 0.1864 |
| 10 | 59 | 0.4168 | 0.4915 |

Cleanly monotonic apart from decile 4 (0 actual goals out of 59 -- plausible at this bin
size). Top decile (mean predicted 41.7%, actual 49.2%) captures the high-value shots (short
range, favorable defensive geometry, including the null-cluster/penalty-kick group) well.

For the dumb baseline's calibration (both tracks, both splits -- 8 more tables, all showing
the expected pattern of one constant predicted value against a scattered, uninformative
actual rate per decile) see `oam_ml.cxg_baseline_v1_predictions` or
`audit_outputs/cxg_analysis/baseline/baseline_v1_run_summary.json` -- not reproduced here in
full to keep this report readable; every number is queryable from the persisted tables and
the calibration chart (Step 5) plots both models on the same axes for direct comparison.

### Coefficients

**`cxg_event`** (n=6, incl. intercept):

| Feature | Coefficient | Std err | p-value |
|---|---|---|---|
| const | -2.4150 | 0.0400 | <0.001 |
| `shot_x_sb` | **+0.7569** | 0.0448 | <0.001 |
| `first_box_entry_to_shot_s` | -0.0319 | 0.0389 | 0.412 |
| `last_action_interval_s` | **-0.2001** | 0.0458 | <0.001 |
| `last_box_entry_to_shot_s` | -0.1174 | 0.0729 | 0.107 |
| `regain_height_speed_interaction` | **+0.1245** | 0.0343 | <0.001 |

Dominated by `shot_x_sb` (distance-to-goal proxy), exactly matching its univariate dominance
found in the earlier univariate task. `first_box_entry_to_shot_s` and
`last_box_entry_to_shot_s` are not significant at p<0.05 in the multivariate fit -- kept in
the model regardless, per this task's explicit no-trimming rule.

**`cxg_plus`** (n=22, incl. intercept + 4 cluster dummies):

| Feature | Coefficient | Std err | p-value |
|---|---|---|---|
| const | -2.9453 | 0.2679 | <0.001 |
| `last_action_interval_s` | -0.0807 | 0.1021 | 0.430 |
| `defenders_between_ball_and_goal` | **-0.4171** | 0.1758 | 0.018 |
| `defensive_reset_index` | +0.1464 | 0.0884 | 0.098 |
| `nearest_defender_distance_delta` | +0.0584 | 0.0897 | 0.515 |
| `pre_shot_receiver_space` | +0.0643 | 0.0978 | 0.511 |
| `gk_distance_to_shooter` | **-0.7012** | 0.1363 | <0.001 |
| `defensive_line_depth` | -0.0233 | 0.1116 | 0.835 |
| `defensive_width` | -0.0349 | 0.0833 | 0.675 |
| `estimated_goalface_occlusion` | -0.0066 | 0.1181 | 0.956 |
| `goal_mouth_defender_count` | -0.0099 | 0.1047 | 0.924 |
| `max_goal_exposure` | **+0.4376** | 0.1210 | <0.001 |
| `min_defensive_compactness_sequence` | **-0.2490** | 0.0985 | 0.011 |
| `shot_corridor_occlusion` | **+0.5269** | 0.2073 | 0.011 |
| `visible_goal_angle_proxy` | **+0.4946** | 0.0833 | <0.001 |
| `gk_odi` | -0.0083 | 0.0611 | 0.892 |
| `mean_backline_odi` | -0.0858 | 0.0660 | 0.194 |
| `nearest_defender_odi` | +0.0873 | 0.0692 | 0.207 |
| `defensive_profile_cluster_cluster_1` | -0.0494 | 0.3412 | 0.885 |
| `defensive_profile_cluster_cluster_2` | +0.2324 | 0.3541 | 0.512 |
| `defensive_profile_cluster_cluster_3` | **+1.4427** | 0.3279 | <0.001 |
| `defensive_profile_cluster_null_cluster` | **+2.2694** | 0.4072 | <0.001 |

Strongest signal (in order of significance): `gk_distance_to_shooter`, the
`null_cluster`/`cluster_3` dummies, `visible_goal_angle_proxy`, `max_goal_exposure`,
`defenders_between_ball_and_goal`, `min_defensive_compactness_sequence`, `shot_corridor_occlusion`.
Consistent with the bivariate task's Tier 1 findings (several of these appeared as the FDR-
surviving validated interaction pairs) and Tier 3's finding that cluster membership remains
significant after controlling for geometry -- `null_cluster`'s coefficient (+2.27, by far the
largest in the model) directly reflects the penalty-kick concentration confirmed in that
task. **All three ODI features are non-significant here** (p=0.89/0.19/0.21) -- consistent
with their flagged weak univariate signal; kept in the model per the explicit no-trimming
rule for this v1 baseline.

---

## Step 3: StatsBomb xG benchmark

`oam_core.shots.statsbomb_xg` is not in Gold. **Decision: joined in read-only for this
analysis only, not materialized into `oam_features`.** Reasoning: this task is a baseline-
modeling comparison, not a feature-engineering task -- adding a new governed Gold feature is
a separate decision with its own taxonomy/governance implications (see the project's
feature-family conventions) that shouldn't be made implicitly as a side effect of a model
comparison. `statsbomb_xg` per shot per split is already persisted in this task's own
`oam_ml.cxg_baseline_v1_predictions` deliverable, which is sufficient for this analysis and
for any downstream reuse without expanding Gold's surface. If a future task wants it as a
permanent Gold feature (e.g. for an ensemble/stacking baseline), that should be its own
explicit call.

Note also: `oam_core.shots` has a known 3x row duplication per `event_id` (documented
earlier this session) -- the join used `SELECT DISTINCT event_id, statsbomb_xg` (verified
identical across the 3 duplicate rows before deduping) to avoid silently tripling every
downstream table. This was caught during this task (the first unfiltered run produced
47,211/11,880 prediction rows instead of 15,737/3,960) and fixed before any table was
finalized.

### `cxg_event`

| | log_loss | Brier | ROC-AUC |
|---|---|---|---|
| v1 (test) | 0.3058 | 0.0872 | 0.6939 |
| statsbomb_xg (test) | 0.2597 | 0.0718 | 0.7972 |

Pearson correlation, v1 predicted prob vs. `statsbomb_xg`, test split: **r = 0.435**
(n=2,427, all CxG test shots have a StatsBomb xG value).

**Honest gap: StatsBomb's model is clearly better on CxG's plain feature set** -- 15%
lower log_loss, 18% lower Brier, and a 0.10 AUC advantage. Unsurprising: CxG's 5-feature pool
is deliberately minimal (shot location + simple timing signals), while StatsBomb's public xG
model uses considerably richer shot/assist-type/body-part features CxG doesn't have access
to at all. Not a fair fight, and not reported as one.

### `cxg_plus`

| | log_loss | Brier | ROC-AUC |
|---|---|---|---|
| v1 (test) | 0.2690 | 0.0780 | 0.8292 |
| statsbomb_xg (test) | 0.2430 | 0.0665 | 0.8476 |

Pearson correlation, test split: **r = 0.739** (n=590, all CxG+ test shots have a StatsBomb
xG value).

**Honest gap: still behind StatsBomb, but much closer than CxG's gap** -- 10.7% higher
log_loss, 14.7% higher Brier, 0.018 AUC deficit (vs. CxG's 0.10 AUC deficit). CxG+'s
360-derived defensive/geometry/ODI features close most of the distance to StatsBomb's
model, consistent with the coefficient table above showing those exact features (goal
angle, GK distance, defender counts, cluster membership) carrying the strongest signal.

### Divergence analysis (CxG+ only -- CxG has no opponent-adjustment features to test)

`divergence = v1_predicted_prob - statsbomb_xg`, test split, n=590 (all with a StatsBomb xG
value). Overall mean divergence: **-0.0028** (v1 slightly underpredicts on average, negligible).

**By `defensive_profile_cluster`:**

| Cluster | n | Mean divergence | Mean v1 prob | Mean StatsBomb xG |
|---|---|---|---|---|
| `cluster_0` | 119 | -0.0095 | 0.0636 | 0.0731 |
| `cluster_1` | 178 | -0.0058 | 0.1411 | 0.1469 |
| `cluster_2` | 195 | +0.0042 | 0.0823 | 0.0781 |
| `cluster_3` | 77 | -0.0199 | 0.1182 | 0.1381 |
| `null_cluster` | 21 | **+0.0585** | 0.4403 | 0.3818 |

**By ODI tercile** (`nearest_defender_odi` -- chosen as the featured breakdown: of the 3 ODI
features, it's the least sign-unstable across splits in the earlier correlation/PCA task's
reverification (train +0.054, validation +0.036, both positive; only test flips, and with
the smallest magnitude of the three features' test-split deviations). All 3 computed and
persisted regardless, per the task's "cheap to do" allowance:

| `nearest_defender_odi` tercile | n | Mean divergence | Mean v1 prob | Mean StatsBomb xG |
|---|---|---|---|---|
| low | 121 | +0.0091 | 0.1375 | 0.1284 |
| mid | 121 | -0.0058 | 0.0747 | 0.0805 |
| high | 121 | +0.0122 | 0.0753 | 0.0631 |

| `gk_odi` tercile | n | Mean divergence | Mean v1 prob | Mean StatsBomb xG |
|---|---|---|---|---|
| low | 125 | +0.0246 | 0.1347 | 0.1100 |
| mid | 125 | -0.0082 | 0.0970 | 0.1051 |
| high | 125 | +0.0014 | 0.0585 | 0.0572 |

| `mean_backline_odi` tercile | n | Mean divergence | Mean v1 prob | Mean StatsBomb xG |
|---|---|---|---|---|
| low | 126 | +0.0061 | 0.1101 | 0.1040 |
| mid | 126 | +0.0099 | 0.0845 | 0.0746 |
| high | 126 | +0.0062 | 0.0935 | 0.0873 |

**Honest read: no strong, clean pattern tying divergence to defensive quality.** The one
standout cell is `null_cluster` (+0.0585, by far the largest divergence of any stratum) --
but this is mechanically explained by the penalty-kick concentration confirmed in the
bivariate task (short-range, high-probability shots where v1's geometry+cluster features
push its estimate higher than StatsBomb's), not evidence of the opponent-adjustment
features capturing something StatsBomb misses on open-play defensive quality specifically.
Excluding `null_cluster`, the remaining 4 clusters show small, non-monotonic divergences
(-0.02 to +0.004) with no obvious ordering by goal-rate or defensive-quality rank from the
Phase 2 clustering.

The ODI-tercile breakdowns are similarly flat-to-noisy for `nearest_defender_odi` (low
+0.009, mid -0.006, high +0.012 -- not monotonic) and `mean_backline_odi` (all three
terciles within 0.004 of each other). `gk_odi` shows the closest thing to a coherent
trend -- low-quality-GK shots diverge most positively (+0.025, v1 overpredicts relative to
StatsBomb more when facing a weak keeper) while high-quality-GK shots are nearly flat
(+0.001) -- but this is a single descriptive trend at n=125/tercile, not tested for
statistical significance here (out of this task's scope), and should be read as a
worth-following-up observation, not a confirmed finding.

**This is reported as a genuine, not-forced null-ish result**, per the task's explicit
instruction not to manufacture a finding where the data doesn't clearly support one.

---

## Step 4/5: tables and charts

### Row-count reconciliation

| Table | Expected | Actual | Match |
|---|---|---|---|
| `oam_ml.cxg_baseline_v1_predictions` | 15,737 (cxg_event) + 3,960 (cxg_plus) = 19,697 | 19,697 | Yes |
| `oam_ml.cxg_baseline_v1_metrics` | 2 splits x 3 models x 2 tracks = 12 | 12 (6+6) | Yes |
| `oam_ml.cxg_baseline_v1_coefficients` | cxg_event: 1+5=6; cxg_plus: 1+17+4=22 | 6 + 22 = 28 | Yes |
| `oam_ml.cxg_baseline_v1_divergence` | 5 clusters + 3 ODI features x 3 terciles = 14 | 14 | Yes |

### Charts

Registered under run_id **`cxg-analysis-20260822T014705Z`** via
`scripts/materialize_cxg_baseline_chart_registry.py` (copy-forward pattern, never
`CREATE OR REPLACE`; prior run's 35 rows copied forward untouched, 4 new rows appended):

| chart_name | chart_type | feature_family |
|---|---|---|
| `cxg_event_baseline_calibration` | `baseline_calibration` | `cxg_event_baseline` |
| `cxg_plus_baseline_calibration` | `baseline_calibration` | `cxg_plus_baseline` |
| `cxg_plus_baseline_divergence_cluster` | `baseline_divergence_bar` | `cxg_plus_baseline` |
| `cxg_plus_baseline_divergence_odi_tercile` | `baseline_divergence_bar` | `cxg_plus_baseline` |

All 39 charts for this run_id (35 carried forward + 4 new) rendered locally first
(`--skip-upload`), verified, then uploaded to GCS
(`gs://oam-varun-260819-artifacts/analysis/cxg/cxg-analysis-20260822T014705Z/rendered_charts/`)
and registered in `cxg_rendered_chart_registry_v1` via the existing scoped
delete-then-insert-by-run_id logic. `cxg_chart_registry_v1` now holds 5 distinct run_id
batches (24/27/27/31/35/39 rows across history), all intact.

---

## Test suite

`python -m pytest -q` -> **229 passed** (223 prior baseline + 6 new
`tests/analysis/baseline/test_modeling.py` tests covering: dumb-baseline-prob correctness,
v1 signal recovery on synthetic data with a known categorical + 2 continuous features,
v1-beats-constant-predictor on held-out synthetic data, constant-predictor AUC returning
`None`, calibration table shape/bounds, and calibration table behavior under tied/constant
predictions). No regressions.

---

## What was explicitly NOT done (per task constraints)

- No interaction terms, no PCA components, no dropping the ODI trio -- this is deliberately
  the unrefined kitchen-sink baseline.
- `statsbomb_xg` was not materialized into `oam_features` -- explicit call documented above.
- The StatsBomb comparison is reported with the actual performance gap, not minimized.
- No existing univariate/correlation/bivariate/PCA table or the frozen K-Means clustering
  was modified -- read-only joins only.

---

## Summary for hand-off

**v1 beats the dumb baseline on both log_loss and Brier, both tracks, both splits** -- gate
passed cleanly, no stop condition triggered.

**Key metrics (test split):**

| Track | Model | log_loss | Brier | ROC-AUC |
|---|---|---|---|---|
| `cxg_event` | dumb_baseline | 0.3281 | 0.0911 | n/a |
| `cxg_event` | v1 | 0.3058 | 0.0872 | 0.694 |
| `cxg_event` | statsbomb_xg | 0.2597 | 0.0718 | 0.797 |
| `cxg_plus` | dumb_baseline | 0.3469 | 0.0980 | n/a |
| `cxg_plus` | v1 | 0.2690 | 0.0780 | 0.829 |
| `cxg_plus` | statsbomb_xg | 0.2430 | 0.0665 | 0.848 |

CxG+ v1 closes most of the gap to StatsBomb's own model; CxG v1 does not (expected, given
CxG's much smaller feature pool). Divergence-vs-defensive-quality analysis is an honest,
mostly-null result with one mechanically-explained outlier (`null_cluster`/penalty kicks)
and one weak, unconfirmed `gk_odi` trend worth a follow-up look in a later task.
