# CxA P_convert Locked-Feature EDA v1

Date: 2026-09-20
Scope: the LOCKED feature sets only --
[`docs/analysis/cxa_event_p_convert_feature_lock_v1.md`](cxa_event_p_convert_feature_lock_v1.md)
(15 event-only features) and
[`docs/analysis/cxa_plus_p_convert_feature_lock_v1.md`](cxa_plus_p_convert_feature_lock_v1.md)
(15 CxA+ features -- **not 17**: that document's own header states "17," but its own
feature table (section 5) lists exactly 15 numbered rows; this EDA uses the actual
table count, 15, and flags the header's number as an error in that document worth a
future correction, not something this task's scope permits fixing here). Does **not**
repeat target usability, sparsity, per-feature signal-vs-target, or any redundancy pair
already covered in
[`docs/analysis/cxa_p_convert_pre_model_analysis.md`](cxa_p_convert_pre_model_analysis.md)
-- this is one level deeper: distribution shape, bounds, rare-level support, and
correlation structure across the *whole locked set together*, on the actual locked
columns (not the pre-model analysis's broader candidate list).

**Scope, deliberately narrower than P_create's own locked-feature EDA:** every check
below runs on `split IN ('train', 'validation')` only -- **not** the full matrix
including test. This is a literal, explicit constraint from this task's own
instructions ("full population (train+validation) only... test stays sealed"), followed
here even though P_create's own locked-feature EDA used the true full population
(including test) under the split policy's general "full-dataset analysis is valid
exploratory work" carve-out. Row counts below: event-only 9,567 rows (7,847 train +
1,720 validation), CxA+ 2,411 rows (1,991 train + 420 validation).

Reproducible via
[`scripts/analyze_cxconvert_locked_feature_eda.py`](../../scripts/analyze_cxconvert_locked_feature_eda.py);
raw output under
[`audit_outputs/cxconvert_analysis/locked_feature_eda/`](../../audit_outputs/cxconvert_analysis/locked_feature_eda/).
Does not train, score, or select a model, and does not reopen either lock decision.

---

## Part A -- Event-only locked set (`cxconvert_event_v1_training_matrix`, 9,567 rows)

### A1. Numeric locked features: distribution, bounds, shape

**Question:** what do the 7 numeric locked features look like -- central tendency,
spread, tails, and any implausible values (out-of-pitch-bounds x-coordinates, negative
distances/counts)?

| feature | mean | median | stddev | min | max | p1 | p5 | p95 | p99 | n_null |
|---|---|---|---|---|---|---|---|---|---|---|
| `start_x` | 95.47 | 98.0 | 18.81 | 5.7 | 120.0 | 34.7 | 59.2 | 120.0 | 120.0 | 0 |
| `pass_end_x` | 100.87 | 104.3 | 12.27 | 20.5 | 120.0 | 59.6 | 78.0 | 115.0 | 117.0 | 0 |
| `shot_x_sb` | 104.56 | 106.1 | 8.20 | 43.7 | 119.5 | 84.8 | 90.2 | 115.3 | 117.4 | 0 |
| `shot_dist_to_goal_m` | 15.95 | 15.30 | 7.18 | 1.15 | 68.68 | 3.96 | 5.96 | 27.87 | 32.62 | 0 |
| `shot_gk_distance_m` | 13.13 | 11.89 | 7.12 | 0.31 | 63.36 | 1.84 | 3.55 | 25.61 | 30.57 | **5** |
| `shot_defenders_within_5m` | 2.30 | 2.0 | 1.40 | 0 | 9 | 0.0 | 1.0 | 5.0 | 7.0 | 0 |
| `shot_defenders_within_8m` | 3.97 | 4.0 | 1.90 | 0 | 11 | 1.0 | 1.0 | 8.0 | 9.0 | 0 |

**Bounds check -- checked, not assumed clean.** All three pitch-x columns
(`start_x`, `pass_end_x`, `shot_x_sb`) have **zero** out-of-`[0,120]`-range values on
this population -- unlike P_create's own locked-feature EDA, which found 2 rows with
`start_x` fractionally past 120 (a benign corner-kick quirk). No equivalent quirk here;
this track's locked `start_x` is clean. `shot_dist_to_goal_m` and `shot_gk_distance_m`
have no negative values (both must be >=0 by construction, both check out).

**`shot_gk_distance_m` has 5 nulls** (0.05% of 9,567 rows) -- not a new finding, this is
the same "6 shots across the full `Y_create=TRUE` population have no identifiable
opposing goalkeeper in their freeze frame" fact the feasibility audit already
documented, restated here because it is directly relevant to this feature's encoding: 5
of those 6 rows fall in train+validation, the 6th presumably in the sealed test split.
Needs a missingness-safe encoding (impute + flag, or an explicit sentinel) before
training -- not resolved here, this is characterization only.

**Long-tail extremes are real shots, not data errors -- investigated, not assumed.**
`shot_dist_to_goal_m`'s max (68.68m) corresponds to `shot_x_sb` = 43.7 (its own
population min) -- a shot struck from just past the halfway line of a ~105m-long pitch.
Checked directly: `shot_x_sb`'s min (43.7) sits well within valid pitch bounds (no
bounds violation), so this is a genuine, if extremely rare, long-range speculative
effort, not a coordinate error. Consistent with this being the *shot* location for an
already-`Y_create=TRUE` pass (the passer created a chance; the shooter chose,
correctly or not, to strike from distance rather than progress further).

**Shape.** `start_x` deciles: `[5.7, 70.9, 82.3, 88.4, 93.1, 97.9, 102.1, 106.8, 111.9,
119.0, 120.0]` -- right-skewed, concentrated in the attacking third as expected for a
chance-creating-pass population (median 98.0, far higher than P_create's own event-only
`start_x` median of 58.7 over its all-passes population -- expected, this population is
already conditioned on `Y_create=TRUE`). `shot_defenders_within_5m`/`_8m` are
low-count, right-skewed but not zero-inflated the way P_create's CxA+ reception counts
were (median 2 / 4, first decile at 0/0 but not a majority-zero distribution) -- these
describe shot-time congestion in a population already selected for created chances,
which tend to occur in the penalty area, so some defensive presence near the shot is
the norm rather than the exception.

**Implication:** no clipping needed anywhere in this track's numeric locked set. The 5
null `shot_gk_distance_m` rows need an explicit missingness-safe encoding before
training (see closing section).

### A2. Categorical/boolean locked features: level counts, rare-level check

**Question:** full train+validation count/percentage for every locked
boolean/categorical level, flagging anything under the 100-row floor P_create's own EDA
used.

| feature | total rows | TRUE rows | % TRUE |
|---|---|---|---|
| `shot_first_time` | 9,567 | 2,495 | 26.079% |
| `is_cross` | 9,567 | 2,000 | 20.905% |
| `pass_type_name = Corner` | 9,567 | 922 | 9.637% |
| `shot_technique_name = Volley` | 9,567 | 463 | 4.840% |
| `shot_one_on_one` | 9,567 | 457 | 4.777% |
| `is_through_ball` | 9,567 | 406 | 4.244% |
| `is_cut_back` | 9,567 | 243 | 2.540% |
| **`shot_open_goal`** | 9,567 | **84** | **0.878%** |
| **`shot_technique_name = Diving Header`** | 9,567 | **67** | **0.700%** |
| **`shot_technique_name = Lob`** | 9,567 | **62** | **0.648%** |
| **`shot_technique_name = Backheel`** | 9,567 | **46** | **0.481%** |

**Rare-level check: 4 flagged, all under the 100-row floor.** `shot_open_goal` (84),
`shot_technique_name = Diving Header` (67), `= Lob` (62), and `= Backheel` (46) are all
below the 100-row floor P_create's own locked-feature EDA used to flag rarity -- a real,
previously-unflagged-at-this-precision finding this task exists to surface. These same
four levels *did* pass the feature-lock's train-vs-validation promotion check (they
held direction and retained enough magnitude on the validation split, per that
document's section 2), but passing a direction/magnitude check on a thin split is a
different property from having enough total population support for a model to learn a
stable pattern from -- exactly the distinction P_create's own EDA drew for
`pass_technique_name = Straight` in CxA+. `shot_technique_name = Volley` (463) and
`is_cut_back` (243) both clear the floor comfortably; `pass_type_name = Corner` (922)
and the four elevated-signal booleans (`shot_first_time`, `is_cross`, `shot_one_on_one`,
`is_through_ball`) all have hundreds to thousands of supporting rows.

**Implication:** the `shot_technique_name` categorical feature's 4 locked levels split
into two support tiers -- `Volley` (463 rows, solid) versus `Diving Header`/`Lob`/
`Backheel` (46-67 rows each, thin). Pooling the three thin levels into a single
"other elevated technique" catch-all (keeping `Volley` and `Normal`/baseline separate)
is worth considering before training, rather than encoding each as its own
single-digit-percent level. `shot_open_goal` (84 rows) is thin in absolute terms but is
a single boolean flag, not a multi-level categorical to pool -- its thinness is a
property to carry into modelling (e.g. expect a wide confidence interval on its
coefficient/importance), not something poolable.

### A3. Full pairwise correlation matrix across the locked set

**Question:** across all 18 encoded locked columns (7 numeric + 11
boolean/one-hot -- `is_through_ball`, `shot_one_on_one`, `is_cross`, `shot_first_time`,
`shot_open_goal`, the 4 `shot_technique_name` levels, `is_cut_back`,
`pass_type_name = Corner`), 153 pairwise correlations -- what reaches the feature-lock
docs' own 0.8 near-duplicate threshold, and what falls in the 0.4-0.8 moderate band
worth noting for modelling?

**3 pairs at or above 0.8 -- a genuine near-duplicate cluster, not previously
quantified this precisely:**

| pair | r | reading |
|---|---|---|
| `shot_dist_to_goal_m` <-> `shot_gk_distance_m` | **0.9644** | Near-duplicate. Both lock docs already flagged this qualitatively ("largely a restatement of shot distance to goal") -- this quantifies it: on this locked set, these two carry almost identical information. |
| `shot_x_sb` <-> `shot_dist_to_goal_m` | **-0.9381** | Also near-duplicate -- expected, since `shot_dist_to_goal_m` is directly derived from `shot_x_sb` (and `shot_y_sb`, not itself locked). |
| `shot_x_sb` <-> `shot_gk_distance_m` | **-0.9302** | Same underlying relationship, one step removed (via `shot_dist_to_goal_m`). |

**These three numeric features (`shot_x_sb`, `shot_dist_to_goal_m`,
`shot_gk_distance_m`) are effectively three views of one signal** ("how close to goal
was this shot"), all three above the 0.8 threshold with each other. This is stronger
and more precise than either lock doc's qualitative caveat -- worth flagging prominently
for the modelling stage (closing section), not a reason to unlock the current feature
list (none of the three individually failed the split-confirmation check, and a
tree-based model tolerates this kind of collinearity far better than a linear one), but
real multicollinearity risk for any linear baseline.

**19 pairs in the 0.4-0.8 moderate band**, most explainable by the same shot-geometry
or set-piece structure already understood:

| pair | r | reading |
|---|---|---|
| `pass_end_x` <-> `shot_x_sb` | 0.7866 | Just under the 0.8 threshold -- the shot is usually taken close to where the pass arrived, same pattern the pre-model analysis found for `pass_end_y`/`shot_y_sb` (r=0.89, already flagged there). |
| `pass_end_x` <-> `shot_dist_to_goal_m` | -0.7806 | Same underlying relationship, one step removed. |
| `start_x` <-> `pass_type_name=Corner` | 0.7409 | Corners originate from a fixed, far-upfield position (the corner arc) -- mechanically high `start_x` whenever `pass_type=Corner` is true. |
| `pass_end_x` <-> `shot_gk_distance_m` | -0.7384 | Same chain as above. |
| `shot_defenders_within_5m` <-> `shot_defenders_within_8m` | 0.7713 | Same nested-radius-band relationship the pre-model analysis already found (r=0.7714 there, reproduced almost exactly on this locked-population subset). |
| `shot_dist_to_goal_m`/`shot_gk_distance_m`/`shot_x_sb` <-> `pass_type_name=Corner` | -0.50 / -0.45 / 0.48 | Corners tend to be headed/struck from further out than open-play chances on average -- a real, moderate relationship, not previously quantified against the *locked shot-geometry trio* specifically. |
| `shot_dist_to_goal_m`/`shot_gk_distance_m` <-> `shot_defenders_within_5m`/`_8m` | -0.44 to -0.52 | The confound the pre-model analysis already flagged qualitatively (closer shots are both more congested and more likely to score) -- quantified here at moderate strength, confirming it's real but not so strong as to make the defender-count features redundant with distance. |

No other pair reaches 0.4. The 11 boolean/categorical flags are otherwise close to
independent of each other and of the numeric block -- collinearity is concentrated in
the shot-geometry trio and the expected pass-position/set-piece relationships, not
scattered across the locked set.

**Implication:** for a linear baseline, consider dropping one of `shot_x_sb` /
`shot_dist_to_goal_m` / `shot_gk_distance_m` (or applying regularization strong enough
to handle the near-0.9-0.96 collinearity cluster) -- keeping all three unmodified risks
unstable coefficients. A tree-based model needs no change; the three features will
simply trade split importance with each other.

### A4. Interaction sanity check

**Question:** selected locked boolean/categorical pairs -- do they co-occur, and how
often?

| pair | both TRUE | A only | B only | reading |
|---|---|---|---|---|
| `is_cross` & `shot_technique_name=Lob` | 2 | 1,998 | 60 | Near-fully exclusive (2/2,000 crosses are also lobs) -- a curved cross delivery and a chipped/lobbed finish are largely distinct shot situations. |
| `shot_one_on_one` & `shot_open_goal` | 16 | 441 | 68 | Partial overlap: 16/84 (19.0%) of open-goal shots are also one-on-one, and 16/457 (3.5%) of one-on-one shots are open-goal -- related but mostly distinct situations, consistent with the pre-model analysis's own full-population finding for this pair (which found `open_goal` "dominates" the rate when both apply). |
| `shot_first_time` & `is_through_ball` | 185 | 2,310 | 221 | **Substantial overlap: 185/406 (45.6%) of through-ball-assisted shots are also first-time finishes.** A new, previously-uncharacterized finding at this precision -- a through ball played into a striker's run is frequently met with an instant, first-time strike, which makes intuitive sense (there often isn't time or need for a controlling touch when running onto a defense-splitting pass). |
| `is_cross` & `is_cut_back` | 179 | 1,821 | 64 | **Strong overlap: 179/243 (73.7%) of cut-backs are also crosses.** Consistent with P_create's own event-only finding for this exact pair (64.8% there) -- the same underlying StatsBomb tagging pattern (a cut-back is frequently also classified as a cross) persists into the P_convert population. |
| `pass_type_name=Corner` & `is_cross` | 48 | 874 | 222 | 48/922 (5.2%) of corners are also crosses -- consistent in direction and rough magnitude with P_create's own event-only finding for this pair (3.2% there). |
| `pass_type_name=Corner` & `shot_open_goal` | 10 | 912 | 7 | 10/84 (11.9%) of open-goal shots came from corner deliveries -- a modest enrichment over corners' 9.6% overall share of this population, worth noting but not a strong effect. |

**Implication:** `is_cross`/`is_cut_back`'s strong overlap and `shot_first_time`/
`is_through_ball`'s newly-quantified substantial overlap are the two relationships a
modeller should expect to see trade off shared credit rather than contribute
independently -- consistent with, and extending, the redundancy findings in A3.

---

## Part B -- CxA+ locked set (`cxconvert_plus_v1_training_matrix`, 2,411 rows,
tournament-only)

Restating the standing caveat: this population is 100% FIFA World Cup + UEFA Euro,
zero Premier League rows (inherited from P_create's own CxA+ population). Every number
below describes tournament football only.

### B1. Numeric locked features: distribution, bounds, shape

| feature | mean | median | stddev | min | max | p1 | p5 | p95 | p99 | n_null |
|---|---|---|---|---|---|---|---|---|---|---|
| `start_x` | 96.35 | 98.7 | 18.10 | 8.7 | 120.0 | 35.2 | 61.5 | 120.0 | 120.0 | 0 |
| `pass_end_x` | 101.57 | 104.9 | 12.03 | 20.5 | 119.3 | 58.3 | 79.8 | 115.0 | 117.2 | 0 |
| `shot_x_sb` | 104.87 | 106.7 | 8.22 | 48.1 | 119.4 | 84.9 | 90.9 | 115.3 | 117.5 | 0 |
| `shot_dist_to_goal_m` | 15.59 | 14.79 | 7.19 | 1.15 | 63.29 | 4.02 | 5.97 | 27.55 | 33.08 | 0 |
| `shot_gk_distance_m` | 12.87 | 11.54 | 7.05 | 0.79 | 55.65 | 1.50 | 3.50 | 25.12 | 29.74 | **0** |
| `reception_nearest_opponent_distance_m` | 3.68 | 2.85 | 3.00 | 0.06 | 25.32 | 0.32 | 0.51 | 9.48 | 13.22 | 0 |
| `reception_opponents_within_5m` | 1.70 | 1.0 | 1.62 | 0 | 9 | 0.0 | 0.0 | 5.0 | 7.0 | 0 |

**Bounds check.** All three pitch-x columns are within `[0,120]` bounds, same clean
result as the event-only track. `shot_gk_distance_m` has **zero** nulls in
train+validation for this track (vs. 5 for event-only) -- consistent with the
feasibility audit's "6 total shots with no identifiable keeper" fact, just distributed
differently across the two tracks' populations. `reception_nearest_opponent_distance_m`'s
min (0.061m, an opponent essentially on top of the receiver) and max (25.32m) are both
physically plausible and notably tighter than P_create's own CxA+ locked-feature EDA
found for the same underlying feature over its `y_create`-defined population (max
42.21m there) -- expected, since this population is a further-conditioned subset (only
chances that already converted or not to a goal, all already `Y_create=TRUE`), which
tends to exclude the most extreme "completely unmarked receiver on a break" outlier
cases that a broader chance-creation population would include.

**Shape.** `reception_opponents_within_5m` remains a low-integer count feature (median
1, not zero-inflated to the extent P_create's own broader-population version was --
that population's median was 0 with 70%+ zero mass; this more-conditioned population's
median is 1, consistent with created-and-then-shot chances tending to occur in more
contested areas than the average reception). `shot_dist_to_goal_m`'s max (63.29m) is
again a genuine long-range effort, same pattern as event-only, not a data error (no
`shot_x_sb` bound violation underlying it, min `shot_x_sb`=48.1 for this track).

**Implication:** no clipping needed. `shot_gk_distance_m`'s nullness needs the same
missingness-safe encoding discussion carried from event-only, even though this
particular track's train+validation split happens to have 0 affected rows (the
underlying feature can still be null in principle, per the full-population fact, and
any encoding decision should be written to handle that case for both tracks
consistently rather than track-specific).

### B2. Categorical/boolean locked features: level counts, rare-level check

| feature | total rows | TRUE rows | % TRUE |
|---|---|---|---|
| `shot_first_time` | 2,411 | 659 | 27.333% |
| `is_cross` | 2,411 | 541 | 22.439% |
| `pass_type_name = Corner` | 2,411 | 214 | 8.876% |
| `shot_one_on_one` | 2,411 | 110 | 4.562% |
| `is_through_ball` | 2,411 | 105 | 4.355% |
| **`is_cut_back`** | 2,411 | **63** | **2.613%** |
| **`shot_open_goal`** | 2,411 | **21** | **0.871%** |
| **`shot_technique_name = Lob`** | 2,411 | **16** | **0.664%** |

**Rare-level check: 3 flagged, more severe than the event-only track's own rare-level
findings.** `is_cut_back` (63), `shot_open_goal` (21), and **`shot_technique_name = Lob`
(only 16 total rows)** are all under the 100-row floor. `is_through_ball` (105) and
`shot_one_on_one` (110) clear the floor, but only barely -- a fifth of the margin
either has on the event-only track (406 and 457 rows there). `shot_technique_name =
Lob`'s 16 total rows is the thinnest support of any locked feature in either track --
comparable in severity to P_create's own CxA+ `pass_technique_name = Straight` finding
(25 rows), which that document explicitly flagged as needing pooling/dropping before
training. **The same treatment applies here.**

This is directly explained by CxA+'s population size (2,411 rows here vs. 9,567 for
event-only, a ~4x difference) combined with each level's rate being *similar* across
tracks (event-only's `shot_technique=Lob` rate was 0.648%, almost identical to this
track's 0.664%) -- the thinness is a population-size effect, not a rate difference (see
Part C for the full cross-track rate comparison).

**Implication:** `shot_technique_name = Lob` needs explicit pooling or dropping before
training for CxA+ specifically -- 16 rows is too few for any model to learn a stable
pattern from, regardless of the direction/magnitude it showed on the (also very thin,
5-row) validation split during feature-lock confirmation. `is_cut_back` and
`shot_open_goal`, while past the mechanical validation-confirmation bar, should be
treated with the same caution the CxA+ lock doc already attached to every
section-2-promoted feature on that track.

### B3. Full pairwise correlation matrix across the locked set

**Question:** across 15 encoded locked columns (7 numeric + 8 boolean/one-hot --
`is_through_ball`, `shot_one_on_one`, `is_cross`, `shot_first_time`, `shot_open_goal`,
`shot_technique_name=Lob`, `is_cut_back`, `pass_type_name=Corner`), 105 pairwise
correlations.

**Same 3-pair near-duplicate cluster as event-only, slightly stronger:**

| pair | r | reading |
|---|---|---|
| `shot_dist_to_goal_m` <-> `shot_gk_distance_m` | **0.9695** | Near-duplicate, same relationship as event-only (0.9644 there). |
| `shot_x_sb` <-> `shot_dist_to_goal_m` | **-0.9429** | Same. |
| `shot_x_sb` <-> `shot_gk_distance_m` | **-0.9406** | Same. |

**22 pairs in the 0.4-0.8 moderate band, including a genuinely new finding this track's
own reception-time features surface:**

| pair | r | reading |
|---|---|---|
| `shot_dist_to_goal_m` <-> `reception_nearest_opponent_distance_m` | **0.6121** | **New, moderate relationship not visible in event-only** (which has no reception-time feature to compare): a receiver marked more tightly *at reception* tends to end up shooting from *closer* to goal too. Two different moments of the same underlying attacking sequence share real correlation -- plausibly because both are driven by the same thing (how far the move penetrated before the shot), not because one causes the other. |
| `shot_gk_distance_m` <-> `reception_nearest_opponent_distance_m` | **0.5964** | Same relationship, one step removed via the shot-distance trio. |
| `reception_nearest_opponent_distance_m` <-> `reception_opponents_within_5m` | **-0.6549** | Same underlying-geometry relationship the pre-model analysis already found for these two (r=0.593-0.745 range across different feature-pair combinations there) -- reproduced here on the further-conditioned population. |
| `pass_end_x` <-> `reception_nearest_opponent_distance_m` | -0.5802 | A pass arriving further upfield tends to be received under tighter marking -- makes sense (deeper/more advanced receptions happen in more congested space). |
| `shot_dist_to_goal_m`/`shot_gk_distance_m` <-> `reception_opponents_within_5m` | -0.60 / -0.56 | Same relationship, inverted sign (more opponents near the receiver at reception correlates with a closer eventual shot). |
| `reception_opponents_within_5m` <-> `pass_type_name=Corner` | 0.5177 | Corners have more opponents packed within 5m of the receiver at reception -- expected, corner deliveries go into a crowded box. |
| `pass_end_x` <-> `shot_x_sb`, `pass_end_x` <-> `shot_dist_to_goal_m`/`shot_gk_distance_m`, `start_x` <-> `pass_type_name=Corner` | 0.79 / -0.78 / -0.75 / 0.73 | Same relationships as event-only (A3), reproduced at very similar magnitude on this track. |

**Implication:** the reception-time and shot-time geometry features (a full moment
apart in the same possession) are not independent of each other -- a moderately strong
relationship (|r| 0.52-0.65) links `reception_nearest_opponent_distance_m` and
`reception_opponents_within_5m` to the shot-geometry trio. This is new information (the
event-only track has no reception-time feature to compare against) and worth carrying
into the modelling stage: the two 360-context features are not redundant with the
shot-geometry trio (correlations stay well under the 0.8 near-duplicate line), but a
modeller should not expect either family to contribute fully independent signal from
the other.

### B4. Interaction sanity check

| pair | both TRUE | A only | B only | reading |
|---|---|---|---|---|
| `is_cross` & `shot_technique_name=Lob` | 0 | 541 | 16 | Fully mutually exclusive on this track (vs. 2/2,000 near-exclusive on event-only) -- consistent direction, and at CxA+'s much smaller Lob count (16 rows), 0 co-occurrences is unsurprising rather than a meaningful divergence. |
| `shot_one_on_one` & `shot_open_goal` | 3 | 107 | 18 | 3/21 (14.3%) of open-goal shots are also one-on-one -- consistent with event-only's 19.0%, same rough relationship. |
| `shot_first_time` & `is_through_ball` | 56 | 603 | 49 | **Even stronger overlap than event-only: 56/105 (53.3%) of through-ball-assisted shots are first-time finishes** (vs. 45.6% event-only) -- same direction, same relationship, somewhat more pronounced on this population. |
| `is_cross` & `is_cut_back` | 50 | 491 | 13 | **Stronger still: 50/63 (79.4%) of cut-backs are also crosses** (vs. 73.7% event-only, 64.8% in P_create's own event-only population) -- the `is_cross`/`is_cut_back` overlap is a consistent, strengthening pattern across every population this project has checked it on. |
| `pass_type_name=Corner` & `is_cross` | **0** | 214 | 541 | **Fully mutually exclusive on CxA+ (vs. 48/922 = 5.2% overlap on event-only).** This exactly reproduces the population-composition difference P_create's own locked-feature EDA already documented for this identical pair (event-only 3.2% overlap, CxA+ 0% there) -- the same underlying reason applies: the tournament-only subset happens not to contain the corner-deliveries-also-tagged-as-crosses cases that exist in the wider, Premier-League-inclusive event population. Confirmed reproducible, not a fluke of one dataset. |
| `pass_type_name=Corner` & `shot_open_goal` | 0 | 214 | 21 | Fully mutually exclusive (vs. 10/84 = 11.9% overlap event-only) -- consistent with the general pattern that CxA+'s smaller counts and tournament-only composition produce tighter mutual exclusivity for corner-related pairs than the wider event-only population. |

**Implication:** every interaction pair checked holds the *same direction* across both
tracks; two (`is_cross`/`is_cut_back`, `shot_first_time`/`is_through_ball`) are
stronger on CxA+, and the `pass_type=Corner`/`is_cross` divergence is a confirmed
reproduction of a pattern this project has now seen in both P_create and P_convert.

---

## Part C -- Cross-track comparison

**Question:** for every feature the two tracks share (12 of the 15: `start_x`,
`pass_end_x`, `shot_x_sb`, `shot_dist_to_goal_m`, `shot_gk_distance_m`,
`is_through_ball`, `shot_one_on_one`, `is_cross`, `shot_first_time`, `shot_open_goal`,
`shot_technique_name=Lob`, `is_cut_back`, `pass_type_name=Corner` -- 13, not counting
event-only's 2 held-out `shot_defenders_within_5m`/`_8m` or CxA+'s own 2
reception-time-only features), does CxA+'s tournament-only, no-Premier-League
composition make a shared feature behave differently in ways visible before modelling?

**Numeric means: closely matched, no material divergence.**

| feature | event mean | plus mean | delta |
|---|---|---|---|
| `start_x` | 95.47 | 96.35 | +0.88 |
| `pass_end_x` | 100.87 | 101.57 | +0.70 |
| `shot_x_sb` | 104.56 | 104.87 | +0.31 |
| `shot_dist_to_goal_m` | 15.95 | 15.59 | -0.36 |
| `shot_gk_distance_m` | 13.13 | 12.87 | -0.26 |

Every shared numeric feature's mean sits within 1 unit (or under half a metre for the
distance features) of its counterpart in the other track. Distribution shapes (deciles,
A1 vs B1) are likewise close. **No evidence tournament football systematically shifts
the shot-geometry features' distributions** relative to the wider (Premier-League-
inclusive) event-only population, at least for this already-conditioned
(`Y_create=TRUE`) population.

**Boolean/categorical rates: also closely matched, with two exceptions.**

| feature | event % TRUE | plus % TRUE | delta |
|---|---|---|---|
| `is_through_ball` | 4.244% | 4.355% | +0.11pp |
| `shot_one_on_one` | 4.777% | 4.562% | -0.22pp |
| `is_cross` | 20.905% | 22.439% | +1.53pp |
| `shot_first_time` | 26.079% | 27.333% | +1.25pp |
| `shot_open_goal` | 0.878% | 0.871% | -0.01pp |
| `shot_technique_name=Lob` | 0.648% | 0.664% | +0.02pp |
| `is_cut_back` | 2.540% | 2.613% | +0.07pp |
| `pass_type_name=Corner` | 9.637% | 8.876% | -0.76pp |

Every rate is within roughly 1.5 percentage points of its counterpart -- broadly
consistent, no feature flips from common to rare (or vice versa) between tracks.
`is_cross` is the largest gap (+1.53pp on CxA+), consistent with a plausible, mild
tournament-football tendency toward wide/crossing play, but not a large effect.

**The real cross-track differences this EDA can see are not in rates or means, but in
two other places:**

1. **Absolute support for already-thin levels shrinks roughly in proportion to
   population size, turning "thin" into "critically thin."** `shot_technique_name=Lob`'s
   *rate* is essentially identical across tracks (0.648% vs 0.664%), but CxA+'s ~4x
   smaller population turns that into 62 rows (event, already flagged as thin) versus
   just 16 rows (CxA+, critically thin) -- see A2/B2. A modeller reading only the rate
   tables in either feature-lock doc would not see this; it only shows up once absolute
   counts are checked on each track's own population, which is exactly what this task's
   rare-level check was for.
2. **Interaction/overlap patterns for the same pair of flags can diverge in direction,
   not just magnitude, between tracks.** `pass_type_name=Corner` & `is_cross` go from a
   real (if modest) 5.2% overlap on event-only to a hard 0% on CxA+ (B4) -- and this
   reproduces a pattern P_create's own locked-feature EDA already found for the
   identical pair, so it is now confirmed reproducible across two different target
   definitions on the same underlying tournament-only population, not a one-off. Two
   other overlaps (`is_cross`/`is_cut_back`, `shot_first_time`/`is_through_ball`) hold
   the same direction on both tracks but are noticeably *stronger* on CxA+ (73.7%->79.4%
   and 45.6%->53.3% respectively) -- a real, consistent tightening of co-occurrence on
   the smaller, tournament-only population, not a contradiction of the event-only
   finding.

**Conclusion:** CxA+'s tournament-only composition does not visibly distort the
*marginal* distribution or rate of any shared feature -- a modeller building on one
track's intuition about means/rates can trust it largely transfers to the other. But it
**does** meaningfully affect (a) how much absolute support a rare level actually has,
and (b) how strongly related flags co-occur -- both of which only show up by checking
each track's own population directly, which is what Parts A/B and this section did.

---

## What would break a first baseline model

Concrete, actionable items surfaced by this EDA that should shape feature encoding
before a model-training prompt gets written -- not new modelling decisions, just
encoding hygiene this analysis is positioned to catch.

1. **`shot_technique_name = Lob` (CxA+) needs pooling or dropping before training.**
   Only 16 total rows in the entire train+validation population (B2) -- the thinnest
   support of any locked feature in either track, comparable in severity to P_create's
   own CxA+ `pass_technique_name = Straight` finding (25 rows), which received the same
   recommendation. Pool into the `Normal`/baseline level for CxA+, or drop the level
   for this track specifically while keeping it for event-only (where it has 62 rows).
2. **`shot_x_sb`, `shot_dist_to_goal_m`, and `shot_gk_distance_m` form a near-duplicate
   cluster (|r| 0.93-0.97) in BOTH tracks** (A3, B3) -- quantitatively confirming what
   both lock docs only flagged qualitatively. For a linear baseline, drop or
   regularize two of the three; a tree-based model needs no change but will split
   shared importance across all three.
3. **`shot_gk_distance_m` has nulls (5 rows event-only train+validation, 0 for
   CxA+'s -- but a real possibility for both tracks per the full-population fact)** --
   needs an explicit missingness-safe encoding (impute + flag, or sentinel) before
   training in either track's pipeline, written to handle the null case generically
   rather than assuming it based on one track's train+validation split happening to
   have zero instances.
4. **`shot_open_goal` (84 event / 21 plus rows), `shot_technique_name = Diving
   Header`/`Backheel` (event-only, 67/46 rows), and `is_cut_back` (243 event / 63 plus
   rows) are all thin enough to warrant a wide-confidence-interval caveat on their
   fitted coefficients/importances**, even though all cleared both the split-lift
   promotion check and (except for the CxA+ trio already covered in #1) the
   population-level 100-row floor -- see A2/B2 for exact counts.
5. **`is_cross` and `is_cut_back` substantially overlap in both tracks, more strongly
   on CxA+ (79.4% of cut-backs are also crosses there, vs. 73.7% event-only)** (A4, B4)
   -- not a reason to drop either, but expect shared/trading-off contribution in a
   fitted model, consistent with the same pattern P_create's own EDA found and this
   task's analysis confirms strengthens on the smaller population.
6. **`shot_first_time` and `is_through_ball` substantially overlap in both tracks
   (45.6% event-only, 53.3% CxA+ of through-ball shots are also first-time
   finishes)** (A4, B4) -- a new finding at this precision, not previously
   characterized; same trade-off-not-redundancy treatment as #5.
7. **CxA+'s `reception_nearest_opponent_distance_m`/`reception_opponents_within_5m`
   correlate moderately (|r| 0.52-0.65) with the shot-geometry trio** (B3) -- a new
   finding (event-only has no reception-time feature to compare against). Not
   redundant (well under the 0.8 threshold), but the two feature families should not
   be expected to contribute fully independent signal.
8. **`pass_type_name=Corner` and `is_cross` diverge sharply between tracks (5.2%
   overlap event-only, 0% CxA+)** (A4, B4, Part C) -- confirmed as a reproducible
   population-composition effect (P_create's own locked-feature EDA found the identical
   divergence for this exact pair), not a bug or a one-off. No action needed beyond
   awareness that intuition from one track's overlap rate for this pair does not
   transfer to the other.
9. **No out-of-pitch-bounds values in either track** (A1, B1) -- unlike P_create's own
   event-only EDA, which found and clipped 2 rows. Nothing to clip here; stated
   explicitly because the task asked this to be checked, not assumed.
10. **No feature needs re-examining on the direction/support grounds the earlier
    feature-lock docs already covered.** This EDA found nothing that contradicts either
    lock's promotion decisions -- findings 1-9 above are additive encoding guidance
    (plus one cross-track-comparison finding, Part C), not a reason to revisit either
    locked list. The one documentation issue worth a future correction (not fixed
    here, out of this task's scope): the CxA+ feature-lock doc's own header states "17"
    locked features where its own table lists 15 -- flagged at the top of this document,
    not silently propagated.
