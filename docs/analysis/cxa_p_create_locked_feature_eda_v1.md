# CxA P_create Locked-Feature EDA v1

Date: 2026-09-19
Scope: the LOCKED feature sets only --
[`docs/analysis/cxa_event_p_create_feature_lock_v1.md`](cxa_event_p_create_feature_lock_v1.md)
(10 event-only features) and
[`docs/analysis/cxa_plus_p_create_feature_lock_v1.md`](cxa_plus_p_create_feature_lock_v1.md)
(9 locked + 1 held-out CxA+ features). Does **not** repeat target usability, sparsity,
per-feature signal-vs-target, or the `pass_length`/`pass_angle`/`start_x`/`start_y`/
`end_x`/`end_y`/`receiver_x`/`receiver_y` redundancy pairs already covered in
[`docs/analysis/cxa_p_create_pre_model_analysis.md`](cxa_p_create_pre_model_analysis.md)
-- this is one level deeper: distribution shape, outliers, missingness in context, and
correlation structure across the *whole locked set together*, which has not been
checked before now.

Runs on the **full population** of both training matrices (not train-only). Per
[`docs/cxa_split_policy_and_parallel_plan.md`](cxa_split_policy_and_parallel_plan.md):
"full-dataset analysis (EDA, null profiling, summary stats) is valid exploratory
work" -- the train-only requirement applies to feature promotion and model evaluation,
both already done in the two feature-lock docs and not repeated here.

Reproducible via
[`scripts/analyze_cxa_locked_feature_eda.py`](../../scripts/analyze_cxa_locked_feature_eda.py);
raw output under
[`audit_outputs/cxa_analysis/locked_feature_eda/`](../../audit_outputs/cxa_analysis/locked_feature_eda/).
Does not train, score, or select a model, and does not reopen either lock decision --
it characterizes features already locked, nothing more.

---

## Part A -- Event-only locked set (`cxa_event_v1_training_matrix`, 608,722 rows)

### A1. Numeric locked features: distribution, bounds, shape

**Question:** what do `start_x` and `end_x` (the 2 numeric event-only locked features)
look like -- central tendency, spread, tails, and any values outside the valid 0-120
pitch range?

**Method:** full-population `AVG`/`STDDEV`/`APPROX_QUANTILES` (mean, median, p1/p5/p95/p99,
min/max) plus a decile-edge histogram for shape.

| feature | mean | median | stddev | min | max | p1 | p5 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|---|
| `start_x` | 59.28 | 58.70 | 27.71 | **0.1** | **120.9** | 5.8 | 11.4 | 105.9 | 119.0 |
| `end_x` | 66.46 | 66.40 | 27.20 | 0.1 | 120.0 | 7.1 | 20.5 | 110.8 | 117.2 |

**Bounds check -- 1 flagged issue.** `start_x` has a max of **120.9**, outside the
valid 0-120 StatsBomb pitch range. Investigated directly (not just counted): exactly
**2 rows** out of 608,722 (0.0003%), both `start_x` = 120.7/120.9, both with `start_y`
near a touchline corner (2.8 / 0.7) and both corner-kick-shaped passes (length 12.1m /
45.5m from a corner-arc origin). This is a known, benign StatsBomb quirk -- a corner is
physically taken from just outside the painted touchline at the corner arc, and
StatsBomb occasionally records that literal starting position a fraction of a unit
past the nominal 120 boundary rather than clamping it. Not a data-integrity problem;
2 rows have no measurable effect on any statistic above. `end_x` has no bound
violations (max exactly 120.0).

**Shape.** Decile edges for `start_x`: `[0.1, 20.7, 34.5, 43.7, 51.8, 58.8, 66.4, 75.0,
84.1, 95.9, 120.9]` -- roughly even decile spacing through the middle of the pitch with
a compressed final decile (84.1 -> 120.9, i.e. the top 10% of passes by origin
concentrate heavily near the byline/box, consistent with `is_cross`/`is_cut_back`/
`pass_type=Corner` all being final-third-or-byline actions). **Not meaningfully
bimodal** -- a true bimodal shape would show a visible gap/trough in the decile
spacing; instead the deciles narrow smoothly toward the attacking end, i.e. a
right-skewed unimodal distribution with a heavy tail into the final third, not two
separate clusters. `end_x`'s deciles show the same right-skew shape, shifted further
forward as expected (a pass's destination sits ahead of its origin on average).

**Implication:** no clipping needed for `end_x`. `start_x` needs a trivial `[0, 120]`
clip before training (2 rows, cosmetic) -- not because it materially changes any
statistic, but because a naive linear-model feature transform (e.g. a distance-to-goal
derived feature) could otherwise silently compute a nonsensical negative
distance-past-the-goal-line for those 2 rows.

### A2. Categorical/boolean locked features: level counts, rare-level check

**Question:** what is the full-population count/percentage for every locked
categorical level, and does any level fall under 100 total rows (too rare for a model
to learn reliably, independent of whether it passed the earlier split-lift check)?

**Method:** `COUNT`/`COUNTIF` per locked level (the 8 event-only categorical/boolean
locked features -- the 7 explicitly named in this task's scope plus
`pass_body_part_name = No Touch`, which is the 10th locked event-only feature and is
included here for completeness even though it wasn't separately named in the task's
part-2 list).

| feature | total rows | TRUE rows | % TRUE |
|---|---|---|---|
| `is_switch` | 608,722 | 17,389 | 2.857% |
| `is_cross` | 608,722 | 15,105 | 2.481% |
| `pass_type_name = Free Kick` | 608,722 | 14,676 | 2.411% |
| `play_pattern_name = From Corner` | 608,722 | 22,128 | 3.635% |
| `is_through_ball` | 608,722 | 2,464 | 0.405% |
| `pass_type_name = Corner` | 608,722 | 6,224 | 1.023% |
| `play_pattern_name = From Counter` | 608,722 | 3,939 | 0.647% |
| `pass_technique_name = Outswinging` | 608,722 | 1,949 | 0.320% |
| `is_cut_back` | 608,722 | 1,206 | 0.198% |
| `pass_body_part_name = No Touch` | 608,722 | 521 | 0.086% |

**Rare-level check: none flagged.** Every locked level clears 100 rows by a wide
margin -- the smallest, `pass_body_part_name = No Touch` at 521 rows, is still 5x the
100-row floor. This is expected: the event-only track's full-population size
(608,722 rows) is large enough that even a 0.086%-prevalence level still has
hundreds of examples. (Contrast with CxA+ below, where the same rare-level check does
flag something.)

**Implication:** no pooling required for event-only locked levels on rarity grounds
alone.

### A3. Full pairwise correlation matrix across the locked set

**Question:** now that all 10 locked features are checked together (rather than the ad
hoc pairs the pre-model analysis checked), does anything new show up at `|r| >= 0.5`?

**Method:** numeric features as-is, each locked categorical level one-hot encoded,
booleans as 0/1 -- 12 encoded columns (`start_x`, `end_x`, plus 10 boolean/one-hot
columns for the 8 categorical/boolean features, 2 of which -- `pass_type_name`,
`play_pattern_name` -- contribute 2 locked levels each), 66 pairwise `CORR()` values
computed in one query. `start_x` <-> `end_x` (r=0.77) is excluded from "new findings"
per the task -- it's already known and accepted from the pre-model analysis.

**1 new finding at `|r| >= 0.5`:**

| pair | r | reading |
|---|---|---|
| `pass_type_name = Corner` <-> `play_pattern_name = From Corner` | **0.8308** | Expected once named: `pass_type = Corner` tags the literal corner-kick delivery pass; `play_pattern = From Corner` tags *every* pass in a possession that started from a corner (the delivery itself, plus any short lay-off/recycling passes before or after it). They describe overlapping but distinct things -- 6,224 rows are `pass_type=Corner`, 22,128 are `play_pattern=From Corner`, and (per A4 below) only a fraction of the `play_pattern=From Corner` rows are the delivery pass itself. Not a near-duplicate (r=0.83, not >0.95), so both are kept as locked, but a modeller should not expect independent information from them -- they largely co-activate. |

No other pair reaches 0.5 -- the remaining 65 pairs (all boolean/categorical
combinations, plus each against `start_x`/`end_x`) are all weaker than this, meaning
the 10 locked features otherwise behave close to independently at the linear-relationship
level, which is a good property for a first baseline (redundancy is concentrated in one
known, explainable pair, not scattered).

**Implication:** no feature needs dropping on correlation grounds. Worth noting for
whoever writes the model-training prompt that `pass_type=Corner` and
`play_pattern=From Corner` will likely trade off importance with each other in a
fitted model (collinearity, not redundancy elimination) -- expected and fine for
tree-based models, worth a coefficient-stability check if a linear baseline is used.

### A4. Interaction sanity check -- set-piece flags vs open-play flags

**Question:** can `is_cross` and `pass_type_name=Corner` co-occur on the same pass?
Can `is_through_ball` and `is_cut_back`? A model needs to know whether these behave as
mutually exclusive categories or as independently-combinable flags.

**Method:** 2x2 crosstab (`both_true` / `a_only` / `b_only` / `neither`) for all 6
pairs among {`is_cross`, `pass_type_name=Corner`} x {`is_through_ball`, `is_cut_back`}
plus the two set-piece flags against each other and the two open-play flags against
each other.

| pair | both TRUE | A only | B only | reading |
|---|---|---|---|---|
| `is_cross` & `pass_type_name=Corner` | 198 | 14,907 | 6,026 | **Mostly but not fully mutually exclusive.** 198/6,224 (3.2%) of corners are also flagged as crosses -- StatsBomb tags an inswinging/outswinging corner delivery into the box as both a corner (by `pass_type`) and a cross (by the `cross` flag) in a minority of cases, likely depth/delivery-style dependent. Model should treat these as two separate, weakly-overlapping flags, not a strict either/or. |
| `is_cross` & `is_through_ball` | 49 | 15,056 | 2,415 | Near-exclusive (49/15,105 = 0.3% of crosses are also through balls) -- a curved delivery and a defense-splitting ground/driven ball are largely distinct techniques, with a small edge-case overlap. |
| `is_cross` & `is_cut_back` | 782 | 14,323 | 424 | **Substantial overlap: 782/1,206 (64.8%) of all cut-backs are also tagged as crosses.** This is the strongest overlap found in this check -- a cut-back (a low ball played back across the box from near the byline) is frequently also classified as a cross by StatsBomb. A model should expect real shared signal between `is_cross` and `is_cut_back`, not independent contributions. |
| `pass_type_name=Corner` & `is_through_ball` | 0 | 6,224 | 2,464 | **Fully mutually exclusive** (0 co-occurrences) -- makes sense, a corner-kick delivery is never simultaneously tagged as a through ball. |
| `pass_type_name=Corner` & `is_cut_back` | 5 | 6,219 | 1,201 | Near-exclusive (5/6,224 = 0.08%). |
| `is_through_ball` & `is_cut_back` | 0 | 2,464 | 1,206 | **Fully mutually exclusive** (0 co-occurrences) -- StatsBomb's pass sub-type tagging treats these as distinct techniques. |

**Implication:** `is_cross` and `is_cut_back` are not independent signals -- roughly
two-thirds of cut-backs are also crosses. This does not disqualify either from the
locked set (their split-confirmed lifts, 9.7x and 13-16x respectively across the two
tracks, are each real on their own), but a modeller should expect their coefficients
(in a linear model) or their split importance (in a tree model) to partially trade off
rather than add independently, and should not be surprised if dropping one changes the
other's apparent importance.

---

## Part B -- CxA+ locked set (`cxa_plus_v1_training_matrix`, 133,143 rows, tournament-only)

Restating the caveat that belongs on every CxA+ artifact: this population is **100%
FIFA World Cup + UEFA Euro, zero Premier League rows**. Every number below describes
tournament football only.

### B1. Numeric locked features: distribution, bounds, shape

**Question:** same as A1, for the 5 CxA+ numeric locked/context features:
`start_x`, `end_x` (shared base), `reception_nearest_opponent_distance_m`,
`reception_opponents_within_5m` (locked), plus `reception_opponents_within_8m`
(context, not itself locked).

| feature | mean | median | stddev | min | max | p1 | p5 | p95 | p99 | n_null |
|---|---|---|---|---|---|---|---|---|---|---|
| `start_x` | 57.67 | 57.30 | 25.75 | 0.6 | 120.0 | 6.0 | 13.6 | 99.3 | 113.2 | 0 |
| `end_x` | 59.84 | 59.80 | 26.21 | 0.1 | 120.0 | 5.3 | 14.7 | 103.1 | 112.8 | 0 |
| `reception_nearest_opponent_distance_m` | 7.31 | 6.54 | 4.63 | 0.02 | **42.21** | 0.53 | 1.21 | 15.77 | 20.63 | 154 |
| `reception_opponents_within_5m` | 0.50 | 0.0 | 0.77 | 0 | 9 | 0.0 | 0.0 | 2.0 | 3.0 | 154 |
| `reception_opponents_within_8m` (context) | 1.13 | 1.0 | 1.24 | 0 | 11 | 0.0 | 0.0 | 3.0 | 5.0 | 154 |

**Bounds check.** `start_x`/`end_x` have no out-of-range values on this population (max
exactly 120.0 both). `reception_nearest_opponent_distance_m`'s min (0.02m) is
physically sensible -- an opponent standing essentially on top of the receiver at the
moment of reception. Its max, **42.21m**, is a genuine outlier worth flagging: on an
assumed-105x68m pitch the maximum possible distance between two points is the
diagonal, ~125m, so 42m is within physical possibility (a completely unmarked
receiver far from the nearest visible opponent, e.g. a switch of play into space on a
break) but sits roughly 2x above the p99 (20.6m) -- a long right tail, not a bug, but
a value a linear model would weight heavily if not addressed.

**Missingness.** 154 of 133,143 rows (0.116%) have no computed reception-geometry
value across all three 360 features (same 154 rows for all three -- one shared root
cause: no valid-coordinate actor row in that reception's 360 frame). Already
documented in the pre-model analysis; restated here because it's directly relevant to
how these 3 numeric features get encoded (see the "what would break a first baseline"
section).

**Shape.** `reception_nearest_opponent_distance_m` decile edges: `[0.02, 1.92, 3.16,
4.25, 5.37, 6.49, 7.83, 9.30, 11.02, 13.59, 42.21]` -- smooth right-skew, no bimodality,
with the same "compressed final decile" long-tail pattern as `start_x` in Part A.
`reception_opponents_within_5m`/`_8m` are **zero-inflated count features**: the 5m
version has a median of 0 and its first 7 deciles are all 0 -- meaning over 70% of all
receptions (not just non-creating ones) have zero opponents within 5m at the moment of
reception. The 8m version is less extreme (median 1, first 4 deciles at 0) but still
clearly a sparse count, not a smooth continuous distribution.

**Implication:** the zero-inflation on `reception_opponents_within_5m`/`_8m` is
expected given the population-wide 2.1% create rate (most receptions genuinely aren't
under close pressure), not a defect -- but it means these two features carry most of
their information in a small number of non-zero values, which matters for
discretization/binning choices at training time (see final section).
`reception_nearest_opponent_distance_m`'s long right tail is a candidate for clipping
or a log-style transform before use in a linear baseline; less of a concern for a
tree-based model, which naturally handles skew via splits.

### B2. Categorical/boolean locked features: level counts, rare-level check

**Question:** same as A2, for CxA+'s locked categorical/boolean features (7 shared
with the event-only list, plus `pass_technique_name = Straight`, which is locked
alongside `Outswinging` as part of the same categorical feature in the CxA+ lock, and
`pass_body_part_name = No Touch`, the held-out feature -- included for context per this
task's instructions).

| feature | total rows | TRUE rows | % TRUE |
|---|---|---|---|
| `play_pattern_name = From Corner` | 133,143 | 3,851 | 2.892% |
| `is_switch` | 133,143 | 2,508 | 1.884% |
| `pass_type_name = Free Kick` | 133,143 | 2,407 | 1.808% |
| `is_cross` | 133,143 | 1,094 | 0.822% |
| `play_pattern_name = From Counter` | 133,143 | 795 | 0.597% |
| `pass_type_name = Corner` | 133,143 | 588 | 0.442% |
| `is_through_ball` | 133,143 | 252 | 0.189% |
| `pass_technique_name = Outswinging` | 133,143 | 213 | 0.160% |
| `pass_body_part_name = No Touch` (held out) | 133,143 | 135 | 0.101% |
| `is_cut_back` | 133,143 | 140 | 0.105% |
| **`pass_technique_name = Straight`** | 133,143 | **25** | **0.019%** |

**Rare-level check: 1 flagged.** `pass_technique_name = Straight` has only **25 total
rows in the entire CxA+ population** -- well under the 100-row floor, and far below
any other locked level. This was not caught by the feature-lock's own split check
(which only individually re-verified `Outswinging`, not `Straight` -- see
`docs/analysis/cxa_plus_p_create_feature_lock_v1.md`, section 1, which explicitly
notes `Straight` was carried into the lock from the pre-model analysis's
full-*event-only*-population finding and "not separately re-verified against the split
here"). With 25 total rows, a train/validation/test split by match could easily put
most or all of them in one split, and a model has no realistic way to learn a stable
pattern from single-digit-to-low-double-digit examples. `is_cut_back` (140 rows) and
`pass_body_part_name = No Touch` (135 rows) clear the 100-row floor but not by a wide
margin, and both were already flagged for thin support at the split-lift stage (21 and
22 validation rows respectively) -- this full-population check confirms that thinness
is a population-size property, not just a split-partition artifact.

**Implication:** `pass_technique_name = Straight` needs explicit handling before
training -- pool it into a catch-all "other technique" level together with the
(regular/untagged) baseline rather than encoding it as its own level, or drop the level
entirely and keep only `Outswinging` as the locked signal from this categorical
feature for CxA+. This is a real, previously-unflagged finding this task exists to
surface -- see the final section.

### B3. Full pairwise correlation matrix across the locked set

**Question:** same as A3, across CxA+'s locked set (15 encoded columns: `start_x`,
`end_x`, `reception_nearest_opponent_distance_m`, `reception_opponents_within_5m`,
`reception_opponents_within_8m` (context), plus 10 boolean/one-hot columns including
the held-out `No Touch` -- 105 pairs).

**7 new findings at `|r| >= 0.5`** (again excluding `start_x`<->`end_x`, r=0.75 on this
population, already known/accepted):

| pair | r | reading |
|---|---|---|
| `reception_opponents_within_5m` <-> `reception_opponents_within_8m` | **0.7622** | Expected -- nested radius bands around the same receiver position; the 8m count is close to a superset of the 5m count. Both are locked/context features built from the same underlying opponent-distance computation, so this is analytically expected, not a coincidental redundancy. |
| `reception_nearest_opponent_distance_m` <-> `reception_opponents_within_8m` | **-0.7065** | Same underlying geometry -- a smaller nearest-opponent distance mechanically tends to co-occur with more opponents captured in an 8m radius. |
| `reception_nearest_opponent_distance_m` <-> `reception_opponents_within_5m` | **-0.6423** | Same relationship, tighter radius, slightly weaker correlation than the 8m version (as expected -- a smaller radius is a noisier summary of the same underlying distance field). |
| `pass_type_name=Corner` <-> `play_pattern_name=From Corner` | **0.7829** | Same phenomenon as Part A3 (r=0.83 there), slightly lower on this smaller tournament-only population -- not a new mechanism, restated for completeness since this is the CxA+-specific number. |
| `play_pattern_name=From Corner` <-> `pass_technique_name=Outswinging` | **0.5584** | New to this population: corner deliveries in tournament football are disproportionately outswinging (a common attacking-corner technique), so the two locked flags co-activate more than half the time one is true. Below the CxA+-specific individual-signal thresholds each already cleared on their own, so both stay locked, but expect them to share credit in a fitted model. |
| `is_through_ball` <-> `pass_technique_name=Outswinging` | **-0.5792** | A genuinely new and sensible finding: through balls and outswinging deliveries are close to mutually exclusive techniques (a through ball is a flat/driven pass into space, an outswinging delivery is a curved cross/corner/free-kick technique) -- this negative correlation is the categorical-feature analogue of A4's mutual-exclusivity crosstabs, just surfaced here via the correlation matrix instead of a crosstab. |
| `start_x` <-> `pass_technique_name=Outswinging` | **0.5117** | Outswinging deliveries in this population originate from high-`start_x` positions (attacking-third corners/wide free kicks), consistent with the population composition (tournament football's outswinging passes are overwhelmingly set-piece deliveries near the byline). |

**Implication:** the three `reception_*` 360 features (`nearest_opponent_distance_m`,
`opponents_within_5m`, `opponents_within_8m`) are meaningfully correlated with each
other (all three pairwise |r| between 0.64 and 0.76) because they are computed from
the same underlying frame geometry -- expected, and the reason `opponents_within_8m`
was only ever context, not itself a locked feature (`reception_nearest_opponent_
distance_m` and `reception_opponents_within_5m` are the two that were independently
split-confirmed and locked; carrying `_8m` into a model alongside both would be adding
a third, highly-correlated view of the same signal for little marginal information).
The categorical correlations (`Corner`<->`From Corner`, `From Corner`<->`Outswinging`,
`is_through_ball`<->`Outswinging`) are all explainable set-piece-technique
co-occurrence patterns, not redundancy to eliminate, but worth knowing about for
coefficient interpretation in a linear baseline.

### B4. Interaction sanity check

**Question:** same pairs as A4, recomputed on the CxA+ population.

| pair | both TRUE | A only | B only | reading |
|---|---|---|---|---|
| `is_cross` & `pass_type_name=Corner` | **0** | 1,094 | 588 | **Different from the event-only track**, where this pair co-occurred 198 times (3.2% of corners). On CxA+'s tournament-only population, `is_cross` and `pass_type_name=Corner` never co-occur -- fully mutually exclusive here. This is a real population-composition difference worth noting explicitly (not an error): the tournament subset happens to contain none of the corner-deliveries-also-tagged-as-crosses cases that exist in the wider Premier-League-inclusive event population. A modeller relying on intuition from the event-only track's 3.2% overlap should not assume the same low-level co-occurrence holds in CxA+. |
| `is_cross` & `is_through_ball` | 10 | 1,084 | 242 | Near-exclusive (10/1,094 = 0.9%), consistent with the event-only track's 0.3%. |
| `is_cross` & `is_cut_back` | 85 | 1,009 | 55 | **Same substantial overlap as the event-only track**: 85/140 (60.7%) of cut-backs are also crosses -- consistent with A4's finding, confirming it holds in the tournament-only population too, not an event-only artifact. |
| `pass_type_name=Corner` & `is_through_ball` | 0 | 588 | 252 | Fully mutually exclusive, same as event-only. |
| `pass_type_name=Corner` & `is_cut_back` | 0 | 588 | 140 | Fully mutually exclusive on this population (event-only had 5/6,224 = 0.08% overlap; at CxA+'s much smaller scale, that rate would predict well under 1 expected co-occurrence, consistent with observing 0). |
| `is_through_ball` & `is_cut_back` | 0 | 252 | 140 | Fully mutually exclusive, same as event-only. |

**Implication:** the `is_cross`/`is_cut_back` overlap (roughly two-thirds of cut-backs
are also crosses) holds consistently across both tracks and both population sizes --
this is a stable, real relationship a modeller should plan around, not a fluke of one
population. The `is_cross`/`pass_type=Corner` difference between tracks (3.2% overlap
in event-only, 0% in CxA+) is population composition, not a bug, but worth stating
explicitly since it's the kind of thing that looks like a discrepancy until explained.

---

## What would break a first baseline model

Concrete, actionable items surfaced by this EDA that should shape feature encoding
before a model-training prompt gets written -- not new modelling decisions, just
encoding hygiene this analysis is positioned to catch:

1. **`pass_technique_name = Straight` (CxA+ only) needs pooling or dropping before
   training.** 25 total rows across the whole population (B2) -- not caught by the
   split-lift check because it was never individually re-verified there. Pool into an
   "other/untagged technique" catch-all alongside the null level, or drop the level
   and keep only `Outswinging` from this categorical feature for CxA+.
2. **`start_x` needs a trivial `[0, 120]` clip (event-only).** 2 rows out of 608,722
   sit fractionally past the pitch boundary (120.7/120.9) -- a known, benign
   corner-kick-position quirk (A1), not worth investigating further, but should be
   clipped before any derived-distance feature is computed from it.
3. **`reception_nearest_opponent_distance_m`'s long right tail (max 42.2m, p99 20.6m)
   is a clipping/transform candidate for a linear baseline** (B1) -- a tree-based
   model handles this natively via splits and needs no change.
4. **154 CxA+ rows (0.12%) have no computed 360-reception geometry at all** (B1,
   restated from the pre-model analysis since it's directly relevant here) -- decide
   drop-vs-impute-with-a-missingness-flag for `reception_nearest_opponent_
   distance_m`/`reception_opponents_within_5m` before training; do not silently
   impute a distance value that would misrepresent "frame had no valid opponent
   coordinates" as "opponent very close" or "opponent very far."
5. **`reception_opponents_within_5m` (and the `_8m` context feature) are
   zero-inflated count features**, not smooth continuous ones (B1) -- over 70% of all
   receptions have 0 opponents within 5m. A linear baseline should consider this
   feature's discreteness (few distinct integer values, heavily mass-at-zero) rather
   than treating it as an ordinary continuous input; a tree-based model handles this
   natively.
6. **`is_cross` and `is_cut_back` substantially overlap in both tracks** (A4, B4 --
   roughly two-thirds of cut-backs are also crosses) -- not a reason to drop either
   (both independently split-confirmed), but a modeller should expect their
   contributions to partially trade off rather than add independently, and should not
   be surprised by shared/unstable coefficients in a linear model.
7. **`pass_type_name=Corner` and `play_pattern_name=From Corner` are meaningfully
   correlated in both tracks** (r=0.83 event-only, r=0.78 CxA+, A3/B3) -- same
   trade-off-not-redundancy note as #6.
8. **CxA+'s three `reception_*` 360 features are mutually correlated (|r| 0.64-0.76,
   B3)** because they share underlying frame geometry -- this is exactly why
   `reception_opponents_within_8m` was kept as context only and never locked; a
   modeller should not add it as a third feature alongside the two that are locked
   without a specific reason, since it would mostly restate information the other two
   already carry.
9. **No feature needs re-examining on the direction/support grounds the earlier
   feature-lock docs already covered** -- this EDA found nothing that contradicts
   either lock's promotion decisions; findings 1-8 above are additive encoding
   guidance, not a reason to revisit either locked list.
