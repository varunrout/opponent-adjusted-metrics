# CxA P_convert Split Policy And Plan

Date: 2026-09-20
Mirrors [`docs/cxa_split_policy_and_parallel_plan.md`](cxa_split_policy_and_parallel_plan.md)
(P_create's split policy). Data facts referenced below are sourced from
[`cxa_convert_data_feasibility_audit.md`](cxa_convert_data_feasibility_audit.md); this
document does not re-derive them.

## Decision

**Reuse `oam_analysis.cxa_match_splits_v1` exactly as-is, read-only.** No new split
table is created for P_convert. Same reasons P_create's own split policy gave for
reusing CxG's split when it built this table in the first place: P_convert's
population (`Y_create = TRUE` passes) is a strict subset of P_create's own
already-split population, over the identical match set. Drawing an independent split
here would risk a match landing in P_create-train but P_convert-test, which would make
it impossible to later build any feature or evaluation spanning both stages without
leakage -- exactly the argument P_create used when it inherited CxG's split instead of
drawing its own. Splitting by shot or pass row (instead of by match) remains explicitly
wrong for the same reason it always has been: two shots/passes from the same match
share possession structure and tactical context.

**Verified, not assumed** (per `cxa_convert_data_feasibility_audit.md` section 5):
every `Y_create = TRUE` row's `match_id` exists in `cxa_match_splits_v1`, zero
exceptions, in both tracks. There is nothing to handle for unmatched rows because
there are none -- this is stated as a verified fact, not a logical inference from "it's
a subset so of course it's covered."

## Split sizes and class balance for the P_convert population

Unlike P_create (600K+ rows per track), P_convert's population is small (11,303 /
2,830 rows) -- checked explicitly whether this causes any split to be too thin,
rather than assuming the same match-level split that worked for P_create's much larger
population automatically works here too:

### Event-only (11,303 rows, 610 matches)

| split | matches | rows | Y_goal=TRUE | goal-conversion rate |
|---|---|---|---|---|
| train | 426 | 7,847 | 738 | 9.405% |
| validation | 92 | 1,720 | 159 | 9.244% |
| test | 92 | 1,736 | 141 | 8.122% |

### CxA+ (2,830 rows, 166 matches)

| split | matches | rows | Y_goal=TRUE | goal-conversion rate |
|---|---|---|---|---|
| train | 119 | 1,991 | 181 | 9.091% |
| validation | 24 | 420 | 46 | 10.952% |
| test | 23 | 419 | 36 | 8.592% |

**Checked, flagged honestly, not smoothed over:** event-only's three splits sit within
a ~1.3 percentage-point band (8.12%-9.41%) -- a real but modest spread, plausible
sampling variance given train has 4.6x validation/test's row count. CxA+'s spread is
wider (8.59%-10.95%, a ~2.4pp band) -- validation's rate is visibly higher than train
or test. With only 420 validation rows (46 positives), this is the kind of swing a
small sample produces; it is **not** evidence the split is broken (the split itself is
inherited unchanged from P_create, which already validated its balance on the much
larger `Y_create`-level population), but it is a real property of this
target at this population size that any future validation-stage work on CxA+
P_convert should account for -- metrics computed on 420 validation rows will be
noisier than P_create's own validation-stage metrics were, and should be read with
that in mind rather than compared at face value to P_create's tighter numbers.

**No split is too thin to use.** Every split in both tracks has at least 36 positive
examples (CxA+ test, the smallest) -- thin compared to P_create's thousands of
positives per split, but not so thin that a split is unusable; this is simply the
reality of modelling a subset-of-a-subset population, to be kept in mind (not solved)
at the feature-promotion and modelling stages, not this one.

## Required design (unchanged from P_create, restated for a self-contained doc)

- Split by `match_id`, never by pass row, never by shot row, never by `possession_id`.
- Train 70% / validation 15% / test 15% of matches -- inherited proportions, not
  re-derived (same match assignment as P_create).
- Test stays sealed until a final P_convert model report, exactly as P_create's own
  test was sealed through steps 1-9 and opened only once, at step 10.
- `oam_analysis.cxa_match_splits_v1` is read-only from this task forward for
  P_convert -- not modified, not re-derived, not re-balanced for this population's
  different class rate. Column semantics carried into the P_convert training matrices
  unchanged: `split`, `split_seed`, `has_360_match`.

## What this stage does NOT do

Mirroring P_create's own staged sequence exactly:

1. This task (data audit + pipeline): materialize
   `oam_features.cxconvert_event_v1_training_matrix` /
   `cxconvert_plus_v1_training_matrix` with the full unfiltered candidate feature set,
   validated for grain/join/split correctness. **Done in this task.**
2. Pre-model target and feature analysis (full population, per the "EDA is valid
   exploratory work" carve-out P_create's own split policy established) -- **not
   started here.**
3. Train-only feature confirmation on the validation split -- **not started here.**
4. Baseline -> candidate model comparison -- **not started here.**
5. Freeze (model family, feature set, hyperparameters) -- **not started here.**
6. Sealed test evaluation, run once -- **not started here.**

This document and the feasibility audit it accompanies stop at step 1.
