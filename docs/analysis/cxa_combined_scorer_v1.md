# CxA Combined Scorer v1 -- Materialized Build

Date: 2026-09-20
The first build step after
[`docs/analysis/cxa_combined_scorer_design_v1.md`](cxa_combined_scorer_design_v1.md)
(the reviewed spec this task implements exactly, not redesigns). Still backend/data
only -- no API routes, no frontend changes, per this task's explicit scope. Six
decisions from the design doc's open questions were binding inputs, restated where
relevant below, not re-litigated.

Reproducible via
[`scripts/materialize_cxa_combined_v1.py`](../../scripts/materialize_cxa_combined_v1.py);
raw output under
[`audit_outputs/cxa_combined/v1/`](../../audit_outputs/cxa_combined/v1/).

**Does not touch, modify, or re-freeze any `oam_ml.*_frozen_v1_config` table** -- both
tracks' P_create and P_convert frozen configs (model family, feature list,
hyperparameters) were read live and used exactly as stored, never edited.

## 1. What was built

`oam_serving.cxa_event_combined_v1` and `oam_serving.cxa_plus_combined_v1` --
**the first two tables ever written to `oam_serving`**, confirmed empty immediately
before this run (per the design doc's own live check, unchanged since). Schema
exactly as the design doc's section 2c proposed, no changes:

`pass_event_id`, `match_id`, `split`, `p_create_predicted_prob`,
`p_create_model_family`, `p_create_model_version`, `y_create`, `shot_event_id`,
`p_convert_predicted_prob`, `p_convert_model_family`, `p_convert_model_version`,
`cxa_combined_score`, `y_goal`, `materialized_at`, `source_docs`.

**Grain: one row per `pass_event_id`, covering every row of that track's P_create
population** (all three splits -- decision 4 below), not just the chance-creating
subset -- `p_convert_predicted_prob`/`cxa_combined_score`/`y_goal`/`shot_event_id` are
NULL, never a placeholder, wherever `y_create = FALSE`.

**Fit approach:** each frozen model (family, feature list, hyperparameters all read
live from `oam_ml.cxa_{track}_frozen_v1_config` / `oam_ml.cxconvert_{track}_frozen_v1_
config`, never hardcoded) was refit on **train+validation combined** -- the same final
fit `materialize_cxa_test_eval_v1.py` / `materialize_cxconvert_test_eval_v1.py`
already used to produce this project's own reported test metrics -- then scored over
the full population (all splits) to produce every row's predicted probability.
P_convert's own per-track model-family split (LightGBM tree for event-only, logistic
MLE for CxA+, per its freeze decision) required no special handling here beyond
branching on `model_family`, confirming the design doc's section 3 claim that this
split never complicates the serving/materialization layer.

### Decisions carried forward from the design doc, binding for this task

1. **"CxA" = the combined product.** `cxa_combined_score = p_create_predicted_prob *
   p_convert_predicted_prob`, computed only where both factors exist.
2. **Scope is the table + the combined-quality metric only.** No per-pass display, no
   API route, no frontend change was built -- see section 4.
4. **All three splits materialized.** `oam_serving.cxa_{track}_combined_v1` contains
   `train`, `validation`, and `test` rows, mirroring CxG's own real precedent
   (`oam_ml.cxg_event_v3_predictions` also carries every split) -- **any future public
   API must filter to `split='test'` only**, matching CxG's display discipline, but
   that filtering is explicitly not built here; nothing in this table forecloses it.

## 2. Row-count verification (checked before writing anything, per this task's
explicit instruction)

Verified in-script, before either table was written -- a `RuntimeError` would have
been raised and nothing written had any of these not matched exactly (see
`materialize_cxa_combined_v1.py`'s `run_track`, the `problems` check). Independently
re-confirmed afterward with a fresh read-only query against the written tables, split
by split:

| track | split | total rows | `cxa_combined_score` non-null | `y_create=TRUE` count |
|---|---|---|---|---|
| event | train | 422,946 | 7,847 | 7,847 |
| event | validation | 92,977 | 1,720 | 1,720 |
| event | test | 92,799 | 1,736 | 1,736 |
| **event total** | | **608,722** | **11,303** | **11,303** |
| plus | train | 95,083 | 1,991 | 1,991 |
| plus | validation | 19,490 | 420 | 420 |
| plus | test | 18,570 | 419 | 419 |
| **plus total** | | **133,143** | **2,830** | **2,830** |

**Every number matches this project's own already-established population sizes
exactly** -- event-only's total (608,722) and coverage (11,303) match the P_create and
P_convert freeze docs' own reported sizes precisely; CxA+'s total (133,143) and
coverage (2,830) likewise. `cxa_combined_score`'s non-null count equals
`p_convert_predicted_prob`'s non-null count equals `y_create=TRUE`'s count, in every
split, in both tracks, with zero discrepancy anywhere -- the join is exact, not
approximately close. Nothing was written until this was confirmed.

## 3. Combined-quality metric (new analysis -- does not validate the freeze
decisions)

Per this task's item 5: `log_loss`/`roc_auc`/`brier_score` of `cxa_combined_score`
against `y_goal`, restricted to `y_create=TRUE`, **test split only**, both tracks --
and, for comparison, the same three metrics for `p_convert_predicted_prob` alone
against the same target and population.

**Stated explicitly, per the task's own instruction: this is a genuinely new question,
not a re-run of anything either frozen model was selected on.** P_create was selected
on `y_create`; P_convert was selected on `y_goal` restricted to `y_create=TRUE` --
neither was ever evaluated on "how good is the *product* of the two." This section
answers that new question; it is not evidence for or against either freeze decision.

| track | metric | `cxa_combined_score` vs `y_goal` | `p_convert_predicted_prob` alone vs `y_goal` | delta |
|---|---|---|---|---|
| event | n | 1,736 | 1,736 | -- |
| event | log_loss | **0.34806** | **0.24646** | +0.10160 (41.2% higher/worse) |
| event | brier_score | 0.07652 | 0.06841 | +0.00810 (11.8% higher/worse) |
| event | roc_auc | 0.73551 | 0.76463 | -0.02912 |
| plus | n | 419 | 419 | -- |
| plus | log_loss | **0.30232** | **0.24964** | +0.05269 (21.1% higher/worse) |
| plus | brier_score | 0.07580 | 0.07042 | +0.00538 (7.6% higher/worse) |
| plus | roc_auc | 0.73557 | 0.79388 | -0.05831 |

**Multiplying by `p_create` makes the combined score a worse predictor of `y_goal`
than `p_convert` alone, on every metric, in both tracks -- reported plainly, not
softened.** This is a real, consistent finding across both tracks and all three
metrics, not a borderline or noisy result: log_loss is 21-41% worse, Brier is 8-12%
worse, and AUC drops by 0.029-0.058 (CxA+'s drop is the larger of the two, consistent
with that track's smaller, noisier population generally producing larger metric
swings throughout this project's history, restated as a caveat here too).

**Why this happens, reasoned from what `p_create` actually measures for this specific
population:** `p_create` answers "how likely was this pass, in general, to create a
chance at all" -- but every row in this evaluation is already restricted to
`y_create=TRUE` (the pass *did* create a chance, by construction). Conditioned on that
fact already being true, `p_create`'s remaining variation among these passes (which
ones looked more or less surprising/likely to succeed) carries close to no information
about whether the *resulting shot* converts -- pass surprisal and shot conversion are
different questions. Multiplying `p_convert`'s well-calibrated, purpose-built
prediction by a second, systematically-compressed factor (event-only's `p_create`
values for real chance-creating passes are necessarily well below 1.0, since the
model was never asked to be confident about a fact already known to be true in this
subset) both shrinks every prediction toward 0 in a way `y_goal`'s own base rate
doesn't warrant, and adds `p_create`'s own noise to the ranking, which is why AUC drops
too, not just calibration. **This is not a flaw in either frozen model** -- both
continue to perform exactly as their own test-eval documents reported, restated here
unchanged (event `p_convert` test log_loss 0.24646/AUC 0.76463; plus 0.24964/0.79388,
both identical to the prior test-eval doc's own numbers, confirming this task's refit
did not silently diverge from the already-reported result). **It is a property of the
product itself as a predictor of `y_goal` specifically** -- a distinct fact from
"is `p_create x p_convert` still the right definition of a combined CxA score" (section
1c of the design doc already answered that question on different grounds -- describing
the *quality of the real assist chain*, not *predicting the goal outcome* -- and this
finding doesn't change that answer, it just means the combined number shouldn't be
marketed as a better goal predictor than `p_convert` alone, because it demonstrably
is not one).

## 4. What's next (explicitly deferred, not done here)

Per the design doc's own open questions and this task's explicit scope boundary:

- **API routes** (`GET /v1/models/cxa-models`, `GET /v1/cxa/coverage`, etc., per the
  design doc's section 4 sketch) -- not built. A future task, reading from the two
  tables materialized here.
- **Frontend changes** -- not built. Includes the design doc's open question 3 (two
  separate model cards, event-only and CxA+, mirroring today's CxG/CxG+ split, vs. one
  merged card) and open question 6 (whether CxA+'s small, tournament-only population
  needs a more prominent badge/treatment than event-only beyond a shared caveat
  sentence) -- both still open, to be decided when that step is scoped, not here.
- **The `split='test'`-only public filtering rule** (design doc decision 4) -- the
  materialized tables intentionally carry all three splits so this filtering can be
  applied at the API/display layer later, matching CxG's actual precedent exactly;
  no filtering logic exists yet anywhere in this codebase for these new tables.
- **A player-level aggregate table** -- the design doc's section 2b was explicit that
  this per-pass table is not, by itself, the full scope of Hard gate 2 (which names
  player-level CxG/CxA values for Track B's quadrant scatter, a different grain) --
  not attempted here.
