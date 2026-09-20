# CxA Combined Scorer: Design and Feasibility v1

Date: 2026-09-20
**This is a design document only.** No scoring pipeline, no BigQuery writes, no API or
frontend changes. It proposes a reviewed spec for combining the now-frozen,
test-evaluated P_create and P_convert models into a single servable "CxA" artifact,
following the same discipline every P_create/P_convert step so far used: read what
exists, reason from it, don't assume, state a recommendation, flag what's genuinely
undecided.

Built on:
[`docs/analysis/cxa_p_create_freeze_v1.md`](cxa_p_create_freeze_v1.md) /
[`cxa_p_create_test_eval_v1.md`](cxa_p_create_test_eval_v1.md) (frozen P_create:
LightGBM tree, both tracks),
[`docs/analysis/cxa_p_convert_freeze_v1.md`](cxa_p_convert_freeze_v1.md) /
[`cxa_p_convert_test_eval_v1.md`](cxa_p_convert_test_eval_v1.md) (frozen P_convert:
LightGBM tree for event-only, logistic MLE for CxA+ -- **not** the same family as
P_create, and not the same family across P_convert's own two tracks either).

All BigQuery facts below were read live during this task, not assumed or copied from
memory of the freeze docs.

---

## 1. The core question: what does a single "CxA score" mean, and for what population?

### 1a. Sanity check against CxG first, as instructed

**CxG does not face this problem at all -- it is architecturally simpler than CxA.**
Checked directly: `oam_ml.cxg_event_v3_coefficients` (and `cxg_plus_v2`/`cxg_plus_v3`)
confirm CxG is a **single-stage** model, `P(shot is a goal | a shot happened)`, fit and
scored over the population of shots that already exist in `oam_core.shots`. There is
no upstream "did a shot happen at all" model CxG multiplies against -- a shot is a shot,
observed directly, not the output of a separate prior-stage prediction. CxG's own
per-shot predictions table (`oam_ml.cxg_event_v3_predictions`) has exactly one
predicted-probability column per model version, keyed on `event_id`, covering every
shot in that split. **There is no existing "combine two models over mismatched
populations" precedent anywhere in this codebase to copy.** CxA's two-stage
create-then-convert structure is genuinely new territory for this project's serving
layer, not a case of "do what CxG already does."

### 1b. Working through the three framings

**Option 1 -- retrospective combined score, chance-creating passes only.** For each
pass where `y_create = TRUE` (a real pass -> shot pair -- exactly the population
P_convert's own training matrices already use, confirmed: `oam_ml.cxconvert_event_
test_v1_predictions` has 1,736 rows for event-only test, matching every P_convert
document's reported test size exactly), `CxA = P_create(pass) x P_convert(pass ->
shot)`. Well-defined, computable today from already-frozen models, and it is the
natural quantity for "how good was this specific, real assist chain" -- the same
framing traditional football analytics uses for "expected assists" (xA): the
probability the pass created a chance at all, times the probability that chance
converts. **Only defined for the passes that actually created a chance** -- every other
pass in the dataset has no P_convert input to multiply against, because P_convert's
shot-level features (`shot_x_sb`, `shot_gk_distance_m`, etc.) only exist because a real
shot happened. There is no way to impute them for a pass that never created one without
inventing a hypothetical shot, which is a different, much bigger modelling problem this
project has not attempted.

**Option 2 -- full-population "publish both, joined where applicable."** On inspection,
this is not a competing definition of "CxA" -- it is the correct **serving shape** for
Option 1, not an alternative to it. Every pass gets its own `P_create` value (defined
over the full population, exactly as P_create was trained and tested); only the subset
where a shot actually resulted additionally gets `P_convert` and the `CxA` product,
present as real values for that subset and **absent (NULL), never a placeholder,**
everywhere else. This is precisely the discipline the existing CxG coverage precedent
already enforces at the API layer (`src/opponent_adjusted/api/cxg_coverage.py`:
"`event_ids` with no coverage are simply absent from the result -- never a placeholder
value") -- just applied to a coverage gate that is now two conditions deep (split
coverage, the same as CxG, **and** `y_create = TRUE`) rather than one.

**Option 3 -- anything else the data supports.** Considered and rejected: scoring every
pass with `P_create` alone and separately, unconditionally, imputing a
population-average `P_convert` value for passes with no real shot (so every pass gets
*some* combined number) was considered and explicitly rejected -- it would fabricate a
shot-quality prediction for a shot that never happened, using features that don't
exist, and would produce a "CxA score" for 97%+ of passes that is not a prediction
about that pass at all, just a constant smuggled in as if it were personalized. This is
exactly the kind of silently-misleading number the task asked to be flagged, so it is
named and dismissed explicitly rather than left as an implicit fourth option.

### 1c. Recommendation

**Adopt Option 1's definition (`CxA = P_create x P_convert`, defined only where
`y_create = TRUE`), served via Option 2's shape (every pass gets `P_create`; `P_convert`
and `CxA` are populated only for the chance-creating subset, NULL -- not zero, not a
dash -- everywhere else).** This is not a compromise between two options; it is the
only technically honest combination once Option 3's alternative is ruled out.

**What would make this actively misleading if shipped without the right label, stated
explicitly per the task's instruction:** a headline "CxA score" presented the way CxG's
score is presented (as if every scored entity gets one) would silently cover only a
small fraction of a season's passes. Concretely, from this project's own already-
established population sizes: P_create's event-only population is
**608,722 passes**; P_convert's (the `y_create=TRUE` subset) is **11,303** -- **1.86%**.
CxA+ is **133,143** vs. **2,830** -- **2.13%**. A visitor seeing "CxA: 0.14" on a
generic pass with no further context would reasonably assume, by analogy with CxG's own
"every shot gets a value" precedent, that this is true for every pass. **It is not, and
the label must say so every place the combined score appears** -- e.g. "Combined CxA
(create x convert) -- chance-creating passes only, ~2% of all passes; undefined,
not zero, elsewhere," not a bare number. `P_create` alone, by contrast, genuinely does
cover (almost) every pass and can be labelled and shown the way CxG's xG is shown today,
with no analogous caveat needed.

---

## 2. Target schema: `oam_serving`

### 2a. Confirmed live: `oam_serving` exists and is currently empty

`bq ls oam_serving` (via the BigQuery MCP `list_table_ids`) returns **zero tables**.
Cross-checked against `docs/dashboard_design_spec_v2.md` §9, Hard gate 2: "`oam_serving`
must be populated with player-level CxG/CxA values... *Status: blocked -- `oam_serving`
has zero tables, confirmed live (26 Aug 2026)*." Still true today. This dataset was
provisioned specifically for exactly this kind of cross-model, dashboard-facing derived
artifact -- confirmed by reading the spec, not assumed from the dataset's name alone.

### 2b. Where should the new table actually live -- `oam_ml` or `oam_serving`?

**Checked, not assumed, and the answer is `oam_serving`, deliberately different from
where CxG's own per-shot predictions currently live.** CxG's `*_predictions` tables
live in `oam_ml` and are read live by the API (`cxg_coverage.py`) with no
`oam_serving` involvement at all -- `oam_serving` has never been used for anything.
But `oam_ml`'s existing tables are all single-model, single-fitting-script artifacts
(one `_predictions`/`_metrics`/`_coefficients` table per model version) -- internal ML
pipeline outputs, not cross-model derived products. The CxA combined score is
categorically different: it is a **derived join-and-multiply across two independently
frozen models**, produced specifically to be served, not a byproduct of fitting any one
model. That is exactly what the design spec's Hard gate 2 language describes
`oam_serving` as being for. **Recommendation: the new combined table(s) are the first
tables ever written to `oam_serving`.** This is a meaningful milestone worth stating
plainly -- it begins to satisfy Hard gate 2's stated condition -- but it does **not**,
by itself, complete Hard gate 2's full scope: that gate specifically names
**player-level** aggregated CxG/CxA values for Track B's quadrant scatter, a different
grain (one row per player-season) from what this design proposes (one row per pass).
Whether a player-level aggregate table is also needed is flagged as an open question in
section 6, not resolved here -- it is a natural follow-on, not the same deliverable.

### 2c. Proposed tables

Following the project's own established per-track naming convention
(`cxa_event_*`/`cxa_plus_*`, `cxconvert_event_*`/`cxconvert_plus_*`) rather than a
single unioned physical table (the two tracks have different frozen feature sets and,
for P_convert, different model families -- keeping them physically separate mirrors
every prior step's own choice, with unioning done at the read layer if ever needed, the
same way `bigquery_analysis_store.py`'s `list_cxg_model_results` already unions across
CxG's own separate per-track/per-version tables at query time):

- **`oam_serving.cxa_event_combined_v1`**
- **`oam_serving.cxa_plus_combined_v1`**

**Grain: one row per `pass_event_id`, covering every row in that track's P_create
population** (i.e. every row `oam_features.cxa_{track}_v1_training_matrix` has, not
just the `y_create=TRUE` subset) -- this is what makes Option 2's "join where
applicable" shape real rather than aspirational: a visitor or a downstream query can
`SELECT * FROM cxa_event_combined_v1 WHERE pass_event_id = ...` for **any** pass and get
a row back, with `p_convert_predicted_prob`/`cxa_combined_score` simply NULL when that
pass never created a chance.

Proposed columns:

| column | type | notes |
|---|---|---|
| `pass_event_id` | STRING | join key, matches every source table's own key |
| `match_id` | INTEGER | carried through for match-level filtering |
| `split` | STRING | `train`/`validation`/`test`, carried through -- see section 6 open question 4 on whether public serving should filter to `test` only, mirroring CxG's own discipline |
| `p_create_predicted_prob` | FLOAT | always populated (P_create's population is ~every pass) |
| `p_create_model_family` | STRING | `'lightgbm_tree'` for both tracks, read live from `oam_ml.cxa_{track}_frozen_v1_config.model_family`, not hardcoded |
| `p_create_model_version` | STRING | e.g. `'cxa_event_frozen_v1'` -- the literal source table name, see versioning below |
| `y_create` | BOOLEAN | ground truth, carried through -- this is the field that determines whether the `p_convert_*`/`cxa_combined_score` columns are populated |
| `shot_event_id` | STRING, NULLABLE | the linked shot's `event_id` where `y_create=TRUE`, for provenance back to `oam_core.shots` -- NULL otherwise |
| `p_convert_predicted_prob` | FLOAT, NULLABLE | **NULL, never a placeholder, where `y_create=FALSE`** |
| `p_convert_model_family` | STRING, NULLABLE | `'lightgbm_tree'` (event) or `'logistic_mle'` (CxA+) -- read live, differs by track, this is exactly the asymmetry the freeze docs already documented |
| `p_convert_model_version` | STRING, NULLABLE | e.g. `'cxconvert_event_frozen_v1'` |
| `cxa_combined_score` | FLOAT, NULLABLE | `p_create_predicted_prob * p_convert_predicted_prob`, NULL wherever `p_convert_predicted_prob` is NULL |
| `y_goal` | BOOLEAN, NULLABLE | ground truth for the resulting shot, only meaningful (non-NULL) where `y_create=TRUE` |
| `materialized_at` | TIMESTAMP | same convention as every prior table in this project |
| `source_docs` | ARRAY\<STRING\> | this doc + the four freeze/test-eval docs, same convention as the frozen-config tables |

### 2d. Model versioning -- so a future retrain doesn't silently overwrite history

**Reuse this project's own existing convention exactly, rather than inventing a new
one:** every prior artifact in this project is versioned by a suffix baked into the
**table name itself** (`cxg_baseline_v1` vs. `cxg_event_v3`, `cxa_event_frozen_v1`,
`cxconvert_event_frozen_v1`), never by a mutable version column inside one
continuously-overwritten table. A future retrain that changes P_create's or
P_convert's frozen config would produce a **new** frozen-config table
(`cxa_event_frozen_v2`, say) rather than modifying `_v1` in place -- confirmed by
inspection, every `write_table` call across this project's scripts uses
`WRITE_TRUNCATE` on a full, versioned table name, never an `UPDATE`/`MERGE`. The
combined scorer should follow the identical pattern: `cxa_event_combined_v1` is
immutable once written; a future change to either upstream frozen model produces
`cxa_event_combined_v2`, and `v1` is retained, not deleted, for historical
reproducibility (matching this project's practice of never deleting a prior version's
tables, e.g. `cxg_baseline_v1` still exists alongside `cxg_event_v3`). The
`p_create_model_version`/`p_convert_model_version` string columns exist specifically so
that, even within one combined-table version, it's traceable which exact frozen config
row produced each score -- relevant if a frozen config table is ever amended in place
(none currently are, but the column costs nothing and removes any ambiguity).

---

## 3. Serving pattern: does P_convert's two-family split complicate "materialize once,
read-only"?

**No.** Checked directly against how this project already materializes predictions:
every `_test_v1_predictions` table (both P_create's and P_convert's, both tracks) is
already produced by loading a fitted model -- LightGBM **or** statsmodels `Logit`,
whichever the frozen config specifies -- and calling `.predict_proba()` /
`.predict()` inside an offline Python script, then writing the resulting floats to
BigQuery (`materialize_cxconvert_test_eval_v1.py` already does exactly this,
branching on `model_family` read from the frozen config, for precisely this reason).
**Once the predicted probability is a float in a BigQuery column, the model family that
produced it is invisible to every downstream consumer** -- the API, the combined-score
join, and any dashboard component only ever read numbers, never call a model. The
two-family split is a fact about how the *materialization script* must branch (reuse
the exact branching logic `materialize_cxconvert_test_eval_v1.py` already implements),
not a fact that propagates to the serving layer at all.

**Confirmed: batch materialization is sufficient for every use case currently
identified** (Models-page display, per-pass coverage lookups, story content, a future
player-level aggregate) -- none of these require a model to be invoked at request time.
**Live inference would only become necessary for a genuinely different feature this
project does not currently have anywhere: scoring a hypothetical, user-supplied pass
that doesn't already exist in the historical dataset** (an interactive "what would CxA
predict for this pass" tool). No such feature exists today, in CxG's precedent or
anywhere else in this codebase, and this design does not propose building one -- flagged
as a possible future direction, not a requirement.

---

## 4. API surface sketch (sketch only -- not implemented)

Mirroring `/v1/models/cxg-models` and `/v1/cxg/coverage`'s existing naming and
ownership pattern:

- **`GET /v1/models/cxa-models`** -- model comparison rows, unioning P_create's and
  P_convert's own already-separate test metrics (`oam_ml.cxa_{track}_test_v1_metrics`,
  `oam_ml.cxconvert_{track}_test_v1_metrics`) the same way `list_cxg_model_results`
  already unions across CxG's four model-version tables. **Note a real asymmetry CxG
  never had:** there is no single well-defined "combined CxA test log_loss" the way
  CxG has one log_loss per model -- P_create and P_convert were each individually
  evaluated against their own targets (`y_create`, `y_goal`), and the *product's*
  quality against `y_goal` (restricted to the chance-creating population) is a
  different, not-yet-computed quantity that was never what either model was selected
  on. This endpoint should show P_create's and P_convert's own metrics as separate
  rows (clearly labelled by stage), not invent a combined metric that doesn't exist
  yet -- if a genuine combined-quality metric is wanted later, it needs its own careful
  definition and evaluation, out of scope for this design.
- **`GET /v1/models/cxa-models/{model_key}/coefficients`** -- mirrors the CxG pattern,
  but **only meaningful for CxA+'s P_convert** (`logistic_mle`, the only one of CxA's
  four sub-models -- P_create x2 tracks, P_convert x2 tracks -- that is a logistic fit
  at all; the other three are LightGBM trees with no coefficient table). CxG never
  faced this because every one of its frozen models is logistic. Propose a parallel
  `feature_importances` shape (LightGBM's built-in split/gain importances, already
  captured in this project's own audit JSON output at the baseline/candidate and
  freeze steps) for the three tree-family sub-models, surfaced under the same general
  "model explainability" concept but a different response shape -- sketch only.
- **`GET /v1/cxa/coverage`** -- mirrors `cxg_coverage.py`'s `CxgCoverageStore` exactly:
  given a track and a list of `pass_event_id`s, return `{pass_event_id: {p_create,
  p_convert, cxa_combined_score}}` for whichever are covered by the relevant
  `oam_serving` table (and whichever split-coverage policy section 6 lands on) -- ids
  with no chance-creation coverage return `p_create` only (never a placeholder
  `p_convert`), ids with no coverage at all are simply absent, identical discipline to
  the existing CxG endpoint.
- **`GET /v1/cxa/matches`** -- optional, mirrors `/v1/cxg/matches`'s per-match scope
  lookup, if "which matches have CxA coverage at all" turns out to be needed by a
  future dashboard component the way it already is for CxG's Explore-zone badges.

---

## 5. Dashboard integration sketch (sketch only)

`web/lib/models-data.ts` already has a placeholder `CxA` entry in its data-driven
`MODELS` array (`status: "planned"`, empty `validationMetrics`, `detailModelKey:
undefined`) -- the exact slot this would fill once real numbers exist, requiring no new
component. Following CxG's own precedent (`status: "evaluated"`, not `"promoted"`,
specifically because `oam_serving` populated status still gated a "promoted" label per
that file's own comment): flip CxA's status to `"evaluated"`, populate
`validationMetrics` with P_create's test AUC/log_loss as the headline pair (it's the
metric that covers the (almost) full population, the fairest single-number summary),
add a `comparisonNote` stating the combined-score coverage caveat from section 1c
explicitly (not left to a detail page a visitor might not click through to), and set
`detailModelKey` once `/v1/models/cxa-models` exists to power a `/models/cxa-event`-
style detail page via the existing `ModelCard`/detail-page components, no new ones
needed. Whether the **per-pass** combined score gets its own visual surface (e.g. a
shot-map-style "assist chain" view on player pages) is a separate, larger design
question this document deliberately does not resolve -- it would need its own
coverage-gating UI treatment at least as careful as CxG's existing "shown alongside
StatsBomb xG, coverage-gated, no placeholder" rule (`dashboard_design_spec_v2.md` §4a),
extended to the two-deep coverage condition this document has described throughout.

---

## 6. Open questions for Varun to decide

Genuinely ambiguous, not something this task is positioned to decide unilaterally:

1. **Naming/product framing.** Should "CxA" refer to the combined product (this
   document's working assumption throughout) or should "CxA" mean P_create alone (the
   full-population "chance creation quality" score), with the conversion-weighted
   product given a distinct name (e.g. "CxA+conversion" or similar) so the
   coverage-gap caveat isn't baked into the primary brand name of the metric at all?
2. **Does the public dashboard need per-pass combined scores in the near term**, or is
   a Models-page aggregate comparison (mirroring CxG's current "evaluated, not
   promoted" treatment) sufficient for now, with per-pass display deferred behind its
   own future Hard-gate-style blocking condition -- the same way Track B's quadrant
   scatter is currently explicitly blocked rather than half-built?
3. **Event-only vs. CxA+ presentation:** shown as two separate model cards (mirroring
   today's CxG/CxG+ split) or merged into one "CxA" card with a track toggle/filter?
4. **Split-coverage display discipline.** Does the served combined score follow CxG's
   conservative "only ever show `split='test'` rows publicly" rule (avoiding any
   confusion between in-sample-fit numbers and genuine generalization), or does it
   show the full historical population (`train`+`validation`+`test`, `split`-labelled)
   for broader browsing use cases like a player's full pass history? This document
   leans toward CxG's conservative choice for anything framed as "model performance,"
   but a full-history browsing view is a legitimate, different use case that would
   need its own explicit label ("in-sample" vs. "held-out") rather than being silently
   conflated -- Varun's call.
5. **Whether a genuine combined-quality metric is worth building at all** (log_loss of
   `cxa_combined_score` against `y_goal`, restricted to the chance-creating population)
   -- it doesn't correspond to what either frozen model was itself selected on
   (section 4), so it would be new analysis work, not a free byproduct of what already
   exists.
6. **Whether CxA+'s small, tournament-only population** (2,830 chance-creating passes
   total, 419 in test, zero Premier League rows -- restated once more here since it
   applies to this design too) warrants a more prominent, separate on-page treatment
   than event-only, beyond a shared caveat sentence.
