# CxA Detail Page v1

Date: 2026-09-21
A working CxA detail page (P_create + P_convert breakdown per track), reachable from
both Models-page cards. Reads already-materialized/already-frozen backend data;
adds one new, minimal explainability materialization step (section 2) after
investigation found the data the task expected to already exist did not actually
exist anywhere.

**Branch note:** [`feature/cxa-dashboard-models-page`](../../docs/analysis/cxa_dashboard_models_page_v1.md)
was not on `main` yet -- merged into this branch first (fast-forward, no conflicts),
same pattern as every prior CxA step.

## 1. Architecture decision: a separate route, not a generalized `/models/[modelKey]`

**Investigated first, per the task's own instruction, not defaulted to the easier
option.** Read `web/app/models/[modelKey]/page.tsx` in full before deciding:

- It hardcodes `REAL_MODEL_KEYS = ["baseline_v1", "event_v3", "plus_v2", "plus_v3"]`
  and calls `getPublicCxgModelResults()`/`getPublicCxgModelCoefficients(modelKey)`
  directly -- CxG-specific fetch functions, not generic ones (already confirmed in
  the prior task's own doc, re-confirmed here by reading the file again).
- Its "Results vs StatsBomb baseline, per track" table assumes **one stage per
  `model_key`** and renders a single flat results table across tracks. CxA needs
  **two stages per track** (P_create, P_convert), each with its own metrics table --
  a real structural difference, not just a data swap.
- Its coefficients panel assumes coefficients **always exist** for the model being
  shown (CxG's `event_v3`/`plus_v2`/`plus_v3`/`baseline_v1` are all logistic --
  confirmed: every one of CxG's four frozen model versions has a `_coefficients`
  table). CxA does not have this luxury: of its four sub-models, **three are
  tree-family** (both tracks' P_create, event-only's P_convert -- feature
  importances only, no coefficients at all) and **only one is logistic**
  (CxA+'s P_convert). A coefficients-only panel breaks for 3 of 4 sub-models.
- The version-history pill row (`REAL_MODEL_KEYS.map(...)`) has no analog for CxA
  at all -- CxA doesn't have a comparable "pick a version" concept per `model_key`,
  it has two tracks each with two stages.

**Every one of these four pieces would need CxA-specific branching if retrofit into
the same component** -- stage separation, family-conditional panels, no
version-pill equivalent, and CxG-specific "Known caveats" prose that says nothing
about CxA. Turning the existing component into a CxG-vs-CxA branch on every major
piece is worse for maintainability than a separate, purpose-built page -- the same
judgment this whole project already made when it built `_cxconvert_modeling_common.py`
as its own module rather than overloading `_cxa_modeling_common.py`.

**Decision: Option B, a separate route -- but ONE parameterized component for both
tracks, not two near-duplicate files.** `web/app/models/cxa/[track]/page.tsx`, with
`track` (`event`/`plus`) as the route param. Both tracks' data comes back from the
same `/v1/models/cxa-models` (+ `/explainability`) response shape, so a single
component driven by the param avoids duplicating markup while still being a
genuinely separate page from CxG's, not a forced generalization of it.

**What IS genuinely reused, checked before deciding to reuse or not:**
- `Card`, `SimpleTable`, `Skeleton`, `Badge`, `PageHead` -- generic layout primitives,
  used as-is.
- `CoefficientForestPlot` -- reused **unmodified** for CxA+'s P_convert
  coefficients. Read its source first: it only ever accesses `.feature` and
  `.coefficient` off each row, never `.model_key`/`.track` despite the prop type
  requiring them -- so CxA's `CxaCoefficient` rows are adapted with a small inline
  mapper (`{model_key: cxconvert_${track}, track, feature, coefficient, std_error,
  p_value}`) rather than forking the component for one field-name difference.

## 2. A gap found during investigation, not assumed away: no live explainability
source existed for any of the four frozen sub-models

The task's own framing assumed feature-importance data was "already captured in
this project's own audit JSON at the baseline/candidate and freeze steps for
P_create." **Checked directly, not trusted:**

```
$ grep -c "feature_importances_" scripts/materialize_cxa_candidate_v1.py \
    scripts/materialize_cxa_freeze_v1.py scripts/materialize_cxa_test_eval_v1.py
0  0  0
```

**Zero hits, in all three P_create scripts, anywhere in this project's history.**
P_create's own tree feature importances were never captured, not in BigQuery, not
even in a local audit JSON file. The one P_create coefficients table that *does*
exist (`oam_ml.cxa_{track}_candidate_v1_coefficients`) belongs to the **logistic
candidate** -- P_create froze `lightgbm_tree` for both tracks, so that table
describes a model that was never chosen, and is not a substitute.

P_convert fares slightly better but is still not a live source: `analyze_cxconvert_
baseline_and_candidate.py` did capture both coefficients and tree
`feature_importances_`, but only into a local JSON file
(`audit_outputs/cxconvert_analysis/baseline_and_candidate/{track}_result.json`),
never BigQuery, and from the **candidate-stage** fit (pre-freeze hyperparameters --
e.g. event-only's candidate tree used `max_depth=4`, the *frozen* tree uses
`max_depth=3`), not the actual frozen configuration.

**Resolution, stated explicitly rather than silently working around it or leaving a
panel empty:** wrote
[`scripts/materialize_cxa_explainability_v1.py`](../../scripts/materialize_cxa_explainability_v1.py),
which refits each of the four frozen sub-models **exactly** as
`materialize_cxa_test_eval_v1.py` / `materialize_cxconvert_test_eval_v1.py` already
legitimately refit them (same family, same feature list, same hyperparameters, all
read live from the already-frozen config tables; same train+validation fit pool) --
making no new modelling decision -- and adds the one extraction step neither
test-eval script happened to include: reading `.feature_importances_`
(`booster_.feature_importance(importance_type="split"|"gain")`) for the tree stages,
or `.params`/`.bse`/`.pvalues` for the one logistic stage, off the exact same fitted
object test-eval already produces. This is the same "materialize once, serve
read-only" precedent this entire project has used throughout, applied to a byproduct
of an already-sanctioned refit -- not a new one, not a hyperparameter change, not a
different feature set.

**Written (WRITE_TRUNCATE):**

| table | rows | shape |
|---|---|---|
| `oam_ml.cxa_event_explainability_v1` | 12 | feature, importance_split, importance_gain |
| `oam_ml.cxa_plus_explainability_v1` | 14 | feature, importance_split, importance_gain |
| `oam_ml.cxconvert_event_explainability_v1` | 19 | feature, importance_split, importance_gain |
| `oam_ml.cxconvert_plus_explainability_v1` | 13 | feature, coefficient, std_error, p_value |

Row counts match each stage's own encoded feature-column count exactly (event
P_create: 10 locked features -> 12 encoded one-hot columns; event P_convert: 15
locked -> 19 encoded; plus P_create: 11 locked -> 14 encoded; plus P_convert: 12
locked + intercept -> 13 rows including `const`) -- confirmed by inspection of the
returned row counts against each track's own already-documented encoded-column
counts, not assumed.

## 3. Backend: new explainability endpoint

- **`src/opponent_adjusted/api/cxa_models.py`** -- extended, not replaced:
  - `CxaModelSummary` gained `p_create_model_family`, `p_create_feature_list`,
    `p_convert_model_family`, `p_convert_feature_list` (read live from the frozen
    config tables' own `model_family`/`feature_list` columns -- the detail page
    needs the feature list and both sub-model families, which the Models-page-only
    version of this endpoint didn't carry).
  - New models: `CxaFeatureImportance`, `CxaCoefficient`, `CxaExplainability`.
  - New method: `BigQueryCxaModelStore.get_explainability(track)` -- reads both
    stages' frozen `model_family` first, then reads the matching
    `oam_ml.*_explainability_v1` table per stage and routes rows into
    `feature_importances` (tree-family stages) or `coefficients` (logistic-family
    stages) -- never both for one stage, never a fabricated coefficients row for a
    tree model.
- **`src/opponent_adjusted/api/routers/cxa_models.py`** -- new route:
  `GET /v1/models/cxa-models/{track}/explainability`, public, no admin gate, mirrors
  `/v1/models/cxg-models/{model_key}/coefficients`'s nesting style exactly.
- **A real, measured performance bug found and fixed during this task's own
  verification pass (section 5), not assumed fine because the code "looked
  right":** `list_model_summaries()` (already existing, from the prior task) was
  measured live at **~20-27 seconds per call** -- 10 sequential, uncached BigQuery
  queries (2 tracks x 5 queries: 2 frozen-config reads + 2 metrics-table reads + 1
  coverage query). This is a real, user-visible latency problem for the page this
  endpoint backs. **Fixed:** added the same `TTLCache`-per-arguments pattern
  `_get_track_coverage` already used, to both `list_model_summaries()` (single
  cache key, `maxsize=1`, since it always returns both tracks) and the new
  `get_explainability(track)` (per-track cache key, same as coverage). Confirmed
  live: first call ~20-27s (cold), every call after within the 300s TTL window
  ~0.2s.

## 4. Live verification (real BigQuery data, no auth header)

### `GET /v1/models/cxa-models/event/explainability`

```json
{
  "track": "event",
  "feature_importances": [
    {"stage": "p_create", "feature": "end_x", "importance_split": 2289, "importance_gain": 175497.0753307042},
    {"stage": "p_create", "feature": "start_x", "importance_split": 2289, "importance_gain": 67822.73349763021},
    {"stage": "p_create", "feature": "play_pattern_Counter", "importance_split": 415, "importance_gain": 16063.029872423083},
    "... 9 more p_create rows ...",
    {"stage": "p_convert", "feature": "shot_gk_distance_m", "importance_split": 215, "importance_gain": 5625.001432383481},
    {"stage": "p_convert", "feature": "shot_dist_to_goal_m", "importance_split": 203, "importance_gain": 2138.6871008351336},
    "... 17 more p_convert rows ..."
  ],
  "coefficients": []
}
```

`coefficients` is empty (`[]`, not omitted, not an error) for the event track --
both of event's stages are tree-family, so there is nothing to put there, and the
response says so honestly rather than being silent about the field.

### `GET /v1/models/cxa-models/plus/explainability`

```json
{
  "track": "plus",
  "feature_importances": [
    {"stage": "p_create", "feature": "end_x", "importance_split": 744, "importance_gain": 60679.56290254615},
    "... 13 more p_create rows ..."
  ],
  "coefficients": [
    {"stage": "p_convert", "feature": "const", "coefficient": -1.6067883888245174, "std_error": 1.353478835237249, "p_value": 0.23516671381339238},
    {"stage": "p_convert", "feature": "is_through_ball", "coefficient": 0.8930055477552372, "std_error": 0.2842135804917112, "p_value": 0.0016778497423639727},
    {"stage": "p_convert", "feature": "shot_gk_distance_m", "coefficient": -0.13082410137387515, "std_error": 0.02122778990570362, "p_value": 7.143832903586337e-10},
    {"stage": "p_convert", "feature": "shot_open_goal", "coefficient": 1.9716377815410158, "std_error": 0.4839879192013981, "p_value": 0.000046265428032428076},
    "... 8 more p_convert rows ..."
  ]
}
```

`feature_importances` here contains **only** `p_create` rows (14, event-only's
tree stage) -- `plus`'s P_convert stage is logistic, so it appears in
`coefficients` (13 rows, including `const`) instead, never both. `feature_importances`
correctly has no `p_convert` rows for this track.

### `GET /v1/models/cxa-models` (confirms the new fields)

```
event lightgbm_tree 10 lightgbm_tree 15
plus  lightgbm_tree 11 logistic_mle  12
```

Matches every prior CxA freeze/test-eval document's own reported feature counts
exactly (10/15 event, 11/12 plus) -- confirmed live, not retyped from memory.

## 5. Frontend

- **`web/app/models/cxa/[track]/page.tsx`** (new) -- the detail page described in
  section 1. Fetches `/v1/models/cxa-models` (filters to the current `track`'s
  summary) and `/v1/models/cxa-models/{track}/explainability`, in parallel, each
  with its own loading/error state (same pattern as CxG's detail page). Renders,
  in order: the combined-score caveat (verbatim), coverage stats, an explicit
  "combined-score quality vs. y_goal ... not shown as a headline metric" note
  (decision 5, both re-stated per the task's instruction), then a P_create `Card`
  and a P_convert `Card`, each with its own metrics table, feature list, and either
  a feature-importance table (tree-family) or a coefficients forest plot + table
  (logistic-family) depending on that stage's own `model_family`. CxA+ additionally
  renders the `Experimental` badge + caption at the top of the page, same wording as
  its Models-page card.
- **`web/lib/types.ts`** -- `CxaModelSummary` extended with the four new fields;
  added `CxaFeatureImportance`, `CxaCoefficient`, `CxaExplainability`.
- **`web/lib/api.ts`** -- added `getPublicCxaExplainability(track)`.
- **`web/lib/models-data.ts`** -- `detailModelKey` flipped from `null` to
  `"cxa/event"` / `"cxa/plus"` for both CxA cards, so `ModelCard`'s existing
  `/models/${detailModelKey}` link construction now resolves to the new route with
  no `ModelCard.tsx` change needed.

## 6. Verification performed

1. **Both new/changed endpoints curled with real data, both tracks** -- section 4.
2. **`npm run build`** -- exit 0, clean TypeScript/lint pass;
   `/models/cxa/[track]` compiles as a new dynamic route (3.93 kB, 91.3 kB first
   load JS).
3. **Live-rendered both detail pages** (local dev server against the real API, not
   just curl) and read the rendered page text directly:
   - Event page: caveat verbatim, coverage stats correct (1,736/92,799, 1.871%),
     combined-score-quality note present and clearly not a headline metric,
     P_create and P_convert sections both correctly labelled `lightgbm_tree` with
     their own metrics tables, feature lists, and feature-importance tables
     (sorted by gain, matching the materialize script's own ordering).
   - CxA+ page: `Experimental` badge + exact copy ("2,830 total chance-creating
     passes, 419 in test, zero Premier League rows") rendered at the top, caveat
     verbatim (identical text to the event page and the Models-page cards),
     coverage stats correct (419/18,570, 2.256%), P_create section shows
     `lightgbm_tree` + feature importances, P_convert section shows
     `logistic_mle` + both the forest plot and the full coefficients table with
     std_error/p-value.
4. **Clicked through from the actual Models page**, not just direct URL
   navigation: read the rendered page's interactive elements, confirmed both
   "View full results & coefficients →" links now resolve to `/models/cxa/event`
   and `/models/cxa/plus` (previously absent entirely, per the prior task's own
   finding), clicked the event-track link, confirmed the detail page loaded with
   the correct content.
5. **A real, measured performance bug found and fixed mid-task** (section 3) --
   not something that would have been caught by curling once and moving on;
   found specifically because the browser-rendered page appeared stuck on
   loading placeholders, which prompted checking response times directly rather
   than assuming the frontend code was at fault.
6. **Full test suites, before and after every change:**
   - Python: `pytest tests/ -q --ignore=tests/features/cxg/test_opponent_adjusted_
     family.py` -> **459 passed** (2 more than the prior task's 457, from the new
     `test_get_explainability_*` tests; the one ignored file remains the same
     pre-existing, unrelated failure already documented in the prior task's doc).
   - `tests/api/test_cxa_models.py` updated for the new `list_model_summaries()`
     query shape (frozen-config reads now precede the metrics queries) and two new
     tests added for `get_explainability` (stage-routing correctness, unknown-track
     guard) -- **6 tests, all passing**.
   - Frontend: `npx vitest run` -> **139 passed** (unchanged from the prior task;
     this task added no new frontend test file, since the new page's behavior was
     verified live against the real backend instead, per item 3 above -- a
     reasonable trade-off given the page is almost entirely data-driven rendering
     with no new client-side logic beyond two straightforward fetches).

## 7. What's still deferred

- **A per-pass display** -- still out of scope, unchanged from the prior task.
- **A player-level aggregate table** -- still not attempted.
- **CxA+'s badge in a future per-pass context** -- the badge now appears on two
  surfaces (Models-page card, this detail page); a future per-pass view should
  carry the same disclosure forward, per the prior task's own note.
- **A genuine combined-quality metric surfaced anywhere as a first-class number**
  -- this page names it in prose (with a link to the doc that computed it) but
  does not compute or display it as a table/chart; if that's ever wanted, it's new
  UI work, not something this task's data already supports rendering nicely.
