# CxA Dashboard: Models Page + API Routes v1

Date: 2026-09-20
Exposes CxA on the public dashboard -- API routes and two Models-page cards
("CxA (event-only)", "CxA+"). Reads already-materialized backend data
([`docs/analysis/cxa_combined_scorer_v1.md`](cxa_combined_scorer_v1.md),
`oam_serving.cxa_{event,plus}_combined_v1`, plus the already-frozen/test-evaluated
`oam_ml.cxa_{track}_test_v1_metrics` / `cxconvert_{track}_test_v1_metrics`) -- **does
not recompute anything.** No per-pass display, no shot maps, no player-page
assist-chain view -- those stay deferred, per this task's own explicit scope.

**Branch note:** neither
[`docs/analysis/cxa_combined_scorer_design_v1.md`](cxa_combined_scorer_design_v1.md)
nor [`cxa_combined_scorer_v1.md`](cxa_combined_scorer_v1.md) were on `main` yet at the
start of this task -- both branches (`design/cxa-combined-scorer-v1`,
`feature/cxa-combined-scorer-pipeline`) were merged into this branch first, cleanly,
no conflicts, same reasoning as every prior CxA step's own branch note.

## 1. Six binding decisions -- confirmed applied, not just read

1. **"CxA" = the combined product.** Every place the combined score's coverage shows
   up, the caveat is the design doc's section 1c wording, reused **verbatim** in three
   places: `src/opponent_adjusted/api/cxa_models.py`'s `COMBINED_SCORE_CAVEAT`
   constant (returned in the API response), and `web/lib/models-data.ts`'s
   `comparisonNote` for both CxA cards -- all three copies read identically:
   *"Combined CxA (create x convert) -- chance-creating passes only, ~2% of all
   passes; undefined, not zero, elsewhere."*
2. **Models-page comparison only.** No per-pass display was built. `getCxaCoverage`
   exists in `web/lib/api.ts` (per decision 4/item 4 below) but is not called by any
   page yet.
3. **Two separate cards.** `web/lib/models-data.ts` now has `"CxA (event-only)"` and
   `"CxA+"` as two distinct `ModelInfo` entries, mirroring the existing `CxG`/`CxG+`
   pair exactly -- not one merged card with a toggle.
4. **`split='test'` only, enforced at the API layer.** Every query in
   `src/opponent_adjusted/api/cxa_models.py` (`BigQueryCxaModelStore.list_model_
   summaries` and `._get_track_coverage`) has `WHERE split = 'test'` -- confirmed by
   inspection, not assumed. The underlying `oam_serving.cxa_{track}_combined_v1`
   tables still carry all three splits, unchanged from
   [`cxa_combined_scorer_v1.md`](cxa_combined_scorer_v1.md) -- this task filters at
   the read layer, matching CxG's own real precedent exactly, and does not touch the
   tables themselves.
5. **P_create's and P_convert's own test metrics, kept separate, never a fake
   combined number.** `CxaStageMetric.stage` (`"p_create"` | `"p_convert"`) keeps
   every row explicitly labelled; `list_model_summaries` reads from two genuinely
   different tables per track (`oam_ml.cxa_{track}_test_v1_metrics`,
   `oam_ml.cxconvert_{track}_test_v1_metrics`) and never computes or returns
   `cxa_combined_score`'s own log_loss/AUC anywhere in this module -- confirmed by
   inspection: the word "combined" never appears next to a computed metric value in
   `cxa_models.py`, only in the caveat string and the coverage-count fields (which are
   population counts, not model-quality metrics).
6. **CxA+'s prominent badge.** Reuses the existing `<Badge status="experimental"
   label="Experimental" />` component and the established "badge + caption sentence"
   layout already used on Match/Player/Team detail pages for CxG+'s own 360-coverage
   disclosures -- not a new component. Applied to CxA+'s `ModelCard` via a new
   `experimentalNote` field, with CxA+-accurate copy: *"2,830 total chance-creating
   passes, 419 in test, zero Premier League rows"* -- the exact figures this project's
   own P_convert documents already established, not paraphrased.

## 2. Backend: new router + store module

- **`src/opponent_adjusted/api/cxa_models.py`** (new) -- `BigQueryCxaModelStore`,
  mirroring `cxg_coverage.py`'s structure: a `CxaModelStore` Protocol, Pydantic
  response models (`CxaStageMetric`, `CxaCoverage`, `CxaModelSummary`,
  `CxaCoverageValues`, `CxaCoverageResponse`), and the same module-level
  `TTLCache`/lock/`hashkey`-based per-track caching pattern
  `BigQueryCxgCoverageStore` already uses (300s TTL, `CACHE_TTL_SECONDS` reused from
  `bigquery_store.py`, not reinvented).
- **`src/opponent_adjusted/api/routers/cxa_models.py`** (new) -- two routes:
  - `GET /v1/models/cxa-models` -- public, no admin gate, no `Role` dependency at
    all, mirroring `routers/models.py`'s existing CxG mirror exactly.
  - `GET /v1/cxa/coverage` -- guest-accessible, `role: Role = Depends(get_role)`
    present (resolves auth context but does not gate access), mirroring
    `routers/cxg_coverage.py`'s own `/coverage` route exactly.
- **`src/opponent_adjusted/api/dependencies.py`** -- added `get_cxa_model_store()`,
  same one-line provider-function shape as every existing `get_*_store` in that file.
- **`src/opponent_adjusted/api/main.py`** -- added the `cxa_models` import and
  `app.include_router(cxa_models.router)`, alphabetically placed next to
  `cxg_coverage` per the existing list's ordering.

## 3. Live verification (both endpoints, no auth header)

Backend started locally (`uvicorn opponent_adjusted.api.main:app`), verified against
live `oam_ml`/`oam_serving` data -- not mocked for this check (the automated test
suite, section 5, uses a mocked BigQuery client; this section is the real end-to-end
check the task's "Verify" section asked for).

### `GET /v1/models/cxa-models`

```
$ curl http://127.0.0.1:8123/v1/models/cxa-models
```

Response (200, real data, abbreviated for readability -- the `event` track's full
body is shown; `plus` follows the identical shape):

```json
[
  {
    "track": "event",
    "stage_metrics": [
      {"stage": "p_create", "model": "dumb_baseline", "split": "test", "n": 92799, "log_loss": 0.09296453191750835, "brier_score": null, "roc_auc": null, "is_frozen": false},
      {"stage": "p_create", "model": "frozen_tree",   "split": "test", "n": 92799, "log_loss": 0.06536308710951319, "brier_score": 0.016343463160857052, "roc_auc": 0.9152592477755437, "is_frozen": true},
      {"stage": "p_create", "model": "v1",            "split": "test", "n": 92799, "log_loss": 0.07482584538959212, "brier_score": 0.017391928687017957, "roc_auc": 0.8635493292459553, "is_frozen": false},
      {"stage": "p_convert", "model": "dumb_baseline", "split": "test", "n": 1736, "log_loss": 0.2827067138344131,  "brier_score": null, "roc_auc": null, "is_frozen": false},
      {"stage": "p_convert", "model": "frozen_candidate", "split": "test", "n": 1736, "log_loss": 0.2464607007277286, "brier_score": 0.06841406513775476, "roc_auc": 0.764627937481936, "is_frozen": true},
      {"stage": "p_convert", "model": "v1",           "split": "test", "n": 1736, "log_loss": 0.2647668758550272,  "brier_score": 0.0721900943856402, "roc_auc": 0.6948620467329198, "is_frozen": false}
    ],
    "coverage": {"split": "test", "population_n": 92799, "chance_creating_n": 1736, "coverage_pct": 1.871},
    "combined_score_caveat": "Combined CxA (create x convert) -- chance-creating passes only, ~2% of all passes; undefined, not zero, elsewhere."
  },
  {
    "track": "plus",
    "stage_metrics": [
      {"stage": "p_create", "model": "frozen_tree", "split": "test", "n": 18570, "log_loss": 0.056730708783402785, "brier_score": 0.014812962455064239, "roc_auc": 0.9565022617871899, "is_frozen": true},
      {"stage": "p_convert", "model": "frozen_candidate", "split": "test", "n": 419, "log_loss": 0.24963538811598207, "brier_score": 0.07042165458918109, "roc_auc": 0.7938787351319988, "is_frozen": true}
    ],
    "coverage": {"split": "test", "population_n": 18570, "chance_creating_n": 419, "coverage_pct": 2.256},
    "combined_score_caveat": "Combined CxA (create x convert) -- chance-creating passes only, ~2% of all passes; undefined, not zero, elsewhere."
  }
]
```

(`plus`'s `dumb_baseline`/`v1` rows omitted above only for length -- the real
response includes all 6 stage-metric rows per track, same shape as `event`.)
**Both tracks present, P_create and P_convert kept as clearly separate stage-labelled
rows, never unioned -- confirmed live, not just by reading the code.**

### `GET /v1/cxa/coverage` (mixed covered/uncovered ids)

Three ids tested: one `y_create=TRUE` (chance-creating) test-split pass, one
`y_create=FALSE` test-split pass, and one fabricated id with no row at all.

```
$ curl "http://127.0.0.1:8123/v1/cxa/coverage?track=event&pass_event_ids=feb79e54-e985-4b48-8b09-27241374b2cb,fd8db8de-8e6a-488a-81cb-87bbf8fdc469,00000000-0000-0000-0000-000000000000"
```

```json
{
  "track": "event",
  "values": {
    "feb79e54-e985-4b48-8b09-27241374b2cb": {
      "p_create_predicted_prob": 0.06672448684463815,
      "p_convert_predicted_prob": 0.08182401373035539,
      "cxa_combined_score": 0.00545966532772659
    },
    "fd8db8de-8e6a-488a-81cb-87bbf8fdc469": {
      "p_create_predicted_prob": 0.016552656162962184,
      "p_convert_predicted_prob": null,
      "cxa_combined_score": null
    }
  }
}
```

**Confirmed exactly the required behavior:** the chance-creating pass has all three
values; the non-chance-creating pass has `p_convert_predicted_prob`/
`cxa_combined_score` as `null` (never `0` or a placeholder); the fabricated id is
**absent from `values` entirely**, not present with nulls.

## 4. Frontend changes

- **`web/lib/types.ts`** -- added `CxaStageMetric`, `CxaCoverage`, `CxaModelSummary`,
  `CxaCoverageValues`, `CxaCoverageResponse`, matching the API response shapes
  exactly (field-for-field against the live JSON in section 3).
- **`web/lib/api.ts`** -- added `getPublicCxaModelSummaries()` (→
  `/v1/models/cxa-models`) and `getCxaCoverage(passEventIds, track)` (→
  `/v1/cxa/coverage`), same shape as the existing `getPublicCxgModelResults`/
  `getCxgCoverage` functions. Per decision 2, neither is called by any page yet --
  they exist so a future per-pass/detail-page task doesn't have to re-derive this
  wiring.
- **`web/lib/models-data.ts`** -- the old single `"CxA"` placeholder entry
  (`status: "planned"`) is now two entries, `"CxA (event-only)"` and `"CxA+"`, both
  `status: "evaluated"` (matching CxG/CxG+'s own convention: `oam_serving` being
  populated doesn't itself mean "promoted" -- no per-pass display exists yet either).
  `validationMetrics` shows each track's P_create-vs-P_convert test numbers
  (log_loss, AUC per stage -- 4 entries per card), not a fabricated combined number.
  `featureFamilyCount` reads `"10 P_create + 15 P_convert features"` (event) /
  `"11 P_create + 12 P_convert features"` (plus) -- counts read from the live frozen
  config tables in the prior task, not guessed. `comparisonNote` carries the verbatim
  caveat (decision 1). CxA+ additionally sets `experimentalNote` (decision 6).
- **A real bug caught and fixed while extending `ModelCard`:** the existing
  `comparisonNote` block unconditionally linked to `/stories/cxg-v3-honest-comparison`
  -- hardcoded to a CxG-specific Stories post. Adding a `comparisonNote` for CxA
  without fixing this would have silently shown "See Stories for the full
  comparison →" on the CxA cards, pointing at a story that says nothing about CxA.
  **Fixed:** extracted a new optional `comparisonStoryHref` field; the link now only
  renders when both `comparisonNote` and `comparisonStoryHref` are set. CxG/CxG+ keep
  their existing link (now explicit rather than hardcoded); CxA/CxA+ leave it unset,
  so their caveat renders as plain text with no dangling link. Existing test
  (`model-card.test.tsx`) updated to match; a new test added confirming the no-link
  case explicitly.
- **`web/components/ui/ModelCard.tsx`** -- added the `experimentalNote` rendering
  block (`<Badge status="experimental" label="Experimental" />` + caption paragraph,
  same layout `app/teams/[teamId]/page.tsx` etc. already use), and the
  `comparisonStoryHref` conditional described above.

### Detail page: confirmed NOT generic, deliberately not built here

Checked before deciding, per this task's own instruction: `/models/[modelKey]`
(`web/app/models/[modelKey]/page.tsx`) is **not** data-driven off `MODELS` the way the
list page is. It hardcodes `REAL_MODEL_KEYS = ["baseline_v1", "event_v3", "plus_v2",
"plus_v3"]` and calls `getPublicCxgModelResults()`/`getPublicCxgModelCoefficients()`
directly -- CxG-specific API functions, not generic ones. Setting `detailModelKey` on
either CxA card and pointing it at this route would either 404 (no matching
`REAL_MODEL_KEYS` entry) or silently render CxG's own data under a CxA-labelled URL,
neither acceptable. **Left `detailModelKey: null` for both CxA cards and did not
touch this page.** A working CxA detail page needs its own bespoke work (a CxA-
specific fetch path, and either a generalized page or a CxA-specific one) -- listed in
section 6 as a deferred item, not attempted here.

## 5. Test coverage

- **New:** `tests/api/test_cxa_models.py` (4 tests, mocked BigQuery client, same
  pattern as `test_cxg_coverage.py`) -- covers the "covered id gets real values / not-
  chance-creating id gets `None` not a placeholder / uncovered id is absent /
  unknown-track raises without querying / caching works / stages stay separate, never
  a unioned combined-metric row" behaviors.
- **Updated:** `web/tests/model-card.test.tsx` -- fixed the now-explicit
  `comparisonStoryHref` on the existing `EVALUATED_MODEL` fixture (preserving that
  test's original intent), plus 3 new tests (no-link-when-unset, experimental badge
  renders, experimental badge absent by default).
- **Full suite run, both languages, before and after:**
  - Python: `pytest tests/ -q --ignore=tests/features/cxg/test_opponent_adjusted_
    family.py` → **457 passed** (that one ignored file has a pre-existing,
    unrelated `ImportError` in `opponent_adjusted.features.cxg.contracts` --
    confirmed via `git log` that this task never touched that file or its last
    commit; not something this task caused or is in scope to fix).
  - Frontend: `npx vitest run` → **139 passed** (28 files), including the updated
    and new `model-card.test.tsx` cases.
  - `npm run build` (`web/`) → **exit 0**, clean TypeScript/lint pass, `/models`
    route compiles as a static page (unchanged size class), no new route errors.

## 6. Two CxA cards, described (Models page, `/models`)

Rendered from `web/lib/models-data.ts` + `ModelCard.tsx`, same grid the existing
CxG/CxG+/CxT cards already sit in (`grid gap-3.5`, `auto-fill minmax(220px, 1fr)`):

- **"CxA (event-only)"** -- title row with an `Evaluated` badge (neutral text2-toned,
  same as CxG's) and a `Core` tier chip. Below: 4 metric rows (P_create test
  log_loss/AUC, P_convert test log_loss/AUC), the feature-count line ("10 P_create +
  15 P_convert features"), and the caveat sentence as plain text (no dangling Stories
  link, per the bug fix in section 4). No detail-page link (per section 4).
- **"CxA+"** -- identical layout, `Spatial` tier chip (matching CxG+'s own tier
  semantics for 360-derived tracks), CxA+'s own 4 metric rows, feature-count line
  ("11 P_create + 12 P_convert features"), the same verbatim caveat sentence, **plus
  an additional row: an amber "Experimental" badge next to "2,830 total
  chance-creating passes, 419 in test, zero Premier League rows"** -- visually
  distinct from the event-only card (the only one of the two carrying this badge),
  matching decision 6's requirement that CxA+ read as clearly limited-data at a
  glance, not just via a caveat sentence a reader might skim past.

## 7. What's next (explicitly deferred, not done here)

- **A per-pass display** (shot-map-style "assist chain" view, coverage badges on
  individual passes) -- explicitly out of scope for this task (decision 2). The API
  functions (`getCxaCoverage`, `getPublicCxaModelSummaries`) and backend routes exist
  and are verified working, ready for that future task to consume.
- **A working CxA detail page** (`/models/cxa-event` or similar) -- needs bespoke
  work beyond data wiring, per section 4's finding that `/models/[modelKey]` is
  CxG-specific today, not generic. A future task should either generalize that page
  or build a separate CxA-specific one, and only then set `detailModelKey` on the two
  `MODELS` entries.
- **CxA+'s badge, in a future per-pass context** -- this task applied the
  `Experimental` badge treatment only to the Models-page card. If/when a per-pass
  display is built, the same disclosure discipline (visible badge, not hover-only,
  per `dashboard_design_spec_v2.md` §4a/§8.1's existing rule for CxG) should carry
  over there too -- noted, not built.
- **A player-level aggregate table** -- still not attempted (per the design doc's own
  section 2b, restated in the prior task's doc); `oam_serving` now has the per-pass
  combined tables plus nothing else.
