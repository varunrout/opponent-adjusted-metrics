# CxA Per-Pass Display v1

Date: 2026-09-21
Closes the first item in [`cxa_detail_page_v1.md`](cxa_detail_page_v1.md) section 7's
"What's still deferred" list: "A per-pass display -- still out of scope, unchanged from
the prior task." Investigated first, per this task's own instruction, before writing
any code -- the finding changed the shape of the build.

## 1. Investigation: does a chance-creating pass have a home to click into today?

**No.** Checked every plausible location:

- No per-match event log or per-player pass list exists anywhere in `web/app/` --
  grepped `web/app` and `web/components` for pass-related UI; the only near-hit,
  `web/components/story/PassFailList.tsx`, is an unrelated model-validation pill
  component (pass/fail on univariate screening checks), not football passes.
- `web/components/shot/ShotFreezeFrame.tsx` already documents this exact gap in its
  own comment: "No assist-arrow: that needs the assisting pass event's own x/y, which
  this table doesn't carry."
- The backend registers 9 routers (`analysis`, `competitions`, `cxa_models`,
  `cxg_coverage`, `matches`, `me`, `models`, `players`, `shots`, `teams`) -- no
  `passes`/`events` router, and the only per-shot read is
  `GET /v1/matches/{match_id}/shots`.

**A chance-creating pass has exactly one existing click-in point: the shot it
created**, already on screen via `ShotDetailModal` everywhere a shot map appears
(Matches/Players/Teams detail pages). There is no "home" for a pass that never
created a chance (~98% of all passes, per the CxA+ Experimental caption's own
coverage numbers), and building one was not attempted -- out of scope for this
pass, named explicitly rather than silently assumed away.

## 2. Decision: extend `ShotDetailModal`, not a new pass-detail view

`ShotDetailModal.tsx` already takes CxG's scores as plain optional props (`cxg?:
number`, `cxgPlus?: number`) and renders them conditionally -- the exact same shape
works for CxA's scores of the pass that created the shot on screen. Building a
separate pass-detail view would require inventing pass-list UI, pass freeze-frame
handling, and a new backend passes endpoint from scratch, none of which has any
precedent in this codebase. Extending the shot modal reuses everything: the modal
component itself, `ShotFreezeFrame`, the existing per-page `Record<string, T>`
bulk-fetch-then-lookup pattern already used for `cxgByEventId`/`cxgPlusByEventId`.

**Scoping caveat, stated explicitly:** this only surfaces CxA for the shot side of
a pass-shot pair -- i.e. chance-creating passes that produced *this specific* shot.
It does not give the pass itself an independent "home"; a pass that never created a
chance still has nowhere to click into. Adequate for this task (CxA-on-a-shot is
the natural unit here), not claimed to solve the general "browse all passes" problem.

## 3. What the source data actually looks like (read live, not assumed)

`oam_serving.cxa_{track}_combined_v1` schema (both event and plus, confirmed via
`get_table_info`): `pass_event_id, match_id, split, y_create, p_create_predicted_prob,
shot_event_id, p_convert_predicted_prob, y_goal, cxa_combined_score,
p_create_model_family/version, p_convert_model_family/version, materialized_at,
source_docs`.

**Grain: one row per `pass_event_id`** (P_create's full population -- every pass,
not just chance-creating ones), left-joined to P_convert by `pass_event_id`.
`shot_event_id`/`p_convert_predicted_prob`/`cxa_combined_score` are NULL for the
~98% of passes that never created a chance. This is NOT one row per pass-shot pair.

The existing `/v1/cxa/coverage` endpoint (built in the prior task, never called by
any page yet) is keyed by `pass_event_id` -- the wrong key for a shot-detail view,
which has `shot_event_id` on screen, not the originating pass's id. This is the
actual gap this task closes: a shot-keyed mirror of that same lookup.

## 4. What was built

**Backend** (`src/opponent_adjusted/api/cxa_models.py` / `routers/cxa_models.py`):

- `CxaShotCoverageValues` / `CxaShotCoverageResponse` -- same shape as the existing
  `CxaCoverageValues`/`CxaCoverageResponse`, plus a `pass_event_id` field so a caller
  can still identify the originating pass. Same no-placeholder contract: a shot
  absent from `values` has no test-split chance-creating pass behind it; a shot
  present there has non-null `p_convert_predicted_prob`/`cxa_combined_score` by
  construction (a row only exists in this index when `shot_event_id IS NOT NULL`).
- `BigQueryCxaModelStore._get_track_shot_coverage(track)` -- a second cached,
  per-track index built from `WHERE split='test' AND shot_event_id IS NOT NULL`,
  separate cache from the existing pass-keyed index (different WHERE clause, can't
  share entries). `get_cxa_for_shots(shot_event_ids, track)` looks up against it.
- `GET /v1/cxa/coverage-by-shot?track=&shot_event_ids=` -- guest-accessible, no
  admin gate, mirrors `/v1/cxa/coverage` exactly. 6 new tests in
  `tests/api/test_cxa_models.py` (9 total in that file now, all passing), covering:
  only-covered-shots-returned, unknown-track rejection, and cache isolation from
  the pass-keyed cache.

**Frontend:**

- `web/lib/types.ts` -- `CxaShotCoverageValues`/`CxaShotCoverageResponse`.
- `web/lib/api.ts` -- `getCxaCoverageByShot(shotEventIds, track)`.
- `web/components/shot/ShotDetailModal.tsx` -- new optional `cxaEvent`/`cxaPlus`
  props (`CxaShotCoverageValues | undefined`), rendered only when at least one is
  set. Shows P(create)/P(convert)/combined score per covered track, with an
  `Experimental` badge (same component/copy pattern as every other CxA/CxG+
  surface in this project) -- no per-value threshold invented, matching the
  established "badge + authored caption" precedent rather than a new automated
  low-n rule.
- Wired into all three existing `ShotDetailModal` callers --
  `web/app/matches/[matchId]/page.tsx`, `web/app/players/[playerId]/page.tsx`,
  `web/app/teams/[teamId]/page.tsx` -- each now bulk-fetches
  `getCxaCoverageByShot(eventIds, "event")` and `(..., "plus")` alongside the
  existing CxG coverage fetches, on the same shot-id list, same
  fetch-then-`Record<string, T>`-lookup pattern already used for CxG.

## 5. Verification

- Backend: `tests/api/test_cxa_models.py` -- 9/9 passing (3 new tests for the
  shot-keyed path, 6 pre-existing).
- Full backend suite: 462 passed (pre-existing, unrelated collection error in
  `tests/features/cxg/test_opponent_adjusted_family.py` excluded, same as every
  prior CxA task's own finding).
- Frontend: `npx tsc --noEmit` clean.
- Live BigQuery spot-check of the new shot-keyed index (not just mocked tests):
  `BigQueryCxaModelStore()._get_track_shot_coverage("event")` /
  `("plus")` queried directly against production data -- **1,736 event-track
  shots and 419 plus-track shots** returned (exactly matching each track's known
  chance-creating-pass test-split population size), every returned value carrying
  a real `pass_event_id` and non-null `p_convert_predicted_prob`/`cxa_combined_score`.

## 6. What's still deferred

- **A general pass browser** (all passes, not just chance-creating ones behind a
  shot) -- no UI or endpoint exists for this; out of scope, per section 2's
  scoping caveat.
- **A player-level aggregate table** -- addressed separately, see
  [`quadrant_scatter_v1.md`](quadrant_scatter_v1.md).
