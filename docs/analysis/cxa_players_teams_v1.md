# CxA on Players/Teams + Teams Attack-Panel Honesty Fix v1

Date: 2026-09-21
Two related pieces: (1) fixes a real honesty gap on the Teams page's "Attack vs
defence, both opponent-adjusted" card, where only the defence side was actually
opponent-adjusted; (2) surfaces CxA as a real metric on the Players and Teams
pages, reusing `oam_serving.player_season_cxg_cxa_v1` (built for the Analysis
tab's quadrant scatter) rather than materializing anything new. Investigated
first, per this task's own instruction, before writing any code.

## 1. The Teams attack-panel honesty gap

### 1a. Confirmed live, not assumed

Read `web/app/teams/[teamId]/page.tsx` in full. The card titled "Attack vs
defence, both opponent-adjusted" (lines 293-316, pre-fix) had exactly three
`DivergingBar`s: Goals vs total xG created (raw StatsBomb xG), Goals conceded vs
xG conceded (also raw), and Goals conceded vs CxG allowed (genuinely
opponent-adjusted, using `cxgByEventIdFaced`/`coveredGoalsConceded`/`cxgAllowed`).
**The attack side never touched `cxgByEventId` at all** despite that state
already being fetched on this page for the shot map and shot-detail modal (line
27, populated at lines 41/57-63) -- confirmed by grep, its only two other uses on
the page were `PitchMap`'s `cxgByEventId` prop and `ShotDetailModal`'s `cxg`
prop. The title's "both opponent-adjusted" claim was false.

### 1b. Fix: a fourth bar, plus an honest split, not just a bolt-on

Added the missing attack-side derivation, mirroring the existing defence-side
one exactly:

```tsx
const coveredShots = scopedShots.filter((s) => cxgByEventId[s.event_id] != null);
const coveredGoals = coveredShots.filter((s) => s.is_goal).length;
const coveredCxg = coveredShots.reduce((sum, s) => sum + (cxgByEventId[s.event_id] ?? 0), 0);
```

Rather than just adding a fourth bar to a `grid-cols-4` and leaving the
misleading title, split the card into two visually distinct groups -- "Raw
(StatsBomb xG, full shot volume)" and "Opponent-adjusted (CxG)" (with the same
`Experimental` badge used everywhere else CxG coverage is shown) -- and renamed
the title from "Attack vs defence, both opponent-adjusted" to plain "Attack vs
defence", since the card now honestly contains two raw bars and two adjusted
bars, not four adjusted ones. A new scoping caption ("Goals-vs-CxG row scoped to
N CxG-covered shots created.") sits alongside the existing "CxG-allowed row
scoped to N CxG-covered shots faced." caption, matching that caption's exact
phrasing style.

### 1c. Live-verified render (Iceland, team_id 793, "All matches" scope)

Local dev QA (see section 4) confirmed the card renders exactly as intended:
Raw row shows Goals 2.00 / Total xG created 4.72 and Goals conceded 5.00 / xG
conceded 4.07; Opponent-adjusted row (with `Experimental` badge) shows Goals
1.00 / Total CxG created 1.86 and Goals conceded 2.00 / CxG allowed 0.73, with
the caption "Goals-vs-CxG row scoped to 17 CxG-covered shots created. CxG-allowed
row scoped to 13 CxG-covered shots faced." -- both numbers matching the shot
map's own "17 of 36 shots have CxG coverage" caption directly above it.

## 2. Surfacing CxA on Players and Teams

### 2a. Investigation: is `/v1/analysis/quadrant-scatter`'s admin gate real access control?

Read `routers/analysis.py`'s `get_quadrant_scatter` endpoint and
`PlayerSeasonQuadrantRow`'s full field list. **`require_admin` here is
convention, not real access control** -- the row contains only player/team
identity, competition/season/split, and CxG/CxA means/totals/n-counts, all
already public elsewhere (Models pages, `/v1/cxg/coverage`, `/v1/cxa/coverage`).
The gate exists purely because the endpoint was "originally scoped for
Analysis" (the module's own docstring), not because the data needs protecting.

**Decision: a new guest-accessible endpoint, not stripping the gate from the
existing one.** `/v1/analysis/quadrant-scatter` stays admin-gated (owned by the
Analysis tab's own contract, list-shaped for the scatter's "every player at
once" access pattern) -- Players/Teams need a different, per-entity access
pattern instead. New guest routes added to `routers/cxa_models.py` (the
existing guest-accessible `/v1/cxa/*` family, using `get_role` not
`require_admin`, exactly mirroring `/v1/cxa/coverage-by-shot`'s own pattern):

- `GET /v1/cxa/player-season?player_id=&competition_id=&season_id=`
- `GET /v1/cxa/team-season?team_id=&competition_id=&season_id=`

Both reuse `BigQueryQuadrantScatterStore`'s existing cached
`list_player_season_rows` (no second BigQuery query path -- the whole
test-split table is 822 rows and already cached per-scope) and filter/aggregate
in Python.

### 2b. Aggregation: sum totals/counts, never average an average

`CxaRollup { n, mean, total }` per track. For a player: sum `cxa_{track}_total`/
`cxa_{track}_n_passes_created` across every matching row (every season the
filters allow), then `mean = total / n` if `n > 0` else `None`. This is
deliberately NOT an average of each season's own `_mean` (which would be
statistically wrong whenever season sample sizes differ) -- verified in a test
(`test_get_player_cxa_sums_across_seasons_and_preserves_null_vs_zero`) that
summing per-season totals/counts gives the correct combined mean where naively
averaging the two per-season means would not.

### 2c. Team grain: live-verified sound, no new materialization

Investigated whether `GROUP BY team_id` over the existing player-grain table is
sound, rather than assuming a separate team-season materialization was needed.
Live BigQuery, test split (`oam_serving.player_season_cxg_cxa_v1 WHERE
split='test'`): **822 rows, 698 distinct players, 48 distinct teams.** Checked
for the one real risk -- a player transferring mid-season and appearing under
two `team_id`s for the same `(player_id, competition_id, season_id, split)` key,
which would double-count that player's passes if summed naively per team:

```sql
SELECT player_id, competition_id, season_id, split,
       COUNT(DISTINCT team_id) AS n_teams, COUNT(*) AS n_rows
FROM `oam-varun-260819.oam_serving.player_season_cxg_cxa_v1`
WHERE split = 'test'
GROUP BY player_id, competition_id, season_id, split
HAVING n_teams > 1 OR n_rows > 1
```

**Zero rows returned** -- every key has exactly one team_id and one row in the
real data. `get_team_cxa` therefore sums every matching player-season row's own
`_total`/`_n` per `team_id`, with no double-counting risk, no `oam_core`
re-join, and no new materialization -- confirmed algebraically equivalent to
computing the mean directly over the team's whole population of chance-creating
passes (sum-of-totals over sum-of-counts, not an average-of-averages).

### 2d. Live-verified numbers (real BigQuery, not test fixtures)

| Query | Result |
|---|---|
| `get_player_cxa(5515)` (Aron Einar Gunnarsson) | event: n=3, mean=0.010866, total=0.032597; plus: n=0, mean=None |
| `get_team_cxa(793)` (Iceland) | event: n=12, mean=0.010598, total=0.127181; plus: n=0, mean=None |
| `get_player_cxa(3009)` | event: n=9, mean=0.011629; plus: n=3, mean=0.054322 -- confirms a real player with BOTH tracks covered exists, for a future screenshot/QA target beyond this task's own spot-check |

Team 793's total (12) is greater than any single player's own count -- confirms
real cross-player summation happened, not just a pass-through of one row.

### 2e. CxA+ Experimental badge: reused verbatim, not invented

Grepped every existing `status="experimental"` usage paired with CxA+ (Models
page, CxA detail page, the per-shot display from the prior task) -- all use the
same shape: a `Badge status="experimental" label="Experimental"` next to a
caption describing coverage. `web/components/analysis/CxaSummaryCard.tsx` (new,
shared between Players and Teams -- avoids duplicating the same render logic and
copy twice) reuses this exact pattern for the CxA+ row only; the CxA (event)
row has no badge, matching how CxG (full-coverage) never carries one either.

**Null-vs-zero, enforced at three levels:** the store's `_rollup` helper
returns `mean=None`/`total=None` whenever `n=0` (never `0.0`); the response
model types this as `float | None`; and `CxaSummaryCard` renders nothing at all
for a track with `n=0` (not a "0.00" tile), and renders nothing at all -- the
whole card is absent -- when BOTH tracks have zero coverage for that
player/team-season, exactly as instructed.

## 3. Backend changes

- `src/opponent_adjusted/api/quadrant_scatter.py`: `CxaRollup`,
  `PlayerCxaResponse`, `TeamCxaResponse` models; `_rollup` helper;
  `BigQueryQuadrantScatterStore.get_player_cxa`/`get_team_cxa`.
- `src/opponent_adjusted/api/routers/cxa_models.py`: two new guest-accessible
  routes, `GET /v1/cxa/player-season` and `GET /v1/cxa/team-season`.
- `tests/api/test_quadrant_scatter.py`: 4 new tests -- season-summation
  correctness (with the "never average an average" check), zero-matching-rows
  returns all-null, team-grain summation with no double-count and no
  cross-team leakage. 9/9 passing in this file, **468/468 passing** across the
  full suite (pre-existing, unrelated `tests/features/cxg/
  test_opponent_adjusted_family.py` collection error excluded, same finding as
  every prior CxA task).

## 4. Frontend changes and live QA

- `web/components/analysis/CxaSummaryCard.tsx` (new, shared component).
- `web/app/players/[playerId]/page.tsx`: fetches `getPlayerCxa(playerId, {
  competition_id, season_id })` in its own effect, renders `CxaSummaryCard`
  after the "Is he actually good" card.
- `web/app/teams/[teamId]/page.tsx`: same pattern with `getTeamCxa`, rendered
  before "Recent matches"; plus the attack-panel honesty fix from section 1.
- `web/lib/types.ts`/`web/lib/api.ts`: `CxaRollup`/`PlayerCxaResponse`/
  `TeamCxaResponse` types, `getPlayerCxa`/`getTeamCxa` client functions.

**Live QA, local dev (not the deployed environment -- this branch is not
deployed):** started both services via `.claude/launch.json` (`web-dev`,
`npm --prefix web run dev`; `api-dev`, a new `scripts/dev_api_server.py` that
inserts `src` onto `sys.path` itself since the package isn't installed in this
machine's active Python environment -- same reasoning `pytest` already handles
via its own rootdir config, made explicit here for a plain `uvicorn` run).
Navigated to `/players/5515` and `/teams/793` in a real browser against the
real BigQuery-backed local API (not mocks): confirmed the "Chances created
(CxA)" card renders `0.011 (n=3 test-split passes)` for Gunnarsson and
`0.011 (n=12 test-split passes)` for Iceland -- exact matches to the live-query
numbers in section 2d -- and confirmed the CxA+ row is correctly absent for
both (real `n=0`). Confirmed the Teams "Attack vs defence" card's live render
matches section 1c exactly. `npx tsc --noEmit` clean.

**One friction point worth recording for a future session:** local concurrent
requests against the dev API were slow (multiple parallel BigQuery-backed
routes cold-starting under this sandbox's network) and looked stuck for
~20-30s before all resolving with real 200s -- not a code defect (confirmed via
an isolated `fetch()` returning the correct response immediately mid-hang), just
this machine's local dev latency under concurrency. Worth a longer loading-state
patience budget in any future local QA pass on this project.

## 5. What's still deferred

- **Competition/season filter UI on the CxA tile itself** -- `getPlayerCxa`/
  `getTeamCxa` both accept `competition_id`/`season_id`, and the Players/Teams
  pages already pass through their own `useMatchFilter` context values, but
  there's no separate UI affordance distinguishing "CxA scoped to this filter"
  from the shot-map's own filtering -- inherited automatically from the
  existing filter bar, not a new gap this task introduces.
- **The "CxG matches only" toggle does not scope the CxA tile** -- confirmed
  live (Gunnarsson's page showed "Shots: 0" under that toggle while the CxA
  tile still showed real data) -- this is by design (CxA is a separate
  season-aggregate table, not filtered by 360-coverage the way the shot map
  is), but worth stating explicitly rather than leaving it as a silent
  surprise.
