# Claude Code prompt — CxG scope toggle

Copy everything below the line into Claude Code, run from `C:\Users\USER\Documents\Python Projects\oam-dashboard` on branch `feature/dashboard-scaffold`.

---

You are working in the `oam-dashboard` repo, on branch `feature/dashboard-scaffold`. Confirm with `git rev-parse --abbrev-ref HEAD` before starting.

Source of truth: `docs/dashboard_content_spec_v3.md`, specifically the **new §2.2a "CxG scope toggle"** and **§9.3 `GET /v1/cxg/matches`**. Read both before writing anything. §8.1 (CxG disclosure rules) still applies unchanged.

Goal: make the whole Explore zone default to showing only the matches that actually carry CxG predictions, so the xG-versus-CxG comparison is visible on every page rather than only on the minority of pages that happen to have coverage. Keep the full dataset one click away.

## Part 1 — Backend: `GET /v1/cxg/matches`

New endpoint, specced in §9.3.

`oam_analysis.cxg_match_splits_v1` holds exactly 610 rows, one per match, with `match_id`, `split` (`test` / `validation` / `train`), `has_360_match`, `event_shot_count`, `plus_shot_count`, `event_goal_count`, `plus_goal_count`. There is one `run_id`, so no version disambiguation is needed.

```
GET /v1/cxg/matches?track=cxg_event|cxg_plus
-> [ { match_id, split, has_360_match,
       event_shot_count, plus_shot_count,
       event_goal_count, plus_goal_count } ]
```

- Filter `split = 'test'` server-side. This must mirror `cxg_coverage.py`'s existing rule exactly. The two must never disagree about what "covered" means, so define the constant once and use it in both places.
- `track=cxg_plus` additionally filters `has_360_match = TRUE`.
- Guest-accessible via `get_role`, not `require_admin`.
- Put it in `cxg_coverage.py` alongside `BigQueryCxgCoverageStore`, **not** in `bigquery_analysis_store.py` (that whole module is admin-gated).
- Wrap it in the same `TTLCache` pattern the other stores use.
- Expected live counts, assert these in a test against a fake store: `cxg_event` returns **92** matches, `cxg_plus` returns **23**.

Add a `getCxgMatches(track)` function to `web/lib/api.ts` and its response type to `web/lib/types.ts`.

## Part 2 — The toggle itself

Add a **CxG scope** control at the top of `web/components/shell/Sidebar.tsx`, above Competition. Two states, **default ON**:

- ON: "CxG matches only"
- OFF: "All matches"

State lives in `MatchFilterProvider` next to `competitionId`/`seasonId`/`teamId`/`metricMode`, so it persists across Explore navigation. Reflect it in the URL as a query param so a scoped view is shareable.

When ON, `/matches`, `/players` and `/teams` filter their results to the match set returned by Part 1. Matches filters directly on `match_id`. Players and Teams currently aggregate server-side across all matches, so **when scope is ON they must derive their rows from the covered matches rather than showing full-season aggregates** — see Part 3, this is the part that needs care, not a one-line filter.

## Part 3 — Labelling obligations (do not skip these)

This is why the spec chose a toggle over a hard restriction, and it is the part most likely to go wrong.

The 92 matches are a **random ~15% sample of matches**, not a coherent slice. Premier League 2015/16 contributes 61 of its 380 matches. Inside the scope there are 2,427 shots, 604 players with at least one shot but only **60 with 10 or more**, and 48 teams averaging two to four matches each.

So:

1. **Teams pages must relabel when scope is ON.** "Season shot record" becomes "CxG sample · N matches", and the page says plainly that these are not full-season totals. A team total computed over three randomly-drawn matches is not a season record and must not be presented as one. This applies to every team-level aggregate on the page and in the Teams list.
2. **Players lowers its minimum-shots default from 10 to 5 when scope is ON.** 60 players at threshold 10 is too thin to rank meaningfully; 152 at 5 is usable. Keep showing the threshold in the crumb as it already does.
3. **Every Explore page header states the active scope and its match count**, e.g. "CxG matches only · 92 matches" or "All matches · 610".
4. When scope is OFF, behaviour is exactly as today.

## Part 4 — What must not change

- The toggle **only changes which matches are listed**. It never changes what a CxG number means and never fabricates coverage. A shot outside the v3 test split has no CxG in either state, and still gets no placeholder, dash or zero (§8.1).
- Do not touch the `split = 'test'` filter in `cxg_coverage.py`.
- Do not regress `silver_schema_version = 'statsbomb_silver_v1_2'` in any `bigquery_store.py` query.
- Do not add a second `bigquery.Client()` construction path.
- Do not remove the `NODE_ENV !== "production"` guard on `RoleSwitch`.
- Do not install a charting library.
- Use existing design tokens from `globals.css`, no hardcoded hex. Every number in `font-data`.

## Tests

- `/v1/cxg/matches` filters to the test split, and `track=cxg_plus` narrows further to 360 matches. Assert the 92 and 23 counts against a fake store.
- The endpoint is guest-reachable (no admin token required).
- The toggle defaults to ON.
- With scope ON, Teams renders the "CxG sample" label and not "Season shot record"; with it OFF, the reverse.
- With scope ON, the Players minimum-shots default is 5.
- Scope state survives navigation between Matches, Players and Teams.

Run `npm test` in `web/` and `pytest` at the repo root before committing. Commit in logical chunks: (1) endpoint + store + tests, (2) provider + sidebar toggle, (3) page filtering, (4) labelling changes.

When done, report what you changed, anything in the spec that turned out to be wrong against the real code, and any decision worth my review.
