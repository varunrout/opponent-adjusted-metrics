# OAM Dashboard — Content Spec v3 (per-tab, build-ready)

Supersedes: nothing. **Complements** `dashboard_design_spec_v2.md`, which defines *personas, zones, and hard gates* (the "why"). This document defines *what goes on every page, where it sits, where its data comes from, and how it behaves* (the "what and how").

Read v2 first for persona rationale. This doc does not re-argue those decisions — it implements them.

---

## 0. How to read this document

Every content block below carries a **source tag**. This is the most important convention in the doc — it tells you whether a thing can be built today or needs new work:

| Tag | Meaning |
|---|---|
| `[LIVE]` | Real data, endpoint already exists, already wired somewhere in the app |
| `[LIVE-NEW-WIRE]` | Real data, endpoint already exists, but this page doesn't call it yet — frontend work only, zero backend |
| `[CLIENT]` | Derived entirely in the browser from data already fetched — zero extra queries, zero backend |
| `[BACKEND]` | Needs a new or changed endpoint. Every one of these is called out explicitly in §9 with its cost implication |
| `[STATIC]` | Authored content in a `.ts` data file or MDX — no query at all |
| `[BLOCKED]` | Cannot be built until a hard gate in v2 §9 clears. Named gate given |

**Rule of thumb that shaped this spec:** roughly 80% of everything below is `[CLIENT]` or `[LIVE-NEW-WIRE]`. The backend is in better shape than the frontend. Most of what makes this dashboard feel empty is data that's *already being fetched and then not used*.

---

## 1. Ground truth — what actually exists today

Written down so no page in this spec invents a field that isn't real.

### 1.1 Live data volumes

- **610 matches**, **1,457 players**, 3 competitions, 5 seasons
- Competitions: Premier League (`competition_id=2`), FIFA World Cup (`43`), UEFA Euro (`55`)
- Seasons: 2015/2016 (`season_id=27`), 2018 (`3`), 2020 (`43`), 2022 (`106`), 2024 (`282`)
- CxG coverage: **v3 test split only, by deliberate dashboard choice, not by data availability** — the prediction tables actually hold all three splits. Confirmed against live BigQuery (26 Aug 2026):

  | Track | train | validation | test (shown today) | total in table |
  |---|---|---|---|---|
  | `cxg_event` | 10,890 | 2,420 | 2,427 | 15,737 |
  | `cxg_plus` | 2,780 | 590 | 590 | 3,960 |

  The dashboard surfaces 2,427 of 15,737 available `cxg_event` predictions and 590 of 3,960 `cxg_plus` predictions — the `split = 'test'` filter in `cxg_coverage.py` is correct and must not change (showing train-split predictions as if they were honest out-of-sample results would be misleading), but the real ceiling is 6.5× larger. See §12 for the open decision on widening coverage with per-split labelling. Both `cxg_event_v3_predictions` and `cxg_plus_v3_predictions` also already carry `statsbomb_xg` and `is_goal` columns — an xG-vs-CxG comparison over these tables needs no join back to `oam_core`.
- `oam_features` contains **9 tables** and is currently **read by zero application code** — neither the FastAPI backend nor any dashboard page queries it. Not scoped into this document's build; noted here so no future page assumes it's already wired.

### 1.2 Endpoints available (all GET, all real)

**Guest-accessible:**

| Endpoint | Returns | Key fields |
|---|---|---|
| `/v1/competitions` | `CompetitionResponse[]` | `competition_id`, `season_id`, `competition_name`, `country_name`, `season_name`, `competition_gender` |
| `/v1/matches?competition_id&season_id&team_id` | `MatchResponse[]` | `match_id`, `match_date`, `kick_off`, `home_team_id/name`, `away_team_id/name`, `home_score`, `away_score`, `competition_stage`, `stadium`, `referee` |
| `/v1/matches/{id}` | `MatchDetailResponse` | above + `lineups: LineupPlayerResponse[]` (`player_id`, `player_name`, `position_name`, `jersey_number`, `formation`, `team_id/name`) |
| `/v1/matches/{id}/shots` | `ShotResponse[]` | `event_id`, `team_id`, `player_id`, `player_name`, `minute`, `period`, `location_x/y`, `end_x/y`, `statsbomb_xg`, `outcome_name`, `body_part_name`, `is_goal` |
| `/v1/players?competition_id&season_id` | `PlayerSeasonResponse[]` | `player_id`, `player_name`, `team_id/name`, `shots`, `goals`, `total_xg` |
| `/v1/players/{id}/shots?…` | `ShotResponse[]` | as above |
| `/v1/teams?competition_id&season_id` | `TeamSeasonResponse[]` | `team_id`, `team_name`, `shots`, `goals`, `total_xg` |
| `/v1/teams/{id}/shots?…` | `ShotResponse[]` | as above |
| `/v1/cxg/coverage?track=&event_ids=` | `CxgCoverageResponse` | `track`, `values: {event_id: prob}` — **absent keys mean no coverage, never a placeholder** |
| `/v1/me` | `MeResponse` | `role`, `uid`, `email` |

**Admin-only** (`/v1/analysis/*`): `features`, `correlation`, `univariate`, `bivariate`, `pca`, `charts`, `cxg-models`, `cxg-models/{key}/coefficients`.

### 1.3 The single most underused fact

`ShotResponse` carries `minute`, `period`, `body_part_name`, `outcome_name`, `end_x`, `end_y` — and **the entire app currently uses none of them except for the pitch dot position**. Every "playground" feature in §6 is built from these six fields with zero new backend. This is the largest single gap between what's fetched and what's shown.

### 1.4 Design tokens (already defined, use these — never hardcode hex)

```
--bg #0b0e12   --surface #12161b   --card #171c22   --card-hi #1c222a   --border #262d35
--text #edeff2   --text2 #97a1ad   --muted #7c8894
--teal #14b8a6 (CxG / primary)     --violet #8b5cf6 (radar / analysis)
--amber #e3a008 (warning / experimental)
--green #22c55e (positive delta)   --red #ef4444 (negative delta)
--home-team #3b82f6                --away-team #f97316
```

Fonts: `font-ui` = Montserrat (all prose/labels), `font-data` = Inconsolata (**every number, without exception**). `rounded` = 10px. Shimmer loading = `.skel`.

### 1.5 Components that exist

`Badge`, `Card`, `ClickableRow`, `TeamLink`, `PlayerLink`, `Leaderboard`*, `MetricTile`, `ModelCard`, `PageHead`, `PitchMap`, `PlaceholderPanel`, `ProfileRadar`*, `Skeleton`, `StoryCard`, `TierChip` + shell (`AppShell`, `Sidebar`, `TopBar`, `PrimaryNav`, `RoleProvider`, `RoleGate`, `MatchFilterProvider`, `RoleSwitch`).

*`Leaderboard` and `ProfileRadar` take **no props** and render hardcoded values. They are decorative shells, not components. Both must be made prop-driven — specced in §10.

---

## 2. Global shell — applies to every tab

### 2.1 TopBar (56px, sticky)

| Slot | Content | Source |
|---|---|---|
| Left | "OA" teal tile + wordmark, links `/overview` | `[STATIC]` |
| Centre | `PrimaryNav` — Explore items heavy, Understand items light, divider between | `[STATIC]` |
| Right | Role-aware: guest → "Sign in"; signed-in → avatar initials + sign-out | `[LIVE]` |

**Change required:** the `VR` avatar currently renders for guests too. A signed-out visitor should see a "Sign in" button only, no fake avatar. The `RoleSwitch` dev control must stay `NODE_ENV !== "production"`-gated (it already is — don't regress this, it's a security-relevant guard).

### 2.2 Sidebar (220px, Explore zone + Playground only)

Four filter groups. Current state and required state:

| Group | Today | Required |
|---|---|---|
| Competition | Works `[LIVE]` | Keep |
| Season | Works, filtered by competition `[LIVE]` | Keep |
| **Team** | **Hardcoded `<option>All teams</option>` — dead** | Populate from `getTeams({competition_id, season_id})` `[LIVE-NEW-WIRE]`; on change, set `team_id` in `MatchFilterProvider` |
| **Metric** | **4 static `<div>` pills, not interactive — dead** | Make `xG` / `CxG` real radio-style toggles `[CLIENT]`; keep `CxA` / `CxT` disabled with "soon" |

**Metric toggle semantics (important — this is the flagship control):**

- `xG` selected → `PitchMap` sizes dots by `statsbomb_xg`, tiles show xG totals. Current behaviour.
- `CxG` selected → dots sized by CxG **where coverage exists**; **uncovered shots render at reduced opacity with a dashed stroke** and are excluded from the CxG total. A caption states `"CxG shown for N of M shots (v3 test split)"`. Never substitute xG and call it CxG.
- The toggle is a *display mode*, not a data refetch — coverage is already fetched on mount.

**Filter persistence:** competition/season/team selection must survive navigation between Matches → Players → Teams (it already lives in `MatchFilterProvider`, so this is free) and should be reflected in the URL as query params `[CLIENT]` so a filtered view is shareable. Currently it is not, which makes every interesting view unlinkable.

### 2.3 Universal states

Every data-backed block on every page must define four states. This is currently inconsistent — some pages show `Skeleton`, some show nothing.

| State | Treatment |
|---|---|
| Loading | `Skeleton` matching the final layout's shape and count — never a spinner, never a layout shift |
| Empty (query worked, zero rows) | `Card` with a plain sentence + one action. E.g. *"No matches for this filter. Clear the season filter."* with a working clear button |
| Error | `Card`, `--amber` left border, plain sentence + Retry button that re-runs the fetch. Never a raw error string |
| Partial (CxG-style) | Render what exists, caption what's missing. Never a dash or zero standing in for absent data |

---

## 3. Overview — the 90-second page

**Route:** `/overview` · **Zone:** Understand · **Persona:** recruiter, first-time visitor
**Current state:** 100% fabricated data, unlabelled. Highest-priority rebuild in the project.

### 3.1 Hard rule

Every number on this page must be real or explicitly labelled sample. No exceptions. This page is the project's credibility claim; fabricated numbers on the page that argues for honest disclosure is the single worst inconsistency in the product.

### 3.2 Layout, top to bottom

**Row 0 — Statement block** (full width, no card, ~120px)
`[STATIC]`

- H1: "Opponent-Adjusted Metrics"
- One sentence, plain language: what the project does and who built it. Not a tagline — a claim a recruiter can evaluate.
- One line of provenance: data source (StatsBomb Open Data), scope (3 competitions, 5 seasons), and a GitHub link.

**Row 1 — Scale tiles** (4 × `MetricTile`, equal width)
`[BACKEND]` — see §9.1 `/v1/summary`

| Tile | Value | Delta line |
|---|---|---|
| Matches | `610` | competitions covered |
| Players | `1,457` | with ≥1 shot |
| Shots analysed | real count | total xG |
| CxG coverage | `N shots` | "v3 test split · experimental" in `--amber` |

Why a new endpoint: computing these client-side means fetching all matches + all players + all teams on the landing page — three full result sets to display four numbers. One cached count query is cheaper and faster. **This is the only genuinely necessary new endpoint in the whole spec.**

**Row 2 — Recent matches strip** (full-width `Card`, 6 rows)
`[LIVE-NEW-WIRE]` — `getMatches({})`, sort by `match_date` desc, `.slice(0,6)`

Each row: date · home `TeamLink` · score (`font-data`) · away `TeamLink` · competition stage. Whole row is a `ClickableRow` → `/matches/{match_id}`. This is v2 §5's "thin live slice" — it makes the page feel current without much query cost, and it's already cached for 300s.

**Row 3 — Two-column split**

*Left (60%) — "The honest result" `Card`* `[STATIC]`

The CxG-vs-StatsBomb comparison, presented as the project's headline intellectual claim rather than buried in `/models`:

- Small table: track × metric × CxG × StatsBomb, 6 rows from v2 §11
- One paragraph explaining that CxG trails the baseline, and why that's disclosed rather than hidden
- Link → `/stories/cxg-v3-honest-comparison`

A recruiter who reads only this block has seen the most senior thing about the project: a candidate who publishes a negative result with its numbers intact.

*Right (40%) — stacked* `[STATIC]`

- "Built with" card: FastAPI · BigQuery · Next.js · Firebase Auth · Cloud Run — plain list, no logos
- "Featured writeup" — one `StoryCard`, the CxG v3 comparison

**Row 4 — Entry points** (3 cards, equal width)
`[STATIC]`

"Browse matches" → `/matches` · "Compare players" → `/playground` · "Read the methodology" → `/stories`. Each with one line of what you'll find. Gives the recruiter somewhere to go instead of bouncing.

### 3.3 Explicitly removed

`DEMO_SHOTS`, the hardcoded `1.82` Team CxG tile, the empty `Team CxA` tile, the two permanently-shimmering `Leaderboard` rows, and the data-free `ProfileRadar`. All deleted, not relabelled — a "sample data" badge is the fallback only if the rebuild is deferred.

---

## 4. Explore zone

### 4.1 Matches — list

**Route:** `/matches` · Sidebar: yes

**Header** `[CLIENT]` — `PageHead` title "Matches", crumb showing the *active filter* ("Premier League · 2015/2016 · 380 matches"), not a hardcoded string.

**Controls bar** (above table, right-aligned)

| Control | Behaviour | Source |
|---|---|---|
| Search | Filters on `home_team_name`/`away_team_name` substring, debounced 200ms | `[CLIENT]` |
| Sort | Date ↓ (default), Date ↑, Total goals ↓ | `[CLIENT]` |
| Result count | "Showing 24 of 610" | `[CLIENT]` |

**Table** — must have a header row (currently has none anywhere in the app):

`Date · Home · Score · Away · Stage · Venue`

- Score in `font-data`; winning side's name at `--text`, losing at `--text2` — instant scannability with zero extra data
- Row = `ClickableRow` → `/matches/{id}`; team names are `TeamLink` with `stopPropagation` (already handled)
- **Pagination: 50 rows/page** `[CLIENT]`. 610 rows in one DOM list is a real performance and usability problem today; 1,457 on `/players` is worse.

**Deferred, not built:** per-match xG columns. `MatchResponse` has no xG field and adding one means aggregating shots per match — a genuine `[BACKEND]` cost. Not worth it while the detail page already shows it.

### 4.2 Matches — detail

**Route:** `/matches/[matchId]` · The strongest page in the app today. Additions below are enhancements, not repairs.

**Row 0 — Match header** `[LIVE]`
Home `TeamLink` — score (`font-data`, 28px) — away `TeamLink`. Below: date · competition stage · stadium · referee. All fields already on `MatchDetailResponse` and currently unused except date.

**Row 1 — Metric tiles** (4 tiles)
`[LIVE]` xG home / xG away (already computed via `summarizeShots`)
`[CLIENT]` Shots home/away, and **xG difference vs actual scoreline** — e.g. *"Home won 1–0 but was out-xG'd 0.8–1.6"*. This single derived line is the most analytically interesting thing on the page and costs one subtraction.

**Row 2 — Shot map** (full-width `Card`)
`[LIVE]` `PitchMap` with `shots`, `homeTeamId`, `cxgByEventId`, `cxgPlusByEventId` — all already wired.

Required additions:
- **Legend** `[CLIENT]` — home/away colour, filled = goal, size = xG. Currently a visitor must guess. Non-negotiable for a non-technical persona.
- **Click a dot → shot detail drawer** `[CLIENT]` — slides from right: player `PlayerLink`, minute, `body_part_name`, `outcome_name`, xG, CxG + CxG+ if covered, and the experimental disclosure as visible text rather than a hover `<title>`.
- Coverage caption stays (`describeCxgCoverage`), promoted to a visible `Badge status="evaluated"` rather than plain text.

**Row 3 — xG race chart** (full-width `Card`)
`[CLIENT]` — cumulative step line, both teams, x-axis 0–90+ from `minute`, y-axis cumulative `statsbomb_xg`. Goals marked. **Zero new data** — `minute` is already in every `ShotResponse` and currently discarded. Listed in v2 §10's component inventory as "xG/CxG race", never built. Highest impact-to-effort item on this page.

**Row 4 — Lineups** (two `Card`s side by side)
`[LIVE]` Already built. Add `formation` in the card header (field exists, unused) and group by `position_name`.

### 4.3 Players — list

**Route:** `/players` · Same controls-bar pattern as Matches.

**Table header (currently missing — this is a confirmed live defect):**

`# · Player · Team · Shots · Goals · xG · xG/shot · G−xG`

- `xG/shot` = `total_xg / shots` `[CLIENT]`
- `G−xG` = `goals − total_xg` `[CLIENT]`, coloured `--green`/`--red`. **This is the "who is overperforming" column** — it's the football analyst persona's entire reason for visiting (v2 §1), it's the project's whole thesis, and it's one subtraction away from existing.
- Sortable on every numeric column `[CLIENT]`
- **Minimum-shots filter** (default 10) `[CLIENT]` — without it the `G−xG` leaderboard is topped by players with one lucky shot, which is statistically meaningless and makes the page look naive. Show the threshold in the crumb.
- Pagination 50/page.

### 4.4 Players — detail

**Route:** `/players/[playerId]`

**Row 0 — Identity** `[CLIENT]` — name from `shots[0].player_name`, team `TeamLink` from `shots[0].team_id`. *Known weakness: if a player has zero shots under the active filter the page renders "Unknown player". Empty state must handle this explicitly rather than showing a broken name.*

**Row 1 — Tiles** (5): Shots, Goals, xG `[LIVE]` · xG/shot, G−xG `[CLIENT]`

**Row 2 — Shot map + shot table, side by side**
- `[LIVE]` `PitchMap` with legend and the same click-to-detail drawer as §4.2
- `[CLIENT]` Shot table beside it: minute · body part · outcome · xG · CxG. Sortable, and **selecting a row highlights the dot on the map** (bidirectional). Turns a dense unreadable cluster (Harry Kane, 220 overlapping dots) into something navigable — this is the fix for the "visually noisy" finding in the cold review.

**Row 3 — Two charts**
- `[CLIENT]` **Shot profile radar** — real `ProfileRadar` driven by 5 axes computed from this player's shots vs the filtered population: shot volume, xG/shot, goal conversion, average shot distance (from `location_x/y`), share of shots in the box. Requires making `ProfileRadar` prop-driven (§10).
- `[CLIENT]` **Body-part breakdown** — horizontal ranked bars, shots and goals by `body_part_name`. Field exists, entirely unused today.

**Row 4 — CxG panel** (only rendered when coverage > 0)
`[LIVE]` For covered shots: a small table of xG vs CxG vs CxG+ per shot, plus mean divergence. This is where a data-scientist visitor sees the model actually behaving on individual shots rather than as an aggregate metric. `Badge status="evaluated"` + full disclosure text.

### 4.5 Teams — list

**Route:** `/teams`

Same table treatment as Players: headers, sortable, `xG/shot`, `G−xG`, pagination.

**Confirmed defect to fix:** rows use a plain `next/link` instead of `ClickableRow`, inconsistent with every other list in the app.

**Club vs national split** `[CLIENT]` — a segmented control: *All · Clubs · National*. Ranking Leicester City against Spain in one list is analytically meaningless. Derive from the active competition (`competition_id=2` → clubs; `43`/`55` → national); when "All competitions" is selected, show the control and default to Clubs.

### 4.6 Teams — detail

**Route:** `/teams/[teamId]` · Thinnest page in the app. **Has no shot map at all**, despite v2 §4a naming Teams as one of the three CxG surfaces.

**Row 0** `[CLIENT]` Team name, competition/season context
**Row 1** `[LIVE]` + `[CLIENT]` 5 tiles: Shots, Goals, xG, xG/shot, G−xG
**Row 2** `[LIVE-NEW-WIRE]` **Shot map** — `getTeamShots` is already called on this page and the shots are used only for tile arithmetic. Rendering a `PitchMap` from them is a component import. Add `getCxgCoverage` for both tracks, exactly as `/matches/[matchId]` does. **This closes the spec-vs-implementation gap the cold review flagged.**
**Row 3** `[LIVE]` Recent matches (already built) — add result W/D/L chips `[CLIENT]` derived from `home_score`/`away_score` vs `team_id`
**Row 4** `[CLIENT]` **Top scorers within this team** — group this team's shots by `player_id`, rank by goals then xG, `PlayerLink` each. Client-side grouping of data already in memory.

---

## 5. Playground — the new tab

**Route:** `/playground` · **Zone:** Explore (sidebar visible) · **Nav:** primary, after Teams
**Roles:** guest gets modules 1–2; **viewer and admin get all four**

This is the answer to "it's a dashboard with a playground." It is also the fix for a structural problem: **the `viewer` role currently grants exactly nothing that `guest` doesn't already have** — it exists in the type union and in `require_admin`'s rejection path, and nowhere else. Signing in is currently pointless. Gating the comparison modules behind viewer gives signing in a reason to exist, and matches v2 §10's own "Comparison card — Viewer-tier feature, not yet built".

**Layout:** module switcher as a tab strip across the top; selected module fills the canvas; sidebar filters apply throughout.

### 5.0 Scope rule — the Playground is CxG-covered matches only

**This is the single most important rule in this section, and it overrides the sidebar's normal behaviour.**

The Explore zone browses all 610 matches, which is correct — it's a general football browser and xG exists for every shot. The Playground is not that. It exists to let someone interrogate *the model*, and the model only has predictions for the test split. Letting a visitor pick any of 610 matches means most picks land on a match with zero CxG, and every CxG panel renders empty. That reads as broken, not as honestly out-of-scope.

So the Playground restricts its entity pickers to matches that actually carry CxG predictions:

| Track | Matches | Shots | Goals |
|---|---|---|---|
| `cxg_event` (8 features) | **92** | 2,427 | 246 |
| `cxg_plus` (24 features) | **23** | 590 | — |

Confirmed against live BigQuery (26 Aug 2026) from `oam_analysis.cxg_match_splits_v1`, which holds exactly 610 rows — one per match — with a `split` label per match (`test` 92, `validation` 92, `train` 426) plus `has_360_match`, `event_shot_count`, `plus_shot_count`, `event_goal_count`, `plus_goal_count`.

Rules that follow from this:

- **Never show train-split or validation-split matches in the Playground.** Not greyed out, not with a "no CxG" label — absent. The Playground's match universe is the 92 test matches.
- **Switching to the `cxg_plus` track narrows the universe further, to the 23 matches with 360 data.** The picker must re-filter, and if the currently-selected match isn't in the CxG+ set, say so plainly and offer the nearest valid choice rather than rendering an empty panel.
- **State the scope on the page, permanently:** *"Playground scope: 92 matches with CxG predictions (v3 test split) — 23 of them also have CxG+."* A visitor should never have to work out why this looks smaller than Matches.
- **The sidebar's competition/season filters still apply**, but *within* the 92, and the result count must reflect that intersection.

This is also the answer to "why is the Playground smaller than the rest of the site" — it's smaller because the model is smaller, and saying so is the honest framing this project already applies everywhere else.

**Cost note:** this needs one new endpoint (§9.3) reading a 610-row table. Everything else in §5.1–5.4 remains `[CLIENT]` — computed in the browser from responses the app already fetches.

### 5.1 Module 1 — Shot Explorer `[CLIENT]` · guest

Load a shot set (by match, player, or team) **from the §5.0 covered universe only**, then filter it live:

| Filter | Field | Control |
|---|---|---|
| Minute range | `minute` | dual-handle slider, 0–90+ |
| Period | `period` | 1st / 2nd / all |
| Body part | `body_part_name` | multi-select chips |
| Outcome | `outcome_name` | multi-select chips |
| xG range | `statsbomb_xg` | dual-handle slider |

Note there is deliberately **no "CxG coverage" toggle** — §5.0 already guarantees every shot in scope is covered, so the toggle would be a no-op. (An earlier draft of this spec had one; it was wrong.)

Output, updating on every change: `PitchMap` of the surviving shots · live counts (shots, goals, total xG, **total CxG**, conversion %) · an xG histogram. **Every count that has an xG form also shows its CxG form**, side by side — that comparison is the entire reason this surface is scoped to covered matches.

Why it matters: it makes the six unused `ShotResponse` fields visible and turns a static map into an instrument. *"Show me every headed shot inside the box in the last 15 minutes"* becomes a five-second interaction.

### 5.2 Module 2 — Compare `[CLIENT]` · guest

Two entity pickers (player vs player, or team vs team). Fetches both via existing endpoints, renders side by side:

- Mirrored shot maps, shared scale
- Stat-by-stat delta table — shots, goals, xG, **CxG**, xG/shot, **CxG/shot**, G−xG, **G−CxG** — with the winner's side highlighted per row
- Overlaid radars

Both entities are drawn from the §5.0 covered universe, so every row in that table has a CxG value — no blanks, and no "this player has xG but no CxG" asymmetry between the two sides.

**G−CxG next to G−xG is the interesting column here.** It asks whether a player's overperformance survives once the defensive context of each shot is priced in — which is the project's actual thesis, stated as a number a visitor can compare.

This is v2 §10's "Comparison card", built as a surface rather than a card.

### 5.3 Module 3 — CxG Lab `[CLIENT]` · **viewer+** · flagship

The single most valuable thing on the site for a data-scientist or hiring-manager visitor, because it demonstrates the model's behaviour honestly rather than describing it.

Over the covered shot set:

- **Scatter: StatsBomb xG (x) vs CxG (y)**, diagonal parity line. Points above the line = CxG rates the chance higher than StatsBomb, below = lower.
- **Track toggle:** `cxg_event` (8 features) / `cxg_plus` (24 features) — makes the feature-pool asymmetry from v2 §11 caveat 1 something you can *see*, not just read.
- **Biggest disagreements table** — shots ranked by `|CxG − xG|`, each row clickable to its shot detail. *Where does this model disagree with the industry standard, and was it right?*
- **Calibration strip** — shots bucketed by predicted probability, actual goal rate per bucket, for both CxG and StatsBomb. This is the honest visual companion to the Brier scores in `/models`.
- **Permanent disclosure banner**, `--amber`: v3 test split only, not scored live, trails the StatsBomb baseline, link to `/models`.
- **Source note:** both prediction tables already carry `statsbomb_xg` and `is_goal` alongside the CxG probability, so the scatter/calibration/disagreements views above need no join back to `oam_core` — they read entirely from `oam_ml.{cxg_event_v3,cxg_plus_v3}_predictions WHERE split = 'test'`.

### 5.4 Module 4 — Leaderboard Builder `[CLIENT]` · **viewer+**

Pick metric (goals · xG · shots · xG/shot · G−xG) → pick population (players or teams) → set minimum shots → set top-N. Renders a ranked bar chart (v2 §10's "Ranked bar", never built) plus an exportable table.

**Guest treatment for locked modules:** render the module's real UI, dimmed, with a single overlay line — *"Sign in to use the CxG Lab"* + button. Never hide it. A visitor must be able to see what signing in gets them.

---

## 6. Understand zone

### 6.1 Stories

**Route:** `/stories` · **Confirmed live defect: all five cards are non-clickable.** `StoryCard` renders a `<div>` with no `href`.

**Required structure:**
- `/stories` — grid of `StoryCard`, each wrapped in a `<Link href={/stories/${slug}}>` `[STATIC]`
- `/stories/[slug]` — **the article pages, which do not exist at all today** `[STATIC]`

`StoryInfo` needs `slug`, `date`, `readingTime`, and body content (MDX file per story, or a `body` field). Category tag stays `--violet`.

**Priority order for actually writing them** — the first is the one every other page links to, so it blocks the most:

1. **CxG v3 — an honest comparison against StatsBomb xG.** The full six-metric table, what was tried, why it trails, what it would take to close the gap. Linked from Overview, Models, and the CxG Lab.
2. **The 3× lineage duplication bug.** Every Explore aggregate was silently inflated 3× until a `silver_schema_version` filter was added. A real production data bug, found, fixed, regression-tested. For a data-engineering-literate reader this is a stronger credibility signal than the model itself.
3. **The £2.50 BigQuery incident and the TTL cache.** Cost discipline as an engineering concern.
4. **Feature-pool asymmetry: 8 vs 24 features.** Why the two tracks aren't a like-for-like comparison (v2 §11 caveat 1).
5. **`zone_displacement`'s unexplained bimodality.** An open question, published as open (v2 §11 caveat 2).

Three of those five are engineering stories rather than modelling stories. That's the right ratio here — the engineering is currently the stronger half of the project and is completely invisible to a visitor.

### 6.2 Models

**Route:** `/models` · Best Understand page today. Two fixes, both small.

- **Confirmed defect:** the "See Stories for the full comparison →" link is dead. It points at `/stories` generally; it must point at the specific article `/stories/cxg-v3-honest-comparison`.
- **Add a comparison table below the cards** `[STATIC]` from v2 §11 — track × metric × CxG × StatsBomb, all six rows, with the losses shown in `--red`. The cards currently state "trails the baseline" without showing by how much.
- Keep the honest `Evaluated` status and the `Planned` CxA/CxT cards exactly as they are. **Note the inconsistency this creates today:** `/models` correctly labels CxA as Planned while `/overview` shows an unlabelled empty CxA tile. §3.3 removes the latter.

**Optional** `[LIVE-NEW-WIRE]`: `/v1/analysis/cxg-models` is admin-gated, so the public page must stay static. Fine — v2 §5 already rules Models static.

### 6.3 About

**Route:** `/about` · **Currently renders "Not yet built" in production.** Any visitor who clicks this concludes the project is abandoned.

Five sections, all `[STATIC]`, all short:

1. **What this is** — one paragraph, plain language
2. **Data source** — StatsBomb Open Data, licence, scope (3 competitions, 5 seasons, 610 matches), what StatsBomb xG is
3. **Glossary** — xG, CxG, CxG+, xG/shot, G−xG, and what "opponent-adjusted" means. Two-column definition list. **Every metric shown anywhere in the app must be defined here**, and the glossary is what tile tooltips link to.
4. **How it's built** — the real architecture: StatsBomb → BigQuery medallion (`oam_core` / `oam_analysis` / `oam_ml`) → FastAPI on Cloud Run → Next.js on Firebase Hosting. One diagram if time allows, prose if not.
5. **Honest limitations** — test-split-only CxG, no live inference, CxG trails the baseline, CxA/CxT not built, single-season depth per competition. Published as a section, not buried.

Section 5 is the one that most distinguishes this from a typical portfolio project. Write it plainly and do not hedge.

---

## 7. Analysis (admin-only)

**Route:** `/analysis` · No persona-facing changes. This is the operator's workbench and the current three-tab structure (Features / Charts / Model results) is correct.

Two items only:

- `[BLOCKED]` **Signed chart URLs** — v2 gate 3. Code is built (`gcs_signing.py`) and degrades gracefully. Blocked solely on the IAM grant `roles/iam.serviceAccountTokenCreator` on `oam-pipeline-sa` for itself. *Deployment note: this was granted on 26 Aug 2026 — verify charts now render as iframes rather than raw `gs://` text, and if so this gate is closed.*
- `[BLOCKED]` **Quadrant scatter (Track B)** — v2 gate 2, needs `oam_serving` populated. Still zero tables. Do not build.

---

## 8. Cross-cutting rules

### 8.1 CxG disclosure — non-negotiable, everywhere

Applies on every surface where a CxG number appears: Explore detail pages, CxG Lab, Models.

1. **Never replaces xG.** Both shown together, always.
2. **Never placeholder.** No dash, zero, "N/A", or greyed value for an uncovered shot. Absent means absent.
3. **Always disclosed visibly** — a `Badge`, not a hover `<title>`. Hover-only disclosure fails on touch devices entirely, and v2 §4a specifies a visible badge.
4. **Feature count named** — 8 (`cxg_event`) or 24 (`cxg_plus`).
5. **Status language:** "Evaluated" and "Experimental" only. Never "Validated", "Production", or "Live" until a real inference pipeline exists (v2 gate 1).
6. **Coverage always quantified** — "N of M shots".

### 8.2 Numbers

Every numeric value uses `font-data`. Fixed decimals: xG 2dp, probabilities 3dp, ratios 2dp. Deltas always signed (`+0.24`, `−0.31`) and coloured `--green`/`--red` — never coloured without a sign, and never a colour alone carrying the meaning.

### 8.3 Roles

| | guest | viewer | admin |
|---|---|---|---|
| Explore + detail pages | ✓ | ✓ | ✓ |
| Understand pages | ✓ | ✓ | ✓ |
| Playground modules 1–2 | ✓ | ✓ | ✓ |
| Playground modules 3–4 | locked preview | ✓ | ✓ |
| `/analysis` | ✗ | ✗ | ✓ |

Backend note: `/v1/*` Explore endpoints use `get_role`, which resolves but never rejects — they are open to all three roles. Module 3/4 gating is therefore **frontend-only**, which is correct here because the underlying data is public anyway. Do not claim it as a security boundary; it's a product tier.

### 8.4 Performance

Pagination at 50 rows on every list (610 matches, 1,457 players today). Debounce all search input 200ms. Client-side filters must never refetch — filter in memory. Respect the existing 300s TTL cache; do not add fetches on hover or scroll.

---

## 9. New backend work — the complete list

Exhaustive. Everything else in this document is frontend-only.

### 9.1 `GET /v1/summary` `[BACKEND]` — required for §3.1

```
{ matches: int, players: int, teams: int, shots: int,
  total_xg: float, cxg_covered_shots: int, competitions: int, seasons: int }
```

One `COUNT`/`SUM` query against `oam_core.shots` + `oam_core.matches`, filtered on `silver_schema_version` (**do not regress the lineage fix**), wrapped in the same `TTLCache` pattern as the other nine store methods. Guest-accessible via `get_role`.

Justification: the alternative is three full result-set fetches on the landing page to display four numbers.

### 9.3 `GET /v1/cxg/matches` `[BACKEND]` — required for §5.0, the Playground's scope

Without this, the Playground cannot know which matches carry CxG predictions, and there is no other way to find out: the `*_predictions` tables key on `event_id` only, with no `match_id`, so match-level coverage cannot be derived from the existing `/v1/cxg/coverage` endpoint.

`oam_analysis.cxg_match_splits_v1` answers it directly — confirmed live (26 Aug 2026): **exactly 610 rows, one per match**, with `match_id`, `split` (`test` 92 / `validation` 92 / `train` 426), `has_360_match`, `event_shot_count`, `plus_shot_count`, `event_goal_count`, `plus_goal_count`, `split_score`, `split_seed`, `run_id`, `created_at`. One `run_id` only, so no version disambiguation is needed today.

Proposed shape:

```
GET /v1/cxg/matches?track=cxg_event|cxg_plus
-> [ { match_id: int, split: str, has_360_match: bool,
       event_shot_count: int, plus_shot_count: int,
       event_goal_count: int, plus_goal_count: int } ]
```

- Filter to `split = 'test'` server-side, mirroring `cxg_coverage.py`'s existing rule — the two must never disagree about what "covered" means.
- `track=cxg_plus` additionally filters to `has_360_match = TRUE`, giving the 23-match universe.
- Guest-accessible via `get_role` (the Explore-zone pattern), and wrapped in the same `TTLCache` used by the other stores.
- **Cost: negligible.** 610 rows, filtered, cached for 300s. This is the cheapest query in the application.

Implementation note: this belongs in `cxg_coverage.py` next to `BigQueryCxgCoverageStore`, not in `bigquery_analysis_store.py` — the latter is entirely `require_admin`-gated, and this endpoint must be guest-reachable.

### 9.2 `GET /v1/shots/{event_id}/opponent-context` `[BACKEND]` — future, not built in this pass

`oam_analysis.cxg_analysis_opponent_adjusted_v1` exists — confirmed live (26 Aug 2026): **3,960 rows across 166 matches and 835 players**, keyed by `event_id`/`match_id`/`player_id`/`team_id`. Nothing in the codebase reads this table today, even though it is the literal content of "opponent-adjusted" in the project's name.

Columns available: `nearest_defender_odi`, `mean_backline_odi`, `gk_odi`, `defensive_profile_cluster`, `nearest_defender_role`, `nearest_defender_zone_displacement`, `nearest_defender_gap`, `nearest_defender_style_archetype` (3,557 of 3,960 non-null), `has_360_frame` (3,960 non-null, i.e. present on every row).

Proposed shape:

```
{ event_id: str, match_id: int, player_id: int, team_id: int,
  nearest_defender_odi: float | null, mean_backline_odi: float | null, gk_odi: float | null,
  defensive_profile_cluster: str | null, nearest_defender_role: str | null,
  nearest_defender_zone_displacement: float | null, nearest_defender_gap: float | null,
  nearest_defender_style_archetype: str | null, has_360_frame: bool }
```

Likely usage: a per-shot detail panel (e.g. the shot-detail drawer specced in §4.2) showing what defensive context the model actually saw for that shot, rather than only the resulting CxG number.

**Cost note:** the table is small (3,960 rows total) and would be queried filtered by `event_id`s or `match_id`, matching the existing `ServingStore` filter-by-key pattern — cheap, no unfiltered-scan risk. Not built in this pass; specced here so a future prompt has a concrete target instead of rediscovering the table.

### 9.3 Everything else: nothing

Table headers, sorting, pagination, search, the xG race chart, shot drawers, radars, body-part breakdowns, all four Playground modules, club/national splitting, top-scorers-within-team, `G−xG`, `xG/shot` — **all `[CLIENT]` or `[LIVE-NEW-WIRE]`.**

Worth stating plainly: **the backend is not what's holding this dashboard back.** One count endpoint closes the entire gap between today and the full spec above.

---

## 10. Component work required

| Component | Change |
|---|---|
| `Leaderboard` | **Currently takes no props and renders 3 hardcoded bars + 2 permanent skeletons.** Rewrite as `{ rows: {label, value, pct, href?}[], loading?, emptyMessage? }` |
| `ProfileRadar` | **Currently takes no props, static SVG.** Rewrite as `{ axes: {label, value, max}[], compareAxes?, size? }` |
| `PitchMap` | Add `onShotClick`, `selectedEventId`, `showLegend`, `sizeBy: "xg" \| "cxg"` |
| `Badge` | Add an `experimental` status variant, `--amber` |
| `StoryCard` | Wrap in `Link` using a new `slug` field — **this is the dead-card fix** |
| **`DataTable`** *(new)* | Sortable headers, pagination, empty/loading/error states, optional row click. Used by Matches, Players, Teams, and the Playground. Removes four divergent hand-rolled table implementations |
| **`RangeSlider`** *(new)* | Dual-handle, for Shot Explorer minute/xG filters |
| **`Chart` primitives** *(new)* | Line (xG race), ranked bar, scatter (CxG Lab), histogram. Note: **no charting library is currently installed** — Recharts or lightweight hand-rolled SVG is a real decision to make before starting §5 |
| **`Drawer`** *(new)* | Right-slide panel for shot detail |
| **`SegmentedControl`** *(new)* | Club/national, track toggle, metric mode |

---

## 11. Build order

Sequenced by impact-to-effort, and by what unblocks what.

**Phase A — stop the bleeding** (the cold review's confirmed live defects; all small)
1. Delete Overview's fabricated data → §3
2. `StoryCard` → `Link` + write story #1 → §6.1
3. Fix Models' dead Stories link → §6.2
4. Write About → §6.3
5. Table headers on Players/Teams → §4.3, §4.5

**Phase B — make it real** (turns fetched-and-discarded data into content)
6. `DataTable` with sort + pagination
7. `G−xG`, `xG/shot`, min-shots filter → §4.3
8. Teams detail shot map + CxG → §4.6
9. Wire the Sidebar's dead Team dropdown and Metric toggle → §2.2
10. `/v1/summary` + Overview scale tiles → §9.1

**Phase C — depth**
11. Chart primitives decision, then xG race → §4.2
12. Shot detail drawer + map legend → §4.2
13. Prop-driven `ProfileRadar` + `Leaderboard` → §10
14. Remaining stories → §6.1

**Phase D — playground**
15. Shot Explorer → §5.1
16. Compare → §5.2
17. CxG Lab → §5.3
18. Leaderboard Builder → §5.4

**Phase A alone** addresses every confirmed defect a first-time visitor hits in their first two minutes. **Phase D** is what makes the project memorable rather than merely competent — but it is worth nothing if a visitor bounces off a fabricated Overview or a dead About page before reaching it.

---

## 12. Open decisions

Four things this spec deliberately does not decide:

1. **Charting library** — Recharts (fast, React-native, adds ~100KB) vs hand-rolled SVG (full theme control, more work, no bundle cost). Blocks Phase C. Note the existing `PitchMap` is already hand-rolled SVG and looks correct, which argues for consistency.
2. **Story authoring format** — MDX files (real articles, needs `@next/mdx`) vs a `body` string in `stories-data.ts` (no new dependency, painful past ~500 words).
3. **Whether `viewer` gating is worth it** — it gives signing in a purpose and matches v2's stated intent, but it does add a friction point on a public portfolio piece where maximum reach may matter more than tiering.
4. **Overview live strip cost** — `getMatches({})` unfiltered on the landing page is the one query every visitor triggers. It is TTL-cached, but if the link is shared widely this is the first thing to watch on the £10/month budget.
5. **Widening CxG coverage beyond the test split.** Confirmed live (26 Aug 2026): the v3 prediction tables hold train/validation/test for both tracks, and the dashboard shows only test (2,427 of 15,737 `cxg_event` rows; 590 of 3,960 `cxg_plus` rows — see §1.1). Showing the full 15,737/3,960 rows would make CxG coverage 6.5× larger overnight with no new model work. **Not decided, and not done in this pass**, because of a real methodological caveat: the model saw every training-split row during fitting, so a train-split prediction is not an honest out-of-sample result — showing it next to test-split predictions without a per-split label would let a visitor mistake a memorized fit for a generalization result. Any future widening must label each shot's split (e.g. a `Badge` reading "train — seen during fitting" vs "test — held out") rather than silently merging all rows into one undifferentiated coverage count. Coverage gain vs. cost of building that per-split labelling UI is the actual open question here — not whether the data exists.
