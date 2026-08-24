# OAM Dashboard — Design Spec (v1, locked)

Status: locked. Ready for implementation. A working HTML/CSS/JS skeleton of the app shell (nav, sidebar, panels, role switch) lives alongside this file at `docs/dashboard_layout_skeleton.html` — treat it as the visual and structural reference. The build prompt for Claude Code is in the appendix at the bottom of this document.

Scope covered: component inventory, colour system, typography, navigation, model roadmap, content/voice direction, hosting and auth, role-based views, phase 1 build scope, and the Claude Code prompt.

---

## 1. Component inventory

One core principle: build each component once, reuse everywhere. A player's shot map and a match's shot map are the same component with different data in.

### Pitch-based components (all share one `PitchMap` primitive, StatsBomb 120x80 coordinate space)

| Component | Shows | Used on |
|---|---|---|
| Shot map | Shot locations, sized by xG/CxG, filled = goal, outline = miss, coloured by team or player | Match page, player page |
| Pass network | Average position per player (node), pass volume and success between pairs (edge thickness) | Match page, team page |
| Touch heatmap | Density of a player's or team's touches, hex-bin or KDE contour | Player page, team page |
| Defensive actions map | Pressures, tackles, interceptions as markers | Player page (defenders), team page |
| Territory / block shape | Defensive line, centroid, convex hull, high line vs low block, with a "before/after" toggle | Team page, methodology stories |
| 360 freeze-frame viewer | Defenders, goalkeeper, and shot angle/occlusion cone at the moment of a single shot | Shot detail drill-down (click a dot on the shot map) |
| Progressive actions map | Arrows for carries and passes that cross a third or enter the box | Player page |

The freeze-frame viewer is the one genuinely distinctive visual here — it's the only place the 360 data (and by extension CxG+) becomes visible rather than just a number in a table. Worth prioritising once CxG+ has anything to show.

### Charts

| Chart | Type | Answers |
|---|---|---|
| xG/CxG race | Step line, cumulative over 90 minutes | Who controlled the match, and when did it swing |
| Profile radar | Polygon vs positional percentile | How does this player compare to peers in their position |
| Ranked bar | Horizontal bar, sorted | Leaderboards — top CxG overperformers, etc. |
| Quadrant scatter | Two metrics, axes crossed at league median | Player-type discovery, e.g. high CxA / low CxG = creator not finisher |
| Rolling form | Line, trailing 5 or 10-match average | Is a player's underlying quality trending, not just results |
| Distribution | Box or violin | Is a player's shot selection unusual for their role |
| Adjustment waterfall | Bridge chart, xG → feature family deltas → CxG | The explainability chart — literally shows what E1–E13/F1–F15 did to the number. Unique to this project, worth a dedicated spot on the shot detail view |
| Sparkline | Inline mini-line | Trend-at-a-glance inside table rows |

The waterfall chart is the one I'd flag as the signature visual — it turns your feature-family taxonomy from an internal contract into something a viewer can actually see working.

### Cards

| Card | Content |
|---|---|
| Metric tile | Label, big number, delta vs baseline, optional sparkline |
| Match card | Teams, score, competition, date, CxG summary — used in list/grid views |
| Player card | Avatar/initials, name, position, team, headline stat — search results, leaderboards |
| Model card | Model name/version, status badge (in training / validated / promoted / deprecated), key validation metric, feature family count |
| Story card | Category tag, headline, one-line takeaway, thumbnail chart |
| Comparison card | Two players or teams, stat-by-stat delta |

A live example of the shot map and metric tiles is in the previous message. A palette-and-card preview is above this one.

---

## 2. Colour palette

Fewer hues, each one meaning exactly one thing, used consistently everywhere it appears. Total working palette: three metric-family hues, two semantic hues, and a neutral scale. That's it.

| Role | Colour | Hex | Rule |
|---|---|---|---|
| CxG (and default brand accent) | Teal | `#14B8A6` | Anywhere CxG appears — charts, badges, active nav |
| CxA | Violet | `#8B5CF6` | Anywhere CxA appears, once it ships |
| CxT | Amber | `#E3A008` | Anywhere CxT appears, once it ships |
| Positive / goal / favourable delta | Green | `#22C55E` | Outcome and delta encoding only, never decorative |
| Negative / error / unfavourable delta | Red | `#EF4444` | Same rule |
| Canvas | Near-black | `#0B0E12` | Page background |
| Surface | `#12161B` | Elevated panels |
| Card | `#171C22` | Cards, table rows |
| Border | `#262D35` | Hairlines only, never a fill |
| Text primary | `#EDEFF2` | |
| Text secondary | `#97A1AD` | |
| Text muted | `#7C8894` | Captions, hints |

Two colours that exist only inside a single match's visuals, never globally: home-team blue (`#3B82F6`) and away-team orange (`#F97316`), assigned per match, not fixed to real clubs.

Why dark: this is a data-dense tool people will stare at for a while, and pitch graphics (which are inherently colourful, green turf plus markers) read far better against a dark canvas than a white one — that's the reason every serious football analytics product (StatsBomb IQ, Wyscout) defaults dark.

The rule that keeps this "efficient and distinguishable": a colour is either a metric-family identity (teal/violet/amber — learn once, recognise everywhere) or a semantic outcome (green/red — always means better/worse, never used for anything else). Nothing is coloured just to look nice.

### Typography — two fonts, split by job

- **Montserrat** — everything you read: nav, headings, body copy, labels, badges, buttons. Weights 400/500/600/700.
- **Inconsolata** — everything that's a number and nothing else: metric tile values and deltas, leaderboard figures and ranks, timestamps. Monospace makes columns of numbers line up and reads as more precise, which is the effect wanted on a stats product. It stays out of headings and prose, where a fixed-width face hurts scanability.
- Rule of thumb for new components: if the text is a sentence or a label, Montserrat. If it's a standalone figure, Inconsolata.

---

## 3. Navigation

Primary tabs (top nav):

- **Overview** — landing page. Competition/season picker, recent results, one featured story.
- **Matches** — browse and search by competition/season/team. Match detail page underneath (shot map, xG/CxG race, lineups, pass network).
- **Players** — search, leaderboards, player profile (radar, rolling form, shot map, comparisons).
- **Teams** — team profile (season summary, block-shape trends, squad table).
- **Analysis** — the free-form workspace: build your own quadrant scatter, custom leaderboards, saved filters. This is the "power user" tab.

Secondary tabs (still top-level, visually lighter weight):

- **Models** — the ML registry made visible. One card per model (CxG, CxG+, CxA once it exists, CxT once it exists), status, validation metrics, feature family list, link to its methodology writeup. This is the tab that shows engineering rigor, not just output — worth having even when only CxG is live.
- **Stories** — long-form writeups: methodology explainers, case studies, release notes per model version. Article-card grid.
- **About** — data source, versioning philosophy, a plain-language glossary of every metric.

Seven tabs is fine at this weight distribution (five primary + two lighter secondary), no dropdown needed yet. Revisit only once Analysis and Models each grow enough to need their own sub-navigation.

---

## 4. Roadmap: CxA, CxG+, CxT, and beyond

Rather than treat CxA/CxA+/CxA_Advanced as one-offs, generalise the tiering you've already established for CxG (event-only vs 360-enhanced) into a pattern every metric family follows. Three tiers:

| Tier | Adds | Constraint |
|---|---|---|
| Core | Event data only | Available for every match |
| Spatial (`+`) | 360 freeze-frame features | Only where 360 coverage exists — this already matters for CxG+, worth surfacing coverage as a filter/badge in the UI |
| Advanced | Possession-chain / sequence-level modelling, ensembles, uncertainty quantification | Research-stage, slowest to ship |

Applied to the current and planned families:

| Family | Core | Spatial | Advanced |
|---|---|---|---|
| CxG | In progress | Planned (CxG+) | Not obviously needed — a shot is a single moment |
| CxA | Planned | Planned (CxA+) | Planned (CxA_Advanced — likely a chained/possession-value model, crediting the buildup, not just the touch) |
| CxT | Planned | Planned (CxT+) | TBD |

Design consequence: the Models tab and every metric badge should render a tier chip (Core / Spatial / Advanced) generically, keyed off data rather than hardcoded per family. When CxA+ or CxT ship, they're new rows in an existing table and new cards in an existing grid — not a redesign. Same logic extends the shot-detail waterfall chart once CxA/CxT are live: it becomes "action value adjustment," not just "goal value adjustment."

---

## 5. Content and voice — phrasing directions, not final copy

Tone: precise, plain, no hype. This is a technical product, the copy should read like it, not like a SaaS landing page.

Tagline directions (pick one later, or write a fourth):
- "Football analytics, adjusted for who you're playing."
- "Expected goals that know the opponent."
- "Context-aware football metrics, built on StatsBomb data."

Model status badges — plain verbs, no marketing:
- In training / Validated / Promoted / Deprecated

Empty or not-yet-live states — say what's true, not "coming soon":
- "CxA is in development. xG and CxG are live for every covered match."

Metric glossary entries — one sentence, defines the adjustment, not just the acronym:
- "CxG — expected goal value adjusted for match context, and, where 360 data exists, defensive shape at the moment of the shot."

Story headline pattern — a question or a claim the article then earns:
- "[Player] is beating xG by 0.3 per 90 — variance, or is CxG explaining it?"

---

## 6. Hosting and authentication (GCP)

Fits the infra you already have: Cloud Run for both `oam-api` and `oam-dashboard`, images in the existing `oam-containers` Artifact Registry repo.

Recommended: **Firebase Authentication** (Google sign-in plus email/password), not Identity-Aware Proxy. IAP is an allowlist gate for internal tools — fine if only you ever log in, wrong if you want a recruiter or a stranger to be able to view the public tiers. Firebase Auth lets anyone sign up, and you assign a role afterwards via custom claims (`role: guest | viewer | admin`), verified server-side in FastAPI with the Firebase Admin SDK.

Auth page: single centred card on the dark canvas, "Continue with Google" as the primary action, email/password as a secondary option, and — because most visitors won't need an account — a "Continue as guest" link into the public read-only tier. No hero copy, no marketing chrome. A dimmed, static pitch graphic behind the card for atmosphere is enough.

---

## 7. Viewer roles and differentiated views

| Role | Access | Sign-in |
|---|---|---|
| Guest | Overview, Matches, Players, Teams, Stories, About — read-only | None required |
| Viewer | Everything Guest has, plus saved views/bookmarks, side-by-side comparisons | Free account |
| Admin (you) | Everything, plus the full Analysis workspace, Models tab internals, and the ability to pin/inspect any historical `model_version`, not just the currently promoted one | Your account only |

Enforced the same way on both ends: FastAPI checks `role` per route (and per field, where a response mixes public and internal data, e.g. a model card's validation internals), Next.js hides nav items and routes the role can't reach rather than just disabling buttons.

---

## Open decisions for you to lock

1. Single accent hue — confirmed teal, doubling as both "brand colour" and "CxG colour." Fine as long as you're comfortable CxG stays visually the flagship metric.
2. Seven tabs at two weights (five primary, two secondary) — or would you rather fold Models and Stories under one "Insights" menu now, before Claude Code builds the nav?
3. Firebase Auth roles as three tiers (guest/viewer/admin) — enough, or do you want a fourth tier between viewer and admin (e.g. "contributor" who can annotate but not see model internals)?

---

## Phase 1 build scope

Status: shipped and verified — `feature/dashboard-scaffold` worktree. `next build` clean (10/10 static routes), `tsc --noEmit` clean, 2/2 web tests and 3/3 API tests passing, FastAPI confirmed live against real `oam_core` in BigQuery via `pip install -e .` + uvicorn. Ready to merge.

Deliberately narrow, so the first Claude Code pass is a short, reviewable run rather than an open-ended build:

- Next.js (App Router, TypeScript) app shell in `web/`, matching `docs/dashboard_layout_skeleton.html` structurally: top nav, sidebar filters, Overview/Models/Stories panels built out, the other five nav targets rendered as loading-skeleton placeholders, and the role switch working client-side.
- Montserrat + Inconsolata wired in exactly as split in this doc, dark palette as CSS variables (or Tailwind theme tokens), not hardcoded hex scattered through components.
- FastAPI skeleton in `src/opponent_adjusted/api/`: a couple of real read-only endpoints (competitions, matches) backed by `oam_core` (there is no `oam_serving` yet), Pydantic response models, no auth wired up yet — stub the role check so the shape is there but it's not load-bearing.
- No auth implementation, no CxA/CxT data, no write paths, no changes to ingestion/features/modeling/pipelines. Those are later phases.

## Phase 2 build scope

Status: shipped and verified — 8/8 API tests, 3/3 web tests, tsc clean. Merged to main.

Modelling (CxG) isn't done and nothing is exposed on `oam_serving` yet. Phase 2 does not wait on it: build out everything that's honestly supportable off `oam_core` alone, using StatsBomb's own `statsbomb_xg`, not CxG. Every metric surfaced this phase is labelled xG, not CxG, no placeholders pretending otherwise.

Scope: the Matches vertical, end to end.

- API: `GET /v1/matches/{match_id}` (single match, including lineups from `starting_xi_players`), `GET /v1/matches/{match_id}/shots` (joins `shots` to `events` for player name/minute, per the same join `publish_core.py` already validates). New `ShotRecord` / `ShotResponse` following the existing `interfaces.py` / `models.py` pattern.
- Web: `Matches` page goes from placeholder to a real list backed by `/v1/matches`, filterable by the competition/season selects already in the sidebar (backed by `/v1/competitions`, already live). New `/matches/[matchId]` route: score header, `PitchMap` (already built in phase 1, just needs real data instead of the mock array) wired to `/v1/matches/{id}/shots`, `MetricTile`s for home/away total xG computed client- or server-side from the shot list, lineups.
- The CxG toggle stays visually present but disabled with the "soon" tag already established in the sidebar, don't build a dead code path for a metric that doesn't exist yet.

Queued right after, same session if you want to keep going: Phase 3 is Players and Teams season-aggregate pages, same query pattern (join shots/events, aggregate), same components (`MetricTile`, `Leaderboard`, `ProfileRadar`) — kept as its own prompt so each Claude Code run stays a small, reviewable diff rather than one long one.

## Phase 3 build scope

Still off `oam_core` alone, still xG not CxG. Players only — Teams (phase 4) is the identical pattern one prompt later, deliberately not bundled in.

No minutes-played tracking exists yet, so no per-90 figures this phase — goals, shots, and total xG only. Honest about what's derivable, not padded out with a stat we can't actually compute yet.

- API: one new aggregation query, `GET /v1/players?competition_id=&season_id=` — `GROUP BY player_id` over shots joined to events, returning player_id/player_name/team_name/shots/goals/total_xg, sorted by total_xg descending. This is the only genuinely new SQL; everything else reuses what phase 2 already built. `GET /v1/players/{player_id}/shots?competition_id=&season_id=` reuses `ShotRecord`/`ShotResponse` as-is, same shape as the match shots endpoint, just filtered by player instead of match.
- Web: `Players` page becomes a real leaderboard off `/v1/players`, using the same competition/season filter state `MatchFilterProvider` already holds (reuse the provider, don't fork a second copy of the same state). `/players/[playerId]` page: season summary tiles (goals, shots, total xG) computed client-side from the shot list exactly the way the match detail page already derives its xG tiles, plus `PitchMap` reusing the same shot list to show every shot that player took across the scoped competition/season.

Phase 4, queued right after: Teams, same shape — one aggregation endpoint (`GROUP BY team_id`), a leaderboard/table page, a team detail page that mostly links back into the already-real Matches list rather than inventing new visuals.

Status (phase 3): shipped and verified — 16/16 API tests, 5/5 web tests, tsc clean.

## Phase 4 build scope

Teams. Same shape as phase 3, off `oam_core`, still xG not CxG.

- API: `GET /v1/teams?competition_id=&season_id=` — identical pattern to `list_player_seasons`, `GROUP BY team_id` over shots joined to events (shots, goals, total_xg). `GET /v1/teams/{team_id}/shots?competition_id=&season_id=` — reuses `ShotRecord`/`ShotResponse` as-is, filtered by team_id. Extend the existing `GET /v1/matches` (and `list_matches` on the Protocol) with an optional `team_id` filter — matches where the team played home or away — rather than inventing a separate matches-by-team endpoint. This is the one place phase 4 touches code that phase 2 already shipped; keep it additive, existing callers with no `team_id` must behave exactly as before.
- Web: `Teams` page — leaderboard off `/v1/teams`, identical structure to the Players page. `/teams/[teamId]` page: summary tiles via `summarizeShots` (reuse the phase 3 utility, don't reimplement it) on that team's shot list, plus a "recent matches" card listing `/v1/matches?team_id=...` results, each row linking into the existing `/matches/[matchId]` page. No new pitch or chart component this phase, this is intentionally the "reuse, don't build" phase.

## Appendix — prompt for Claude Code (phase 1, shipped)

```
Work in a new git worktree off main: git worktree add ../oam-dashboard -b feature/dashboard-scaffold
Do not touch anything under src/opponent_adjusted/ingestion, features, modeling, or pipelines — this branch is dashboard scaffolding only, read-only against oam_core.

Read docs/dashboard_design_spec.md and docs/dashboard_layout_skeleton.html first. The skeleton is the structural and visual reference — match it, don't redesign it.

Build phase 1 only, as scoped in the "Phase 1 build scope" section of the design doc:

1. web/ — Next.js App Router, TypeScript. App shell (top nav, sidebar, role switch) plus Overview, Models, and Stories panels as real components. The remaining five nav targets (Matches, Players, Teams, Analysis, About) render a shared loading-skeleton placeholder component, not full pages yet. Pull typography and colour tokens from the design doc's palette and typography sections, define them once as CSS variables or Tailwind theme config, not inline per component.

2. src/opponent_adjusted/api/ — FastAPI app with routers for GET /v1/competitions and GET /v1/matches, reading from oam_core via BigQuery. Pydantic v2 response models. Add a ServingStore-style Protocol for the query layer, matching the pattern already used in src/opponent_adjusted/storage/interfaces.py, so routers are testable without hitting real BigQuery. Stub a role dependency (guest/viewer/admin) that always resolves to admin for now — don't implement Firebase yet.

3. Tests: at minimum, router tests using a fake store, and a basic render test for the app shell's role-switching logic.

Stop after phase 1. Don't implement auth, CxA/CxT, oam_serving, or Cloud Run deployment — those are separate follow-up prompts. End with a short summary of what you built and any deviations from the design doc, not a full narration.
```

## Appendix — prompt for Claude Code (phase 2)

```
Continue in the existing feature/dashboard-scaffold worktree (oam-dashboard). Still read-only against oam_core, still no changes under src/opponent_adjusted/ingestion, features, modeling, or pipelines.

Read docs/dashboard_design_spec.md's "Phase 2 build scope" section first.

CxG does not exist yet and oam_serving is empty — every metric this phase is StatsBomb's own statsbomb_xg, labelled xG, not CxG. Don't build a CxG data path, don't fake CxG values.

1. API: add GET /v1/matches/{match_id} (single match plus lineups, join matches to starting_xi_players) and GET /v1/matches/{match_id}/shots (join shots to events for player_name/minute/period, matching the join oam_core's own publish_core.py already validates as shots_join_events_matches). Add ShotRecord to interfaces.py and ShotResponse to models.py following the existing pattern exactly — don't invent a different shape. Extend the ServingStore Protocol and BigQueryServingStore accordingly, and extend the FakeServingStore in tests/api/conftest.py with fake shots so the new router is tested the same way matches/competitions are: no real BigQuery in tests.

2. Web: replace the Matches PlaceholderPanel with a real page reading from /v1/matches, filterable using the competition/season selects already in the sidebar (which should now actually call /v1/competitions instead of being static options). Add a /matches/[matchId] route: score header, the existing PitchMap component wired to /v1/matches/{id}/shots instead of its current mock data, MetricTiles for home/away total xG summed from the shot list, and a simple lineup list from the match detail endpoint. Leave the CxG toggle visible but disabled with the "soon" tag already established, don't wire it to anything.

3. Tests: router tests for the two new endpoints using a fake store, and at least one component test asserting PitchMap renders the right number of shot markers for a given shot list.

Stop after phase 2 — Players, Teams, and Analysis are the next prompt, not this one. End with a short summary and any deviations, not a full narration.
```

## Appendix — prompt for Claude Code (phase 3)

```
Continue in the feature/dashboard-scaffold worktree (oam-dashboard), now merged to main — pull/rebase if needed before starting. Still read-only against oam_core, still no changes under ingestion, features, modeling, or pipelines.

Read docs/dashboard_design_spec.md's "Phase 3 build scope" section first.

Players only this phase — do not touch Teams or Analysis, those are separate follow-up prompts. Still xG, not CxG. No per-90 stats: there's no minutes-played data yet, so don't compute or fake one — goals, shots, and total xG only.

1. API: add GET /v1/players?competition_id=&season_id= — a GROUP BY player_id aggregation over shots joined to events (reuse the exact join shots_router already uses), returning player_id, player_name, team_name, shots, goals, total_xg, sorted by total_xg descending. Add a PlayerSeasonRecord to interfaces.py and PlayerSeasonResponse to models.py following the existing pattern. Add GET /v1/players/{player_id}/shots?competition_id=&season_id= reusing the existing ShotRecord/ShotResponse as-is, filtered by player_id instead of match_id. Extend ServingStore, BigQueryServingStore, and the FakeServingStore in tests/api/conftest.py the same way prior phases did.

2. Web: Players page becomes a real leaderboard off /v1/players, reusing MatchFilterProvider's existing competition/season state rather than forking a second copy of it. Add /players/[playerId] page: season summary tiles (goals, shots, total xG) computed client-side from the player's shot list the same way the match detail page derives its xG tiles, and PitchMap reusing that same shot list.

3. Tests: router tests for both new endpoints using a fake store, and a component/page test asserting the player detail page's summary tiles compute correctly from a known shot list.

Stop after phase 3. Teams is the next prompt. End with a short summary and any deviations, not a full narration.
```

## Appendix — prompt for Claude Code (phase 4)

```
Continue in the feature/dashboard-scaffold worktree (oam-dashboard). Pull/rebase against main first if phase 3 was already merged. Still read-only against oam_core, still no changes under ingestion, features, modeling, or pipelines.

Read docs/dashboard_design_spec.md's "Phase 4 build scope" section first.

Teams only. Still xG, not CxG. This phase is explicitly "reuse, don't build" — no new pitch or chart component.

1. API: add GET /v1/teams?competition_id=&season_id=, identical pattern to list_player_seasons (GROUP BY team_id over shots joined to events: shots, goals, total_xg). Add GET /v1/teams/{team_id}/shots?competition_id=&season_id=, reusing ShotRecord/ShotResponse as-is. Extend the existing GET /v1/matches endpoint and list_matches on the ServingStore Protocol with an optional team_id filter (matches where the team played home or away) — additive only, existing calls without team_id must be unaffected. Extend BigQueryServingStore and the FakeServingStore in tests/api/conftest.py accordingly.

2. Web: Teams page as a leaderboard off /v1/teams, same structure as the Players page. Add /teams/[teamId] page: summary tiles computed via the existing summarizeShots utility from lib/shot-summary.ts (reuse it, don't reimplement it) on that team's shot list, plus a "recent matches" card listing /v1/matches?team_id=... results, each linking to the existing /matches/[matchId] page.

3. Tests: router tests for the two new team endpoints and the team_id filter on the matches endpoint, all via the fake store.

Stop after phase 4. Auth (Firebase, replacing the stubbed role dependency) is the next prompt, not this one. End with a short summary and any deviations, not a full narration.
```

Status (phase 4): shipped and verified — 27/27 API tests, 5/5 web tests, tsc clean. The `team_id` extension to `/v1/matches` was additive; phase 2's original two tests for that endpoint pass unmodified.

## Phase 5 build scope — Auth

This one is different from phases 1–4: it needs real infrastructure Claude Code can't provision itself. Two manual steps happen outside the prompt, on your side, before or right after the code lands:

1. Create (or attach) a Firebase project on the same GCP project (`oam-varun-260819`), enable the Google and Email/Password sign-in providers in the Firebase console.
2. Decide the default-role policy and set custom claims accordingly — new sign-ups default to `guest`, you promote your own account to `admin` by hand (a one-off script or the Firebase console, not something to automate blindly — deciding who gets admin isn't a call to hand to an agent).

Claude Code's job is the code path, correct end to end, gracefully doing nothing destructive if credentials aren't configured yet in dev.

- API: replace the stubbed `get_role()` with real verification — read the `Authorization: Bearer <token>` header if present, verify it via `firebase-admin`'s `verify_id_token`, read the `role` custom claim. No token at all resolves to `guest`, not an error — guest is anonymous access by design, per the role table. An invalid/expired token is a 401, not a silent fallback to guest. Add `GET /v1/me` returning the resolved role (plus uid/email if authenticated) so the frontend has one place to ask "who am I" after sign-in. Firebase Admin initializes via Application Default Credentials (works on Cloud Run as-is); for local dev, read the service account path from an env var, and don't crash on import if it's unset — degrade to guest-only, log it, don't pretend auth works when it can't initialize.
- Web: add the Firebase client SDK, config from `NEXT_PUBLIC_FIREBASE_*` env vars. New `/login` page matching the design doc's section 6: centred card, dark canvas, "Continue with Google" primary, email/password secondary, "Continue as guest" link, a dimmed static `PitchMap` behind the card for atmosphere (reuse the component, not a new asset). `RoleProvider` stops taking its role from local state and instead derives it from the real Firebase auth session (`onAuthStateChanged` + `/v1/me`), defaulting to guest when signed out. The existing manual "view as" `RoleSwitch` dev tool doesn't get ripped out, gate it behind `process.env.NODE_ENV !== "production"` so it's still there for local QA but never ships live once real auth exists.
- Tests: role-dependency tests using a fake token verifier (no real Firebase project needed to run tests) covering no-token→guest, valid-token-with-claim→that role, and invalid-token→401.

## Appendix — prompt for Claude Code (phase 5)

```
Continue in the feature/dashboard-scaffold worktree (oam-dashboard), rebased on main. Still no changes under ingestion, features, modeling, or pipelines.

Read docs/dashboard_design_spec.md's "Phase 5 build scope — Auth" section first.

This phase touches real auth infrastructure. I will create the Firebase project, enable providers, and set custom claims myself outside this prompt — don't invent or assume real credentials exist yet. Code must degrade gracefully (to guest-only, with a clear log line) if Firebase Admin can't initialize in the current environment, not crash the API on startup.

1. API: add firebase-admin. Replace the stubbed get_role() in dependencies.py with real verification: parse an Authorization: Bearer <token> header if present, verify via firebase_admin.auth.verify_id_token, read the role custom claim. Missing token resolves to "guest" (not an error — guest is anonymous by design). An invalid or expired token is a 401. Add GET /v1/me returning the resolved role and, if authenticated, uid/email. Initialize Firebase Admin with Application Default Credentials, reading a service account path from an env var for local dev, guarded so import/startup doesn't fail if unset.

2. Web: add the Firebase client SDK, reading config from NEXT_PUBLIC_FIREBASE_* env vars (don't hardcode a project's config). Add /login: centred card on the dark canvas, "Continue with Google" as the primary action, email/password as secondary, a "Continue as guest" link, and a dimmed static PitchMap behind the card (reuse the component). Change RoleProvider to derive role from the real Firebase auth session (onAuthStateChanged plus a call to /v1/me) instead of local state, defaulting to guest when signed out. Keep the existing manual RoleSwitch dev tool but gate its rendering behind process.env.NODE_ENV !== "production" — don't delete it, don't ship it live.

3. Tests: role-dependency tests using a fake/mocked token verifier, no real Firebase project required to run them — cover no-token → guest, valid token with a role claim → that role, invalid token → 401.

Stop after phase 5. End with a short summary, any deviations, and an explicit list of what I still need to do manually in the Firebase console before this works end to end in a live environment.
```

