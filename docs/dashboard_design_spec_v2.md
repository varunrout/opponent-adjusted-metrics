# OAM Dashboard — Design Spec v2 (structure-first, persona-driven)

Status: draft for review. Not yet committed to git — review and lock this before any Claude Code prompt references it. Once locked, commit it alongside `nav-config.ts` and any structural code changes in the same commit, so doc and code can't drift the way `dashboard_design_spec.md`'s Phase 6 section did.

## 0. Why this supersedes v1's structure

`dashboard_design_spec.md` (v1) fixed a nav structure (Overview/Matches/Players/Teams/Analysis + Models/Stories/About) before anyone had defined what each type of visitor actually needs from this project. That produced a flat, arbitrary structure — filters that may or may not apply per page, a nav that treats a recruiter's 90-second visit the same as a data scientist's deep methodology read, and no explicit rule for what should be static versus live-queried.

This version starts from the other end: who looks at this, what do they each need, and only then — what pages, what's static, what's dynamic, what's in the nav, where do filters live.

**What is not being scrapped:** the component inventory (pitch-based components, charts, cards — table below, unchanged from v1), the colour palette and typography system (unchanged from v1), and the Firebase auth implementation (phases 5 and the `/analysis` admin-gating work — unchanged, keeps working exactly as built).

**What is being scrapped:** the seven-tab flat nav, the assumption that every page needs the same filter sidebar, and the implicit assumption that content should default to live BigQuery queries unless someone thinks to optimise it later.

---

## 1. Personas — the actual starting point

Five real visitor types, agreed in review:

| Persona | Primary need | Depth wanted | Static or dynamic content | What makes them bounce |
|---|---|---|---|---|
| **Recruiter / hiring manager** | "Is this person credible" — a 60–90 second decision | Minimal — one strong signal, not a deep dive | Static — a narrative overview, no explore/filter needed | Too much scrolling, numbers with no context |
| **Football analyst (fan-technical)** | Explore real insight — who's overperforming, browse a match | Medium — wants interactivity, no jargon | Dynamic — browsing and filtering (Matches/Players/Teams) | Everything frozen/static, nothing to click into |
| **Data scientist (peer)** | Verify rigor — methodology, validation, what failed and what worked | High — wants raw numbers, feature families, honest negative results (e.g. CxG v1 losing to the StatsBomb baseline is a credibility signal for this persona, not something to hide) | Static/curated (Stories, Models registry) plus a link to the real GitHub repo. Doesn't need live explore. | Marketing language, no way to verify claims are real |
| **Coach / stakeholder** | "Can this tell me something useful" — practical value, not methodology | Low–medium — case-study style, visual, plain language | Static — curated example narratives, not raw tables | Technical jargon, raw stats with no framing |
| **Guest / fan** | Casual browsing, fun findings | Low — football-literate, not technical | Mixed — some dynamic browsing (matches/players), some static (glossary) | Overwhelming technical depth |

Consequence worth stating plainly: three of five personas (recruiter, data scientist, coach) are fully served by **static, curated content** — none of them need a live filter sidebar. Only two (analyst, guest/fan) genuinely need to **explore** dynamically. That imbalance is what drives the structure below — it is not a 50/50 split between static and dynamic, it's closer to 60/40 toward static/curated.

---

## 2. Content zones

Two zones, not seven flat tabs:

### Understand (static / curated)
Overview narrative, Stories (methodology writeups, case studies, release notes, dev-log entries), Models registry (public layer — status/tier/validation metrics), About (data source, versioning, glossary). Serves recruiter, data scientist, coach, and the non-exploring half of guest/fan.

### Explore (dynamic / filterable)
Matches, Players, Teams — browse, search, filter by competition/season/team, drill into detail pages with shot maps, leaderboards, radar. Serves the football analyst persona and the exploring half of guest/fan.

### Analysis (admin-only, separate from both)
Not persona-facing at all — this is an operator's own research workbench (feature/model research browser, per Phase 6), not content built for any of the five visitor types. Stays admin-gated exactly as built. No visitor-scaling cost risk by construction, since only the admin account can reach it.

---

## 3. Navigation structure

| Zone | Nav items | Weight |
|---|---|---|
| Understand | Overview, Stories, Models, About | Secondary (lighter visual weight — these are read, not operated) |
| Explore | Matches, Players, Teams | Primary (heavier weight — these are the interactive core) |
| Admin-only | Analysis | Admin-only, unchanged |

This keeps roughly the same visual weighting v1 already had (five primary + lighter secondary tabs) but the placement is now justified by persona need rather than arbitrary judgement — Understand items are secondary because none of their audiences need them to be an interactive focal point, Explore items are primary because that's where genuine interaction happens.

---

## 4. Filters — where they apply, explicitly

Filters (competition/season/team) apply **only inside the Explore zone** — Matches, Players, Teams, and their detail pages. They do not apply to Understand-zone content: Stories, Models, About, and the Overview narrative are curated/authored, not queried, and filtering authored content doesn't mean anything (there's no "which competition's About page" — About is one fixed page).

Guest role has full access to both zones (per the roles table below), so filters are visible to every non-admin visitor who's actually browsing Explore content, not gated behind sign-in.

### 4a. CxG in the Explore zone

Unblocked by Hard gate 1's reframing (§9) now that CxG v3 results exist and are honestly documented (§11). CxG values may appear on Matches/Players/Teams pages under these rules:

- **Shown alongside, never replacing, StatsBomb xG.** Both numbers are visible together — CxG is an additional, clearly-labelled data point, not a swap-in.
- **Coverage-gated, not full `oam_core` coverage.** A shot only gets a CxG value where it falls inside `oam_ml`'s v3 test-set coverage — a fixed train/test split, not every shot in `oam_core`. Where a shot falls outside that split, show xG only. No CxG placeholder, dash, zero, or "N/A" — a value that looks like it should be there but isn't reads as broken, not as honestly out-of-scope.
- **Visible "Experimental · limited data & features" badge or tooltip on every CxG value**, disclosing plainly: 8 features for the event-wide track, 24 features for CxG+, evaluated on a fixed test split rather than scored live per request. This is the UI-level expression of Hard gate 1's "don't overclaim" language — CxG reads as evaluated/experimental, not validated or production, anywhere it appears.
- **Data source: `oam_ml.cxg_event_v3_*` / `oam_ml.cxg_plus_v3_*`, not `oam_serving`.** `oam_serving` stays empty — Hard gate 2 (a proper serving layer, and Track B's quadrant scatter) is a separate, still-blocked condition. This section covers surfacing already-computed v3 test-set predictions as read-only display data; it is not a serving-layer build.

---

## 5. Static vs dynamic — the rule, and per-section decisions

**Rule:** content is static (curated, pre-authored, or served from an already-rendered artifact with no live query) when the audience needs authority and consistency, not personalisation, and the content doesn't meaningfully change per visit. Content is dynamic (live-queried) only when it's inherently per-entity or combinatorial — you cannot pre-render every match/player/team/filter combination — or it's the admin's own workbench, where visitor-scaling cost isn't a concern.

| Section | Static or dynamic | Why |
|---|---|---|
| Overview | Static narrative + a small live "recent matches" strip | The pitch itself is authored; a thin live slice keeps it feeling current without much query cost |
| Stories | Static | Authored writeups, don't change per visit |
| Models (public layer) | Static, rebuilt on deploy or on model promotion, not per-visit | Model status doesn't change second-to-second; no reason to query live |
| About | Static | Fixed reference content |
| Matches / Players / Teams (list + detail) | Dynamic | Genuinely combinatorial — can't pre-render every competition/season/player |
| Analysis (admin) | Dynamic | Real-time research tool by nature; no visitor-scaling cost since admin-only |

**Caching, not just static-vs-dynamic:** the Explore zone stays dynamic, but needs a TTL cache layer in front of the BigQuery-backed queries (see Hard gate 4) — match/player/team aggregates don't need to be query-fresh to the second, and repeat visits to the same competition/season shouldn't re-bill BigQuery every time.

---

## 6. Visual assets — charts, three separate paths

Corrected from an earlier draft of this conversation, which wrongly lumped PNGs into dashboard scope:

- **PNG exports** — exclusively for offline markdown reports, generated once the whole project is complete. Out of dashboard scope entirely. Never embedded in the web app.
- **Plotly HTML (pre-rendered by the analysis pipeline, already in GCS, tracked in `cxg_rendered_chart_registry_v1`)** — used only inside the Analysis tab (admin-only), embedded via `<iframe src="{signed URL}">` (a full standalone Plotly HTML document can't be pasted directly into a React component). Theme mismatch versus the dashboard's dark/teal palette is accepted deliberately here — this audience (data scientist, via the admin's own research use) values seeing the pipeline's real, unedited output over visual consistency; rebuilding these exact chart types (correlation heatmaps, PCA scree plots, bivariate significance grids) as new theme-matched components would be a large lift serving a single-user, admin-only surface.
- **Reusable dashboard chart components** (`PitchMap`, `MetricTile`, `Leaderboard`, `ProfileRadar`, xG race line, ranked bar — full inventory below) — used exclusively in the Explore zone and the Overview's live strip, where theme consistency and interactivity genuinely matter to a broad audience.

---

## 7. Cost discipline

Grounded in real findings from this session, not hypothetical:

- BigQuery client is already a lazy singleton (`8da338c`) — don't regress this, don't add a second `bigquery.Client()` construction path anywhere.
- No unfiltered full-table scans. Every `ServingStore`/`AnalysisStore` query must filter by `competition_id`/`season_id`/`family`/etc. The one confirmed real cost incident this session (£2.5, dominated by 34 `oam_core` queries billing 6.1GB — average ~180MB/query, some over 1GB) came from exactly this pattern. Audit `bigquery_store.py` for any query missing a `WHERE`/partition-style filter before this ships more broadly.
- Analysis tab carries no visitor-scaling cost risk (admin-only, single user) — no caching urgency there beyond normal query hygiene.
- Explore zone (guest-visible) is the real cost-scaling risk once this link is shared — see Hard gate 4, must not ship widely without a cache layer.

---

## 8. Auth — unchanged, one known open bug

Firebase login stays exactly as built in phase 5: Google sign-in primary, email/password secondary, "Continue as guest" link, role resolved via custom claims through `/v1/me`, `RoleProvider` deriving role from the real auth session. The dry-run-guarded admin-claim script (`scripts/set_admin_claim.py`) stays as the one-off way to promote an account to admin.

One open reliability bug, diagnosed but not yet confirmed/fixed this session: `RoleProvider`'s `await getMe(idToken)` call has no timeout. If it hangs (slow/unresponsive backend), `roleResolved` never becomes `true`, and any page gating on that flag (currently only `/analysis`) is stuck on its loading skeleton forever, with zero backend requests ever firing. See Hard gate 5.

---

## 9. Hard gates

Explicit blocking conditions — nothing past a gate ships until its condition is met.

1. **CxG results (any version) may be shown in Models/Stories/Analysis, and CxG values may be shown in the Explore zone, only once an honest, documented comparison against the StatsBomb baseline exists — including where CxG underperforms.** Status language must not overclaim ("Evaluated," not "Validated" or "Production," until a production inference pipeline exists — see gate 2). *Status: unblocked — CxG v3 training is complete on both tracks, and the comparison (including CxG trailing StatsBomb on log_loss on both tracks) is documented in §11 below, satisfying the honest-comparison condition. This reframes the gate from "must beat the baseline" to "must be honestly compared to the baseline" — CxG v1 losing to StatsBomb was already framed in §1 as a credibility signal for the data-scientist persona, not something to hide, and v3 landing in the same place doesn't change that. A public-facing Stories writeup presenting this comparison to visitors is a separate, still-pending follow-up build task — this document's own disclosure is what satisfies the gate, not a claim that the writeup already exists. See §4a for where/how CxG may now appear in the Explore zone.*
2. **`oam_serving` must be populated with player-level CxG/CxA values** before: Track B (build-your-own quadrant scatter, originally scoped for Analysis) can be built at all. *Status: blocked — `oam_serving` has zero tables, confirmed live.*
3. **Signed URL generation for `gs://` chart URIs must exist** before: Analysis tab charts render as actual images/iframes instead of literal text paths. *Status: blocked — not yet built.*
4. **A TTL cache layer on `ServingStore`'s `oam_core` queries must exist** before: this dashboard link is shared with real external visitors at any scale beyond Varun's own testing. *Status: blocked — not yet built; real cost already observed from unfiltered testing traffic alone.*
5. **`RoleProvider`'s `/v1/me` call needs a timeout guard** before: this reliability bug can be considered closed. *Status: diagnosed, not yet fixed or confirmed.*
6. **This document must be locked and committed** before: any further Claude Code prompt references the old flat-tab structure in `nav-config.ts` / v1's `dashboard_design_spec.md`. Don't resume phase-numbered prompts against the old structure until this is final.

---

## 10. Carried over unchanged from v1

### Component inventory

| Component | Shows | Used on |
|---|---|---|
| Shot map | Shot locations, sized by xG/CxG, filled = goal, outline = miss, coloured by team or player | Match page, player page |
| Pass network | Average position per player (node), pass volume and success between pairs (edge thickness) | Match page, team page |
| Touch heatmap | Density of a player's or team's touches, hex-bin or KDE contour | Player page, team page |
| Defensive actions map | Pressures, tackles, interceptions as markers | Player page (defenders), team page |
| Territory / block shape | Defensive line, centroid, convex hull, high line vs low block, with a "before/after" toggle | Team page, methodology stories |
| 360 freeze-frame viewer | Defenders, goalkeeper, shot angle/occlusion cone at the moment of a shot | Shot detail drill-down |
| Progressive actions map | Arrows for carries and passes that cross a third or enter the box | Player page |
| xG/CxG race (chart) | Step line, cumulative over 90 minutes | Match page |
| Profile radar (chart) | Polygon vs positional percentile | Player page |
| Ranked bar (chart) | Horizontal bar, sorted | Leaderboards |
| Quadrant scatter (chart) | Two metrics, axes crossed at league median | Blocked — see Hard gate 2 |
| Rolling form (chart) | Line, trailing 5–10 match average | Player page |
| Distribution (chart) | Box or violin | Player page |
| Adjustment waterfall (chart) | Bridge chart, xG → feature family deltas → CxG | Shot detail — blocked on Hard gate 1 |
| Sparkline (chart) | Inline mini-line | Table rows |
| Metric tile (card) | Label, big number, delta, optional sparkline | Everywhere |
| Match card | Teams, score, competition, date, xG/CxG summary | List/grid views |
| Player card | Avatar/initials, name, position, team, headline stat | Search, leaderboards |
| Model card | Name/version, status badge, validation metric, feature family count | Models tab |
| Story card | Category tag, headline, takeaway, thumbnail | Stories tab |
| Comparison card | Two players/teams, stat-by-stat delta | Viewer-tier feature, not yet built |

Colour palette and typography (Montserrat for text, Inconsolata for figures, teal/violet/amber metric-family hues, green/red semantic-only) — unchanged from v1, see `dashboard_design_spec.md` section 2.

Firebase auth implementation — unchanged from v1 phase 5, see section 8 above.

---

## 11. CxG v3 results (reference)

Written down once here so it doesn't need re-explaining in every future prompt — this is the honest comparison that satisfies Hard gate 1 (§9), and the source data for the §4a Explore-zone badge copy and any future Stories writeup.

### Test-set results vs StatsBomb baseline

| Track | Metric | CxG v3 | StatsBomb xG | Result |
|---|---|---|---|---|
| Event-wide | log_loss | 0.3003 | 0.2597 | CxG trails |
| CxG+ | log_loss | 0.2555 | 0.2430 | CxG trails |

Brier score and AUC for both tracks were not captured in this session — pull the exact values from the v3 training run artifacts before they're quoted in a Stories writeup or anywhere public-facing. Do not infer or estimate them from the log_loss figures above.

Consistent with v1: CxG trails the StatsBomb baseline on both tracks, same direction of result as v1's evaluation. Per §1's persona framing, this is disclosed as a credibility signal for the data-scientist persona, not hidden or softened.

### Frozen version history

- `cxg_baseline_v1_*` — event-wide, v1 (frozen)
- `cxg_plus_v2_*` — CxG+, v2 (frozen)
- `cxg_v3_*` — event-wide, v3 (frozen, current)
- `cxg_plus_v3_*` — CxG+, v3 (frozen, current)

### Known caveats

1. **Feature-pool asymmetry.** Event-wide uses 8 features; CxG+ uses 24. The two tracks are not a like-for-like comparison of "does adding 360 data help" — some of the gap between tracks reflects feature count, not just data richness. Any comparison drawn between the two tracks should say this explicitly.
2. **`zone_displacement`'s unexplained bimodality.** This feature shows a bimodal distribution with no documented cause yet. Flagged as an open question for future investigation, not resolved — don't present `zone_displacement` as a clean, well-understood feature until this is explained.
