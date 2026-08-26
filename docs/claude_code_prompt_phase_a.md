# Claude Code prompt — Phase A (confirmed defects + dataset corrections)

Copy everything below the line into Claude Code.

---

You are working in the `oam-dashboard` repo, on branch `feature/dashboard-scaffold`.

Two documents are your source of truth, read both before starting:
- `docs/dashboard_design_spec_v2.md` — personas, zones, hard gates (the "why")
- `docs/dashboard_content_spec_v3.md` — per-tab content, layout, data sources (the "what")

A cold external review (`docs/cold_review_dashboard.md`) found the live deployed dashboard scores 5/10 against an 8/10 potential, with the gap almost entirely in confirmed frontend defects. This prompt fixes those defects and corrects two spec inaccuracies I found by querying BigQuery directly.

**Do not** attempt the Playground (v3 §5), charts (v3 §4.2 Row 3), or anything in v3 Phase C/D. Those are later work and need a charting-library decision first.

---

## Part 0 — Correct the specs against real BigQuery state

I queried the live datasets. Two things in the specs are wrong or incomplete. Fix the docs first so nothing downstream inherits the error.

**Finding 1 — CxG prediction coverage is 6.5× larger than the specs assume.**

`oam_ml.cxg_event_v3_predictions` and `cxg_plus_v3_predictions` contain **all three splits**, not just test:

| Track | train | validation | test | total |
|---|---|---|---|---|
| `cxg_event` | 10,890 | 2,420 | 2,427 | 15,737 |
| `cxg_plus` | 2,780 | 590 | 590 | 3,960 |

`src/opponent_adjusted/api/cxg_coverage.py`'s `_get_track_coverage()` hardcodes `WHERE split = 'test'`, so the app currently surfaces 2,427 of 15,737 available event-track predictions.

Both prediction tables also already carry `statsbomb_xg` and `is_goal` columns — meaning an xG-vs-CxG comparison needs no join back to `oam_core`.

Update `docs/dashboard_design_spec_v2.md` §4a and `docs/dashboard_content_spec_v3.md` §1.1/§5.3 to state the real split counts and the fact that `statsbomb_xg`/`is_goal` are available in `oam_ml`.

**Do not change the `split = 'test'` filter in code in this prompt.** Widening coverage to train/validation is a real product decision with a methodological caveat (the model saw the training rows, so showing those predictions requires labelling them per-split so nobody reads a train-set prediction as an honest out-of-sample result). Write it up as a decision in v3 §12 "Open decisions" instead, noting the coverage gain and the caveat.

**Finding 2 — the actual opponent-adjustment data is unsurfaced.**

`oam_analysis.cxg_analysis_opponent_adjusted_v1` exists with **3,960 rows across 166 matches and 835 players**, keyed by `event_id`/`match_id`/`player_id`/`team_id`. Columns include `nearest_defender_odi`, `mean_backline_odi`, `gk_odi`, `defensive_profile_cluster`, `nearest_defender_role`, `nearest_defender_zone_displacement`, `nearest_defender_gap`, `nearest_defender_style_archetype` (3,557 non-null), `has_360_frame` (3,960 non-null).

Nothing in the codebase reads this table. It is the literal content of the phrase "opponent-adjusted" that the whole project is named after.

Add a subsection to `docs/dashboard_content_spec_v3.md` §9 describing a future `[BACKEND]` endpoint that would expose per-shot opponent context from this table. Do not build the endpoint in this prompt — just spec it, with its cost note (3,960 rows is small; a filtered-by-match_id query is cheap).

Also record in v3 §1 that `oam_features` contains 9 tables and is read by zero application code.

---

## Part 1 — Overview page rebuild

`web/app/overview/page.tsx` is currently 100% fabricated: `DEMO_SHOTS` (5 invented shots), a hardcoded `Team CxG 1.82` tile, an empty `Team CxA` tile, a `Leaderboard` with hardcoded bars and two permanently-shimmering rows, and a data-free `ProfileRadar`. None of it is labelled as sample data.

This is the project's worst inconsistency: the one page arguing for honest disclosure is the one page showing invented numbers.

Rebuild per `docs/dashboard_content_spec_v3.md` §3, with this scope adjustment: **skip Row 1 (scale tiles) for now** — it needs the `/v1/summary` endpoint which is out of scope here. Build Rows 0, 2, 3, 4:

- **Row 0** — statement block: H1, one sentence on what the project does, one provenance line (StatsBomb Open Data, 3 competitions, 5 seasons, GitHub link).
- **Row 2** — recent matches strip: call `getMatches({})`, sort by `match_date` descending, take 6. Each row is a `ClickableRow` to `/matches/{match_id}` with `TeamLink`s and the score in `font-data`. This makes the page a client component — that's expected and correct.
- **Row 3 left** — "The honest result" card: the six-row CxG-vs-StatsBomb table from v2 §11, one paragraph on why a trailing result is disclosed rather than hidden, and a link to the story slug from Part 2.
- **Row 3 right** — "Built with" list (FastAPI, BigQuery, Next.js, Firebase Auth, Cloud Run) and one featured `StoryCard`.
- **Row 4** — three entry-point cards: Matches, Players, Stories.

Delete `DEMO_SHOTS`, `DEMO_HOME_TEAM_ID`, and the fabricated tiles entirely. Do not relabel them as "sample" — remove them.

Give the strip proper loading (`Skeleton` matching final shape), empty, and error states per v3 §2.3.

---

## Part 2 — Stories: make the cards work, write the first article

**Confirmed defect:** `web/components/ui/StoryCard.tsx` renders a `<div>` with no `href`. All five cards on `/stories` are dead. `web/components/ui/ModelCard.tsx`'s "See Stories for the full comparison →" link points at `/stories` generally rather than the specific article, so it also reads as dead.

1. Add `slug: string` to `StoryInfo` in `web/lib/stories-data.ts` and give all five existing entries slugs.
2. Wrap `StoryCard`'s root in `<Link href={`/stories/${story.slug}`}>`.
3. Create the `/stories/[slug]` route, which does not currently exist at all.
4. Write the first article in full: **"CxG v3 — an honest comparison against StatsBomb xG"**, slug `cxg-v3-honest-comparison`. Content comes from `docs/dashboard_design_spec_v2.md` §11 — all six metrics for both tracks, the feature-pool asymmetry caveat (8 vs 24 features), and the `zone_displacement` bimodality open question. State plainly that CxG trails the baseline on every metric and why that's published rather than buried. Do not invent numbers — every figure must come from §11.
5. Point `ModelCard`'s link at `/stories/cxg-v3-honest-comparison`.
6. For the other four stories, render the detail route with a short honest "writeup in progress" state rather than a fake article — but make the route work.

Choose the authoring format yourself (MDX with `@next/mdx`, or a `body` field in the data file). Note the tradeoff in your summary — MDX is better long-term but adds a dependency.

---

## Part 3 — About page

`web/app/about/page.tsx` currently renders `<PlaceholderPanel title="About" variant="text" />`, which displays "Not yet built" **live in production**. Any visitor who clicks About concludes the project is abandoned.

Write it per `docs/dashboard_content_spec_v3.md` §6.3 — five sections: what this is, data source, glossary, how it's built, honest limitations.

The glossary must define every metric that appears anywhere in the app: xG, CxG, CxG+, xG/shot, G−xG, and what "opponent-adjusted" actually means.

The architecture section should describe the real stack you can verify from the repo: StatsBomb Open Data → BigQuery medallion (`oam_core` / `oam_analysis` / `oam_ml` / `oam_features`) → FastAPI on Cloud Run → Next.js on Firebase Hosting.

The limitations section is the most important one and should not be hedged: CxG is evaluated on a fixed test split and not scored live, CxG trails the StatsBomb baseline, CxA and CxT are not built, `oam_serving` is empty so there is no serving layer, and coverage is one season per competition.

---

## Part 4 — Explore tables: headers, sorting, pagination, derived columns

**Confirmed defect:** neither `/players` nor `/teams` has table headers. Numeric columns render bare (e.g. Harry Kane `220 / 42 / 35.97`) with no indication of what they mean.

Build a shared `DataTable` component in `web/components/ui/` rather than patching three divergent hand-rolled tables. Props roughly: columns (with key, label, alignment, optional numeric formatter, sortable flag), rows, optional `rowHref`, `loading`, `error`, `onRetry`, `emptyMessage`, `pageSize`. Sorting and pagination client-side.

Then apply it:

- **`/players`** — columns `# · Player · Team · Shots · Goals · xG · xG/shot · G−xG`. `xG/shot` = `total_xg / shots`; `G−xG` = `goals - total_xg`, coloured `--green` when positive and `--red` when negative, always signed. Add a minimum-shots filter defaulting to 10, surfaced in the `PageHead` crumb — without it the G−xG ranking is topped by one-shot players and reads as statistically naive.
- **`/teams`** — same treatment. Also fix the inconsistency that team rows use a plain `next/link` instead of `ClickableRow`.
- **`/matches`** — headers `Date · Home · Score · Away · Stage · Venue`. `competition_stage` and `stadium` are already on `MatchResponse` and currently unused. Render the winning side's name at `--text` and the losing side at `--text2`.

Pagination at 50 rows on all three — there are 610 matches and 1,457 players, currently all rendered in one DOM list.

All numeric values in `font-data`, per v3 §8.2: xG 2dp, ratios 2dp, deltas always signed.

---

## Part 5 — Teams detail: shot map and CxG

`web/app/teams/[teamId]/page.tsx` already calls `getTeamShots()` and uses the result only for tile arithmetic via `summarizeShots`. It renders no `PitchMap` at all, despite `dashboard_design_spec_v2.md` §4a naming Teams as one of three Explore surfaces where CxG appears.

Mirror what `web/app/matches/[matchId]/page.tsx` already does:
- Render a `PitchMap` from the shots already in memory
- Call `getCxgCoverage(eventIds, "cxg_event")` and `getCxgCoverage(eventIds, "cxg_plus")`, each degrading independently to `{}` on failure
- Add the coverage caption via `describeCxgCoverage`
- Add two derived tiles: `xG/shot` and `G−xG`

Also add a "Top scorers" card — group this team's shots by `player_id`, rank by goals then xG, each name a `PlayerLink`. Pure client-side grouping of data already fetched.

---

## Part 6 — Sidebar: wire the two dead controls

`web/components/shell/Sidebar.tsx` has two non-functional controls that look interactive:

1. **Team dropdown** — hardcoded to a single `<option>All teams</option>`. Populate it from `getTeams({competition_id, season_id})` and wire selection to a `teamId` value in `MatchFilterProvider`, which `/matches` should then pass to `getMatches`. The `team_id` query parameter already exists on the endpoint.

2. **Metric pills** — four static `<div>`s. Make `xG` and `CxG` genuine toggles; leave `CxA`/`CxT` disabled with their "soon" labels. Per v3 §2.2, the toggle is a **display mode, not a refetch** — coverage is already loaded. When `CxG` is selected, `PitchMap` sizes dots by CxG where coverage exists and renders uncovered shots at reduced opacity with a dashed stroke, excluded from the CxG total. Never substitute xG and label it CxG.

This needs `PitchMap` to accept a `sizeBy: "xg" | "cxg"` prop.

---

## Part 7 — CxG disclosure and PitchMap legend

Per `dashboard_design_spec_v2.md` §4a and v3 §8.1, the "Experimental" disclosure must be a **visible badge, not a hover `<title>`**. Hover-only disclosure fails entirely on touch devices.

- Add an `experimental` variant to `web/components/ui/Badge.tsx` using `--amber`
- Render it next to every CxG value or coverage caption, on match, player, and team detail pages
- Keep the `<title>` tooltip as well — it's additive, not a replacement
- Add a legend to `PitchMap` (behind a `showLegend` prop): home/away colour, filled = goal, dot size = xG. Visitors currently have to guess.

Disclosure text stays exactly as specified: 8 features for `cxg_event`, 24 for `cxg_plus`, evaluated on a fixed test split, not scored live. Status language stays "Evaluated"/"Experimental" — never "Validated", "Production", or "Live".

---

## Part 8 — Update the hard-gate status table

`dashboard_design_spec_v2.md` §9's status lines are stale in the codebase's favour, which undersells completed work to exactly the technical reader the doc is aimed at.

Verify each against the actual code and update:
- **Gate 3** (signed URLs) — `src/opponent_adjusted/api/gcs_signing.py` exists and the IAM grant `roles/iam.serviceAccountTokenCreator` was applied to `oam-pipeline-sa` on 26 Aug 2026. Check whether Analysis-tab charts now render as iframes rather than raw `gs://` text before declaring it closed.
- **Gate 4** (TTL cache) — implemented in `bigquery_store.py` with tests. Close it.
- **Gate 5** (RoleProvider timeout) — implemented in `web/components/shell/RoleProvider.tsx` with two passing tests. Close it.
- **Gate 2** (`oam_serving`) — I confirmed live: still zero tables. Leave blocked.

---

## Constraints

- **Do not** change the `split = 'test'` filter in `cxg_coverage.py` (Part 0 explains why).
- **Do not** regress the `silver_schema_version = 'statsbomb_silver_v1_2'` filter in any `bigquery_store.py` query — it fixes a real 3× row-duplication bug and has a dedicated regression test.
- **Do not** add a second `bigquery.Client()` construction path; the lazy singleton is deliberate.
- **Do not** remove the `NODE_ENV !== "production"` guard on `RoleSwitch`.
- **Do not** install a charting library — no charts in this prompt.
- Use existing design tokens from `globals.css`. No hardcoded hex values.
- Every number in `font-data` (Inconsolata).
- Add tests for new logic: `DataTable` sorting/pagination, the derived `G−xG` and `xG/shot` computations, `StoryCard`'s link, and the metric-toggle display-mode switch. Follow the existing Vitest patterns in `web/tests/`.
- Run `npm test` in `web/` and `pytest` at the repo root before committing.

## Commits

Commit in logical chunks rather than one large commit — roughly: (0) spec corrections, (1) Overview, (2) Stories, (3) About, (4) DataTable + tables, (5) Teams detail, (6) Sidebar, (7) disclosure + legend, (8) gate status.

When done, report what you changed per part, anything you found that contradicts either spec, and any decision you made that I should review — particularly the story authoring format choice.
