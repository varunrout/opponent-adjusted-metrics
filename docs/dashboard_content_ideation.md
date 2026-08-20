# OAM Dashboard — Content Ideation

Status: draft for review. Phrasing directions and content ideas, not final copy.

## Two lenses, don't conflate them

**Role** (guest/viewer/admin) controls *access* — which tabs render at all, per the design spec's role table.

**Persona** controls *content* — what a tab should actually say to be worth someone's time, since two anonymous guests can want completely different things from the same page. Content ideation has to run on personas; role-gating stays exactly as already locked.

Across every shared tab, guest/viewer/admin see the *same underlying content*. The tiers add capability (save, compare, pin a historical model version), they don't fragment the information itself. That's worth stating plainly so nothing gets built three times for no reason.

## The four personas

1. **Recruiter / hiring manager** — semi-technical at best, 60-90 seconds of patience, evaluating "is this person credible," arrives from a CV or LinkedIn link, guest tier, one visit.
2. **Peer practitioner** — a working or aspiring data scientist / football analyst, technical, evaluating rigor and defensibility, might sign up (viewer) to save or compare things, arrives from a technical write-up or the GitHub repo itself.
3. **Football enthusiast** — knows the sport, not necessarily technical, wants interesting findings, browses casually, guest tier.
4. **Varun (admin)** — the operator. Not really a "content" audience, needs the tabs to serve a real working purpose, not a portfolio purpose.

---

## Overview

Already real: competition/season picker, recent results, headline tiles (currently mock CxG values, will be real xG once wired).

- Recruiter: needs the pitch without scrolling — one line framing the whole project ("opponent-adjusted football analytics, built end to end from raw event data to a production dashboard"), a couple of headline stats, one link straight to "how this was built."
- Peer: wants a fast route to the interesting technical surfaces, a link into Models/methodology, maybe a "what's new" feel if a Story or model checkpoint just landed.
- Enthusiast: wants "what's happening" — recent match results, one featured Story teaser, an easy way to start browsing.
- Content ideas: hero stat strip (matches covered, shots analyzed, competitions), a small grid of recent/featured match cards, one featured Story teaser, a single sentence with a link into About ("built on StatsBomb Open Data — see how").

## Matches

Already real: filterable list, match detail with shot map, xG tiles, lineups (Phase 2).

- Recruiter: not a primary destination for this persona, likely arrives here only by drilling in from Overview.
- Peer: uses this to sanity-check data quality and breadth — filters by competition, inspects a match's shot map and lineup as evidence the pipeline is actually correct, not just pretty.
- Enthusiast: primary destination — browsing matches, opening shot maps, this is the "fun" surface.
- Ideas beyond what's built: pass network (already in the component inventory, not yet wired), a one-line auto-generated summary per match ("home controlled xG until the 70th minute"), lineup names linking through to Player pages.
- Viewer adds: bookmark a match, side-by-side match comparison.

## Players

Already real: leaderboard (shots/goals/total xG), player detail with shot map (Phase 3).

- Recruiter: unlikely destination unless following a direct link.
- Peer: checks whether the per-player aggregation is sound, spot-checks a known player's numbers against intuition.
- Enthusiast: high-interest — "who's overperforming xG," search for a favourite player.
- Ideas: a short natural-language takeaway per player card ("scoring above expected shot quality this season"), per-90 figures once minutes-played tracking exists (currently blocked, noted in the phase 3 scope), radar once CxG/CxA exist.
- Viewer adds: watchlist, player comparison.

## Teams

Already real: leaderboard, team detail with recent matches (Phase 4).

- Same shape as Players. Peer checks the `team_id` home/away OR-filter is behaving. Enthusiast wants season summary and recent form. Longer-term: defensive block-shape trends (the F2 feature family) once that's presentable, not before.
- Viewer adds: watchlist, team comparison.

## Analysis — admin only, single persona

This tab has one real audience: Varun-as-analyst. Don't try to ideate a guest or peer version of it, the role table gates it to admin on purpose, it's a working tool, not a showcase. Content: the free-form quadrant-scatter/custom-leaderboard workspace, plus the ability to pin and compare specific historical `model_version`s against each other, exactly as scoped in the design doc's role table.

## Models — flagging a real decision, not just ideating

Current build has this fully admin-gated (`nav-config.ts`: `roles: ["admin"]`). Worth reconsidering: the design doc's own stated reason for having a Models tab at all is "shows engineering rigor, not just output" and is explicitly called out as worth having "even when only CxG is live" — that's a portfolio argument, and portfolio arguments don't work if only Varun can see the tab. This is arguably the single strongest credibility surface in the whole product for both the recruiter and peer personas, hiding it undercuts its own purpose.

Recommendation: split it. A public layer, guest-visible, read-only — model cards with status badges (Promoted/In training/Planned), tier chips (Core/Spatial/Advanced), validation metrics, feature-family counts (e.g. "69/75 event-context candidates frozen"), links to methodology Stories, and eventually links into the real rendered analysis charts (`oam_analysis.cxg_rendered_chart_registry_v1`, already real in BigQuery). Keep genuine internals, admin-only, admin-only: raw validation logs, the ability to pin/inspect arbitrary superseded versions, promote/retire controls.

Per persona once split:
- Recruiter: quick scan, "CxG: Promoted, validated against StatsBomb xG as baseline." Doesn't need to understand the maths, needs to trust it exists.
- Peer: the substance — feature family breakdown, validation methodology, calibration metrics, links to the actual analysis charts. This is where they judge rigor.
- Enthusiast: mostly passes through, maybe curious enough to follow a "what does CxG mean" link into the glossary.
- Admin: everything above, plus the internals described above.

This is a call for you to make, not something to build on my say-so, flagging it because it's the kind of thing better caught during ideation than after Claude Code has already built three more phases on top of the current gating.

## Stories

Content types already scoped: methodology explainers, case studies, release notes per model version.

- Recruiter: arguably the second most important tab after Overview/Models for this persona if they have a spare minute, a well-written methodology piece reads as a writing sample, not just a code sample.
- Peer: the most-read tab for this persona, judges rigor through prose and reasoning, not just numbers, case studies and release notes are the currency here.
- Enthusiast: player/match case studies are the most naturally readable and shareable content type on the whole site ("Player X is beating xG by 0.3 per 90 — variance, or is CxG explaining it?").
- One addition worth ideating in: a behind-the-scenes dev-log style post. The actual engineering process already has a genuinely good story in it, the Silver `_SUCCESS` ordering defect, catching it, and the controlled repair, that's a real "here's how I handle production data issues" narrative, more credible than a polished case study because it's about a mistake and the fix, not a clean result.
- Viewer adds: bookmark / notify on new stories, minor, not essential.

## About

- Recruiter: quick credibility and context — what StatsBomb Open Data is, why this project exists, and it should link back out to Varun (CV/LinkedIn/GitHub), a portfolio piece that doesn't link to the person behind it is a missed step.
- Peer: full metric glossary, versioning philosophy, a direct link to the GitHub repo, real commits and tests are a bigger credibility signal here than anything the dashboard itself can show.
- Enthusiast: plain-language "what is xG / CxG" explainer.
- No real persona differentiation needed here beyond depth-on-demand, glossary entries can expand rather than forcing a wall of text on everyone.

---

## Summary of what's genuinely open

1. Models tab visibility, admin-only vs public-with-gated-internals, the one real decision above.
2. Whether Stories gets a dev-log/behind-the-scenes category alongside methodology and case studies.
3. Cross-linking as content: lineup names → Player pages, team names → Team pages, these aren't new tabs, just make the existing ones feel connected rather than isolated.
