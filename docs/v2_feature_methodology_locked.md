# v2 Feature Engineering Methodology — Locked 2026-08-22

**Execution split:** Phase A and Phase B are independent workstreams, run in separate Claude Code sessions in parallel (Phase A in the existing session, Phase B in a fresh session — Phase B's prompt is fully self-contained for that reason). After both complete, a follow-up task re-runs the full governed analysis chain (Summary Stats → EDA → Univariate → Correlation/Redundancy → PCA → Bivariate) on the enlarged CxG+ candidate pool before v2 is trained. Phase C is out of scope for v2 entirely — locked as v3 work.

## Cost discipline (standing rule, applies to every future Claude Code prompt, not just v2)

Traced a £2.50 BigQuery overage to a real inefficiency: a past backfill script looped per-match against `oam_core.events`/`three_sixty_frames` (634 + 292 separate calls, ~734 GB total) instead of batching all `match_id`s into one query — these tables aren't clustered by `match_id`, so each per-match call still scanned a large chunk of the table. Going forward: batch multi-row filters into single queries, never loop a query per match/player/row; check whether large tables are clustered/partitioned appropriately before writing per-entity join logic against them; avoid unnecessary full-table scans. Bake this into every future Claude Code prompt explicitly.

## Guiding constraints (non-negotiable, apply to every new feature below)

- No `player_id`/`team_id` exposed as an identity feature to any model. `team_id`/`player_id` may be used internally as join/filter keys during computation, but the feature the model sees must be role-level, archetype-level, or a continuous rate — never a lookup of who someone is.
- No `statsbomb_xg` as an input ingredient to any new feature (only used read-only for the existing benchmark/divergence analysis, per the v1 decision — not touched here).
- Every new feature goes through the same governed analysis chain already established (Summary Stats → EDA → Univariate → Correlation/Redundancy Screen → PCA → Bivariate) before being trusted in a model. Nothing gets added to v2 straight from feature engineering.

## Phase A — Geometric/categorical, ships first, no clustering validation needed

- `nearest_defender_role` — categorical, bucketed `position_name` (GK/CB/Fullback-WingBack/Midfield/Attack), extends existing `BACKLINE_POSITION_MARKERS` convention.
- `nearest_defender_zone_displacement` — continuous, distance between the nearest defender's actual location at the shot and their role's typical/expected zone centroid (computed dataset-wide per role from `oam_core.events.location_x/location_y`). Requires verifying pitch-coordinate attack-direction normalization before implementation.
- `second_nearest_defender_role` — same categorical construction as `nearest_defender_role`, applied to the second-nearest defender.
- `nearest_defender_gap` — continuous, distance between the first- and second-nearest defenders at the shot. Measures local defensive cover vs. isolation.

## Phase B — Defender-style archetype, own governed clustering phase (like Phase 2 defensive-profile clustering)

- K-means cluster over nearest defender's action-type mix: rates of `Interception`, `Duel`, `Clearance`, `Block`, `Foul Committed`, `Pressure` (normalized per-90 or per-total-actions), computed from ALL of that player's defensive events dataset-wide (median 186 events/player — confirmed sufficient sample, unlike the shot-facing-involvement data that killed ODI).
- Needs its own stability/interpretability validation report before being trusted as a v2 input, same rigor as Phase 2's defensive-profile clustering.
- Open implementation question: whether duel sub-type (aerial vs. ground, won vs. lost) is available in nested event attributes — needed for the "physical" vs. "ground-dueller" distinction specifically. Verify before committing to that level of granularity; interceptor/pressurer/general-duel split is already confirmed feasible.
- **Phase B follow-up (next Phase B prompt, after this one completes):** apply the same style-archetype clustering lookup to the *second*-nearest defender too — `second_nearest_defender_style_archetype`. Reuses the cluster assignment table from the first Phase B run; just needs the join applied to the second-nearest defender identified in Phase A's `second_nearest_defender_role`/`nearest_defender_gap` work. Not part of the first Phase B prompt — deliberately sequenced after, since it depends on the cluster definitions already existing and being validated first.

## Phase C — Rolling-window features, larger engineering lift, LOCKED as v3 scope (not v2)

- Defensive action frequency, rolling within-match windows (15/30/45/60 min) — extends the existing `_momentum()` rolling-window pattern in `event_context_extended.py` to defensive event types.
- Field tilt extension — extends the existing `territorial_dominance_last_5m` pattern to a 15-min window.
- Cross-match rolling (last 1-2 matches' defensive activity rate for the currently-defending team) — genuinely new: first feature in this project to look before the current match. `team_id` used only as a join/filter key to select the correct prior matches, never exposed as an identity feature. Will have a cold-start null case (a team's first 1-2 matches in the dataset) — same explicit-null-reason discipline as ODI's cold-start guard, not a silent zero/impute.

## Explicitly deferred, not dropped

- Individual defender/GK quality scoring (season/history-long, xG-adjusted) — deferred due to statsbomb_xg circularity risk and shot-facing-involvement sparsity (median 4 events/player). Documented alternative if revisited: hierarchical/empirical-Bayes shrinkage estimator instead of a hard eligibility cutoff.
- ODI trio (`nearest_defender_odi`, `mean_backline_odi`, `gk_odi`) — dropped from v2 for the reasons already logged (unstable univariate, null bivariate, near-zero PCA loadings, near-zero/non-significant v1 coefficients). Superseded by Phase A/B's role- and archetype-based features, which address the same underlying idea (opponent defensive quality) without ODI's sample-size and identity problems.
