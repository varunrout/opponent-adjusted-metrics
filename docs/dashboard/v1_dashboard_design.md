# V1 Dashboard Design

## Product Goal

The v1 dashboard should turn the modelling repository into a reviewable football analytics product. A reviewer should be able to understand what the project does, inspect player and team outputs, compare CxG, CxA, and baseline CxT, and trace example insights back to generated model reports.

The v1 dashboard shell lives at `app/streamlit_app.py` and can be run with `make dashboard`. This document defines the product shape and data contract for that first dashboard build.

The dashboard reads generated outputs when they exist and degrades gracefully when they are missing. Missing generated files should produce status messages and empty tables, not application crashes.

## Intended Users

- Football analysts evaluating player, team, and action value.
- Technical reviewers checking whether the modelling outputs are reproducible and defensible.
- Recruiters or portfolio reviewers who need a fast but credible read of the project.
- Developers extending the metrics or dashboard after v1.

## Dashboard Pages

### Project Overview

Purpose: explain the project in one screen and show the current modelling status.

Key questions:

- What problem does opponent-adjusted and contextual football modelling solve?
- Which metric families are implemented?
- What outputs are generated locally and ignored by Git?
- What is baseline versus future roadmap work?

Visual sections:

- Metric family cards for CxG, CxA, and CxT.
- Generated output status table.
- Reproducibility command panel.
- Example insight callouts.

Required inputs:

- `outputs/modeling/cxg/reports/metrics.json`
- `outputs/modeling/cxa/reports/metrics.json`
- `outputs/modeling/cxt/reports/metrics.json`
- CxT interpretation summary when available.

### Player Analysis

Purpose: rank and compare player contributions across shot quality, chance creation, and ball progression.

Key questions:

- Which players generate the most CxA?
- Which players add the most baseline CxT through progression?
- Which players combine chance creation and territorial threat?
- Which players produce high-value actions repeatedly rather than from one outlier?

Visual sections:

- Player leaderboard.
- Player profile panel.
- Metric comparison scatter.
- Action mix summary for passes, carries, final-third entries, and box entries.

Required inputs:

- `outputs/modeling/cxa/aggregates/player_cxa.parquet`
- `outputs/modeling/cxt/aggregates/player_cxt.parquet`
- `outputs/modeling/cxg/aggregates/player_cxg.parquet`

Filters:

- Player
- Team
- Match
- Metric family
- Minimum action count

### Team Analysis

Purpose: show how teams create shots, chances, and progression threat.

Key questions:

- Which teams create the most CxG?
- Which teams create the most CxA?
- Which teams add threat through CxT progression?
- Do teams rely more on passes, carries, final-third entries, or box entries?

Visual sections:

- Team leaderboard.
- Team profile panel.
- CxA versus CxT comparison.
- Team progression breakdown.

Required inputs:

- `outputs/modeling/cxg/aggregates/team_cxg.parquet`
- `outputs/modeling/cxa/aggregates/team_cxa.parquet`
- `outputs/modeling/cxt/aggregates/team_cxt.parquet`

Filters:

- Team
- Match
- Metric family
- Minimum action count

### CxG Analysis

Purpose: inspect shot quality and model validation.

Key questions:

- Which shots have the highest CxG?
- Which players and teams create high-quality shots?
- How does the baseline model perform?
- Where are calibration, slice, or baseline-comparison limitations visible?

Visual sections:

- Shot prediction table.
- Player/team CxG summaries.
- Metrics summary.
- Calibration and slice diagnostics.

Required inputs:

- `outputs/modeling/cxg/predictions/shot_predictions.parquet`
- `outputs/modeling/cxg/reports/metrics.json`
- `outputs/modeling/cxg/reports/validation_summary.json`
- `outputs/modeling/cxg/reports/calibration_table.csv`
- `outputs/modeling/cxg/reports/slice_metrics.csv`

### CxA Analysis

Purpose: inspect baseline chance-creation value and attribution.

Key questions:

- Which actions create or progress toward chances?
- Which players and teams accumulate CxA?
- How much value comes from high-value actions?
- What does the baseline attribution method assume?

Visual sections:

- Action prediction table.
- Player/team CxA leaderboards.
- Sequence aggregate view.
- Attribution summary panel.

Required inputs:

- `outputs/modeling/cxa/predictions/action_predictions.parquet`
- `outputs/modeling/cxa/aggregates/player_cxa.parquet`
- `outputs/modeling/cxa/aggregates/team_cxa.parquet`
- `outputs/modeling/cxa/aggregates/sequence_cxa.parquet`
- `outputs/modeling/cxa/reports/attribution_summary.json`
- `outputs/modeling/cxa/reports/metrics.json`

### CxT Analysis

Purpose: inspect baseline territorial and threat progression.

Key questions:

- Which players add the most threat by moving the ball?
- Which teams add threat through progression?
- Which zone transitions create the most threat?
- Which actions are the highest positive and most negative threat movements?
- Which possessions accumulate threat?

Visual sections:

- Player/team CxT leaderboards.
- Sequence CxT table.
- Zone transition heatmap or table.
- Top positive and negative action table.
- Interpretation summary for final-third entries, box entries, pass threat, carry threat, and progressive threat.

Required inputs:

- `outputs/modeling/cxt/predictions/action_threat.parquet`
- `outputs/modeling/cxt/aggregates/player_cxt.parquet`
- `outputs/modeling/cxt/aggregates/team_cxt.parquet`
- `outputs/modeling/cxt/aggregates/sequence_cxt.parquet`
- `outputs/modeling/cxt/reports/zone_transition_summary.csv`
- `outputs/modeling/cxt/reports/top_actions.csv`
- `outputs/modeling/cxt/reports/interpretation_summary.json`
- `outputs/modeling/cxt/reports/metrics.json`

### Action-Level Explorer

Purpose: allow a reviewer to inspect individual actions across CxA and CxT.

Key questions:

- What happened in this action?
- Which player and team were involved?
- How much CxA or CxT did the action add?
- Was this action a pass, carry, final-third entry, box entry, or high-value movement?

Visual sections:

- Searchable action table.
- Metric detail side panel.
- Filters for player, team, match, action type, zone, and value range.

Required inputs:

- CxA action predictions.
- CxT action threat predictions.
- Optional CxG shot predictions for shot-linked context.

### Model and Report Diagnostics

Purpose: make the modelling layer inspectable without requiring a reader to inspect raw files manually.

Key questions:

- Which generated outputs exist locally?
- What model versions and metadata are available?
- Which validation checks passed?
- Which comparisons or diagnostics were skipped because inputs were unavailable?

Visual sections:

- Output inventory.
- Metrics cards.
- Validation summary table.
- Model metadata viewer.

Required inputs:

- CxG/CxA/CxT metrics JSON files.
- CxG validation outputs.
- CxA attribution summary.
- CxT interpretation summary.
- Model metadata JSON files where available.

## Metric Explanations

- CxG: contextual expected goal value for shots. It answers "how good was this shot?"
- CxA: contextual expected assist or chance-creation value for eligible attacking actions. It answers "which actions created or moved toward chances?"
- Baseline CxT: expected threat gained by moving from one pitch zone to another. It answers "how much territorial threat did this movement add?"

The dashboard should label CxG, CxA, and baseline CxT as baseline modelling outputs unless a future PR adds stronger model cards and validation for production-style claims.

## Storytelling Flow

1. Start with the project overview: why contextual and opponent-adjusted metrics matter.
2. Show the three metric families and what each one answers.
3. Move from teams to players to individual actions.
4. Use CxG to explain shot quality.
5. Use CxA to explain chance creation.
6. Use baseline CxT to explain ball progression and territorial threat.
7. End with diagnostics and limitations so the reviewer can trust what is implemented and what is still roadmap work.

## Example Insights

- A player may not shoot often but can rank highly in CxA and CxT by creating chances and progressing play.
- A team may generate moderate CxG but strong baseline CxT, suggesting territorial dominance without enough final shot quality.
- A zone-transition report can reveal whether threat comes from central progression, wide entries, or box entries.
- Top-action reports can surface individual movements that explain aggregate leaderboard positions.
