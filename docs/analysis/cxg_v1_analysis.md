# CxG V1 Analysis

The CxG v1 analysis layer is the first DB-backed football analysis report. It reads the local SQLite database and model-output tables, then writes reproducible local analysis artifacts under:

```text
outputs/analysis/cxg/
```

Generated analysis outputs are ignored by Git.

## Football Purpose

The report answers practical CxG review questions:

- How many shots and goals are represented in the current model output?
- What is the goal rate and mean CxG?
- How is shot quality distributed?
- How large are opponent/context neutralization adjustments?
- Which players combine shot volume and shot quality?
- Which teams generate the most total CxG?
- Which optional tactical slices are available from the DB-backed feature layer?

## Inputs

Primary source:

```text
data/opponent_adjusted.db
```

The loader reads from SQLite tables where available:

- `model_registry`
- `shots`
- `shot_predictions`
- `shot_features`
- `events`
- `aggregates_player`
- `aggregates_team`
- `players`
- `teams`

The primary CxG shot table joins `shots`, `shot_predictions`, and `model_registry`. Optional context columns from `shot_features` and `events` are used when available.

## Run

```bash
make analysis-cxg
```

Direct command:

```bash
poetry run python scripts/run_cxg_analysis_v1.py
```

Optional paths:

```bash
poetry run python scripts/run_cxg_analysis_v1.py --db-path data/opponent_adjusted.db --output-dir outputs/analysis/cxg
```

The v1 analysis suite target currently runs the CxG report:

```bash
make analysis-v1
```

CxA, CxT, and cross-metric analysis reports are intentionally deferred to later PRs.

## Outputs

EDA tables:

```text
outputs/analysis/cxg/eda/tables/shot_population_summary.csv
outputs/analysis/cxg/eda/tables/shot_outcome_summary.csv
```

Distribution tables and plots:

```text
outputs/analysis/cxg/distributions/tables/cxg_distribution_summary.csv
outputs/analysis/cxg/distributions/tables/opponent_adjustment_summary.csv
outputs/analysis/cxg/distributions/plots/cxg_distribution.png
outputs/analysis/cxg/distributions/plots/opponent_adjustment_distribution.png
```

Optional slice tables:

```text
outputs/analysis/cxg/slices/tables/by_body_part.csv
outputs/analysis/cxg/slices/tables/by_pressure.csv
outputs/analysis/cxg/slices/tables/by_minute_bucket.csv
outputs/analysis/cxg/slices/tables/by_opponent.csv
```

Slice tables are written only when the required columns exist. Missing optional slices are recorded in `report.md`.

Player outputs:

```text
outputs/analysis/cxg/players/tables/top_players_by_cxg.csv
outputs/analysis/cxg/players/tables/shot_quality_vs_volume.csv
outputs/analysis/cxg/players/plots/player_shot_quality_vs_volume.png
```

Team outputs:

```text
outputs/analysis/cxg/teams/tables/top_teams_by_cxg.csv
outputs/analysis/cxg/teams/tables/team_quality_vs_volume.csv
outputs/analysis/cxg/teams/plots/team_shot_quality_vs_volume.png
```

Narrative report:

```text
outputs/analysis/cxg/report.md
```

## How To Read The Artifacts

- `shot_population_summary.csv`: dataset coverage, goals, goal rate, mean CxG, provider xG, and neutral CxG.
- `shot_outcome_summary.csv`: how CxG varies by shot outcome.
- `cxg_distribution.png`: whether most shots are low-probability and how many high-value chances exist.
- `opponent_adjustment_distribution.png`: whether observed-context CxG is usually close to or far from neutralized CxG.
- `by_body_part.csv`: whether headers, feet, or other body parts differ in average CxG.
- `by_pressure.csv`: whether shots under pressure differ in goal rate or mean CxG.
- `by_minute_bucket.csv`: whether chance quality changes by game phase.
- `by_opponent.csv`: which opponent IDs appear in high or low CxG shot populations.
- `shot_quality_vs_volume.csv`: whether high-volume players/teams also take high-quality shots.

## Limitations

- This report is CxG-only. CxA, CxT, and cross-metric reports are not included in this PR.
- It reads local generated SQLite state and is only as current as the last model run.
- Optional slice columns may be missing in small fixtures or partially generated DBs; those slices are skipped rather than failing the report.
- Opponent adjustment values are model outputs, not causal defensive claims.
- The report uses matplotlib only and keeps plots intentionally simple for reproducibility.
