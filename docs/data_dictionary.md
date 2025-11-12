# Data Dictionary

Comprehensive reference for all database tables and columns in the opponent-adjusted metrics system.

## Reference Tables

### `competitions`
Competition metadata from StatsBomb.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| statsbomb_competition_id | INTEGER | NO | StatsBomb competition identifier (unique) |
| name | VARCHAR(255) | NO | Competition name (e.g., "FIFA World Cup") |
| season | VARCHAR(50) | NO | Season identifier (e.g., "2018") |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `statsbomb_competition_id` (unique), `(name, season)`

---

### `teams`
Team reference data.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| statsbomb_team_id | INTEGER | NO | StatsBomb team identifier (unique) |
| name | VARCHAR(255) | NO | Team name |
| country | VARCHAR(100) | YES | Country/region |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `statsbomb_team_id` (unique)

---

### `players`
Player reference data.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| statsbomb_player_id | INTEGER | NO | StatsBomb player identifier (unique) |
| name | VARCHAR(255) | NO | Player name |
| position | VARCHAR(50) | YES | Primary position |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `statsbomb_player_id` (unique)

---

## Match Data

### `matches`
Match-level information.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| statsbomb_match_id | INTEGER | NO | StatsBomb match identifier (unique) |
| competition_id | INTEGER | NO | Foreign key to competitions |
| home_team_id | INTEGER | NO | Foreign key to teams (home) |
| away_team_id | INTEGER | NO | Foreign key to teams (away) |
| kickoff_time | TIMESTAMP | YES | Match kickoff time |
| match_date | DATE | YES | Match date |
| season | VARCHAR(50) | YES | Season identifier |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `statsbomb_match_id` (unique), `competition_id`

---

### `raw_events`
Raw event JSON from StatsBomb (for lineage and auditing).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| match_id | INTEGER | NO | Foreign key to matches |
| statsbomb_event_id | VARCHAR(100) | NO | StatsBomb event UUID |
| raw_json | JSON | NO | Complete event JSON |
| type | VARCHAR(50) | NO | Event type name |
| period | INTEGER | NO | Match period (1, 2, etc.) |
| minute | INTEGER | NO | Match minute |
| second | INTEGER | NO | Match second |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `(match_id, statsbomb_event_id)` (unique), `match_id`, `type`

---

### `possessions`
Possession sequences within matches.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| match_id | INTEGER | NO | Foreign key to matches |
| possession_number | INTEGER | NO | Possession sequence number in match |
| team_id | INTEGER | NO | Foreign key to teams (possessing team) |
| start_event_id | INTEGER | YES | Foreign key to events (first event) |
| end_event_id | INTEGER | YES | Foreign key to events (last event) |
| start_minute | INTEGER | YES | Start minute |
| end_minute | INTEGER | YES | End minute |
| duration_seconds | FLOAT | YES | Duration in seconds |
| event_count | INTEGER | NO | Number of events in possession |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `(match_id, possession_number)` (unique), `match_id`, `team_id`

---

### `events`
Normalized event data.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| raw_event_id | INTEGER | NO | Foreign key to raw_events |
| match_id | INTEGER | NO | Foreign key to matches |
| team_id | INTEGER | NO | Foreign key to teams |
| player_id | INTEGER | YES | Foreign key to players |
| type | VARCHAR(50) | NO | Event type name |
| period | INTEGER | NO | Match period |
| minute | INTEGER | NO | Match minute |
| second | INTEGER | NO | Match second |
| timestamp | VARCHAR(20) | YES | Event timestamp string |
| possession | INTEGER | YES | Possession sequence number |
| location_x | FLOAT | YES | X coordinate (0-120) |
| location_y | FLOAT | YES | Y coordinate (0-80) |
| under_pressure | BOOLEAN | NO | Under defensive pressure flag |
| outcome | VARCHAR(100) | YES | Event outcome |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `match_id`, `(match_id, possession)`, `team_id`, `type`

---

## Shot Data

### `shots`
Shot base information.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| event_id | INTEGER | NO | Foreign key to events (unique) |
| match_id | INTEGER | NO | Foreign key to matches |
| team_id | INTEGER | NO | Foreign key to teams (shooting team) |
| player_id | INTEGER | YES | Foreign key to players |
| opponent_team_id | INTEGER | NO | Foreign key to teams (defending team) |
| statsbomb_xg | FLOAT | YES | StatsBomb's xG value |
| body_part | VARCHAR(50) | YES | Body part used (e.g., "Right Foot") |
| technique | VARCHAR(50) | YES | Shot technique (e.g., "Normal", "Volley") |
| shot_type | VARCHAR(50) | YES | Shot type (e.g., "Open Play", "Penalty") |
| outcome | VARCHAR(50) | NO | Shot outcome (e.g., "Goal", "Saved") |
| first_time | BOOLEAN | NO | First-time shot flag |
| is_blocked | BOOLEAN | NO | Blocked shot flag |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `event_id` (unique), `match_id`, `team_id`, `opponent_team_id`

---

### `shot_features`
Engineered features for shots (versioned).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| shot_id | INTEGER | NO | Foreign key to shots (unique) |
| version_tag | VARCHAR(20) | NO | Feature version (e.g., "v1") |
| shot_distance | FLOAT | YES | Distance to goal center (meters) |
| shot_angle | FLOAT | YES | Angle between goalposts (radians) |
| centrality | FLOAT | YES | Distance from center line |
| distance_to_goal_line | FLOAT | YES | Distance to goal line |
| score_diff_at_shot | INTEGER | YES | Team goals - opponent goals |
| is_leading | BOOLEAN | NO | Team is leading flag |
| is_trailing | BOOLEAN | NO | Team is trailing flag |
| is_drawing | BOOLEAN | NO | Game is tied flag |
| minute_bucket | VARCHAR(20) | YES | Minute bucket (e.g., "46-60") |
| possession_sequence_length | INTEGER | YES | Events in possession before shot |
| possession_duration | FLOAT | YES | Possession duration (seconds) |
| previous_action_gap | FLOAT | YES | Time since last action (seconds) |
| recent_def_actions_count | INTEGER | NO | Recent defensive actions |
| pressure_proxy_score | FLOAT | YES | Composite pressure score |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `shot_id` (unique), `version_tag`

---

## Opponent Profiles

### `opponent_def_profile`
Opponent defensive ratings (versioned).

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| team_id | INTEGER | NO | Foreign key to teams |
| version_tag | VARCHAR(20) | NO | Profile version (e.g., "v1") |
| zone_id | VARCHAR(10) | YES | Zone identifier (NULL for global) |
| global_rating | FLOAT | YES | Global defensive rating |
| block_rate | FLOAT | YES | Smoothed block rate |
| zone_rating | FLOAT | YES | Zone-specific rating |
| shots_sample | INTEGER | NO | Number of shots in sample |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `(team_id, version_tag, zone_id)` (unique), `(team_id, version_tag)`

**Zone IDs**: A, B, C, D, E, F (NULL for global metrics)

---

## Models and Predictions

### `model_registry`
Model metadata and versioning.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| model_name | VARCHAR(100) | NO | Model name (e.g., "cxg") |
| version | VARCHAR(50) | NO | Model version (e.g., "cxg_v1") |
| algorithm | VARCHAR(100) | NO | Algorithm (e.g., "LightGBM") |
| hyperparams | JSON | YES | Hyperparameters JSON |
| trained_on_version_tag | VARCHAR(20) | NO | Feature version used |
| artifact_path | TEXT | NO | Path to model artifact file |
| calibration_metrics | JSON | YES | Calibration metrics JSON |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `(model_name, version)` (unique)

---

### `shot_predictions`
Model predictions for shots.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| shot_id | INTEGER | NO | Foreign key to shots |
| model_id | INTEGER | NO | Foreign key to model_registry |
| version_tag | VARCHAR(20) | NO | Feature version used |
| is_neutralized | BOOLEAN | NO | Neutralized prediction flag |
| raw_probability | FLOAT | NO | Raw CxG probability |
| neutral_probability | FLOAT | YES | Neutralized CxG probability |
| opponent_adjusted_diff | FLOAT | YES | Raw - neutral difference |
| opponent_adjusted_ratio | FLOAT | YES | Raw / neutral ratio |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `(shot_id, model_id, is_neutralized)` (unique), `model_id`, `shot_id`

---

## Aggregates

### `aggregates_player`
Player-level aggregate metrics.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| player_id | INTEGER | NO | Foreign key to players |
| model_id | INTEGER | NO | Foreign key to model_registry |
| version_tag | VARCHAR(20) | NO | Feature version |
| shots_count | INTEGER | NO | Number of shots |
| summed_cxg | FLOAT | NO | Sum of raw CxG |
| summed_neutral_cxg | FLOAT | NO | Sum of neutral CxG |
| summed_oppadj_diff | FLOAT | NO | Sum of opponent-adjusted difference |
| avg_oppadj_diff | FLOAT | YES | Average opponent-adjusted difference |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `(player_id, model_id, version_tag)` (unique), `model_id`

---

### `aggregates_team`
Team-level aggregate metrics (same structure as aggregates_player, with team_id instead).

---

## Evaluation

### `evaluation_metrics`
Model evaluation metrics.

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| id | INTEGER | NO | Primary key |
| model_id | INTEGER | NO | Foreign key to model_registry |
| metric_name | VARCHAR(100) | NO | Metric name (e.g., "brier_score") |
| metric_value | FLOAT | NO | Metric value |
| slice_name | VARCHAR(100) | YES | Slice name (NULL for overall) |
| slice_filter | TEXT | YES | SQL filter for slice |
| created_at | TIMESTAMP | NO | Record creation timestamp |
| updated_at | TIMESTAMP | NO | Record update timestamp |

**Indices**: `model_id`, `metric_name`

---

## Naming Conventions

- **Tables**: Lowercase with underscores (snake_case)
- **Primary keys**: `id` (integer, auto-increment)
- **Foreign keys**: `<referenced_table>_id`
- **Timestamps**: `created_at`, `updated_at` (automatic)
- **Booleans**: Prefix with `is_` or `has_`
- **Versioning**: `version_tag` for feature/profile versions

## Data Types

- **Coordinates**: FLOAT (StatsBomb pitch: 120 x 80)
- **Probabilities**: FLOAT (0.0 to 1.0)
- **Identifiers**: INTEGER or VARCHAR (depending on source)
- **JSON**: Complex nested structures from StatsBomb
- **Timestamps**: TIMESTAMP WITH TIME ZONE (recommended for production)
