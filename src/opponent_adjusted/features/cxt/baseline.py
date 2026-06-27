"""Leakage-safe baseline CxT feature and aggregate helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from opponent_adjusted.features.cxt.xt_model import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    XT_GRID,
    get_zone,
)

MODEL_VERSION = "cxt-baseline-v1"
BASELINE_FORMULA = "cxt_value = end_threat - start_threat"
LEAKAGE_NOTE = (
    "Baseline CxT uses only action type, identifiers, and start/end locations. "
    "Future shot or goal outcomes are not used as action-level inputs."
)
ELIGIBLE_ACTION_TYPES = {
    "pass",
    "carry",
    "dribble",
    "cross",
    "progressive_pass",
    "progressive_carry",
}
PROHIBITED_LEAKAGE_COLUMNS = {
    "future_shot_xg",
    "future_shot_location",
    "future_goal",
    "future_shot_outcome",
    "next_action_is_shot",
    "actions_until_shot",
    "total_future_possession_length",
    "goal_outcome",
    "shot_outcome",
}
IDENTITY_COLUMNS = [
    "action_id",
    "event_id",
    "match_id",
    "possession_id",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
]
LOCATION_COLUMNS = ["start_x", "start_y", "end_x", "end_y"]
BASELINE_VALUE_COLUMNS = [
    "start_zone",
    "end_zone",
    "start_threat",
    "end_threat",
    "cxt_value",
]


@dataclass(frozen=True)
class CxTBaselineOutputs:
    feature_path: Path
    threat_grid_path: Path
    predictions_path: Path
    player_aggregates_path: Path
    team_aggregates_path: Path
    metrics_path: Path


def _zone_id(x: float, y: float) -> str:
    x_zone, y_zone = get_zone(x, y)
    return f"x{x_zone:02d}_y{y_zone:02d}"


def build_threat_grid() -> pd.DataFrame:
    """Build the deterministic 12x8 baseline threat grid."""

    rows = []
    x_step = PITCH_LENGTH / XT_GRID.shape[1]
    y_step = PITCH_WIDTH / XT_GRID.shape[0]
    for y_zone in range(XT_GRID.shape[0]):
        for x_zone in range(XT_GRID.shape[1]):
            rows.append(
                {
                    "zone_id": f"x{x_zone:02d}_y{y_zone:02d}",
                    "x_zone": x_zone,
                    "y_zone": y_zone,
                    "x_min": float(x_zone * x_step),
                    "x_max": float((x_zone + 1) * x_step),
                    "y_min": float(y_zone * y_step),
                    "y_max": float((y_zone + 1) * y_step),
                    "threat": float(XT_GRID[y_zone, x_zone]),
                    "model_version": MODEL_VERSION,
                }
            )
    return pd.DataFrame(rows)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported CxT input format: {path.suffix}")


def _normalise_action_type(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _successful_action(row: pd.Series) -> bool:
    for column in ("successful_action", "action_success", "is_complete", "completed"):
        if column in row and pd.notna(row[column]):
            return bool(row[column])
    if "action_outcome" in row and pd.notna(row["action_outcome"]):
        return str(row["action_outcome"]).lower() not in {
            "incomplete",
            "out",
            "pass_offside",
            "pass offside",
        }
    return True


def _ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in IDENTITY_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    if "action_type" not in df.columns:
        df["action_type"] = "pass"
    return df


def build_action_features(actions: pd.DataFrame) -> pd.DataFrame:
    """Build baseline action-level CxT features.

    The calculation intentionally ignores leakage-sensitive future outcome
    columns even when they are present in the source data.
    """

    df = _ensure_optional_columns(actions)
    df = df.drop(columns=sorted(PROHIBITED_LEAKAGE_COLUMNS & set(df.columns)))
    df["action_type"] = df["action_type"].map(_normalise_action_type)
    df = df[df["action_type"].isin(ELIGIBLE_ACTION_TYPES)].copy()

    for column in LOCATION_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")

    valid_locations = df[LOCATION_COLUMNS].notna().all(axis=1)
    df = df[valid_locations].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                *IDENTITY_COLUMNS,
                "action_type",
                *LOCATION_COLUMNS,
                "successful_action",
                *BASELINE_VALUE_COLUMNS,
                "model_version",
            ]
        )

    for column, upper in (
        ("start_x", PITCH_LENGTH),
        ("end_x", PITCH_LENGTH),
        ("start_y", PITCH_WIDTH),
        ("end_y", PITCH_WIDTH),
    ):
        df[column] = df[column].clip(lower=0.0, upper=upper)

    start_zones = [get_zone(x, y) for x, y in zip(df["start_x"], df["start_y"])]
    end_zones = [get_zone(x, y) for x, y in zip(df["end_x"], df["end_y"])]
    df["start_zone"] = [_zone_id(x, y) for x, y in zip(df["start_x"], df["start_y"])]
    df["end_zone"] = [_zone_id(x, y) for x, y in zip(df["end_x"], df["end_y"])]
    df["start_threat"] = [float(XT_GRID[y_zone, x_zone]) for x_zone, y_zone in start_zones]
    df["end_threat"] = [float(XT_GRID[y_zone, x_zone]) for x_zone, y_zone in end_zones]
    df["cxt_value"] = df["end_threat"] - df["start_threat"]
    df["successful_action"] = df.apply(_successful_action, axis=1)
    df["model_version"] = MODEL_VERSION

    output_columns = [
        *IDENTITY_COLUMNS,
        "action_type",
        *LOCATION_COLUMNS,
        "successful_action",
        *BASELINE_VALUE_COLUMNS,
        "model_version",
    ]
    return df[output_columns].reset_index(drop=True)


def _aggregate(features: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "actions",
                "total_cxt",
                "mean_cxt",
                "positive_cxt_actions",
                "negative_cxt_actions",
                "max_cxt",
                "min_cxt",
            ]
        )

    grouped = features.groupby(group_columns, dropna=False)
    summary = grouped["cxt_value"].agg(
        actions="size",
        total_cxt="sum",
        mean_cxt="mean",
        max_cxt="max",
        min_cxt="min",
    )
    positive = grouped["cxt_value"].apply(lambda values: int((values > 0).sum()))
    negative = grouped["cxt_value"].apply(lambda values: int((values < 0).sum()))
    summary["positive_cxt_actions"] = positive
    summary["negative_cxt_actions"] = negative

    ordered_columns = [
        "actions",
        "total_cxt",
        "mean_cxt",
        "positive_cxt_actions",
        "negative_cxt_actions",
        "max_cxt",
        "min_cxt",
    ]
    return summary.reset_index()[group_columns + ordered_columns]


def aggregate_players(features: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(features, ["player_id", "player_name", "team_id", "team_name"])


def aggregate_teams(features: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(features, ["team_id", "team_name"])


def build_metrics(
    features: pd.DataFrame,
    *,
    threat_grid_path: Path,
    predictions_path: Path,
    player_aggregates_path: Path,
    team_aggregates_path: Path,
) -> dict[str, Any]:
    cxt = features["cxt_value"] if "cxt_value" in features else pd.Series(dtype=float)
    number_of_actions = int(len(features))
    number_of_players = (
        int(features["player_id"].nunique(dropna=True)) if "player_id" in features else 0
    )
    number_of_teams = int(features["team_id"].nunique(dropna=True)) if "team_id" in features else 0
    return {
        "model_version": MODEL_VERSION,
        "baseline_formula": BASELINE_FORMULA,
        "leakage_note": LEAKAGE_NOTE,
        "number_of_actions": number_of_actions,
        "number_of_players": number_of_players,
        "number_of_teams": number_of_teams,
        "actions": number_of_actions,
        "players": number_of_players,
        "teams": number_of_teams,
        "total_cxt": float(cxt.sum()) if not cxt.empty else 0.0,
        "mean_cxt": float(cxt.mean()) if not cxt.empty else 0.0,
        "min_cxt": float(cxt.min()) if not cxt.empty else 0.0,
        "max_cxt": float(cxt.max()) if not cxt.empty else 0.0,
        "positive_action_count": int((cxt > 0).sum()) if not cxt.empty else 0,
        "negative_action_count": int((cxt < 0).sum()) if not cxt.empty else 0,
        "zero_action_count": int((cxt == 0).sum()) if not cxt.empty else 0,
        "threat_grid_path": str(threat_grid_path),
        "prediction_path": str(predictions_path),
        "player_aggregates_path": str(player_aggregates_path),
        "team_aggregates_path": str(team_aggregates_path),
    }


def synthetic_actions() -> pd.DataFrame:
    """Small deterministic fixture used when the runner has no input path."""

    return pd.DataFrame(
        [
            {
                "action_id": "a1",
                "event_id": "e1",
                "match_id": 1,
                "possession_id": 10,
                "team_id": 100,
                "team_name": "Home",
                "player_id": 1000,
                "player_name": "Player One",
                "action_type": "pass",
                "start_x": 35.0,
                "start_y": 40.0,
                "end_x": 85.0,
                "end_y": 40.0,
                "successful_action": True,
            },
            {
                "action_id": "a2",
                "event_id": "e2",
                "match_id": 1,
                "possession_id": 11,
                "team_id": 100,
                "team_name": "Home",
                "player_id": 1001,
                "player_name": "Player Two",
                "action_type": "carry",
                "start_x": 90.0,
                "start_y": 35.0,
                "end_x": 55.0,
                "end_y": 35.0,
                "successful_action": True,
            },
            {
                "action_id": "a3",
                "event_id": "e3",
                "match_id": 2,
                "possession_id": 20,
                "team_id": 200,
                "team_name": "Away",
                "player_id": 2000,
                "player_name": "Player Three",
                "action_type": "dribble",
                "start_x": 60.0,
                "start_y": 55.0,
                "end_x": 74.0,
                "end_y": 42.0,
                "successful_action": True,
            },
        ]
    )


def run_baseline(
    *,
    input_path: Path | None = None,
    feature_store_dir: Path = Path("feature_store/cxt"),
    output_dir: Path = Path("outputs/modeling/cxt"),
    write_csv: bool = False,
) -> CxTBaselineOutputs:
    raw_actions = _read_table(input_path) if input_path else synthetic_actions()
    threat_grid = build_threat_grid()
    features = build_action_features(raw_actions)
    player_aggregates = aggregate_players(features)
    team_aggregates = aggregate_teams(features)

    feature_store_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    aggregates_dir = output_dir / "aggregates"
    reports_dir = output_dir / "reports"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    aggregates_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    feature_path = feature_store_dir / "action_features.parquet"
    threat_grid_path = output_dir / "threat_grid.parquet"
    predictions_path = predictions_dir / "action_threat.parquet"
    player_aggregates_path = aggregates_dir / "player_cxt.parquet"
    team_aggregates_path = aggregates_dir / "team_cxt.parquet"
    metrics_path = reports_dir / "metrics.json"

    features.to_parquet(feature_path, index=False)
    threat_grid.to_parquet(threat_grid_path, index=False)
    features.to_parquet(predictions_path, index=False)
    player_aggregates.to_parquet(player_aggregates_path, index=False)
    team_aggregates.to_parquet(team_aggregates_path, index=False)

    if write_csv:
        features.to_csv(predictions_dir / "action_threat.csv", index=False)
        player_aggregates.to_csv(aggregates_dir / "player_cxt.csv", index=False)
        team_aggregates.to_csv(aggregates_dir / "team_cxt.csv", index=False)

    metrics = build_metrics(
        features,
        threat_grid_path=threat_grid_path,
        predictions_path=predictions_path,
        player_aggregates_path=player_aggregates_path,
        team_aggregates_path=team_aggregates_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return CxTBaselineOutputs(
        feature_path=feature_path,
        threat_grid_path=threat_grid_path,
        predictions_path=predictions_path,
        player_aggregates_path=player_aggregates_path,
        team_aggregates_path=team_aggregates_path,
        metrics_path=metrics_path,
    )
