"""Leakage-safe baseline CxT feature and aggregate helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from opponent_adjusted.db.model_output_persistence import persist_cxt_outputs_to_database
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
HIGH_VALUE_CXT_THRESHOLD = 0.01


@dataclass(frozen=True)
class CxTBaselineOutputs:
    feature_path: Path
    threat_grid_path: Path
    predictions_path: Path
    player_aggregates_path: Path
    team_aggregates_path: Path
    sequence_aggregates_path: Path
    metrics_path: Path
    zone_transition_summary_path: Path
    top_actions_path: Path
    interpretation_summary_path: Path


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


def _in_penalty_box(x: pd.Series, y: pd.Series) -> pd.Series:
    return (x >= 102.0) & y.between(18.0, 62.0, inclusive="both")


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
                "entered_final_third",
                "entered_box",
                "progressive_action",
                "action_type_group",
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
    df["entered_final_third"] = (df["start_x"] < 80.0) & (df["end_x"] >= 80.0)
    df["entered_box"] = ~_in_penalty_box(df["start_x"], df["start_y"]) & _in_penalty_box(
        df["end_x"], df["end_y"]
    )
    df["progressive_action"] = (df["end_x"] - df["start_x"] >= 10.0) | df["action_type"].isin(
        {"progressive_pass", "progressive_carry"}
    )
    df["action_type_group"] = "other"
    df.loc[df["action_type"].isin({"pass", "cross", "progressive_pass"}), "action_type_group"] = (
        "pass"
    )
    df.loc[df["action_type"].isin({"carry", "progressive_carry"}), "action_type_group"] = "carry"
    df["model_version"] = MODEL_VERSION

    output_columns = [
        *IDENTITY_COLUMNS,
        "action_type",
        *LOCATION_COLUMNS,
        "successful_action",
        *BASELINE_VALUE_COLUMNS,
        "entered_final_third",
        "entered_box",
        "progressive_action",
        "action_type_group",
        "model_version",
    ]
    return df[output_columns].reset_index(drop=True)


def _conditional_sum(
    features: pd.DataFrame, group_columns: list[str], mask: pd.Series
) -> pd.Series:
    return features.loc[mask].groupby(group_columns, dropna=False)["cxt_value"].sum()


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
                "pass_cxt",
                "carry_cxt",
                "final_third_entry_cxt",
                "box_entry_cxt",
                "progressive_cxt",
                "high_value_actions",
                "cxt_per_action",
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
    summary["pass_cxt"] = _conditional_sum(
        features, group_columns, features["action_type_group"] == "pass"
    )
    summary["carry_cxt"] = _conditional_sum(
        features, group_columns, features["action_type_group"] == "carry"
    )
    summary["final_third_entry_cxt"] = _conditional_sum(
        features, group_columns, features["entered_final_third"]
    )
    summary["box_entry_cxt"] = _conditional_sum(features, group_columns, features["entered_box"])
    summary["progressive_cxt"] = _conditional_sum(
        features, group_columns, features["progressive_action"]
    )
    summary["high_value_actions"] = grouped["cxt_value"].apply(
        lambda values: int((values >= HIGH_VALUE_CXT_THRESHOLD).sum())
    )
    summary["cxt_per_action"] = summary["total_cxt"] / summary["actions"]
    summary = summary.fillna(
        {
            "pass_cxt": 0.0,
            "carry_cxt": 0.0,
            "final_third_entry_cxt": 0.0,
            "box_entry_cxt": 0.0,
            "progressive_cxt": 0.0,
        }
    )

    ordered_columns = [
        "actions",
        "total_cxt",
        "mean_cxt",
        "positive_cxt_actions",
        "negative_cxt_actions",
        "max_cxt",
        "min_cxt",
        "pass_cxt",
        "carry_cxt",
        "final_third_entry_cxt",
        "box_entry_cxt",
        "progressive_cxt",
        "high_value_actions",
        "cxt_per_action",
    ]
    return summary.reset_index()[group_columns + ordered_columns]


def aggregate_players(features: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(features, ["player_id", "player_name", "team_id", "team_name"])


def aggregate_teams(features: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(features, ["team_id", "team_name"])


def _threat_direction(total_cxt: float) -> str:
    if total_cxt > 0:
        return "positive"
    if total_cxt < 0:
        return "negative"
    return "neutral"


def aggregate_sequences(features: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["match_id", "possession_id", "team_id", "team_name"]
    if features.empty:
        return pd.DataFrame(
            columns=[
                *group_columns,
                "action_count",
                "total_cxt",
                "mean_cxt",
                "max_cxt",
                "min_cxt",
                "positive_cxt_actions",
                "negative_cxt_actions",
                "start_zone",
                "end_zone",
                "sequence_threat_direction",
                "dominant_transition",
            ]
        )

    sequence_features = features.assign(
        transition=features["start_zone"].astype(str) + "->" + features["end_zone"].astype(str)
    )
    grouped = sequence_features.groupby(group_columns, dropna=False, sort=False)
    summary = grouped["cxt_value"].agg(
        action_count="size",
        total_cxt="sum",
        mean_cxt="mean",
        max_cxt="max",
        min_cxt="min",
    )
    summary["positive_cxt_actions"] = grouped["cxt_value"].apply(
        lambda values: int((values > 0).sum())
    )
    summary["negative_cxt_actions"] = grouped["cxt_value"].apply(
        lambda values: int((values < 0).sum())
    )
    summary["start_zone"] = grouped["start_zone"].first()
    summary["end_zone"] = grouped["end_zone"].last()
    summary["sequence_threat_direction"] = summary["total_cxt"].map(_threat_direction)
    summary["dominant_transition"] = grouped["transition"].agg(
        lambda values: values.mode(dropna=False).iloc[0]
    )
    return summary.reset_index()


def build_zone_transition_summary(features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "start_zone",
        "end_zone",
        "actions",
        "total_cxt",
        "mean_cxt",
        "max_cxt",
        "positive_cxt_actions",
        "negative_cxt_actions",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)

    grouped = features.groupby(["start_zone", "end_zone"], dropna=False)
    summary = grouped["cxt_value"].agg(
        actions="size",
        total_cxt="sum",
        mean_cxt="mean",
        max_cxt="max",
    )
    summary["positive_cxt_actions"] = grouped["cxt_value"].apply(
        lambda values: int((values > 0).sum())
    )
    summary["negative_cxt_actions"] = grouped["cxt_value"].apply(
        lambda values: int((values < 0).sum())
    )
    return summary.reset_index()[columns]


def build_top_actions(features: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    columns = [
        "rank",
        "direction",
        "match_id",
        "team_name",
        "player_name",
        "action_type",
        "start_zone",
        "end_zone",
        "start_threat",
        "end_threat",
        "cxt_value",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)

    report_parts = []
    for direction, ascending in (("top_positive", False), ("top_negative", True)):
        ranked = features.sort_values(
            ["cxt_value", "action_id"], ascending=[ascending, True], kind="mergesort"
        ).head(top_n)
        ranked = ranked.copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["direction"] = direction
        report_parts.append(ranked)

    report = pd.concat(report_parts, ignore_index=True)
    return report[columns]


def _sum_where(features: pd.DataFrame, mask: pd.Series) -> float:
    if features.empty:
        return 0.0
    return float(features.loc[mask, "cxt_value"].sum())


def build_interpretation_summary(
    features: pd.DataFrame,
    *,
    zone_transition_summary_path: Path,
    top_actions_path: Path,
    sequence_aggregates_path: Path,
) -> dict[str, Any]:
    cxt = features["cxt_value"] if "cxt_value" in features else pd.Series(dtype=float)
    return {
        "model_version": MODEL_VERSION,
        "baseline_formula": BASELINE_FORMULA,
        "summary": (
            "Baseline CxT interpretation for deterministic zone/grid threat values. "
            "CxT+ and opponent-adjusted variants are not included."
        ),
        "total_cxt": float(cxt.sum()) if not cxt.empty else 0.0,
        "pass_cxt": _sum_where(features, features["action_type_group"] == "pass"),
        "carry_cxt": _sum_where(features, features["action_type_group"] == "carry"),
        "final_third_entry_cxt": _sum_where(features, features["entered_final_third"]),
        "box_entry_cxt": _sum_where(features, features["entered_box"]),
        "progressive_action_cxt": _sum_where(features, features["progressive_action"]),
        "top_positive_action_count": int((cxt > 0).sum()) if not cxt.empty else 0,
        "top_negative_action_count": int((cxt < 0).sum()) if not cxt.empty else 0,
        "zone_transition_report_path": str(zone_transition_summary_path),
        "top_actions_report_path": str(top_actions_path),
        "sequence_aggregate_path": str(sequence_aggregates_path),
    }


def build_metrics(
    features: pd.DataFrame,
    *,
    threat_grid_path: Path,
    predictions_path: Path,
    player_aggregates_path: Path,
    team_aggregates_path: Path,
    sequence_aggregates_path: Path,
    interpretation_summary_path: Path,
    zone_transition_summary_path: Path,
    top_actions_path: Path,
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
        "sequence_aggregates_path": str(sequence_aggregates_path),
        "interpretation_summary_path": str(interpretation_summary_path),
        "zone_transition_summary_path": str(zone_transition_summary_path),
        "top_actions_path": str(top_actions_path),
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
    persist_db: bool = False,
) -> CxTBaselineOutputs:
    raw_actions = _read_table(input_path) if input_path else synthetic_actions()
    threat_grid = build_threat_grid()
    features = build_action_features(raw_actions)
    player_aggregates = aggregate_players(features)
    team_aggregates = aggregate_teams(features)
    sequence_aggregates = aggregate_sequences(features)
    zone_transition_summary = build_zone_transition_summary(features)
    top_actions = build_top_actions(features)

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
    sequence_aggregates_path = aggregates_dir / "sequence_cxt.parquet"
    metrics_path = reports_dir / "metrics.json"
    zone_transition_summary_path = reports_dir / "zone_transition_summary.csv"
    zone_transition_summary_parquet_path = reports_dir / "zone_transition_summary.parquet"
    top_actions_path = reports_dir / "top_actions.csv"
    interpretation_summary_path = reports_dir / "interpretation_summary.json"

    features.to_parquet(feature_path, index=False)
    threat_grid.to_parquet(threat_grid_path, index=False)
    features.to_parquet(predictions_path, index=False)
    player_aggregates.to_parquet(player_aggregates_path, index=False)
    team_aggregates.to_parquet(team_aggregates_path, index=False)
    sequence_aggregates.to_parquet(sequence_aggregates_path, index=False)
    zone_transition_summary.to_csv(zone_transition_summary_path, index=False)
    zone_transition_summary.to_parquet(zone_transition_summary_parquet_path, index=False)
    top_actions.to_csv(top_actions_path, index=False)

    interpretation_summary = build_interpretation_summary(
        features,
        zone_transition_summary_path=zone_transition_summary_path,
        top_actions_path=top_actions_path,
        sequence_aggregates_path=sequence_aggregates_path,
    )
    interpretation_summary_path.write_text(
        json.dumps(interpretation_summary, indent=2), encoding="utf-8"
    )

    if write_csv:
        features.to_csv(predictions_dir / "action_threat.csv", index=False)
        player_aggregates.to_csv(aggregates_dir / "player_cxt.csv", index=False)
        team_aggregates.to_csv(aggregates_dir / "team_cxt.csv", index=False)
        sequence_aggregates.to_csv(aggregates_dir / "sequence_cxt.csv", index=False)

    metrics = build_metrics(
        features,
        threat_grid_path=threat_grid_path,
        predictions_path=predictions_path,
        player_aggregates_path=player_aggregates_path,
        team_aggregates_path=team_aggregates_path,
        sequence_aggregates_path=sequence_aggregates_path,
        interpretation_summary_path=interpretation_summary_path,
        zone_transition_summary_path=zone_transition_summary_path,
        top_actions_path=top_actions_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if persist_db:
        metadata = {
            "model_name": "cxt",
            "model_version": MODEL_VERSION,
            "model_type": "baseline_grid_threat",
            "artifact_path": str(threat_grid_path),
            "features": {
                "identity": IDENTITY_COLUMNS,
                "location": LOCATION_COLUMNS,
                "value": BASELINE_VALUE_COLUMNS,
            },
        }
        persist_cxt_outputs_to_database(
            metadata=metadata,
            metrics=metrics,
            predictions=features,
            player_aggregates=player_aggregates,
            team_aggregates=team_aggregates,
            sequence_aggregates=sequence_aggregates,
        )

    return CxTBaselineOutputs(
        feature_path=feature_path,
        threat_grid_path=threat_grid_path,
        predictions_path=predictions_path,
        player_aggregates_path=player_aggregates_path,
        team_aggregates_path=team_aggregates_path,
        sequence_aggregates_path=sequence_aggregates_path,
        metrics_path=metrics_path,
        zone_transition_summary_path=zone_transition_summary_path,
        top_actions_path=top_actions_path,
        interpretation_summary_path=interpretation_summary_path,
    )
