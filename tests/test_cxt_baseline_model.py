import json
from pathlib import Path

import pandas as pd

from opponent_adjusted.features.cxt.baseline import (
    PROHIBITED_LEAKAGE_COLUMNS,
    build_action_features,
    run_baseline,
)
from scripts.check_cxg_outputs import assert_git_ignored


def _synthetic_cxt_actions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_id": "forward-pass",
                "event_id": "event-1",
                "match_id": 1,
                "possession_id": 10,
                "team_id": 100,
                "team_name": "Home",
                "player_id": 1000,
                "player_name": "Player One",
                "action_type": "pass",
                "start_x": 35.0,
                "start_y": 40.0,
                "end_x": 100.0,
                "end_y": 40.0,
                "successful_action": True,
            },
            {
                "action_id": "backward-carry",
                "event_id": "event-2",
                "match_id": 1,
                "possession_id": 11,
                "team_id": 100,
                "team_name": "Home",
                "player_id": 1001,
                "player_name": "Player Two",
                "action_type": "carry",
                "start_x": 100.0,
                "start_y": 40.0,
                "end_x": 45.0,
                "end_y": 40.0,
                "successful_action": True,
            },
            {
                "action_id": "wide-dribble",
                "event_id": "event-3",
                "match_id": 2,
                "possession_id": 20,
                "team_id": 200,
                "team_name": "Away",
                "player_id": 2000,
                "player_name": "Player Three",
                "action_type": "dribble",
                "start_x": 65.0,
                "start_y": 65.0,
                "end_x": 82.0,
                "end_y": 55.0,
                "successful_action": True,
            },
            {
                "action_id": "ignored-shot",
                "event_id": "event-4",
                "match_id": 2,
                "possession_id": 21,
                "team_id": 200,
                "team_name": "Away",
                "player_id": 2001,
                "player_name": "Player Four",
                "action_type": "shot",
                "start_x": 110.0,
                "start_y": 40.0,
                "end_x": 120.0,
                "end_y": 40.0,
            },
            {
                "action_id": "invalid-location",
                "event_id": "event-5",
                "match_id": 3,
                "team_id": 300,
                "player_id": 3000,
                "action_type": "pass",
                "start_x": 50.0,
                "start_y": 40.0,
                "end_x": None,
                "end_y": 40.0,
            },
        ]
    )


def test_cxt_baseline_emits_grid_predictions_aggregates_and_metrics(tmp_path: Path):
    input_path = tmp_path / "actions.parquet"
    _synthetic_cxt_actions().to_parquet(input_path, index=False)

    outputs = run_baseline(
        input_path=input_path,
        feature_store_dir=tmp_path / "feature_store" / "cxt",
        output_dir=tmp_path / "outputs" / "modeling" / "cxt",
    )

    assert outputs.feature_path.exists()
    assert outputs.threat_grid_path.exists()
    assert outputs.predictions_path.exists()
    assert outputs.player_aggregates_path.exists()
    assert outputs.team_aggregates_path.exists()
    assert outputs.sequence_aggregates_path.exists()
    assert outputs.metrics_path.exists()
    assert outputs.zone_transition_summary_path.exists()
    assert outputs.top_actions_path.exists()
    assert outputs.interpretation_summary_path.exists()

    threat_grid = pd.read_parquet(outputs.threat_grid_path)
    predictions = pd.read_parquet(outputs.predictions_path)
    player_aggregates = pd.read_parquet(outputs.player_aggregates_path)
    team_aggregates = pd.read_parquet(outputs.team_aggregates_path)
    sequence_aggregates = pd.read_parquet(outputs.sequence_aggregates_path)
    zone_transitions = pd.read_csv(outputs.zone_transition_summary_path)
    top_actions = pd.read_csv(outputs.top_actions_path)
    metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))
    interpretation = json.loads(outputs.interpretation_summary_path.read_text(encoding="utf-8"))

    assert len(threat_grid) == 96
    assert {"zone_id", "x_zone", "y_zone", "threat"}.issubset(threat_grid.columns)
    assert len(predictions) == 3
    assert {"start_zone", "end_zone", "start_threat", "end_threat", "cxt_value"}.issubset(
        predictions.columns
    )
    assert {
        "entered_final_third",
        "entered_box",
        "progressive_action",
        "action_type_group",
    }.issubset(predictions.columns)
    assert (
        predictions["cxt_value"] == predictions["end_threat"] - predictions["start_threat"]
    ).all()

    forward = predictions.loc[predictions["action_id"] == "forward-pass"].iloc[0]
    backward = predictions.loc[predictions["action_id"] == "backward-carry"].iloc[0]
    assert forward["cxt_value"] > 0
    assert backward["cxt_value"] < 0

    total_cxt = float(predictions["cxt_value"].sum())
    assert abs(float(player_aggregates["total_cxt"].sum()) - total_cxt) < 1e-12
    assert abs(float(team_aggregates["total_cxt"].sum()) - total_cxt) < 1e-12
    assert abs(float(sequence_aggregates["total_cxt"].sum()) - total_cxt) < 1e-12
    assert abs(float(zone_transitions["total_cxt"].sum()) - total_cxt) < 1e-12

    interpretation_columns = {
        "pass_cxt",
        "carry_cxt",
        "final_third_entry_cxt",
        "box_entry_cxt",
        "progressive_cxt",
        "cxt_per_action",
    }
    assert interpretation_columns | {"high_value_actions"} <= set(player_aggregates.columns)
    assert interpretation_columns <= set(team_aggregates.columns)
    assert {
        "match_id",
        "possession_id",
        "team_id",
        "team_name",
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
    } <= set(sequence_aggregates.columns)
    assert {
        "start_zone",
        "end_zone",
        "actions",
        "total_cxt",
        "mean_cxt",
        "max_cxt",
        "positive_cxt_actions",
        "negative_cxt_actions",
    } <= set(zone_transitions.columns)
    assert {
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
    } <= set(top_actions.columns)
    top_positive = top_actions[top_actions["direction"] == "top_positive"]
    top_negative = top_actions[top_actions["direction"] == "top_negative"]
    assert top_positive["cxt_value"].is_monotonic_decreasing
    assert top_negative["cxt_value"].is_monotonic_increasing

    assert metrics["actions"] == 3
    assert metrics["players"] == 3
    assert metrics["teams"] == 2
    assert metrics["number_of_actions"] == 3
    assert metrics["number_of_players"] == 3
    assert metrics["number_of_teams"] == 2
    assert abs(metrics["total_cxt"] - total_cxt) < 1e-12
    assert metrics["baseline_formula"] == "cxt_value = end_threat - start_threat"
    assert "Future shot or goal outcomes are not used" in metrics["leakage_note"]
    assert metrics["threat_grid_path"] == str(outputs.threat_grid_path)
    assert metrics["prediction_path"] == str(outputs.predictions_path)
    assert metrics["player_aggregates_path"] == str(outputs.player_aggregates_path)
    assert metrics["team_aggregates_path"] == str(outputs.team_aggregates_path)
    assert metrics["sequence_aggregates_path"] == str(outputs.sequence_aggregates_path)
    assert metrics["interpretation_summary_path"] == str(outputs.interpretation_summary_path)
    assert metrics["zone_transition_summary_path"] == str(outputs.zone_transition_summary_path)
    assert metrics["top_actions_path"] == str(outputs.top_actions_path)

    assert abs(interpretation["total_cxt"] - total_cxt) < 1e-12
    assert interpretation["pass_cxt"] == float(
        predictions.loc[predictions["action_type_group"] == "pass", "cxt_value"].sum()
    )
    assert interpretation["carry_cxt"] == float(
        predictions.loc[predictions["action_type_group"] == "carry", "cxt_value"].sum()
    )
    assert interpretation["final_third_entry_cxt"] == float(
        predictions.loc[predictions["entered_final_third"], "cxt_value"].sum()
    )
    assert interpretation["box_entry_cxt"] == float(
        predictions.loc[predictions["entered_box"], "cxt_value"].sum()
    )
    assert interpretation["progressive_action_cxt"] == float(
        predictions.loc[predictions["progressive_action"], "cxt_value"].sum()
    )
    assert interpretation["zone_transition_report_path"] == str(
        outputs.zone_transition_summary_path
    )
    assert interpretation["top_actions_report_path"] == str(outputs.top_actions_path)
    assert interpretation["sequence_aggregate_path"] == str(outputs.sequence_aggregates_path)


def test_cxt_baseline_handles_missing_optional_names_and_drops_leakage_columns():
    actions = pd.DataFrame(
        [
            {
                "match_id": 1,
                "team_id": 10,
                "action_type": "cross",
                "start_x": 80.0,
                "start_y": 5.0,
                "end_x": 102.0,
                "end_y": 38.0,
                "future_shot_xg": 0.99,
                "future_goal": True,
            }
        ]
    )

    features = build_action_features(actions)

    assert len(features) == 1
    assert features.loc[0, "player_id"] is pd.NA
    assert features.loc[0, "player_name"] is pd.NA
    assert not (PROHIBITED_LEAKAGE_COLUMNS & set(features.columns))
    assert features.loc[0, "cxt_value"] == (
        features.loc[0, "end_threat"] - features.loc[0, "start_threat"]
    )


def test_cxt_baseline_outputs_handle_missing_optional_names(tmp_path: Path):
    input_path = tmp_path / "actions.parquet"
    pd.DataFrame(
        [
            {
                "match_id": 1,
                "team_id": 10,
                "action_type": "pass",
                "start_x": 50.0,
                "start_y": 40.0,
                "end_x": 85.0,
                "end_y": 40.0,
            }
        ]
    ).to_parquet(input_path, index=False)

    outputs = run_baseline(
        input_path=input_path,
        feature_store_dir=tmp_path / "feature_store" / "cxt",
        output_dir=tmp_path / "outputs" / "modeling" / "cxt",
    )

    player_aggregates = pd.read_parquet(outputs.player_aggregates_path)
    team_aggregates = pd.read_parquet(outputs.team_aggregates_path)
    sequence_aggregates = pd.read_parquet(outputs.sequence_aggregates_path)

    assert len(player_aggregates) == 1
    assert len(team_aggregates) == 1
    assert len(sequence_aggregates) == 1
    assert "player_name" in player_aggregates.columns
    assert "team_name" in team_aggregates.columns


def test_cxt_baseline_can_write_csv_mirrors(tmp_path: Path):
    input_path = tmp_path / "actions.csv"
    _synthetic_cxt_actions().to_csv(input_path, index=False)
    output_dir = tmp_path / "outputs" / "modeling" / "cxt"

    run_baseline(
        input_path=input_path,
        feature_store_dir=tmp_path / "feature_store" / "cxt",
        output_dir=output_dir,
        write_csv=True,
    )

    assert (output_dir / "predictions" / "action_threat.csv").exists()
    assert (output_dir / "aggregates" / "player_cxt.csv").exists()
    assert (output_dir / "aggregates" / "team_cxt.csv").exists()
    assert (output_dir / "aggregates" / "sequence_cxt.csv").exists()


def test_cxt_generated_paths_are_git_ignored():
    assert_git_ignored(
        (
            Path("feature_store/cxt/action_features.parquet"),
            Path("outputs/modeling/cxt/threat_grid.parquet"),
            Path("outputs/modeling/cxt/predictions/action_threat.parquet"),
            Path("outputs/modeling/cxt/predictions/action_threat.csv"),
            Path("outputs/modeling/cxt/aggregates/player_cxt.parquet"),
            Path("outputs/modeling/cxt/aggregates/player_cxt.csv"),
            Path("outputs/modeling/cxt/aggregates/team_cxt.parquet"),
            Path("outputs/modeling/cxt/aggregates/team_cxt.csv"),
            Path("outputs/modeling/cxt/aggregates/sequence_cxt.parquet"),
            Path("outputs/modeling/cxt/aggregates/sequence_cxt.csv"),
            Path("outputs/modeling/cxt/reports/metrics.json"),
            Path("outputs/modeling/cxt/reports/interpretation_summary.json"),
            Path("outputs/modeling/cxt/reports/zone_transition_summary.csv"),
            Path("outputs/modeling/cxt/reports/zone_transition_summary.parquet"),
            Path("outputs/modeling/cxt/reports/top_actions.csv"),
        ),
        Path.cwd(),
    )
