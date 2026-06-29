import inspect
from pathlib import Path

import pandas as pd
import pytest

from opponent_adjusted.analysis.cxa import build_pre_model_cxa_analysis, detect_cxa_target_column
from opponent_adjusted.analysis.cxa import core


def _action_features_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_id": "a1",
                "event_id": "1",
                "match_id": 10,
                "team_id": 100,
                "player_id": 1000,
                "possession_id": "10-1",
                "sequence_id": "10-1",
                "action_type": "Pass",
                "start_x": 45.0,
                "start_y": 35.0,
                "end_x": 75.0,
                "end_y": 38.0,
                "dx": 30.0,
                "dy": 3.0,
                "distance_progressed": 30.1,
                "goal_distance_start": 75.0,
                "goal_distance_end": 45.0,
                "goal_distance_reduction": 30.0,
                "progressive_distance": 30.0,
                "sequence_position": 1,
                "time_since_possession_start": 2.0,
                "is_progressive": True,
                "final_third_entry": False,
                "box_entry": False,
                "cross": False,
                "through_ball": True,
                "cutback": False,
                "carry": False,
                "dribble": False,
                "start_zone": "middle_central",
                "end_zone": "final_central",
                "under_pressure": False,
                "shot_created": 1,
                "created_shot_cxg": 0.18,
                "created_shot_id": "s1",
            },
            {
                "action_id": "a2",
                "event_id": "2",
                "match_id": 10,
                "team_id": 100,
                "player_id": 1001,
                "possession_id": "10-1",
                "sequence_id": "10-1",
                "action_type": "Carry",
                "start_x": 60.0,
                "start_y": 20.0,
                "end_x": 72.0,
                "end_y": 22.0,
                "dx": 12.0,
                "dy": 2.0,
                "distance_progressed": 12.2,
                "goal_distance_start": 62.0,
                "goal_distance_end": 50.0,
                "goal_distance_reduction": 12.0,
                "progressive_distance": 12.0,
                "sequence_position": 2,
                "time_since_possession_start": 5.0,
                "is_progressive": True,
                "final_third_entry": False,
                "box_entry": False,
                "cross": False,
                "through_ball": False,
                "cutback": False,
                "carry": True,
                "dribble": False,
                "start_zone": "middle_left",
                "end_zone": "middle_left",
                "under_pressure": True,
                "shot_created": 0,
                "created_shot_cxg": 0.0,
                "created_shot_id": None,
            },
            {
                "action_id": "a3",
                "event_id": "3",
                "match_id": 11,
                "team_id": 101,
                "player_id": 1002,
                "possession_id": "11-5",
                "sequence_id": "11-5",
                "action_type": "Dribble",
                "start_x": 88.0,
                "start_y": 50.0,
                "end_x": 104.0,
                "end_y": 45.0,
                "dx": 16.0,
                "dy": -5.0,
                "distance_progressed": 16.8,
                "goal_distance_start": 35.0,
                "goal_distance_end": 18.0,
                "goal_distance_reduction": 17.0,
                "progressive_distance": 16.0,
                "sequence_position": 3,
                "time_since_possession_start": 9.0,
                "is_progressive": True,
                "final_third_entry": True,
                "box_entry": True,
                "cross": False,
                "through_ball": False,
                "cutback": True,
                "carry": False,
                "dribble": True,
                "start_zone": "final_central",
                "end_zone": "final_central",
                "under_pressure": False,
                "shot_created": 1,
                "created_shot_cxg": 0.31,
                "created_shot_id": "s2",
            },
            {
                "action_id": "a4",
                "event_id": "4",
                "match_id": 11,
                "team_id": 101,
                "player_id": 1003,
                "possession_id": "11-5",
                "sequence_id": "11-5",
                "action_type": "Pass",
                "start_x": 30.0,
                "start_y": 65.0,
                "end_x": 34.0,
                "end_y": 63.0,
                "dx": 4.0,
                "dy": -2.0,
                "distance_progressed": 4.5,
                "goal_distance_start": 92.0,
                "goal_distance_end": 88.0,
                "goal_distance_reduction": 4.0,
                "progressive_distance": 4.0,
                "sequence_position": 1,
                "time_since_possession_start": 1.0,
                "is_progressive": False,
                "final_third_entry": False,
                "box_entry": False,
                "cross": True,
                "through_ball": False,
                "cutback": False,
                "carry": False,
                "dribble": False,
                "start_zone": "defensive_right",
                "end_zone": "defensive_right",
                "under_pressure": False,
                "shot_created": 0,
                "created_shot_cxg": 0.0,
                "created_shot_id": None,
            },
        ]
    )


def test_target_detection_accepts_known_aliases():
    assert detect_cxa_target_column(pd.DataFrame({"created_shot": [0, 1]})) == "created_shot"
    assert detect_cxa_target_column(pd.DataFrame({"target_shot_created": [False]})) == (
        "target_shot_created"
    )


def test_target_detection_fails_clearly_when_missing():
    with pytest.raises(ValueError, match="Expected one of"):
        detect_cxa_target_column(pd.DataFrame({"action_type": ["Pass"]}))


def test_pre_model_cxa_analysis_writes_required_artifacts(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "analysis" / "cxa"

    result = build_pre_model_cxa_analysis(
        _action_features_fixture(),
        output_dir=output_dir,
        min_slice_size=1,
    )

    assert result.row_count == 4
    assert result.target_column == "shot_created"
    assert result.target_rate == 0.5

    expected_paths = [
        "00_target/tables/target_summary.csv",
        "00_target/plots/target_balance.png",
        "00_target/tables/created_shot_value_summary.csv",
        "00_target/plots/created_shot_value_distribution.png",
        "01_action_coverage/tables/action_type_coverage.csv",
        "01_action_coverage/plots/action_type_coverage.png",
        "01_action_coverage/tables/id_coverage.csv",
        "01_action_coverage/tables/location_coverage.csv",
        "02_feature_distributions/tables/numeric_feature_profiles.csv",
        "02_feature_distributions/tables/categorical_feature_profiles.csv",
        "02_feature_distributions/plots/start_x_distribution.png",
        "02_feature_distributions/plots/action_type_levels.png",
        "03_feature_target_relationships/tables/action_type_target_rate.csv",
        "03_feature_target_relationships/plots/action_type_target_rate.png",
        "03_feature_target_relationships/tables/end_zone_target_rate.csv",
        "03_feature_target_relationships/plots/end_zone_target_rate.png",
        "03_feature_target_relationships/tables/progression_feature_target_rates.csv",
        "03_feature_target_relationships/plots/progression_feature_target_rates.png",
        "03_feature_target_relationships/tables/sequence_position_target_rate.csv",
        "03_feature_target_relationships/plots/sequence_position_target_rate.png",
        "03_feature_target_relationships/tables/numeric_target_relationships.csv",
        "04_feature_correlations/tables/numeric_correlations.csv",
        "04_feature_correlations/tables/high_correlations.csv",
        "04_feature_correlations/plots/correlation_heatmap.png",
        "04_feature_correlations/tables/targeted_redundancy_checks.csv",
        "05_sequence_window_stability/tables/window_coverage.csv",
        "05_sequence_window_stability/plots/window_coverage.png",
        "05_sequence_window_stability/tables/sequence_position_positive_rate.csv",
        "05_sequence_window_stability/plots/sequence_position_positive_rate.png",
        "05_sequence_window_stability/tables/missing_window_fields.csv",
        "06_slice_stability/tables/slice_stability.csv",
        "06_slice_stability/plots/slice_stability.png",
        "07_data_quality/tables/feature_quality.csv",
        "07_data_quality/tables/football_value_checks.csv",
        "07_data_quality/tables/cleaning_recommendations.csv",
        "08_leakage_checks/tables/leakage_checks.csv",
        "08_leakage_checks/tables/feature_training_eligibility.csv",
        "report.md",
    ]
    for relative_path in expected_paths:
        assert (output_dir / relative_path).exists()

    assert not (output_dir / "02_feature_distributions" / "numeric_distributions.png").exists()

    report = result.report_path.read_text(encoding="utf-8")
    for heading in (
        "# CxA Pre-Model Target and Action-Feature Analysis",
        "1. Target usability",
        "2. Target sparsity and imbalance",
        "3. Action coverage",
        "4. Action-type signal",
        "5. Movement and spatial feature signal",
        "6. Sequence/window stability",
        "7. Feature redundancy",
        "8. Slice stability",
        "9. Data quality and cleaning recommendations",
        "10. Leakage risks and training eligibility",
        "11. Modelling recommendations",
    ):
        assert heading in report


def test_pre_model_cxa_analysis_handles_missing_optional_columns(tmp_path: Path):
    frame = _action_features_fixture()[["action_id", "action_type", "shot_created"]]

    result = build_pre_model_cxa_analysis(
        frame,
        output_dir=tmp_path / "cxa",
        min_slice_size=2,
    )

    assert result.row_count == 4
    missing = pd.read_csv(
        result.output_dir / "05_sequence_window_stability" / "tables" / "missing_window_fields.csv"
    )
    assert "actions_until_shot" in set(missing["missing_field"])
    assert (
        result.output_dir / "08_leakage_checks" / "tables" / "feature_training_eligibility.csv"
    ).exists()


def test_pre_model_cxa_analysis_does_not_depend_on_post_model_tables():
    source = inspect.getsource(core.load_action_feature_dataset)

    assert "ActionFeature" in source
    assert "ActionPrediction" not in source
    assert "ModelRegistry" not in source
    assert "AggregatesPlayer" not in source
    assert "AggregatesTeam" not in source
    assert "AggregatesSequence" not in source
