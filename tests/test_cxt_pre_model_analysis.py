import inspect
from pathlib import Path

import pandas as pd

from opponent_adjusted.analysis.cxt import build_pre_model_cxt_analysis, detect_target_proxy_column
from opponent_adjusted.analysis.cxt import pre_model


def _progression_fixture(include_target: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
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
                "start_x": 30.0,
                "start_y": 40.0,
                "end_x": 70.0,
                "end_y": 42.0,
                "dx": 40.0,
                "dy": 2.0,
                "distance_moved": 40.1,
                "distance_progressed": 40.0,
                "goal_distance_start": 90.0,
                "goal_distance_end": 50.0,
                "goal_distance_reduction": 40.0,
                "final_third_entry": False,
                "box_entry": False,
                "zone14_entry": False,
                "under_pressure": False,
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
                "start_x": 65.0,
                "start_y": 22.0,
                "end_x": 84.0,
                "end_y": 25.0,
                "dx": 19.0,
                "dy": 3.0,
                "distance_moved": 19.2,
                "distance_progressed": 19.0,
                "goal_distance_start": 58.0,
                "goal_distance_end": 39.0,
                "goal_distance_reduction": 19.0,
                "final_third_entry": True,
                "box_entry": False,
                "zone14_entry": False,
                "under_pressure": True,
            },
            {
                "action_id": "a3",
                "event_id": "3",
                "match_id": 11,
                "team_id": 101,
                "player_id": 1002,
                "possession_id": "11-4",
                "sequence_id": "11-4",
                "action_type": "Dribble",
                "start_x": 90.0,
                "start_y": 50.0,
                "end_x": 105.0,
                "end_y": 45.0,
                "dx": 15.0,
                "dy": -5.0,
                "distance_moved": 15.8,
                "distance_progressed": 15.0,
                "goal_distance_start": 32.0,
                "goal_distance_end": 16.0,
                "goal_distance_reduction": 16.0,
                "final_third_entry": True,
                "box_entry": True,
                "zone14_entry": False,
                "under_pressure": False,
            },
            {
                "action_id": "a4",
                "event_id": "4",
                "match_id": 11,
                "team_id": 101,
                "player_id": 1003,
                "possession_id": "11-4",
                "sequence_id": "11-4",
                "action_type": "Pass",
                "start_x": 78.0,
                "start_y": 36.0,
                "end_x": 64.0,
                "end_y": 38.0,
                "dx": -14.0,
                "dy": 2.0,
                "distance_moved": 14.1,
                "distance_progressed": -14.0,
                "goal_distance_start": 43.0,
                "goal_distance_end": 56.0,
                "goal_distance_reduction": -13.0,
                "final_third_entry": False,
                "box_entry": False,
                "zone14_entry": False,
                "under_pressure": False,
            },
        ]
    )
    if include_target:
        frame["xt_delta"] = [0.05, 0.03, 0.08, -0.02]
    return frame


def test_target_proxy_detection():
    assert detect_target_proxy_column(pd.DataFrame({"xt_delta": [0.1]})) == "xt_delta"
    assert detect_target_proxy_column(pd.DataFrame({"future_shot_value": [0.2]})) == (
        "future_shot_value"
    )
    assert detect_target_proxy_column(pd.DataFrame({"action_type": ["Pass"]})) is None


def test_pre_model_cxt_analysis_writes_required_outputs(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "analysis" / "cxt"

    result = build_pre_model_cxt_analysis(
        _progression_fixture(),
        output_dir=output_dir,
        min_sample_size=1,
    )

    assert result.row_count == 4
    assert result.target_proxy_column is None

    expected_paths = [
        "00_action_coverage/tables/action_type_coverage.csv",
        "00_action_coverage/plots/action_type_coverage.png",
        "00_action_coverage/tables/id_coverage.csv",
        "00_action_coverage/tables/location_coverage.csv",
        "01_spatial_coverage/tables/start_zone_coverage.csv",
        "01_spatial_coverage/tables/end_zone_coverage.csv",
        "01_spatial_coverage/tables/transition_coverage.csv",
        "01_spatial_coverage/plots/start_zone_coverage.png",
        "01_spatial_coverage/plots/end_zone_coverage.png",
        "01_spatial_coverage/plots/transition_coverage.png",
        "02_feature_distributions/tables/numeric_feature_profiles.csv",
        "02_feature_distributions/tables/categorical_feature_profiles.csv",
        "02_feature_distributions/plots/start_x_distribution.png",
        "02_feature_distributions/plots/action_type_levels.png",
        "03_feature_target_relationships/tables/missing_target_proxy.csv",
        "03_feature_target_relationships/tables/action_type_progression_summary.csv",
        "03_feature_target_relationships/plots/action_type_progression_summary.png",
        "03_feature_target_relationships/tables/zone_progression_summary.csv",
        "03_feature_target_relationships/plots/zone_progression_summary.png",
        "03_feature_target_relationships/tables/final_third_box_entry_summary.csv",
        "03_feature_target_relationships/plots/final_third_box_entry_summary.png",
        "04_feature_correlations/tables/numeric_correlations.csv",
        "04_feature_correlations/tables/high_correlations.csv",
        "04_feature_correlations/tables/targeted_redundancy_checks.csv",
        "04_feature_correlations/plots/correlation_heatmap.png",
        "05_transition_stability/tables/transition_stability.csv",
        "05_transition_stability/plots/transition_stability.png",
        "05_transition_stability/tables/sparse_transitions.csv",
        "05_transition_stability/tables/zone_resolution_recommendations.csv",
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
        "# CxT Pre-Model Ball Progression Feature Analysis",
        "1. Action/progression table usability",
        "2. Action coverage",
        "3. Spatial coverage",
        "4. Progression feature distributions",
        "5. Target/proxy availability",
        "6. Progression signal without target/proxy",
        "7. Feature redundancy",
        "8. Transition stability",
        "9. Slice stability",
        "10. Data quality and cleaning recommendations",
        "11. Leakage risks and training eligibility",
        "12. Modelling recommendations",
    ):
        assert heading in report
    assert "No supervised CxT target/proxy is currently available" in report


def test_pre_model_cxt_analysis_handles_optional_target_proxy(tmp_path: Path):
    result = build_pre_model_cxt_analysis(
        _progression_fixture(include_target=True),
        output_dir=tmp_path / "cxt",
        min_sample_size=1,
    )

    assert result.target_proxy_column == "xt_delta"
    assert (
        result.output_dir
        / "03_feature_target_relationships"
        / "tables"
        / "numeric_target_relationships.csv"
    ).exists()


def test_pre_model_cxt_analysis_handles_missing_optional_columns(tmp_path: Path):
    frame = _progression_fixture()[
        ["action_id", "action_type", "start_x", "start_y", "end_x", "end_y"]
    ]

    result = build_pre_model_cxt_analysis(frame, output_dir=tmp_path / "cxt", min_sample_size=1)

    assert result.row_count == 4
    assert (
        result.output_dir / "05_transition_stability" / "tables" / "transition_stability.csv"
    ).exists()
    assert (
        result.output_dir / "08_leakage_checks" / "tables" / "feature_training_eligibility.csv"
    ).exists()


def test_pre_model_cxt_analysis_does_not_depend_on_post_model_tables():
    source = inspect.getsource(pre_model.load_progression_feature_dataset)

    assert "ActionFeature" in source
    assert "ActionThreatPrediction" not in source
    assert "ModelRegistry" not in source
    assert "action_threat_predictions" not in source
    assert "model_registry" not in source
