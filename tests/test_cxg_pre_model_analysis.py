import inspect
from pathlib import Path

import pandas as pd

from opponent_adjusted.analysis.cxg import build_pre_model_cxg_analysis
from opponent_adjusted.analysis.cxg import core


def _shot_features_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "shot_id": 1,
                "shot_distance": 7.0,
                "shot_angle": 0.60,
                "centrality": 0.8,
                "distance_to_goal_line": 7.0,
                "possession_sequence_length": 3,
                "possession_duration": 8.0,
                "previous_action_gap": 2.0,
                "recent_def_actions_count": 1,
                "pressure_proxy_score": 0.1,
                "is_leading": False,
                "minute_bucket": "0-15",
            },
            {
                "shot_id": 2,
                "shot_distance": 18.0,
                "shot_angle": 0.20,
                "centrality": 0.2,
                "distance_to_goal_line": 18.0,
                "possession_sequence_length": 7,
                "possession_duration": 18.0,
                "previous_action_gap": 5.0,
                "recent_def_actions_count": 3,
                "pressure_proxy_score": 0.9,
                "is_leading": True,
                "minute_bucket": "16-30",
            },
            {
                "shot_id": 3,
                "shot_distance": 10.0,
                "shot_angle": 0.50,
                "centrality": 0.6,
                "distance_to_goal_line": 10.0,
                "possession_sequence_length": 4,
                "possession_duration": 10.0,
                "previous_action_gap": 1.0,
                "recent_def_actions_count": 0,
                "pressure_proxy_score": 0.3,
                "is_leading": False,
                "minute_bucket": "0-15",
            },
            {
                "shot_id": 4,
                "shot_distance": 24.0,
                "shot_angle": 0.10,
                "centrality": 0.1,
                "distance_to_goal_line": 24.0,
                "possession_sequence_length": 9,
                "possession_duration": 22.0,
                "previous_action_gap": 7.0,
                "recent_def_actions_count": 5,
                "pressure_proxy_score": 1.4,
                "is_leading": True,
                "minute_bucket": "16-30",
            },
        ]
    )


def _shots_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "match_id": 100,
                "team_id": 10,
                "player_id": 20,
                "opponent_team_id": 11,
                "outcome": "Goal",
                "statsbomb_xg": 0.35,
                "shot_type": "Open Play",
                "body_part": "Right Foot",
                "technique": "Normal",
                "first_time": False,
                "is_blocked": False,
            },
            {
                "id": 2,
                "match_id": 100,
                "team_id": 10,
                "player_id": 21,
                "opponent_team_id": 11,
                "outcome": "Saved",
                "statsbomb_xg": 0.08,
                "shot_type": "Open Play",
                "body_part": "Left Foot",
                "technique": "Normal",
                "first_time": True,
                "is_blocked": False,
            },
            {
                "id": 3,
                "match_id": 101,
                "team_id": 12,
                "player_id": 22,
                "opponent_team_id": 13,
                "outcome": "Goal",
                "statsbomb_xg": 0.28,
                "shot_type": "Corner",
                "body_part": "Head",
                "technique": "Header",
                "first_time": False,
                "is_blocked": False,
            },
            {
                "id": 4,
                "match_id": 101,
                "team_id": 12,
                "player_id": 23,
                "opponent_team_id": 13,
                "outcome": "Off T",
                "statsbomb_xg": 0.03,
                "shot_type": "Free Kick",
                "body_part": "Right Foot",
                "technique": "Normal",
                "first_time": False,
                "is_blocked": True,
            },
        ]
    )


def test_pre_model_cxg_analysis_writes_required_artifacts(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "analysis" / "cxg"

    result = build_pre_model_cxg_analysis(
        _shot_features_fixture(),
        _shots_fixture(),
        output_dir=output_dir,
        min_slice_size=1,
    )

    assert result.row_count == 4
    assert result.goal_rate == 0.5
    assert result.report_path == output_dir / "report.md"
    assert result.report_path.exists()

    expected_paths = [
        "00_target/target_summary.csv",
        "00_target/plots/target_balance.png",
        "01_feature_distributions/feature_missingness.csv",
        "01_feature_distributions/tables/numeric_feature_profiles.csv",
        "01_feature_distributions/plots/shot_distance_distribution.png",
        "01_feature_distributions/plots/shot_angle_distribution.png",
        "02_feature_target_relationships/numeric_target_relationships.csv",
        "02_feature_target_relationships/tables/shot_distance_bins.csv",
        "02_feature_target_relationships/plots/shot_distance_vs_goal_rate.png",
        "02_feature_target_relationships/tables/shot_angle_bins.csv",
        "02_feature_target_relationships/plots/shot_angle_vs_goal_rate.png",
        "02_feature_target_relationships/tables/pressure_goal_rate.csv",
        "02_feature_target_relationships/plots/pressure_vs_goal_rate.png",
        "03_feature_correlations/high_correlations.csv",
        "03_feature_correlations/tables/targeted_redundancy_pairs.csv",
        "04_slice_stability/slice_stability.csv",
        "05_data_quality/data_quality.csv",
        "05_data_quality/tables/football_value_checks.csv",
        "05_data_quality/tables/cleaning_recommendations.csv",
        "06_leakage_checks/tables/feature_training_eligibility.csv",
        "report.md",
    ]
    for relative_path in expected_paths:
        assert (output_dir / relative_path).exists()

    report = result.report_path.read_text(encoding="utf-8")
    assert not (output_dir / "01_feature_distributions" / "numeric_distributions.png").exists()

    for heading in (
        "# CxG Pre-Model Target and Feature Analysis",
        "1. Target usability",
        "2. Target imbalance",
        "3. Feature distribution findings",
        "4. Shot geometry signal",
        "5. Shot context signal",
        "6. Possession context signal",
        "7. Feature redundancy",
        "8. Slice stability",
        "9. Data quality and cleaning recommendations",
        "10. Leakage risks and training eligibility",
        "11. Modelling recommendations",
    ):
        assert heading in report
    for required_label in (
        "Question",
        "Calculation",
        "Visual/Table",
        "Interpretation",
        "Modelling implication",
        "Limitation",
    ):
        assert f"**{required_label}:**" in report


def test_pre_model_cxg_analysis_handles_missing_optional_columns(tmp_path: Path):
    features = _shot_features_fixture()[["shot_id", "shot_distance"]]
    shots = _shots_fixture()[["id", "outcome"]]

    result = build_pre_model_cxg_analysis(
        features,
        shots,
        output_dir=tmp_path / "cxg",
        min_slice_size=2,
    )

    assert result.row_count == 4
    assert (result.output_dir / "04_slice_stability" / "slice_stability.csv").exists()
    assert (result.output_dir / "05_data_quality" / "tables" / "football_value_checks.csv").exists()
    assert (
        result.output_dir / "06_leakage_checks" / "tables" / "feature_training_eligibility.csv"
    ).exists()


def test_pre_model_cxg_analysis_does_not_depend_on_post_model_tables():
    source = inspect.getsource(core.load_shot_feature_dataset)

    assert "ShotFeature" in source
    assert "Shot" in source
    assert "ShotPrediction" not in source
    assert "ModelRegistry" not in source
    assert "AggregatesPlayer" not in source
    assert "AggregatesTeam" not in source
