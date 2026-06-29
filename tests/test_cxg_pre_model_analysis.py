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
                "pressure_proxy_score": 0.1,
                "is_leading": False,
                "minute_bucket": "0-15",
            },
            {
                "shot_id": 2,
                "shot_distance": 18.0,
                "shot_angle": 0.20,
                "centrality": 0.2,
                "pressure_proxy_score": 0.9,
                "is_leading": True,
                "minute_bucket": "16-30",
            },
            {
                "shot_id": 3,
                "shot_distance": 10.0,
                "shot_angle": 0.50,
                "centrality": 0.6,
                "pressure_proxy_score": 0.3,
                "is_leading": False,
                "minute_bucket": "0-15",
            },
            {
                "shot_id": 4,
                "shot_distance": 24.0,
                "shot_angle": 0.10,
                "centrality": 0.1,
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
        "00_target/target_balance.png",
        "01_feature_distributions/feature_missingness.csv",
        "01_feature_distributions/numeric_distributions.png",
        "02_feature_target_relationships/numeric_target_relationships.csv",
        "03_feature_correlations/high_correlations.csv",
        "04_slice_stability/slice_stability.csv",
        "05_data_quality/data_quality.csv",
        "06_leakage_checks/leakage_checks.csv",
        "report.md",
    ]
    for relative_path in expected_paths:
        assert (output_dir / relative_path).exists()

    report = result.report_path.read_text(encoding="utf-8")
    for heading in (
        "Is the target usable?",
        "Is the target imbalanced?",
        "Which features show signal against the goal target?",
        "Which features are redundant or highly correlated?",
        "Which relationships are stable or unstable across slices?",
        "Which features need cleaning",
        "Which columns are leakage risks?",
        "What should the modelling layer do next?",
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


def test_pre_model_cxg_analysis_does_not_depend_on_post_model_tables():
    source = inspect.getsource(core.load_shot_feature_dataset)

    assert "ShotFeature" in source
    assert "Shot" in source
    assert "ShotPrediction" not in source
    assert "ModelRegistry" not in source
    assert "AggregatesPlayer" not in source
    assert "AggregatesTeam" not in source
