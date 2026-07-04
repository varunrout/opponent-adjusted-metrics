import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_cxg_portfolio_summary import (
    PortfolioPaths,
    build_cxg_portfolio_summary,
    build_scorecard,
    category_insights,
    feature_driver_summary,
    metric_comparison,
    player_rankings,
    team_rankings,
)


def _promotion_summary() -> dict:
    return {
        "model_version": "diagnostic_v1",
        "selected_model_candidate": "calibrated_gradient_boosting_sigmoid",
        "promotion_status": "promoted",
        "promotion_gate_passed": True,
        "baseline_comparison": {"join_rate": 1.0},
        "governance_summary": {"status": "passed"},
        "validation_metrics": {
            "baseline": {
                "log_loss": 0.277314,
                "brier": 0.078278,
                "roc_auc": 0.778879,
                "expected_calibration_error": 0.002608,
            },
            "diagnostic_v1:calibrated_gradient_boosting_sigmoid": {
                "log_loss": 0.269903,
                "brier": 0.076413,
                "roc_auc": 0.796411,
                "expected_calibration_error": 0.005963,
            },
        },
    }


def _impact_summary() -> dict:
    return {
        "model_version": "diagnostic_v1",
        "selected_model_candidate": "calibrated_gradient_boosting_sigmoid",
        "promotion_status": "promoted",
        "promotion_gate_passed": True,
        "governance_status": "passed",
        "baseline_join_rate": 1.0,
        "selected_feature_count": 32,
        "result_integrity_checks": {
            "feature_frame_shot_id_missing_count": 0,
            "feature_frame_player_id_missing_count": 0,
            "feature_frame_team_id_missing_count": 0,
            "shot_predictions_shot_id_missing_count": 0,
            "shot_predictions_player_id_missing_count": 0,
            "shot_predictions_team_id_missing_count": 0,
        },
    }


def _team_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 2, 3],
            "team_name": ["Alpha", "Beta", "Gamma"],
            "shots": [20, 18, 15],
            "goals": [5, 2, 1],
            "total_cxg": [4.2, 3.4, 2.0],
            "mean_cxg_per_shot": [0.21, 0.19, 0.13],
            "goals_minus_cxg": [0.8, -1.4, -1.0],
            "baseline_total_cxg": [4.0, 3.1, 2.2],
            "total_cxg_delta_vs_baseline": [0.2, 0.3, -0.2],
            "rank_total_cxg": [1, 2, 3],
        }
    )


def _player_frame(missing_player_id: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [10, None if missing_player_id else 11, 12],
            "player_name": ["A One", "B Two", "C Three"],
            "team_id": [1, 2, 3],
            "team_name": ["Alpha", "Beta", "Gamma"],
            "shots": [8, 7, 6],
            "goals": [3, 1, 1],
            "total_cxg": [2.1, 1.4, 1.1],
            "mean_cxg_per_shot": [0.26, 0.2, 0.18],
            "goals_minus_cxg": [0.9, -0.4, -0.1],
            "baseline_total_cxg": [2.0, 1.2, 1.0],
            "total_cxg_delta_vs_baseline": [0.1, 0.2, 0.1],
            "rank_total_cxg": [1, 2, 3],
        }
    )


def _group_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_group": ["geometry", "shot_execution"],
            "log_loss_delta": [0.04, 0.02],
            "brier_delta": [0.01, 0.004],
            "roc_auc_delta": [0.03, 0.01],
            "absolute_probability_delta_mean": [0.05, 0.02],
        }
    )


def _permutation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["shot_distance", "body_part", "statsbomb_xg"],
            "feature_group": ["geometry", "shot_execution", "reference"],
            "log_loss_delta": [0.03, 0.01, 0.99],
            "brier_delta": [0.006, 0.002, 0.2],
            "roc_auc_delta": [0.02, 0.005, 0.8],
            "absolute_probability_delta_mean": [0.04, 0.01, 0.5],
            "impact_rank": [1, 2, 99],
        }
    )


def _category_frame(category_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category_column": [category_column, category_column],
            "category": ["A", "B"],
            "shots": [10, 5],
            "goals": [2, 1],
            "goal_rate": [0.2, 0.2],
            "mean_predicted_cxg": [0.18, 0.12],
            "total_predicted_cxg": [1.8, 0.6],
            "mean_baseline_cxg": [0.16, 0.11],
            "total_baseline_cxg": [1.6, 0.55],
            "mean_delta_vs_baseline": [0.02, 0.01],
            "total_delta_vs_baseline": [0.2, 0.05],
        }
    )


def _write_artifacts(root: Path) -> PortfolioPaths:
    results_dir = root / "outputs" / "results" / "cxg" / "diagnostic_v1"
    impact_dir = root / "outputs" / "modeling" / "cxg" / "diagnostic_v1" / "feature_impact"
    results_dir.mkdir(parents=True)
    impact_dir.mkdir(parents=True)
    (results_dir / "model_promotion_summary.json").write_text(
        json.dumps(_promotion_summary()),
        encoding="utf-8",
    )
    pd.DataFrame({"shot_id": [1, 2], "player_id": [10, 11]}).to_parquet(
        results_dir / "shot_predictions.parquet",
        index=False,
    )
    _team_frame().to_csv(results_dir / "team_cxg_summary.csv", index=False)
    _team_frame().to_csv(results_dir / "team_cxg_rankings.csv", index=False)
    _player_frame().to_csv(results_dir / "player_cxg_summary.csv", index=False)
    _player_frame().head(2).to_csv(results_dir / "top_players_by_cxg.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "baseline_log_loss", "value": 0.277314},
            {"metric": "diagnostic_log_loss", "value": 0.269903},
        ]
    ).to_csv(results_dir / "baseline_vs_diagnostic_summary.csv", index=False)
    (impact_dir / "feature_impact_summary.json").write_text(
        json.dumps(_impact_summary()),
        encoding="utf-8",
    )
    (impact_dir / "feature_impact_report.md").write_text("feature impact", encoding="utf-8")
    _group_frame().to_csv(impact_dir / "group_perturbation_summary.csv", index=False)
    _permutation_frame().to_csv(impact_dir / "permutation_importance.csv", index=False)
    _category_frame("body_part").to_csv(impact_dir / "category_lift_body_part.csv", index=False)
    _category_frame("shot_type").to_csv(impact_dir / "category_lift_shot_type.csv", index=False)
    _category_frame("set_piece_category").to_csv(
        impact_dir / "category_lift_set_piece_category.csv",
        index=False,
    )
    return PortfolioPaths.from_roots(
        results_dir=results_dir,
        feature_impact_dir=impact_dir,
        output_dir=root / "outputs" / "portfolio" / "cxg",
    )


def test_scorecard_extracts_metrics_and_join_rate():
    comparison = metric_comparison(_promotion_summary(), _impact_summary(), pd.DataFrame())
    scorecard = build_scorecard(
        promotion_summary=_promotion_summary(),
        impact_summary=_impact_summary(),
        baseline_summary=pd.DataFrame(),
        outputs={"summary_md": Path("summary.md")},
        charts={"model_metric_comparison": Path("chart.png")},
    )

    assert scorecard["baseline_join_rate"] == 1.0
    assert comparison["baseline"]["log_loss"] == 0.277314
    assert comparison["diagnostic"]["log_loss"] == 0.269903
    assert comparison["diagnostic_minus_baseline"]["log_loss"] == pytest.approx(-0.007411)


def test_team_rankings_output_has_expected_columns():
    ranked = team_rankings(_team_frame(), pd.DataFrame())

    assert list(ranked.columns) == [
        "team_id",
        "team_name",
        "shots",
        "goals",
        "total_cxg",
        "mean_cxg_per_shot",
        "goals_minus_cxg",
        "baseline_total_cxg",
        "total_cxg_delta_vs_baseline",
        "rank_total_cxg",
    ]
    assert ranked.iloc[0]["team_name"] == "Alpha"


def test_player_rankings_validate_non_null_player_id():
    ranked = player_rankings(_player_frame(), pd.DataFrame())

    assert ranked["player_id"].isna().sum() == 0
    with pytest.raises(ValueError, match="missing player_id"):
        player_rankings(_player_frame(missing_player_id=True), pd.DataFrame())


def test_feature_driver_summary_combines_groups_and_features_without_statsbomb_xg():
    drivers = feature_driver_summary(_group_frame(), _permutation_frame())

    assert set(drivers["driver_type"]) == {"feature_group", "feature"}
    assert "statsbomb_xg" not in set(drivers["name"])


def test_category_insights_combines_available_tables_and_skips_missing(tmp_path: Path):
    _category_frame("body_part").to_csv(tmp_path / "category_lift_body_part.csv", index=False)
    _category_frame("shot_type").to_csv(tmp_path / "category_lift_shot_type.csv", index=False)

    insights, skipped = category_insights(tmp_path)

    assert set(insights["category_column"]) == {"body_part", "shot_type"}
    assert "set_piece_category" in skipped
    assert "pressure_state" in skipped


def test_portfolio_summary_writes_outputs_and_charts(tmp_path: Path):
    paths = _write_artifacts(tmp_path)

    outputs = build_cxg_portfolio_summary(
        paths=paths,
        top_n_teams=2,
        top_n_players=2,
        top_n_features=2,
    )

    required = [
        "summary_md",
        "scorecard_json",
        "team_rankings_csv",
        "player_rankings_csv",
        "feature_driver_summary_csv",
        "category_insights_csv",
    ]
    for key in required:
        assert outputs[key].exists()
    chart_keys = [key for key in outputs if key.startswith("chart_")]
    assert chart_keys
    for key in chart_keys:
        assert outputs[key].exists()
        assert outputs[key].suffix == ".png"
    markdown = outputs["summary_md"].read_text(encoding="utf-8")
    assert "# Diagnostic-informed CxG Portfolio Summary" in markdown
    assert "## Model Scorecard" in markdown
    assert "## Charts" in markdown
    scorecard = json.loads(outputs["scorecard_json"].read_text(encoding="utf-8"))
    assert scorecard["governance_status"] == "passed"
    assert scorecard["metric_comparison"]["diagnostic"]["roc_auc"] == 0.796411
