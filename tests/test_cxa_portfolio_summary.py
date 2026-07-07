import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_cxa_portfolio_summary import (
    CxAPortfolioPaths,
    build_cxa_portfolio_summary,
    build_headline_metrics,
    feature_driver_summary,
    metric_comparison,
)


def _promotion_summary(status: str = "provisionally_promoted", gate: bool = True) -> dict:
    return {
        "metric": "cxa",
        "model_version": "diagnostic_v1",
        "selected_model_candidate": "calibrated_gradient_boosting_sigmoid",
        "validation_recommendation": "provisional_promote",
        "promotion_status": status,
        "promotion_gate_passed": gate,
        "baseline_is_fair_comparator": False,
        "strict_promotion_comparison_enabled": False,
        "governance_summary": {
            "status": "passed",
            "selected_feature_count": 37,
            "selected_features": ["safe_feature", "action_type"],
        },
    }


def _actions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "action_id": [f"a{i}" for i in range(8)],
            "player_id": [10, 10, 11, 12, 12, 13, 14, 14],
            "team_id": [1, 1, 1, 2, 2, 2, 3, 3],
            "shot_created": [1, 0, 0, 1, 0, 0, 0, 1],
            "predicted_shot_created_probability": [0.4, 0.1, 0.2, 0.5, 0.05, 0.08, 0.03, 0.6],
            "diagnostic_cxa": [0.4, 0.1, 0.2, 0.5, 0.05, 0.08, 0.03, 0.6],
        }
    )


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [10, 12, 14, 11],
            "team_id": [1, 2, 3, 1],
            "actions": [2, 2, 2, 1],
            "shot_creating_actions": [1, 1, 1, 0],
            "total_diagnostic_cxa": [0.5, 0.55, 0.63, 0.2],
            "mean_diagnostic_cxa": [0.25, 0.275, 0.315, 0.2],
            "max_diagnostic_cxa": [0.4, 0.5, 0.6, 0.2],
            "rank": [3, 2, 1, 4],
        }
    )


def _teams() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_id": [1, 2, 3],
            "actions": [3, 3, 2],
            "shot_creating_actions": [1, 1, 1],
            "total_diagnostic_cxa": [0.7, 0.63, 0.63],
            "mean_diagnostic_cxa": [0.233, 0.21, 0.315],
            "max_diagnostic_cxa": [0.4, 0.5, 0.6],
            "rank": [1, 2, 3],
        }
    )


def _sequences() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3"],
            "match_id": [1, 1, 2],
            "team_id": [1, 2, 3],
            "possession": [1, 2, 3],
            "actions": [4, 2, 2],
            "shot_creating_actions": [1, 1, 1],
            "total_diagnostic_cxa": [0.8, 0.7, 0.46],
            "max_diagnostic_cxa": [0.4, 0.5, 0.6],
            "mean_diagnostic_cxa": [0.2, 0.35, 0.23],
            "sequence_led_to_shot": [1, 1, 1],
            "rank": [1, 2, 3],
        }
    )


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "log_loss",
                "brier",
                "roc_auc",
                "average_precision",
                "expected_calibration_error",
                "precision_at_top_1pct",
                "precision_at_top_5pct",
            ],
            "baseline": [0.15, 0.04, 0.85, 0.28, 0.002, 0.46, 0.35],
            "diagnostic": [0.14, 0.039, 0.86, 0.33, 0.001, 0.58, 0.37],
            "diagnostic_minus_baseline": [-0.01, -0.001, 0.01, 0.05, -0.001, 0.12, 0.02],
        }
    )


def _feature_impact() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_name": ["end_x", "action_type", "created_shot_cxg"],
            "feature_group": ["progression/location", "action-type/context", "reference"],
            "impact": [0.08, 0.02, 0.99],
            "mean_probability_shift": [0.01, 0.002, 0.5],
            "rank": [1, 2, 99],
        }
    )


def _group_impact() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_group": ["progression/location", "action-type/context"],
            "feature_count": [12, 14],
            "impact": [0.10, 0.03],
            "mean_probability_shift": [0.01, 0.003],
        }
    )


def _impact_summary() -> dict:
    return {
        "metric": "cxa",
        "model_version": "diagnostic_v1",
        "selected_model_candidate": "calibrated_gradient_boosting_sigmoid",
        "promotion_status": "provisionally_promoted",
        "promotion_gate_passed": True,
        "selected_feature_count": 37,
        "top_feature_driver": {"name": "end_x"},
        "top_feature_group_driver": {"name": "progression/location"},
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_artifacts(
    root: Path,
    *,
    promotion_status: str = "provisionally_promoted",
    gate: bool = True,
    include_feature_impact: bool = True,
    missing_optional_columns: bool = False,
) -> CxAPortfolioPaths:
    results_dir = root / "outputs" / "results" / "cxa" / "diagnostic_v1"
    validation_dir = root / "outputs" / "validation" / "cxa" / "diagnostic_v1"
    impact_dir = root / "outputs" / "modeling" / "cxa" / "diagnostic_v1" / "feature_impact"
    results_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    impact_dir.mkdir(parents=True)
    _actions().to_parquet(results_dir / "action_predictions.parquet", index=False)
    players = _players().drop(columns=["max_diagnostic_cxa"] if missing_optional_columns else [])
    players.to_csv(results_dir / "player_cxa_summary.csv", index=False)
    _teams().to_csv(results_dir / "team_cxa_summary.csv", index=False)
    _sequences().to_csv(results_dir / "sequence_cxa_summary.csv", index=False)
    _write_json(
        results_dir / "model_promotion_summary.json",
        _promotion_summary(promotion_status, gate),
    )
    pd.DataFrame([{"check_name": "promotion_gate_passed", "status": "passed"}]).to_csv(
        results_dir / "prediction_quality_checks.csv",
        index=False,
    )
    _metrics().to_csv(validation_dir / "baseline_vs_diagnostic_metrics.csv", index=False)
    if include_feature_impact:
        _feature_impact().to_csv(impact_dir / "feature_impact_summary.csv", index=False)
        _group_impact().to_csv(impact_dir / "feature_group_impact.csv", index=False)
        _write_json(impact_dir / "feature_impact_summary.json", _impact_summary())
    return CxAPortfolioPaths.from_roots(
        results_dir=results_dir,
        validation_dir=validation_dir,
        feature_impact_dir=impact_dir,
        output_dir=root / "outputs" / "portfolio" / "cxa",
    )


def test_metric_comparison_and_headline_fields():
    comparison = metric_comparison(_metrics())
    drivers = feature_driver_summary(_feature_impact(), _group_impact())
    headline = build_headline_metrics(
        promotion=_promotion_summary(),
        impact_summary=_impact_summary(),
        actions=_actions(),
        comparison=comparison,
        drivers=drivers,
    )

    assert comparison["diagnostic_minus_baseline"]["log_loss"] == pytest.approx(-0.01)
    assert headline["action_row_count"] == 8
    assert headline["promotion_status"] == "provisionally_promoted"
    assert headline["top_feature_driver"]["name"] == "end_x"
    assert headline["top_feature_group_driver"]["name"] == "progression/location"


def test_portfolio_summary_writes_outputs_and_charts(tmp_path: Path):
    paths = _write_artifacts(tmp_path)

    outputs = build_cxa_portfolio_summary(paths=paths, top_n_players=2, top_n_teams=2)

    for key in (
        "summary_md",
        "headline_metrics_json",
        "top_players_csv",
        "top_teams_csv",
        "top_sequences_csv",
        "feature_driver_summary_csv",
    ):
        assert outputs[key].exists()
    chart_keys = [key for key in outputs if key.startswith("chart_")]
    assert set(chart_keys) == {
        "chart_top_players_by_cxa",
        "chart_top_teams_by_cxa",
        "chart_feature_group_impact",
        "chart_baseline_vs_diagnostic_metrics",
        "chart_prediction_distribution",
    }
    for key in chart_keys:
        assert outputs[key].exists()
        assert outputs[key].suffix == ".png"


def test_markdown_contains_required_portfolio_story(tmp_path: Path):
    outputs = build_cxa_portfolio_summary(paths=_write_artifacts(tmp_path))
    markdown = outputs["summary_md"].read_text(encoding="utf-8")

    assert "# Diagnostic CxA Portfolio Summary" in markdown
    assert "provisionally_promoted" in markdown
    assert "reference-only/in-sample" in markdown
    assert "Baseline vs diagnostic improvement" in markdown
    assert "Feature impact interpretation" in markdown
    assert "`created_shot_cxg` and `cxa_value` are not model features" in markdown
    assert "CxA+ and Advanced CxA come later" in markdown


def test_headline_metrics_json_contains_required_fields(tmp_path: Path):
    outputs = build_cxa_portfolio_summary(paths=_write_artifacts(tmp_path))
    headline = json.loads(outputs["headline_metrics_json"].read_text(encoding="utf-8"))

    assert headline["selected_model"] == "calibrated_gradient_boosting_sigmoid"
    assert headline["promotion_gate_passed"] is True
    assert headline["selected_feature_count"] == 37
    assert (
        "precision_at_top_1pct" in headline["baseline_vs_diagnostic"]["diagnostic_minus_baseline"]
    )


def test_blocked_promotion_prevents_portfolio_build(tmp_path: Path):
    paths = _write_artifacts(tmp_path, promotion_status="blocked", gate=False)

    with pytest.raises(ValueError, match="blocked promotion gate"):
        build_cxa_portfolio_summary(paths=paths)


def test_missing_feature_impact_input_fails_clearly(tmp_path: Path):
    paths = _write_artifacts(tmp_path, include_feature_impact=False)

    with pytest.raises(FileNotFoundError, match="feature_impact_summary_csv"):
        build_cxa_portfolio_summary(paths=paths)


def test_missing_optional_display_columns_do_not_crash(tmp_path: Path):
    outputs = build_cxa_portfolio_summary(
        paths=_write_artifacts(tmp_path, missing_optional_columns=True)
    )

    players = pd.read_csv(outputs["top_players_csv"])
    assert "max_diagnostic_cxa" in players.columns


def test_no_retraining_or_validation_functions_are_imported():
    source = Path("scripts/build_cxa_portfolio_summary.py").read_text(encoding="utf-8")

    assert "run_cxa_diagnostic_training" not in source
    assert "validate_cxa_diagnostic_model" not in source
    assert "generate_cxa_diagnostic_results" not in source
