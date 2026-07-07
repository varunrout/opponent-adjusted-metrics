import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_cxa_portfolio_summary as portfolio
from scripts.build_cxa_portfolio_summary import (
    CxAPortfolioPaths,
    build_cxa_portfolio_summary,
    build_headline_metrics,
    build_player_team_lookup,
    display_player_label,
    display_team_label,
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
            "player_name": [
                "Alex Action",
                "Alex Action",
                "Bea Ball",
                "Cam Carry",
                "Cam Carry",
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "team_name": [
                "Alpha FC",
                "Alpha FC",
                "Alpha FC",
                "Beta FC",
                "Beta FC",
                "Beta FC",
                pd.NA,
                pd.NA,
            ],
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
            "feature_group": ["progression/location", "action-type/context", "pressure"],
            "feature_count": [12, 14, 0],
            "impact": [0.10, 0.03, pd.NA],
            "mean_probability_shift": [0.01, 0.003, pd.NA],
            "status": ["computed", "computed", "skipped"],
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
        feature_path=root / "feature_store" / "cxa" / "action_features.parquet",
        database_path=root / "data" / "opponent_adjusted.db",
    )


def test_metric_comparison_and_headline_fields():
    comparison = metric_comparison(_metrics())
    drivers = feature_driver_summary(_feature_impact(), _group_impact())
    headline = build_headline_metrics(
        promotion=_promotion_summary(),
        impact_summary=_impact_summary(),
        actions=_actions(),
        players=_players(),
        teams=_teams(),
        name_source_used="action_predictions",
        comparison=comparison,
        drivers=drivers,
    )

    assert comparison["diagnostic_minus_baseline"]["log_loss"] == pytest.approx(-0.01)
    assert headline["action_row_count"] == 8
    assert headline["promotion_status"] == "provisionally_promoted"
    assert headline["top_feature_driver"]["name"] == "end_x"
    assert headline["top_feature_group_driver"]["name"] == "progression/location"
    assert headline["name_source_used"] == "action_predictions"


def test_feature_group_skipped_rows_do_not_break_rankings():
    drivers = feature_driver_summary(_feature_impact(), _group_impact())

    groups = drivers.loc[drivers["driver_type"] == "feature_group"].copy()
    assert groups.loc[groups["name"] == "progression/location", "rank"].iloc[0] == 1
    assert groups.loc[groups["name"] == "action-type/context", "rank"].iloc[0] == 2
    assert pd.isna(groups.loc[groups["name"] == "pressure", "rank"].iloc[0])


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
    drivers = pd.read_csv(outputs["feature_driver_summary_csv"])
    skipped = drivers.loc[drivers["name"] == "pressure"].iloc[0]
    assert skipped["status"] == "skipped"
    assert pd.isna(skipped["rank"])


def test_portfolio_outputs_are_enriched_with_player_and_team_names(tmp_path: Path):
    outputs = build_cxa_portfolio_summary(
        paths=_write_artifacts(tmp_path),
        top_n_players=4,
        top_n_teams=3,
        top_n_sequences=3,
    )

    players = pd.read_csv(outputs["top_players_csv"])
    teams = pd.read_csv(outputs["top_teams_csv"])
    sequences = pd.read_csv(outputs["top_sequences_csv"])

    assert players.columns[:10].tolist() == [
        "player_name",
        "team_name",
        "player_id",
        "team_id",
        "actions",
        "shot_creating_actions",
        "total_diagnostic_cxa",
        "mean_diagnostic_cxa",
        "max_diagnostic_cxa",
        "rank",
    ]
    assert teams.columns[:8].tolist() == [
        "team_name",
        "team_id",
        "actions",
        "shot_creating_actions",
        "total_diagnostic_cxa",
        "mean_diagnostic_cxa",
        "max_diagnostic_cxa",
        "rank",
    ]
    assert sequences.columns[:12].tolist() == [
        "sequence_id",
        "match_id",
        "team_name",
        "team_id",
        "possession",
        "actions",
        "shot_creating_actions",
        "total_diagnostic_cxa",
        "max_diagnostic_cxa",
        "mean_diagnostic_cxa",
        "sequence_led_to_shot",
        "rank",
    ]
    assert set(["player_id", "team_id"]).issubset(players.columns)
    assert "Alpha FC" in set(teams["team_name"])
    assert "Unknown player 14" in set(players["player_name"])
    assert "Unknown team 3" in set(teams["team_name"])


def test_portfolio_markdown_displays_names_instead_of_raw_ids(tmp_path: Path):
    outputs = build_cxa_portfolio_summary(paths=_write_artifacts(tmp_path), top_n_players=4)
    markdown = outputs["summary_md"].read_text(encoding="utf-8")

    assert "`Alex Action` (`Alpha FC`, player_id `10`)" in markdown
    assert "`Alpha FC` (team_id `1`)" in markdown
    assert "`s1` (`Alpha FC`, team_id `1`)" in markdown
    assert "- `10`:" not in markdown


def test_chart_labels_use_names_where_available(monkeypatch, tmp_path: Path):
    captured: dict[str, list[str]] = {}

    def fake_plot(
        frame: pd.DataFrame, *, label_col: str, value_col: str, path: Path, title: str, xlabel: str
    ) -> None:
        captured[path.name] = frame[label_col].astype(str).tolist()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")

    monkeypatch.setattr(portfolio, "_plot_horizontal_bars", fake_plot)
    build_cxa_portfolio_summary(
        paths=_write_artifacts(tmp_path),
        top_n_players=4,
        top_n_teams=3,
    )

    assert "Alex Action (Alpha FC)" in captured["top_players_by_cxa.png"]
    assert "Alpha FC" in captured["top_teams_by_cxa.png"]
    assert "Unknown player 14 (Unknown team 3)" in captured["top_players_by_cxa.png"]


def test_name_lookup_uses_most_frequent_non_null_name_and_fallbacks():
    actions = pd.DataFrame(
        {
            "player_id": [7, 7, 7, 8],
            "player_name": ["Frequent Name", "Rare Name", "Frequent Name", pd.NA],
            "team_id": [2, 2, 3, 4],
            "team_name": ["Two FC", "Two FC", "Wrong FC", pd.NA],
        }
    )
    lookup = build_player_team_lookup(actions)
    fallback_row = pd.Series({"player_id": 8, "team_id": 4})

    assert lookup["player_names"][7] == "Frequent Name"
    assert lookup["team_names"][2] == "Two FC"
    named_row = pd.Series(
        {
            "player_id": 7,
            "player_name": lookup["player_names"][7],
            "team_id": 2,
            "team_name": "Two FC",
        }
    )
    assert display_player_label(named_row) == "Frequent Name (Two FC)"
    assert display_player_label(fallback_row) == "Unknown player 8 (Unknown team 4)"
    assert display_team_label(fallback_row) == "Unknown team 4"


def test_name_lookup_can_fill_from_feature_store_when_action_predictions_lack_names():
    actions = pd.DataFrame({"player_id": [9], "team_id": [5]})
    feature_frame = pd.DataFrame(
        {
            "player_id": [9, 9],
            "player_name": ["Feature Player", "Feature Player"],
            "team_id": [5, 5],
            "team_name": ["Feature Team", "Feature Team"],
        }
    )

    lookup = build_player_team_lookup(actions, feature_frame)

    assert lookup["name_source_used"] == "feature_store"
    assert lookup["player_names"][9] == "Feature Player"
    assert lookup["team_names"][5] == "Feature Team"


def test_portfolio_can_enrich_names_from_sqlite_when_parquet_names_are_absent(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    actions = _actions().drop(columns=["player_name", "team_name"])
    actions.to_parquet(paths.action_predictions, index=False)
    paths.database_path.parent.mkdir(parents=True)
    with sqlite3.connect(paths.database_path) as conn:
        conn.execute("CREATE TABLE players (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.execute("CREATE TABLE teams (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO players (id, name) VALUES (?, ?)",
            [(10, "Database Player"), (14, "Database Finisher")],
        )
        conn.executemany(
            "INSERT INTO teams (id, name) VALUES (?, ?)",
            [(1, "Database Team"), (3, "Database Third")],
        )

    outputs = build_cxa_portfolio_summary(paths=paths, top_n_players=4, top_n_teams=3)
    players = pd.read_csv(outputs["top_players_csv"])
    teams = pd.read_csv(outputs["top_teams_csv"])

    assert "Database Player" in set(players["player_name"])
    assert "Database Team" in set(teams["team_name"])


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
    assert headline["name_source_used"] == "action_predictions"
    assert headline["player_name_coverage"] < 1.0
    assert headline["team_name_coverage"] < 1.0
    assert headline["name_quality_warnings"]
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
