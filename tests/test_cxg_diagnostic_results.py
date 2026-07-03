import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.generate_cxg_diagnostic_results import (
    ResultPaths,
    _blocked_limitations,
    build_entity_summary,
    generate_cxg_diagnostic_results,
    join_baseline_predictions,
    prediction_quality_checks,
    selected_feature_columns,
)


class DummyCxGModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = (0.04 + (0.01 * X["shot_angle"].astype(float))).clip(0.01, 0.99)
        return np.column_stack([1.0 - probs, probs])


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "shot_id": i,
                "event_id": f"event-{i}",
                "match_id": i // 2,
                "team_id": 10 + (i % 2),
                "team_name": f"Team {i % 2}",
                "player_id": 100 + (i % 3),
                "player_name": f"Player {i % 3}",
                "opponent_team_id": 20 + (i % 2),
                "opponent_team_name": f"Opp {i % 2}",
                "is_goal": int(i in {1, 4}),
                "shot_distance": 8.0 + i,
                "shot_angle": 0.2 + (i * 0.03),
                "body_part": "Head" if i % 3 == 0 else "Right Foot",
                "technique": "Normal",
                "shot_type": "Open Play",
                "play_pattern": "Regular Play",
                "under_pressure": i % 2 == 0,
                "pressure_state": "under_pressure" if i % 2 == 0 else "no_pressure",
                "minute": 10 + i,
                "score_state": "drawing",
                "def_label": "average",
                "statsbomb_xg": 0.1,
            }
            for i in range(6)
        ]
    )


def _write_result_artifacts(
    root: Path, recommendation: str = "promote"
) -> tuple[ResultPaths, Path]:
    diagnostic_dir = root / "outputs" / "modeling" / "cxg" / "diagnostic_v1"
    validation_dir = root / "outputs" / "validation" / "cxg" / "diagnostic_v1"
    baseline_dir = root / "outputs" / "modeling" / "cxg" / "baseline"
    output_dir = root / "outputs" / "results" / "cxg" / "diagnostic_v1"
    feature_path = root / "feature_store" / "cxg" / "shot_features.parquet"
    for directory in (
        diagnostic_dir / "models",
        diagnostic_dir / "contracts",
        diagnostic_dir / "diagnostics",
        validation_dir,
        baseline_dir / "predictions",
        baseline_dir / "aggregates",
        baseline_dir / "reports",
        feature_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    _feature_frame().to_parquet(feature_path, index=False)
    joblib.dump(DummyCxGModel(), diagnostic_dir / "models" / "selected_model.joblib")
    metadata = {
        "selected_model": "diagnostic_logistic",
        "model_candidates": [
            {
                "name": "diagnostic_logistic",
                "features": {
                    "numeric": ["shot_distance", "shot_angle"],
                    "binary": [],
                    "categorical": [],
                },
            }
        ],
    }
    (diagnostic_dir / "models" / "selected_model_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    (diagnostic_dir / "contracts" / "feature_contract.json").write_text(
        json.dumps(
            {
                "reference_only_columns": ["statsbomb_xg"],
                "excluded_leakage_columns": ["outcome"],
            }
        ),
        encoding="utf-8",
    )
    resolved = {
        "source_available": {
            "numeric": ["shot_distance", "shot_angle"],
            "binary": [],
            "categorical": [],
        },
        "synthetic_default_features": {"numeric": [], "binary": [], "categorical": []},
        "synthetic_default_excluded": {"numeric": [], "binary": [], "categorical": []},
        "training_features": {
            "numeric": ["shot_distance", "shot_angle"],
            "binary": [],
            "categorical": [],
        },
    }
    (diagnostic_dir / "diagnostics" / "resolved_features.json").write_text(
        json.dumps(resolved),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "feature": "shot_distance",
                "feature_group": "numeric",
                "availability_source": "source",
            },
            {
                "feature": "statsbomb_xg",
                "feature_group": "reference",
                "availability_source": "source",
            },
        ]
    ).to_csv(diagnostic_dir / "diagnostics" / "feature_group_summary.csv", index=False)
    pd.DataFrame([{"column": "statsbomb_xg", "reason": "reference_only"}]).to_csv(
        diagnostic_dir / "diagnostics" / "excluded_columns.csv",
        index=False,
    )

    (validation_dir / "promotion_recommendation.json").write_text(
        json.dumps({"recommendation": recommendation, "known_limitations": []}),
        encoding="utf-8",
    )
    (validation_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "promotion_recommendation": recommendation,
                "selected_diagnostic_model": "diagnostic_logistic",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"model_version": "baseline", "brier": 0.08, "log_loss": 0.3},
            {"model_version": "diagnostic_v1:diagnostic_logistic", "brier": 0.07, "log_loss": 0.28},
        ]
    ).to_csv(validation_dir / "model_comparison_validation.csv", index=False)
    pd.DataFrame().to_csv(validation_dir / "fold_stability.csv", index=False)
    pd.DataFrame().to_csv(validation_dir / "calibration_bins.csv", index=False)
    pd.DataFrame().to_csv(validation_dir / "slice_calibration.csv", index=False)
    (validation_dir / "validation_report.md").write_text("# Validation", encoding="utf-8")

    baseline = _feature_frame()[["shot_id"]].copy()
    baseline["cxg_raw"] = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    baseline.to_parquet(baseline_dir / "predictions" / "shot_predictions.parquet", index=False)
    pd.DataFrame().to_parquet(baseline_dir / "aggregates" / "player_cxg.parquet", index=False)
    pd.DataFrame().to_parquet(baseline_dir / "aggregates" / "team_cxg.parquet", index=False)
    (baseline_dir / "reports" / "metrics.json").write_text("{}", encoding="utf-8")

    return (
        ResultPaths.from_roots(
            diagnostic_dir=diagnostic_dir,
            validation_dir=validation_dir,
            baseline_dir=baseline_dir,
            output_dir=output_dir,
        ),
        feature_path,
    )


def test_promotion_gate_allows_promote_and_writes_outputs(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="promote")

    outputs = generate_cxg_diagnostic_results(input_path=feature_path, paths=paths)

    assert outputs["shot_predictions"].exists()
    assert outputs["player_cxg_summary_csv"].exists()
    assert outputs["team_cxg_summary_csv"].exists()
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))
    assert summary["promotion_status"] == "promoted"
    assert summary["promotion_gate_passed"] is True
    assert summary["validation_model_matches_selected"] is True
    assert summary["validation_selected_model"] == "diagnostic_logistic"
    assert summary["current_selected_model"] == "diagnostic_logistic"
    assert summary["stale_validation_detected"] is False


def test_promotion_gate_allows_provisional_promote(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="provisional_promote")

    outputs = generate_cxg_diagnostic_results(input_path=feature_path, paths=paths)
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))

    assert outputs["shot_predictions"].exists()
    assert summary["promotion_status"] == "provisionally_promoted"


def test_promoted_player_summary_uses_real_player_ids(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="promote")

    outputs = generate_cxg_diagnostic_results(input_path=feature_path, paths=paths)
    player_summary = pd.read_csv(outputs["player_cxg_summary_csv"])

    assert not player_summary["player_id"].isna().any()
    assert set(player_summary["player_id"]).issubset(set(_feature_frame()["player_id"]))


def test_promotion_gate_blocks_rejected_recommendations(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="needs_revision")

    outputs = generate_cxg_diagnostic_results(input_path=feature_path, paths=paths)

    assert "shot_predictions" not in outputs
    assert outputs["model_promotion_summary"].exists()
    assert outputs["results_report"].exists()
    assert not (paths.output_dir / "shot_predictions.parquet").exists()
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))
    assert summary["promotion_status"] == "blocked"


def test_allow_non_promoted_generates_exploratory_outputs(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="needs_revision")

    outputs = generate_cxg_diagnostic_results(
        input_path=feature_path,
        paths=paths,
        allow_non_promoted=True,
    )
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))

    assert outputs["shot_predictions"].exists()
    assert summary["promotion_status"] == "exploratory"
    shots = pd.read_parquet(outputs["shot_predictions"])
    assert set(shots["prediction_source"]) == {"exploratory_model"}


def test_promoted_outputs_use_promoted_prediction_source(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="promote")

    outputs = generate_cxg_diagnostic_results(input_path=feature_path, paths=paths)
    shots = pd.read_parquet(outputs["shot_predictions"])

    assert set(shots["prediction_source"]) == {"promoted_model"}


def test_missing_validation_summary_blocks_promoted_outputs(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="promote")
    paths.validation_summary.unlink()

    outputs = generate_cxg_diagnostic_results(input_path=feature_path, paths=paths)
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))

    assert "shot_predictions" not in outputs
    assert summary["promotion_status"] == "blocked"
    assert summary["validation_selected_model"] is None
    assert summary["validation_model_matches_selected"] is False
    assert summary["stale_validation_detected"] is True
    assert any("missing or stale" in item for item in summary["known_limitations"])


def test_missing_validation_selected_model_blocks_promoted_outputs(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="promote")
    paths.validation_summary.write_text(
        json.dumps({"promotion_recommendation": "promote"}),
        encoding="utf-8",
    )

    outputs = generate_cxg_diagnostic_results(input_path=feature_path, paths=paths)
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))

    assert "shot_predictions" not in outputs
    assert summary["promotion_status"] == "blocked"
    assert summary["validation_selected_model"] is None
    assert summary["validation_model_matches_selected"] is False
    assert summary["stale_validation_detected"] is True
    assert any("missing or stale" in item for item in summary["known_limitations"])


def test_stale_validation_blocks_promoted_outputs(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="promote")
    paths.validation_summary.write_text(
        json.dumps(
            {
                "promotion_recommendation": "promote",
                "selected_diagnostic_model": "old_diagnostic_logistic",
            }
        ),
        encoding="utf-8",
    )

    outputs = generate_cxg_diagnostic_results(input_path=feature_path, paths=paths)
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))

    assert "shot_predictions" not in outputs
    assert summary["promotion_status"] == "blocked"
    assert summary["validation_selected_model"] == "old_diagnostic_logistic"
    assert summary["validation_model_matches_selected"] is False
    assert summary["stale_validation_detected"] is True
    assert any("missing or stale" in item for item in summary["known_limitations"])


def test_stale_validation_allows_exploratory_outputs_with_flag(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="promote")
    paths.validation_summary.write_text(
        json.dumps(
            {
                "promotion_recommendation": "promote",
                "selected_diagnostic_model": "old_diagnostic_logistic",
            }
        ),
        encoding="utf-8",
    )

    outputs = generate_cxg_diagnostic_results(
        input_path=feature_path,
        paths=paths,
        allow_non_promoted=True,
    )
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))

    assert outputs["shot_predictions"].exists()
    assert summary["promotion_status"] == "exploratory"
    assert summary["validation_selected_model"] == "old_diagnostic_logistic"
    assert summary["validation_model_matches_selected"] is False
    assert summary["stale_validation_detected"] is True
    assert any("despite stale validation" in item for item in summary["known_limitations"])
    shots = pd.read_parquet(outputs["shot_predictions"])
    assert set(shots["prediction_source"]) == {"exploratory_model"}


def test_selected_model_metadata_drives_feature_selection(tmp_path: Path):
    paths, _ = _write_result_artifacts(tmp_path)
    metadata = json.loads(paths.selected_model_metadata.read_text(encoding="utf-8"))

    features, grouped = selected_feature_columns(metadata)

    assert features == ["shot_distance", "shot_angle"]
    assert grouped["numeric"] == ["shot_distance", "shot_angle"]


def test_baseline_join_by_shot_id_works(tmp_path: Path):
    shots = _feature_frame()[["shot_id", "is_goal"]].copy()
    shots["predicted_cxg"] = 0.1
    baseline_path = tmp_path / "baseline.parquet"
    pd.DataFrame({"shot_id": shots["shot_id"], "cxg_raw": 0.08}).to_parquet(
        baseline_path,
        index=False,
    )

    joined, metadata = join_baseline_predictions(shots, baseline_path)

    assert metadata["join_key"] == ["shot_id"]
    assert metadata["join_rate"] == 1.0
    assert "cxg_delta_vs_baseline" in joined.columns


def test_missing_baseline_does_not_fail_and_records_zero_join_rate(tmp_path: Path):
    shots = _feature_frame()[["shot_id", "is_goal"]].copy()
    shots["predicted_cxg"] = 0.1

    joined, metadata = join_baseline_predictions(shots, tmp_path / "missing.parquet")

    assert metadata["join_rate"] == 0.0
    assert joined["baseline_cxg"].isna().all()


def test_player_and_team_summaries_aggregate_correctly():
    shots = _feature_frame()
    shots["predicted_cxg"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    shots["baseline_cxg"] = [0.05] * 6

    player = build_entity_summary(shots, "player")
    team = build_entity_summary(shots, "team")

    assert player["shots"].sum() == 6
    assert team["shots"].sum() == 6
    assert np.isclose(player["total_cxg"].sum(), 2.1)
    assert np.isclose(team["total_cxg"].sum(), 2.1)


def test_prediction_quality_checks_catch_invalid_probabilities():
    shots = pd.DataFrame(
        {
            "shot_id": [1, 1],
            "match_id": [1, 1],
            "team_id": [10, 10],
            "player_id": [100, 101],
            "is_goal": [0, 1],
            "predicted_cxg": [1.2, np.nan],
        }
    )

    checks = prediction_quality_checks(
        shots,
        baseline_join_rate=0.0,
        model_loaded=True,
        promotion_gate_passed=True,
        governance_artifacts_present=True,
        validation_recommendation="promote",
    )

    statuses = dict(zip(checks["check_name"], checks["status"], strict=False))
    assert statuses["prediction_null_count"] == "failed"
    assert statuses["outside_0_1_count"] == "failed"
    assert statuses["duplicate_shot_id_count"] == "warning"


def test_governance_missing_blocks_promoted_outputs(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="promote")
    paths.resolved_features.unlink()

    outputs = generate_cxg_diagnostic_results(input_path=feature_path, paths=paths)
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))

    assert "shot_predictions" not in outputs
    assert summary["promotion_status"] == "blocked"
    assert summary["governance_summary"]["status"] == "failed"


def test_governance_missing_blocks_exploratory_outputs(tmp_path: Path):
    paths, feature_path = _write_result_artifacts(tmp_path, recommendation="needs_revision")
    paths.resolved_features.unlink()

    outputs = generate_cxg_diagnostic_results(
        input_path=feature_path,
        paths=paths,
        allow_non_promoted=True,
    )
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))

    assert "shot_predictions" not in outputs
    assert not (paths.output_dir / "shot_predictions.parquet").exists()
    assert summary["promotion_status"] == "blocked"
    assert summary["governance_summary"]["status"] == "failed"
    assert any("governance" in item.lower() for item in summary["known_limitations"])


def test_blocked_limitations_explain_governance_without_validation_blame():
    limitations = _blocked_limitations(
        "promote",
        {
            "missing_governance_artifacts": ["resolved_features.json"],
            "forbidden_features_used": [],
        },
        blockers=["Feature governance failed; scoring outputs are blocked."],
    )

    assert limitations[0] == "Feature governance failed; scoring outputs are blocked."
    assert not any("not allowed for promoted outputs" in item for item in limitations)
