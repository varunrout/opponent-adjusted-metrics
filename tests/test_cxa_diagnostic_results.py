import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.generate_cxa_diagnostic_results import (
    CxAResultPaths,
    generate_cxa_diagnostic_results,
)


class DummyCxAModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        if "created_shot_cxg" in frame.columns:
            raise AssertionError("created_shot_cxg must not be passed as a model feature")
        if "cxa_value" in frame.columns:
            raise AssertionError("cxa_value must not be passed as a model feature")
        base = pd.to_numeric(frame["safe_feature"], errors="coerce").fillna(0.0)
        probabilities = np.clip(0.05 + (base * 0.4), 0.01, 0.95).to_numpy()
        return np.column_stack([1 - probabilities, probabilities])


def _feature_frame(*, include_optional_ids: bool = True) -> pd.DataFrame:
    rows = []
    for idx in range(12):
        row = {
            "action_id": f"action-{idx}",
            "event_id": f"event-{idx}",
            "match_id": idx // 4,
            "team_id": idx % 3,
            "player_id": idx % 5,
            "action_type": "Pass" if idx % 2 == 0 else "Carry",
            "shot_created": int(idx % 5 == 0),
            "safe_feature": float(idx % 4) / 4,
            "created_shot_cxg": 0.2 if idx % 5 == 0 else 0.0,
            "created_shot_id": f"shot-{idx}" if idx % 5 == 0 else None,
            "cxa_value": 0.1,
        }
        if include_optional_ids:
            row["sequence_id"] = f"seq-{idx // 3}"
            row["possession"] = idx // 2
        rows.append(row)
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_artifacts(
    root: Path,
    *,
    recommendation: str = "provisional_promote",
    metadata_model: str = "calibrated_gradient_boosting_sigmoid",
    validation_model: str = "calibrated_gradient_boosting_sigmoid",
    include_contract: bool = True,
    include_optional_ids: bool = True,
) -> CxAResultPaths:
    feature_path = root / "feature_store" / "cxa" / "action_features.parquet"
    diagnostic_dir = root / "outputs" / "modeling" / "cxa" / "diagnostic_v1"
    validation_dir = root / "outputs" / "validation" / "cxa" / "diagnostic_v1"
    output_dir = root / "outputs" / "results" / "cxa" / "diagnostic_v1"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    _feature_frame(include_optional_ids=include_optional_ids).to_parquet(feature_path, index=False)

    (diagnostic_dir / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(DummyCxAModel(), diagnostic_dir / "models" / "selected_model.joblib")
    _write_json(
        diagnostic_dir / "models" / "selected_model_metadata.json",
        {"selected_model_candidate": metadata_model},
    )
    if include_contract:
        _write_json(
            diagnostic_dir / "contracts" / "feature_contract.json",
            {
                "metric": "cxa",
                "model_version": "diagnostic_v1",
                "primary_target": "shot_created",
                "selected_feature_candidates": {
                    "numeric": ["safe_feature", "created_shot_cxg"],
                    "binary": [],
                    "categorical": [],
                },
                "excluded_columns": {
                    "target_columns": ["shot_created"],
                    "reference_only_columns": ["created_shot_cxg", "created_shot_id"],
                    "output_prediction_columns": ["cxa_value"],
                    "identifier_columns": [
                        "action_id",
                        "event_id",
                        "match_id",
                        "team_id",
                        "player_id",
                    ],
                    "leakage_excluded_columns": [],
                    "requires_review_columns": [],
                    "excluded_unknown_columns": [],
                },
            },
        )
    _write_json(
        validation_dir / "validation_summary.json",
        {
            "selected_diagnostic_model": validation_model,
            "baseline_is_fair_comparator": False,
            "strict_promotion_comparison_enabled": False,
            "baseline_prediction_provenance": "full_data_in_sample",
            "metric_deltas": {
                "log_loss": -0.01,
                "brier": -0.002,
                "average_precision": 0.04,
            },
        },
    )
    _write_json(
        validation_dir / "promotion_recommendation.json",
        {
            "recommendation": recommendation,
            "known_limitations": ["Reference-only baseline comparator."],
        },
    )
    pd.DataFrame(
        [
            {
                "metric": "log_loss",
                "baseline": 0.16,
                "diagnostic": 0.14,
                "diagnostic_minus_baseline": -0.02,
            }
        ]
    ).to_csv(validation_dir / "baseline_vs_diagnostic_metrics.csv", index=False)

    return CxAResultPaths.from_roots(
        feature_path=feature_path,
        diagnostic_dir=diagnostic_dir,
        validation_dir=validation_dir,
        output_dir=output_dir,
    )


def test_blocked_recommendation_writes_only_summary_checks_and_report(tmp_path: Path):
    paths = _write_artifacts(tmp_path, recommendation="blocked")

    outputs = generate_cxa_diagnostic_results(paths)

    assert set(outputs) == {
        "model_promotion_summary",
        "prediction_quality_checks",
        "cxa_results_report",
    }
    assert not (paths.output_dir / "action_predictions.parquet").exists()
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))
    assert summary["promotion_status"] == "blocked"
    assert summary["promotion_gate_passed"] is False


def test_provisional_promote_writes_full_outputs_with_expected_status(tmp_path: Path):
    paths = _write_artifacts(tmp_path)

    outputs = generate_cxa_diagnostic_results(paths)

    predictions = pd.read_parquet(outputs["action_predictions"])
    summary = json.loads(outputs["model_promotion_summary"].read_text(encoding="utf-8"))
    checks = pd.read_csv(outputs["prediction_quality_checks"])

    assert summary["promotion_status"] == "provisionally_promoted"
    assert summary["promotion_gate_passed"] is True
    assert summary["baseline_is_fair_comparator"] is False
    assert (predictions["prediction_source"] == "provisional_promoted_model").all()
    assert (predictions["promotion_status"] == "provisionally_promoted").all()
    assert predictions["predicted_shot_created_probability"].between(0, 1).all()
    assert "created_shot_cxg_reference" in predictions.columns
    assert checks.loc[checks["check_name"] == "provisional_promotion", "status"].item() == "warning"


def test_selected_model_mismatch_blocks_full_outputs(tmp_path: Path):
    paths = _write_artifacts(tmp_path, validation_model="stale_model")

    outputs = generate_cxa_diagnostic_results(paths)
    checks = pd.read_csv(outputs["prediction_quality_checks"])

    assert not (paths.output_dir / "action_predictions.parquet").exists()
    assert (
        checks.loc[checks["check_name"] == "selected_model_matches_validation", "status"].item()
        == "failed"
    )


def test_missing_governance_artifact_blocks_full_outputs(tmp_path: Path):
    paths = _write_artifacts(tmp_path, include_contract=False)

    outputs = generate_cxa_diagnostic_results(paths)
    checks = pd.read_csv(outputs["prediction_quality_checks"])

    assert not (paths.output_dir / "action_predictions.parquet").exists()
    assert checks.loc[checks["check_name"] == "feature_contract", "status"].item() == "failed"


def test_aggregates_reconcile_to_action_level_total(tmp_path: Path):
    paths = _write_artifacts(tmp_path)

    outputs = generate_cxa_diagnostic_results(paths)

    predictions = pd.read_parquet(outputs["action_predictions"])
    players = pd.read_csv(outputs["player_cxa_summary_csv"])
    teams = pd.read_csv(outputs["team_cxa_summary_csv"])
    sequences = pd.read_csv(outputs["sequence_cxa_summary_csv"])
    total = predictions["diagnostic_cxa"].sum()

    assert players["total_diagnostic_cxa"].sum() == pytest.approx(total)
    assert teams["total_diagnostic_cxa"].sum() == pytest.approx(total)
    assert sequences["total_diagnostic_cxa"].sum() == pytest.approx(total)
    assert players["player_id"].notna().all()


def test_report_mentions_provisional_promotion_and_baseline_caveat(tmp_path: Path):
    paths = _write_artifacts(tmp_path)

    outputs = generate_cxa_diagnostic_results(paths)
    report = outputs["cxa_results_report"].read_text(encoding="utf-8")

    assert "provisionally_promoted" in report
    assert "full-data/in-sample" in report
    assert "`created_shot_cxg`, `cxa_value`" in report


def test_missing_optional_ids_do_not_crash_result_generation(tmp_path: Path):
    paths = _write_artifacts(tmp_path, include_optional_ids=False)

    outputs = generate_cxa_diagnostic_results(paths)

    predictions = pd.read_parquet(outputs["action_predictions"])
    sequences = pd.read_csv(outputs["sequence_cxa_summary_csv"])

    assert len(predictions) == 12
    assert sequences.empty
