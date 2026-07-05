import json
from pathlib import Path

import pandas as pd

from scripts.validate_cxa_diagnostic_model import (
    CxAValidationPaths,
    calibration_summary,
    detect_baseline_prediction_column,
    join_predictions,
    metric_summary,
    promotion_recommendation,
    threshold_summary,
    validate_cxa_diagnostic,
)


def _prediction_rows(count: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_rows = []
    diagnostic_rows = []
    for idx in range(count):
        positive = int(idx % 10 == 0)
        base_prob = 0.15 if positive else 0.04 + ((idx % 5) * 0.004)
        diag_prob = 0.36 if positive else 0.03 + ((idx % 5) * 0.003)
        common = {
            "action_id": f"action-{idx}",
            "event_id": f"event-{idx}",
            "match_id": idx // 8,
            "team_id": idx % 4,
            "player_id": idx % 20,
            "sequence_id": f"seq-{idx // 4}",
            "possession": idx // 3,
            "shot_created": positive,
            "action_type": "Pass" if idx % 2 == 0 else "Carry",
            "is_pass": idx % 2 == 0,
            "is_carry": idx % 2 == 1,
            "is_progressive": idx % 3 == 0,
            "enters_final_third": idx % 4 == 0,
            "enters_penalty_area": idx % 9 == 0,
            "start_third": "middle" if idx % 2 == 0 else "defensive",
            "end_third": "attacking" if idx % 3 == 0 else "middle",
        }
        baseline_rows.append(
            {
                **common,
                "predicted_shot_created_probability": base_prob,
            }
        )
        diagnostic_rows.append(
            {
                **{
                    key: common[key]
                    for key in (
                        "action_id",
                        "event_id",
                        "match_id",
                        "team_id",
                        "player_id",
                        "sequence_id",
                        "possession",
                        "shot_created",
                    )
                },
                "predicted_shot_created_probability": diag_prob,
                "model_candidate": "calibrated_gradient_boosting_sigmoid",
                "fold": idx % 4,
                "split": "group_kfold_match_id",
            }
        )
    return pd.DataFrame(baseline_rows), pd.DataFrame(diagnostic_rows)


def _write_artifacts(
    root: Path,
    *,
    baseline_count: int = 80,
    bad_probability: bool = False,
    selected_model: str = "calibrated_gradient_boosting_sigmoid",
    write_oof_baseline: bool = False,
    baseline_metrics: dict | None = None,
):
    baseline, diagnostic = _prediction_rows()
    baseline = baseline.head(baseline_count).copy()
    if bad_probability:
        diagnostic.loc[0, "predicted_shot_created_probability"] = 1.5

    diagnostic_dir = root / "outputs" / "modeling" / "cxa" / "diagnostic_v1"
    baseline_dir = root / "outputs" / "modeling" / "cxa"
    output_dir = root / "outputs" / "validation" / "cxa" / "diagnostic_v1"
    (diagnostic_dir / "predictions").mkdir(parents=True)
    (diagnostic_dir / "models").mkdir(parents=True)
    (diagnostic_dir / "reports").mkdir(parents=True)
    (diagnostic_dir / "contracts").mkdir(parents=True)
    (baseline_dir / "predictions").mkdir(parents=True)
    (baseline_dir / "reports").mkdir(parents=True)

    diagnostic.to_parquet(
        diagnostic_dir / "predictions" / "cross_validated_predictions.parquet",
        index=False,
    )
    baseline.to_parquet(baseline_dir / "predictions" / "action_predictions.parquet", index=False)
    if write_oof_baseline:
        baseline.to_parquet(
            baseline_dir / "predictions" / "cross_validated_predictions.parquet",
            index=False,
        )
    (diagnostic_dir / "models" / "selected_model_metadata.json").write_text(
        json.dumps({"selected_model_candidate": selected_model}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"candidate_name": "calibrated_gradient_boosting_sigmoid", "log_loss": 0.14}]
    ).to_csv(diagnostic_dir / "reports" / "model_comparison.csv", index=False)
    (diagnostic_dir / "contracts" / "feature_contract.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (baseline_dir / "reports" / "metrics.json").write_text(
        json.dumps(baseline_metrics or {}),
        encoding="utf-8",
    )

    return CxAValidationPaths.from_roots(
        diagnostic_dir=diagnostic_dir,
        baseline_dir=baseline_dir,
        output_dir=output_dir,
    )


def test_baseline_prediction_column_is_detected():
    assert (
        detect_baseline_prediction_column(pd.DataFrame({"predicted_chance_action": [0.1]}))
        == "predicted_chance_action"
    )
    assert (
        detect_baseline_prediction_column(
            pd.DataFrame(
                {
                    "predicted_shot_created_probability": [0.1],
                    "predicted_chance_action": [True],
                }
            )
        )
        == "predicted_shot_created_probability"
    )


def test_baseline_and_diagnostic_predictions_join_by_action_id():
    baseline, diagnostic = _prediction_rows()
    joined, join_key, quality = join_predictions(
        diagnostic,
        baseline,
        selected_model="calibrated_gradient_boosting_sigmoid",
        baseline_prediction_column="predicted_shot_created_probability",
    )

    assert join_key == ["action_id"]
    assert quality["baseline_join_rate"] == 1.0
    assert len(joined) == len(diagnostic)


def test_metrics_include_average_precision_and_top_k_summaries():
    baseline, diagnostic = _prediction_rows()
    joined, _, _ = join_predictions(
        diagnostic,
        baseline,
        selected_model="calibrated_gradient_boosting_sigmoid",
        baseline_prediction_column="predicted_shot_created_probability",
    )

    metrics = metric_summary(
        joined,
        prediction_column="diagnostic_probability",
        model_name="diagnostic_v1",
    )
    thresholds = threshold_summary(
        joined,
        prediction_column="diagnostic_probability",
        model_name="diagnostic_v1",
    )
    calibration = calibration_summary(
        joined,
        prediction_column="diagnostic_probability",
        model_name="diagnostic_v1",
    )

    assert "average_precision" in metrics
    assert "precision_at_top_1pct" in metrics
    assert {"top_1pct", "top_5pct", "top_10pct"}.issubset(set(thresholds["threshold_or_cut"]))
    assert {"model", "bin", "absolute_gap"}.issubset(calibration.columns)


def test_quality_checks_fail_when_join_rate_is_too_low(tmp_path: Path):
    paths = _write_artifacts(tmp_path, baseline_count=20)
    outputs = validate_cxa_diagnostic(paths, min_slice_rows=1)
    checks = pd.read_csv(outputs["validation_quality_checks"])
    recommendation = json.loads(outputs["promotion_recommendation"].read_text(encoding="utf-8"))

    assert checks.loc[checks["check_name"] == "baseline_join_rate", "status"].item() == "failed"
    assert recommendation["recommendation"] == "blocked"


def test_quality_checks_fail_when_probabilities_are_outside_bounds(tmp_path: Path):
    paths = _write_artifacts(tmp_path, bad_probability=True)
    outputs = validate_cxa_diagnostic(paths, min_slice_rows=1)
    checks = pd.read_csv(outputs["validation_quality_checks"])
    recommendation = json.loads(outputs["promotion_recommendation"].read_text(encoding="utf-8"))

    assert (
        checks.loc[checks["check_name"] == "diagnostic_probability_bounds", "status"].item()
        == "failed"
    )
    assert recommendation["recommendation"] == "blocked"


def test_full_data_baseline_is_reference_only_and_caps_promotion(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    outputs = validate_cxa_diagnostic(paths, min_slice_rows=1)

    summary = json.loads(outputs["validation_summary"].read_text(encoding="utf-8"))
    recommendation = json.loads(outputs["promotion_recommendation"].read_text(encoding="utf-8"))
    checks = pd.read_csv(outputs["validation_quality_checks"])
    report = outputs["validation_report"].read_text(encoding="utf-8")

    assert summary["baseline_prediction_provenance"] == "full_data_in_sample"
    assert summary["baseline_is_fair_comparator"] is False
    assert summary["strict_promotion_comparison_enabled"] is False
    assert recommendation["recommendation"] != "promote"
    assert "reference-only/in-sample" in report
    assert (
        checks.loc[checks["check_name"] == "baseline_prediction_provenance", "status"].item()
        == "warning"
    )


def test_oof_baseline_enables_strict_comparison(tmp_path: Path):
    paths = _write_artifacts(tmp_path, write_oof_baseline=True)
    outputs = validate_cxa_diagnostic(paths, min_slice_rows=1)

    summary = json.loads(outputs["validation_summary"].read_text(encoding="utf-8"))
    checks = pd.read_csv(outputs["validation_quality_checks"])
    report = outputs["validation_report"].read_text(encoding="utf-8")

    assert summary["baseline_prediction_provenance"] == "out_of_fold"
    assert summary["baseline_is_fair_comparator"] is True
    assert summary["strict_promotion_comparison_enabled"] is True
    assert "fair OOF/holdout comparator" in report
    assert (
        checks.loc[checks["check_name"] == "baseline_prediction_provenance", "status"].item()
        == "passed"
    )


def test_stale_selected_candidate_blocks_without_crashing(tmp_path: Path):
    paths = _write_artifacts(tmp_path, selected_model="stale_candidate")

    outputs = validate_cxa_diagnostic(paths, min_slice_rows=1)

    recommendation = json.loads(outputs["promotion_recommendation"].read_text(encoding="utf-8"))
    summary = json.loads(outputs["validation_summary"].read_text(encoding="utf-8"))
    checks = pd.read_csv(outputs["validation_quality_checks"])
    report = outputs["validation_report"].read_text(encoding="utf-8")

    assert recommendation["recommendation"] == "blocked"
    assert summary["selected_diagnostic_model"] == "stale_candidate"
    selected_check = checks.loc[checks["check_name"] == "selected_candidate_match"].iloc[0]
    assert selected_check["status"] == "failed"
    assert selected_check["severity"] == "blocker"
    assert "selected diagnostic model is missing" in report


def test_promotion_recommendation_logic_categories():
    passed_checks = pd.DataFrame(
        [{"check_name": "joined_row_count", "status": "passed", "severity": "info"}]
    )
    empty_slices = pd.DataFrame()
    baseline = {
        "log_loss": 0.20,
        "brier": 0.05,
        "average_precision": 0.30,
        "expected_calibration_error": 0.01,
    }
    promote = {
        "log_loss": 0.19,
        "brier": 0.04,
        "average_precision": 0.35,
        "expected_calibration_error": 0.012,
    }
    provisional = {
        "log_loss": 0.215,
        "brier": 0.055,
        "average_precision": 0.34,
        "expected_calibration_error": 0.015,
    }
    revise = {
        "log_loss": 0.25,
        "brier": 0.08,
        "average_precision": 0.25,
        "expected_calibration_error": 0.03,
    }
    blocked_checks = pd.DataFrame(
        [{"check_name": "baseline_join_rate", "status": "failed", "severity": "blocker"}]
    )

    assert (
        promotion_recommendation(
            baseline_metrics=baseline,
            diagnostic_metrics=promote,
            checks=passed_checks,
            slices=empty_slices,
            strict_promotion_comparison_enabled=True,
        )["recommendation"]
        == "promote"
    )
    assert (
        promotion_recommendation(
            baseline_metrics=baseline,
            diagnostic_metrics=provisional,
            checks=passed_checks,
            slices=empty_slices,
            strict_promotion_comparison_enabled=True,
        )["recommendation"]
        == "provisional_promote"
    )
    assert (
        promotion_recommendation(
            baseline_metrics=baseline,
            diagnostic_metrics=revise,
            checks=passed_checks,
            slices=empty_slices,
            strict_promotion_comparison_enabled=True,
        )["recommendation"]
        == "needs_revision"
    )
    assert (
        promotion_recommendation(
            baseline_metrics=baseline,
            diagnostic_metrics=promote,
            checks=blocked_checks,
            slices=empty_slices,
            strict_promotion_comparison_enabled=True,
        )["recommendation"]
        == "blocked"
    )
    assert (
        promotion_recommendation(
            baseline_metrics=baseline,
            diagnostic_metrics=promote,
            checks=passed_checks,
            slices=empty_slices,
            strict_promotion_comparison_enabled=False,
        )["recommendation"]
        == "provisional_promote"
    )


def test_validation_outputs_are_written_and_missing_optional_slices_do_not_crash(tmp_path: Path):
    paths = _write_artifacts(tmp_path)
    baseline = pd.read_parquet(paths.baseline_predictions)
    baseline = baseline.drop(columns=["start_third", "end_third"])
    baseline.to_parquet(paths.baseline_predictions, index=False)

    outputs = validate_cxa_diagnostic(paths, min_slice_rows=1)

    assert outputs["validation_summary"].exists()
    assert outputs["promotion_recommendation"].exists()
    assert outputs["validation_report"].exists()
    assert outputs["baseline_vs_diagnostic_metrics"].exists()
    assert outputs["calibration_summary"].exists()
    assert outputs["threshold_summary"].exists()
    assert outputs["slice_summary"].exists()
    assert outputs["validation_quality_checks"].exists()

    report = outputs["validation_report"].read_text(encoding="utf-8")
    summary = json.loads(outputs["validation_summary"].read_text(encoding="utf-8"))
    assert "# Diagnostic CxA Validation Report" in report
    assert summary["selected_diagnostic_model"] == "calibrated_gradient_boosting_sigmoid"
