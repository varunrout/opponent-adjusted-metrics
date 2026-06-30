import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.validate_cxg_diagnostic_model import (
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    ValidationPaths,
    calibration_bins,
    normalize_diagnostic_predictions,
    promotion_recommendation,
    safe_metric_summary,
    slice_calibration,
    validate_diagnostic_cxg,
)


def _baseline_predictions() -> pd.DataFrame:
    rows = []
    for i in range(48):
        goal = int(i % 6 == 0)
        rows.append(
            {
                "shot_id": i,
                "event_id": f"event-{i}",
                "match_id": i // 6,
                "team_id": 10 + (i % 3),
                "player_id": 100 + (i % 5),
                "is_goal": goal,
                "cxg_raw": 0.08 + (0.2 * goal) + ((i % 4) * 0.01),
                "body_part": "Head" if i % 8 == 0 else "Right Foot",
                "technique": "Normal",
                "shot_type": "Free Kick" if i % 10 == 0 else "Open Play",
                "pressure_state": "under_pressure" if i % 3 == 0 else "no_pressure",
                "under_pressure": i % 3 == 0,
                "minute_bucket": "0-15" if i < 24 else "46-60",
                "score_state": "drawing" if i % 2 else "leading",
            }
        )
    return pd.DataFrame(rows)


def _diagnostic_predictions() -> pd.DataFrame:
    baseline = _baseline_predictions()
    rows = []
    for candidate, offset in (("diagnostic_logistic", -0.01), ("extra_trees", 0.04)):
        for record in baseline.to_dict(orient="records"):
            rows.append(
                {
                    "shot_id": record["shot_id"],
                    "event_id": record["event_id"],
                    "match_id": record["match_id"],
                    "team_id": record["team_id"],
                    "player_id": record["player_id"],
                    "is_goal": record["is_goal"],
                    "predicted_cxg": min(max(record["cxg_raw"] + offset, 0.01), 0.99),
                    "model_candidate": candidate,
                    "fold": int(record["match_id"]) % 4,
                    "prediction_source": "cross_validated",
                }
            )
    return pd.DataFrame(rows)


def _write_artifacts(root: Path) -> ValidationPaths:
    baseline_dir = root / "outputs" / "modeling" / "cxg" / "baseline"
    diagnostic_dir = root / "outputs" / "modeling" / "cxg" / "diagnostic_v1"
    analysis_dir = root / "outputs" / "analysis" / "cxg"
    output_dir = root / "outputs" / "validation" / "cxg" / "diagnostic_v1"

    (baseline_dir / "predictions").mkdir(parents=True)
    (baseline_dir / "reports").mkdir(parents=True)
    (baseline_dir / "models").mkdir(parents=True)
    (diagnostic_dir / "predictions").mkdir(parents=True)
    (diagnostic_dir / "reports").mkdir(parents=True)
    (diagnostic_dir / "models").mkdir(parents=True)
    (diagnostic_dir / "contracts").mkdir(parents=True)
    (diagnostic_dir / "diagnostics").mkdir(parents=True)
    (analysis_dir / "04_slice_stability").mkdir(parents=True)
    (analysis_dir / "06_leakage_checks").mkdir(parents=True)

    _baseline_predictions().to_parquet(
        baseline_dir / "predictions" / "shot_predictions.parquet", index=False
    )
    _diagnostic_predictions().to_parquet(
        diagnostic_dir / "predictions" / "cross_validated_predictions.parquet",
        index=False,
    )
    (baseline_dir / "reports" / "metrics.json").write_text("{}", encoding="utf-8")
    (baseline_dir / "reports" / "validation_summary.json").write_text(
        "{}",
        encoding="utf-8",
    )
    pd.DataFrame().to_csv(baseline_dir / "reports" / "calibration_table.csv", index=False)
    pd.DataFrame().to_csv(baseline_dir / "reports" / "slice_metrics.csv", index=False)
    (baseline_dir / "models" / "contextual_model.json").write_text("{}", encoding="utf-8")

    selected = {
        "selected_model": "diagnostic_logistic",
        "resolved_features": {
            "source_available": {"numeric": ["shot_distance"], "binary": [], "categorical": []},
            "synthetic_default_excluded": {
                "numeric": ["opponent_def_rating_global"],
                "binary": [],
                "categorical": ["def_label"],
            },
            "training_features": {
                "numeric": ["shot_distance"],
                "binary": [],
                "categorical": [],
            },
        },
    }
    (diagnostic_dir / "models" / "selected_model_metadata.json").write_text(
        json.dumps(selected),
        encoding="utf-8",
    )
    (diagnostic_dir / "reports" / "training_summary.json").write_text(
        json.dumps({"selected_model": "diagnostic_logistic"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "model_candidate": "diagnostic_logistic",
                "fold": fold,
                "brier": 0.07 + (fold * 0.001),
                "log_loss": 0.25 + (fold * 0.002),
                "roc_auc": 0.78 + (fold * 0.002),
            }
            for fold in range(1, 5)
        ]
        + [
            {
                "model_candidate": "extra_trees",
                "fold": fold,
                "brier": 0.09,
                "log_loss": 0.32,
                "roc_auc": 0.7,
            }
            for fold in range(1, 5)
        ]
    ).to_csv(diagnostic_dir / "reports" / "fold_metrics.csv", index=False)
    pd.DataFrame([{"model_candidate": "diagnostic_logistic", "log_loss_mean": 0.25}]).to_csv(
        diagnostic_dir / "reports" / "model_comparison.csv", index=False
    )
    (diagnostic_dir / "contracts" / "feature_contract.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (diagnostic_dir / "diagnostics" / "resolved_features.json").write_text(
        json.dumps(selected["resolved_features"]),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "feature": "opponent_def_rating_global",
                "excluded_reason": "constant_synthetic_default",
            }
        ]
    ).to_csv(diagnostic_dir / "diagnostics" / "feature_group_summary.csv", index=False)
    pd.DataFrame([{"column": "statsbomb_xg", "reason": "reference_only"}]).to_csv(
        diagnostic_dir / "diagnostics" / "excluded_columns.csv",
        index=False,
    )
    (analysis_dir / "report.md").write_text("# CxG analysis", encoding="utf-8")

    return ValidationPaths.from_roots(
        baseline_dir=baseline_dir,
        diagnostic_dir=diagnostic_dir,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
    )


def test_metric_helpers_compute_probability_metrics_safely():
    frame = pd.DataFrame(
        {
            TARGET_COLUMN: [0, 1, 0, 1],
            PREDICTION_COLUMN: [0.1, 0.8, 0.2, 0.7],
        }
    )

    metrics = safe_metric_summary(frame, model_version="test")

    assert metrics["row_count"] == 4
    assert metrics["brier"] < 0.1
    assert metrics["log_loss"] < 0.5
    assert metrics["roc_auc_status"] == "computed"
    assert metrics["probability_clipping_used"] is False


def test_roc_auc_skips_single_class_slices_safely():
    frame = pd.DataFrame(
        {
            TARGET_COLUMN: [0, 0, 0],
            PREDICTION_COLUMN: [0.1, 0.2, 0.3],
        }
    )

    metrics = safe_metric_summary(frame, model_version="single")

    assert np.isnan(metrics["roc_auc"])
    assert metrics["roc_auc_status"] == "skipped_single_class"


def test_calibration_bins_are_generated_correctly():
    frame = pd.DataFrame(
        {
            TARGET_COLUMN: [0, 1, 0, 1],
            PREDICTION_COLUMN: [0.05, 0.15, 0.55, 0.95],
        }
    )

    bins = calibration_bins(frame, model_version="test", n_bins=5)

    assert len(bins) == 5
    assert bins["rows"].sum() == 4
    assert {"calibration_error", "absolute_calibration_error"}.issubset(bins.columns)


def test_slice_calibration_handles_missing_slice_columns():
    frame = pd.DataFrame(
        {
            "model_version": ["m"] * 4,
            TARGET_COLUMN: [0, 1, 0, 1],
            PREDICTION_COLUMN: [0.1, 0.8, 0.2, 0.7],
            "body_part": ["Head", "Foot", "Foot", "Head"],
        }
    )

    slices, missing = slice_calibration([frame], slice_columns=("body_part", "shot_type"))

    assert not slices.empty
    assert missing == ["shot_type"]


def test_diagnostic_predictions_are_filtered_to_selected_model_only():
    selected = normalize_diagnostic_predictions(
        _diagnostic_predictions(),
        selected_model="diagnostic_logistic",
        baseline_context=_baseline_predictions(),
    )

    assert set(selected["model_candidate"]) == {"diagnostic_logistic"}
    assert "body_part" in selected.columns
    assert len(selected) == len(_baseline_predictions())


def test_promotion_recommendation_categories():
    baseline = {
        "log_loss": 0.3,
        "brier": 0.08,
        "expected_calibration_error": 0.04,
        "probability_null_count": 0,
        "probability_min": 0.01,
        "probability_max": 0.9,
    }
    good = {
        **baseline,
        "log_loss": 0.28,
        "brier": 0.07,
        "expected_calibration_error": 0.035,
    }
    leakage = {"status": "passed"}
    empty_slices = pd.DataFrame()

    assert (
        promotion_recommendation(
            baseline,
            good,
            selected_fold_status="stable",
            leakage=leakage,
            slice_calibration_df=empty_slices,
        )["recommendation"]
        == "promote"
    )

    mixed = {**baseline, "log_loss": 0.31, "brier": 0.079}
    assert (
        promotion_recommendation(
            baseline,
            mixed,
            selected_fold_status="moderate_variance",
            leakage=leakage,
            slice_calibration_df=empty_slices,
        )["recommendation"]
        == "provisional_promote"
    )

    ece_only = {
        **baseline,
        "log_loss": 0.34,
        "brier": 0.09,
        "expected_calibration_error": 0.035,
    }
    assert (
        promotion_recommendation(
            baseline,
            ece_only,
            selected_fold_status="stable",
            leakage=leakage,
            slice_calibration_df=empty_slices,
        )["recommendation"]
        == "needs_revision"
    )

    worse = {**baseline, "log_loss": 0.34, "brier": 0.09, "expected_calibration_error": 0.08}
    assert (
        promotion_recommendation(
            baseline,
            worse,
            selected_fold_status="stable",
            leakage=leakage,
            slice_calibration_df=empty_slices,
        )["recommendation"]
        == "needs_revision"
    )

    broken = {**good, "probability_max": 1.2}
    assert (
        promotion_recommendation(
            baseline,
            broken,
            selected_fold_status="stable",
            leakage=leakage,
            slice_calibration_df=empty_slices,
        )["recommendation"]
        == "do_not_promote"
    )

    governance_unknown = {
        "status": "unknown",
        "missing_governance_artifacts": ["resolved_features.json"],
    }
    assert (
        promotion_recommendation(
            baseline,
            good,
            selected_fold_status="stable",
            leakage=governance_unknown,
            slice_calibration_df=empty_slices,
        )["recommendation"]
        == "do_not_promote"
    )


def test_missing_governance_artifacts_block_promotion(tmp_path: Path):
    for missing_name, path_attr in (
        ("resolved_features.json", "diagnostic_resolved_features"),
        ("excluded_columns.csv", "diagnostic_excluded_columns"),
        ("feature_group_summary.csv", "diagnostic_feature_group_summary"),
    ):
        paths = _write_artifacts(tmp_path / missing_name.replace(".", "_"))
        getattr(paths, path_attr).unlink()

        outputs = validate_diagnostic_cxg(paths, n_bins=5, make_plots=False)
        summary = json.loads(outputs["validation_summary"].read_text(encoding="utf-8"))
        recommendation = json.loads(outputs["promotion_recommendation"].read_text(encoding="utf-8"))

        assert summary["governance_status"] == "unknown"
        assert summary["leakage_status"] == "unknown"
        assert any(missing_name in item for item in summary["missing_governance_artifacts"])
        assert recommendation["recommendation"] == "do_not_promote"
        assert recommendation["known_limitations"]


def test_validation_command_writes_expected_outputs(tmp_path: Path):
    paths = _write_artifacts(tmp_path)

    outputs = validate_diagnostic_cxg(paths, n_bins=5, make_plots=False)

    expected = {
        "validation_summary",
        "model_comparison_validation",
        "fold_stability",
        "calibration_bins",
        "slice_calibration",
        "promotion_recommendation",
        "validation_report",
        "calibration_curve",
        "predicted_vs_actual_by_slice",
    }
    assert set(outputs) == expected
    for key, path in outputs.items():
        if key not in {"calibration_curve", "predicted_vs_actual_by_slice"}:
            assert path.exists()

    summary = json.loads(outputs["validation_summary"].read_text(encoding="utf-8"))
    comparison = pd.read_csv(outputs["model_comparison_validation"])
    recommendation = json.loads(outputs["promotion_recommendation"].read_text(encoding="utf-8"))

    assert summary["selected_diagnostic_model"] == "diagnostic_logistic"
    assert set(comparison["model_version"]) == {"baseline", "diagnostic_v1:diagnostic_logistic"}
    assert recommendation["recommendation"] in {
        "promote",
        "provisional_promote",
        "needs_revision",
        "do_not_promote",
    }
