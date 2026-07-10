import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run_cxa_plus_diagnostic_training import (
    TARGET_COLUMN,
    assert_feature_leakage_guard,
    grouped_train_test_split,
    run_training,
    selected_features_from_summary,
)


def _matrix() -> pd.DataFrame:
    rows = []
    for match_id in range(1, 9):
        for position in range(1, 7):
            positive = int(position == 6 or (match_id % 2 == 0 and position == 5))
            rows.append(
                {
                    "action_id": f"{match_id}-{position}",
                    "event_id": f"event-{match_id}-{position}",
                    "match_id": match_id,
                    "possession": match_id * 10,
                    "sequence_id": f"seq-{match_id}",
                    "team_id": match_id + 100,
                    "player_id": position + 1000,
                    "action_position": position,
                    TARGET_COLUMN: positive,
                    "length": float(position * 3 + match_id),
                    "x_progression": float(position) / 10.0,
                    "is_pass": int(position % 2 == 0),
                    "enters_final_third": int(position >= 4),
                    "action_type": "Pass" if position % 2 == 0 else "Carry",
                    "start_zone": "middle" if position < 4 else "final",
                    "shot_created": positive,
                    "created_shot_cxg": 0.99,
                    "future_leak": positive,
                    "model_score": 0.8,
                }
            )
    return pd.DataFrame(rows)


def _write_inputs(root: Path, *, features: list[str] | None = None) -> tuple[Path, Path]:
    matrix_path = (
        root
        / "outputs"
        / "modeling"
        / "cxa_plus"
        / "diagnostic_v1"
        / "datasets"
        / "feature_matrix.parquet"
    )
    summary_path = (
        root
        / "outputs"
        / "modeling"
        / "cxa_plus"
        / "diagnostic_v1"
        / "datasets"
        / "feature_matrix_summary.json"
    )
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    frame = _matrix()
    frame.to_parquet(matrix_path, index=False)
    summary = {
        "metric": "cxa_plus",
        "model_version": "diagnostic_v1",
        "primary_target": TARGET_COLUMN,
        "eligible_model_features": features
        or [
            "length",
            "x_progression",
            "is_pass",
            "enters_final_third",
            "action_type",
            "start_zone",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return matrix_path, summary_path


def test_selected_features_are_loaded_only_from_summary():
    summary = {"eligible_model_features": ["length", "is_pass"]}

    assert selected_features_from_summary(summary) == ["length", "is_pass"]


@pytest.mark.parametrize(
    "feature",
    [
        TARGET_COLUMN,
        "shot_created",
        "created_shot_id",
        "created_shot_cxg",
        "shot_within_next_3_actions",
        "discounted_downstream_value",
        "future_label",
        "shot_outcome",
        "result_name",
        "predicted_cxa_plus",
        "model_score",
        "event_probability",
        "score_state",
        "action_id",
        "event_id",
        "match_id",
        "possession",
        "sequence_id",
        "team_id",
        "player_id",
    ],
)
def test_target_leakage_reference_and_identifier_columns_cannot_be_features(feature: str):
    with pytest.raises(ValueError, match="leakage guard failed"):
        assert_feature_leakage_guard(["length", feature])


def test_grouped_split_has_no_shared_match_id():
    frame = _matrix()

    train_idx, test_idx = grouped_train_test_split(frame, random_state=3)

    train_matches = set(frame.iloc[train_idx]["match_id"])
    test_matches = set(frame.iloc[test_idx]["match_id"])
    assert train_matches.isdisjoint(test_matches)
    assert frame.iloc[train_idx][TARGET_COLUMN].sum() > 0
    assert frame.iloc[test_idx][TARGET_COLUMN].sum() > 0


def test_training_writes_model_metrics_calibration_and_report(tmp_path: Path):
    matrix_path, summary_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "outputs" / "modeling" / "cxa_plus" / "diagnostic_v1"

    outputs = run_training(
        matrix_path=matrix_path,
        summary_path=summary_path,
        output_dir=output_dir,
        random_state=3,
    )

    for path in outputs.values():
        assert path.exists()
    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    assert metrics["target"] == TARGET_COLUMN
    assert metrics["feature_count"] == 6
    assert metrics["split"]["shared_match_count"] == 0
    assert metrics["promotion_status"] == "not_promoted"
    for metric in (
        "log_loss",
        "brier",
        "roc_auc",
        "average_precision",
        "positive_rate",
        "baseline_log_loss",
        "baseline_brier",
        "log_loss_lift_over_baseline",
        "brier_lift_over_baseline",
    ):
        assert metric in metrics["metrics"]

    calibration = pd.read_csv(outputs["calibration"])
    assert {
        "bin",
        "row_count",
        "mean_predicted_probability",
        "absolute_calibration_error",
    }.issubset(calibration.columns)
    coefficients = pd.read_csv(outputs["coefficients"])
    assert not coefficients["feature"].str.contains("future_leak|created_shot|model_score").any()
    assert not (output_dir / "results" / "model_promotion_summary.json").exists()
    assert not (output_dir / "results" / "promotion_recommendation.json").exists()


def test_training_uses_summary_allowlist_not_extra_dataframe_columns(tmp_path: Path):
    matrix_path, summary_path = _write_inputs(tmp_path, features=["length", "is_pass"])
    output_dir = tmp_path / "outputs" / "modeling" / "cxa_plus" / "diagnostic_v1"

    outputs = run_training(
        matrix_path=matrix_path,
        summary_path=summary_path,
        output_dir=output_dir,
        random_state=3,
    )

    metrics = json.loads(outputs["metrics"].read_text(encoding="utf-8"))
    coefficients = pd.read_csv(outputs["coefficients"])
    assert metrics["feature_count"] == 2
    assert not coefficients["feature"].str.contains("x_progression|action_type|future_leak").any()


def test_training_fails_when_summary_allowlist_contains_forbidden_column(tmp_path: Path):
    matrix_path, summary_path = _write_inputs(tmp_path, features=["length", "created_shot_cxg"])

    with pytest.raises(ValueError, match="leakage guard failed"):
        run_training(
            matrix_path=matrix_path, summary_path=summary_path, output_dir=tmp_path / "out"
        )


def test_training_fails_when_summary_allowlist_contains_shot_created(tmp_path: Path):
    matrix_path, summary_path = _write_inputs(tmp_path, features=["length", "shot_created"])

    with pytest.raises(ValueError, match="leakage guard failed"):
        run_training(
            matrix_path=matrix_path, summary_path=summary_path, output_dir=tmp_path / "out"
        )


def test_training_script_does_not_import_dashboard_or_promotion_logic():
    source = Path("scripts/run_cxa_plus_diagnostic_training.py").read_text(encoding="utf-8")

    assert "streamlit" not in source
    assert "app/streamlit_app" not in source
    assert "generate_cxa_diagnostic_results" not in source
    assert "promotion_recommendation" not in source
