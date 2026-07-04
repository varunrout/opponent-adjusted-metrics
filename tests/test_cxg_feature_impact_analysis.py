import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.analyze_cxg_feature_impact import (
    FeatureImpactPaths,
    align_selected_feature_matrix,
    analyze_cxg_feature_impact,
    category_lift_table,
    group_perturbation_summary,
    map_feature_groups,
    model_impact_features,
    result_integrity_checks,
    selected_features_from_metadata,
)


class DummyImpactModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        distance = frame["shot_distance"].astype(float).to_numpy()
        angle = frame["shot_angle"].astype(float).to_numpy()
        first_time = frame["first_time"].astype(float).to_numpy()
        logits = -0.12 * distance + 0.9 * angle + 0.35 * first_time
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probabilities, probabilities])


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": [1, 2, 3, 4, 5, 6],
            "event_id": ["a", "b", "c", "d", "e", "f"],
            "match_id": [10, 10, 11, 11, 12, 12],
            "team_id": [100, 100, 101, 101, 102, 102],
            "team_name": ["A", "A", "B", "B", "C", "C"],
            "player_id": [1000, 1001, 1002, 1003, 1004, 1005],
            "player_name": ["P0", "P1", "P2", "P3", "P4", "P5"],
            "opponent_team_id": [101, 101, 100, 100, 100, 100],
            "opponent_team_name": ["B", "B", "A", "A", "A", "A"],
            "is_goal": [0, 1, 0, 0, 1, 0],
            "shot_distance": [24.0, 8.0, 18.0, 14.0, 7.0, 28.0],
            "shot_angle": [0.12, 0.65, 0.22, 0.3, 0.7, 0.08],
            "first_time": [0, 1, 0, 0, 1, 0],
            "body_part": ["Right Foot", "Head", "Left Foot", "Right Foot", "Head", "Left Foot"],
            "shot_type": [
                "Open Play",
                "Open Play",
                "Free Kick",
                "Open Play",
                "Corner",
                "Open Play",
            ],
            "statsbomb_xg": [0.05, 0.4, 0.08, 0.12, 0.5, 0.02],
        }
    )


def _metadata() -> dict:
    return {
        "selected_model": "calibrated_gradient_boosting_sigmoid",
        "selected_features": [
            "shot_distance",
            "shot_angle",
            "first_time",
            "body_part",
            "shot_type",
        ],
        "selected_feature_groups": {
            "numeric": ["shot_distance", "shot_angle"],
            "binary": ["first_time"],
            "categorical": ["body_part", "shot_type"],
        },
    }


def _write_artifacts(
    root: Path, *, drop_category: str | None = None
) -> tuple[FeatureImpactPaths, Path]:
    diagnostic_dir = root / "outputs" / "modeling" / "cxg" / "diagnostic_v1"
    results_dir = root / "outputs" / "results" / "cxg" / "diagnostic_v1"
    feature_path = root / "feature_store" / "cxg" / "shot_features.parquet"
    for directory in (
        diagnostic_dir / "models",
        diagnostic_dir / "contracts",
        diagnostic_dir / "diagnostics",
        results_dir,
        feature_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    feature_frame = _feature_frame()
    if drop_category:
        feature_frame = feature_frame.drop(columns=[drop_category])
    feature_frame.to_parquet(feature_path, index=False)

    predictions = feature_frame[
        [
            "shot_id",
            "event_id",
            "match_id",
            "team_id",
            "team_name",
            "player_id",
            "player_name",
            "opponent_team_id",
            "opponent_team_name",
            "is_goal",
        ]
    ].copy()
    model = DummyImpactModel()
    predictions["predicted_cxg"] = model.predict_proba(
        feature_frame[_metadata()["selected_features"]]
    )[
        :,
        1,
    ]
    predictions["baseline_cxg"] = predictions["predicted_cxg"] * 0.95
    if drop_category is None:
        predictions["body_part"] = feature_frame["body_part"]
        predictions["shot_type"] = feature_frame["shot_type"]
    predictions.to_parquet(results_dir / "shot_predictions.parquet", index=False)

    joblib.dump(model, diagnostic_dir / "models" / "selected_model.joblib")
    (diagnostic_dir / "models" / "selected_model_metadata.json").write_text(
        json.dumps(_metadata()),
        encoding="utf-8",
    )
    (diagnostic_dir / "contracts" / "feature_contract.json").write_text(
        json.dumps(
            {
                "version": "cxg_diagnostic_v1",
                "target_column": "is_goal",
                "reference_only_columns": ["statsbomb_xg"],
                "excluded_leakage_columns": [],
            }
        ),
        encoding="utf-8",
    )
    (diagnostic_dir / "diagnostics" / "resolved_features.json").write_text(
        json.dumps({"source_available": {"numeric": ["shot_distance", "shot_angle"]}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"feature": "shot_distance", "availability_source": "source"},
            {"feature": "body_part", "availability_source": "source"},
        ]
    ).to_csv(diagnostic_dir / "diagnostics" / "feature_group_summary.csv", index=False)
    (results_dir / "model_promotion_summary.json").write_text(
        json.dumps(
            {
                "promotion_status": "promoted",
                "promotion_gate_passed": True,
                "baseline_comparison": {"baseline_join_rate": 1.0},
                "governance_summary": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"metric": "baseline_log_loss", "value": 0.276},
            {"metric": "diagnostic_log_loss", "value": 0.261},
        ]
    ).to_csv(results_dir / "baseline_vs_diagnostic_summary.csv", index=False)

    paths = FeatureImpactPaths.from_roots(
        diagnostic_dir=diagnostic_dir,
        results_dir=results_dir,
        output_dir=diagnostic_dir / "feature_impact",
    )
    return paths, feature_path


def test_selected_features_are_loaded_from_metadata():
    assert selected_features_from_metadata(_metadata()) == _metadata()["selected_features"]


def test_statsbomb_xg_and_identifiers_are_not_impact_features():
    selected = ["shot_id", "player_id", "shot_distance", "statsbomb_xg"]

    assert model_impact_features(selected) == ["shot_distance"]


def test_feature_grouping_maps_known_selected_features():
    grouped = map_feature_groups(_metadata()["selected_features"])

    assert grouped["geometry"] == ["shot_distance", "shot_angle"]
    assert grouped["shot_execution"] == ["first_time", "body_part", "shot_type"]
    assert grouped["team_player_identifiers_only"] == []


def test_selected_feature_matrix_uses_known_coordinate_aliases():
    frame = pd.DataFrame({"shot_x": [101.0], "shot_y": [40.0], "shot_distance": [12.0]})

    matrix = align_selected_feature_matrix(
        frame,
        ["location_x", "location_y", "shot_distance"],
    )

    assert list(matrix.columns) == ["location_x", "location_y", "shot_distance"]
    assert matrix.loc[0, "location_x"] == 101.0
    assert matrix.loc[0, "location_y"] == 40.0


def test_group_perturbation_produces_one_row_per_non_empty_group():
    frame = _feature_frame()
    metadata = _metadata()
    matrix = frame[metadata["selected_features"]]
    grouped = map_feature_groups(metadata["selected_features"])

    summary = group_perturbation_summary(DummyImpactModel(), matrix, frame["is_goal"], grouped)

    assert set(summary["feature_group"]) == {"geometry", "shot_execution"}


def test_category_lift_table_returns_expected_columns():
    frame = _feature_frame()
    frame["predicted_cxg"] = DummyImpactModel().predict_proba(
        frame[_metadata()["selected_features"]]
    )[:, 1]
    frame["baseline_cxg"] = frame["predicted_cxg"] * 0.9

    table = category_lift_table(frame, "body_part")

    assert {
        "category_column",
        "category",
        "shots",
        "goals",
        "goal_rate",
        "mean_predicted_cxg",
        "total_predicted_cxg",
        "mean_baseline_cxg",
        "total_baseline_cxg",
        "mean_delta_vs_baseline",
        "total_delta_vs_baseline",
    }.issubset(table.columns)


def test_report_and_summary_files_are_written(tmp_path: Path):
    paths, feature_path = _write_artifacts(tmp_path)

    outputs = analyze_cxg_feature_impact(
        feature_path=feature_path,
        paths=paths,
        n_repeats=2,
        random_state=7,
    )

    assert outputs["permutation_importance"].exists()
    assert outputs["group_perturbation_summary"].exists()
    assert outputs["feature_impact_summary"].exists()
    assert outputs["feature_impact_report"].exists()
    summary = json.loads(outputs["feature_impact_summary"].read_text(encoding="utf-8"))
    assert summary["reference_columns_selected"] == []
    assert summary["result_integrity_checks"]["shot_predictions_player_id_missing_count"] == 0
    assert (paths.output_dir / "category_lift_body_part.csv").exists()


def test_missing_optional_category_lift_columns_are_skipped(tmp_path: Path):
    paths, feature_path = _write_artifacts(tmp_path)

    outputs = analyze_cxg_feature_impact(
        feature_path=feature_path,
        paths=paths,
        n_repeats=1,
        random_state=7,
    )

    summary = json.loads(outputs["feature_impact_summary"].read_text(encoding="utf-8"))
    assert "technique" in summary["skipped_category_lift_tables"]
    assert not (paths.output_dir / "category_lift_technique.csv").exists()


def test_player_id_integrity_checks_report_zero_missing_when_present():
    checks = result_integrity_checks(_feature_frame(), _feature_frame())

    assert checks["feature_frame_player_id_missing_count"] == 0
    assert checks["shot_predictions_player_id_missing_count"] == 0
