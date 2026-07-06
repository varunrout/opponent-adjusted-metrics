import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from scripts.analyze_cxa_feature_impact import (
    CxAFeatureImpactPaths,
    analyze_cxa_feature_impact,
    forbidden_columns,
    model_impact_features,
)


class DummyCxAImpactModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        start_x = pd.to_numeric(frame["start_x"], errors="coerce").fillna(0.0).to_numpy()
        length = pd.to_numeric(frame["length"], errors="coerce").fillna(0.0).to_numpy()
        minute = pd.to_numeric(frame["minute"], errors="coerce").fillna(0.0).to_numpy()
        is_pass = pd.to_numeric(frame["is_pass"], errors="coerce").fillna(0.0).to_numpy()
        under_pressure = (
            pd.to_numeric(frame["under_pressure"], errors="coerce").fillna(0.0).to_numpy()
        )
        progressive = pd.to_numeric(frame["is_progressive"], errors="coerce").fillna(0.0).to_numpy()
        action_type = (frame["action_type"].astype(str) == "Pass").astype(float).to_numpy()
        logits = (
            -4.8
            + (start_x * 0.03)
            + (length * 0.08)
            + (minute * 0.004)
            + (is_pass * 0.4)
            - (under_pressure * 0.35)
            + (progressive * 0.45)
            + (action_type * 0.25)
        )
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probabilities, probabilities])


def _feature_frame(*, include_optional_columns: bool = True) -> pd.DataFrame:
    rows = []
    for idx in range(40):
        row = {
            "action_id": f"action-{idx}",
            "event_id": idx + 1000,
            "match_id": idx // 5,
            "team_id": 10 + (idx % 4),
            "player_id": 200 + (idx % 9),
            "shot_created": int(idx % 7 == 0),
            "created_shot_cxg": 0.22 if idx % 7 == 0 else 0.0,
            "created_shot_id": f"shot-{idx}" if idx % 7 == 0 else None,
            "cxa_value": 0.22 if idx % 7 == 0 else 0.0,
            "start_x": 35.0 + idx,
            "length": 4.0 + (idx % 6),
            "minute": idx % 90,
            "is_pass": int(idx % 2 == 0),
            "under_pressure": int(idx % 5 == 0),
            "is_progressive": int(idx % 3 == 0),
            "enters_final_third": int(idx % 4 == 0),
            "enters_penalty_area": int(idx % 8 == 0),
            "action_type": "Pass" if idx % 2 == 0 else "Carry",
            "play_pattern": "Open Play" if idx % 3 else "From Throw In",
            "start_zone": "middle_central" if idx % 2 == 0 else "middle_left",
            "distance_to_goal_before": 60.0 - (idx % 12),
            "teammate_receipt_pressure": None,
        }
        if include_optional_columns:
            row["sequence_id"] = f"seq-{idx // 4}"
            row["possession"] = idx // 2
            row["score_state"] = "drawing" if idx % 2 == 0 else "leading"
            row["end_zone"] = "final_central" if idx % 4 == 0 else "middle_central"
        rows.append(row)
    return pd.DataFrame(rows)


def _contract(
    *,
    leakage_excluded_columns: list[str] | None = None,
    selected_feature_candidates: dict[str, list[str]] | None = None,
) -> dict:
    selected = selected_feature_candidates or {
        "numeric": ["start_x", "length", "minute"],
        "binary": [
            "is_pass",
            "under_pressure",
            "is_progressive",
            "enters_final_third",
            "enters_penalty_area",
        ],
        "categorical": ["action_type", "play_pattern", "start_zone"],
    }
    return {
        "metric": "cxa",
        "model_version": "diagnostic_v1",
        "primary_target": "shot_created",
        "selected_feature_candidates": selected,
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
                "sequence_id",
                "possession",
            ],
            "leakage_excluded_columns": leakage_excluded_columns or [],
            "requires_review_columns": ["distance_to_goal_before"],
            "excluded_unknown_columns": ["teammate_receipt_pressure"],
        },
    }


def _metadata(
    *,
    selected_feature_count: int = 10,
    numeric_feature_count: int = 3,
    binary_feature_count: int = 5,
    categorical_feature_count: int = 2,
) -> dict:
    return {
        "metric": "cxa",
        "model_version": "diagnostic_v1",
        "selected_model_candidate": "calibrated_gradient_boosting_sigmoid",
        "selected_feature_count": selected_feature_count,
        "numeric_feature_count": numeric_feature_count,
        "binary_feature_count": binary_feature_count,
        "categorical_feature_count": categorical_feature_count,
    }


def _promotion_summary(
    *,
    status: str | None = "provisionally_promoted",
    gate_passed: bool = True,
    selected_features: list[str] | None = None,
) -> dict:
    selected = selected_features or [
        "start_x",
        "length",
        "minute",
        "is_pass",
        "under_pressure",
        "is_progressive",
        "enters_final_third",
        "enters_penalty_area",
        "action_type",
        "play_pattern",
    ]
    return {
        "metric": "cxa",
        "model_version": "diagnostic_v1",
        "selected_model_candidate": "calibrated_gradient_boosting_sigmoid",
        "validation_selected_model": "calibrated_gradient_boosting_sigmoid",
        "promotion_status": status,
        "promotion_recommendation": (
            "provisional_promote" if status == "provisionally_promoted" else "promote"
        ),
        "promotion_gate_passed": gate_passed,
        "governance_summary": {
            "status": "passed",
            "selected_feature_count": len(selected),
            "selected_features": selected,
            "forbidden_features_used": [],
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_artifacts(
    root: Path,
    *,
    promotion_status: str = "provisionally_promoted",
    gate_passed: bool = True,
    include_optional_columns: bool = True,
    include_promotion_status: bool = True,
    leakage_excluded_columns: list[str] | None = None,
    selected_feature_candidates: dict[str, list[str]] | None = None,
    selected_features: list[str] | None = None,
) -> tuple[CxAFeatureImpactPaths, Path]:
    feature_path = root / "feature_store" / "cxa" / "action_features.parquet"
    diagnostic_dir = root / "outputs" / "modeling" / "cxa" / "diagnostic_v1"
    results_dir = root / "outputs" / "results" / "cxa" / "diagnostic_v1"
    output_dir = diagnostic_dir / "feature_impact"

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame = _feature_frame(include_optional_columns=include_optional_columns)
    feature_frame.to_parquet(feature_path, index=False)

    (diagnostic_dir / "models").mkdir(parents=True, exist_ok=True)
    (diagnostic_dir / "contracts").mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(DummyCxAImpactModel(), diagnostic_dir / "models" / "selected_model.joblib")
    contract = _contract(
        leakage_excluded_columns=leakage_excluded_columns,
        selected_feature_candidates=selected_feature_candidates,
    )
    promotion_summary = _promotion_summary(
        status=promotion_status,
        gate_passed=gate_passed,
        selected_features=selected_features,
    )
    selected_set = set(promotion_summary["governance_summary"]["selected_features"])
    grouped_selected_counts = {
        key: len(
            [
                feature
                for feature in contract["selected_feature_candidates"].get(key, [])
                if feature in selected_set
            ]
        )
        for key in ("numeric", "binary", "categorical")
    }
    _write_json(
        diagnostic_dir / "models" / "selected_model_metadata.json",
        _metadata(
            selected_feature_count=len(selected_set),
            numeric_feature_count=grouped_selected_counts["numeric"],
            binary_feature_count=grouped_selected_counts["binary"],
            categorical_feature_count=grouped_selected_counts["categorical"],
        ),
    )
    _write_json(diagnostic_dir / "contracts" / "feature_contract.json", contract)
    if not include_promotion_status:
        promotion_summary.pop("promotion_status", None)

    predictions = feature_frame[
        [
            column
            for column in (
                "action_id",
                "event_id",
                "match_id",
                "team_id",
                "player_id",
                "sequence_id",
                "possession",
                "action_type",
                "shot_created",
            )
            if column in feature_frame.columns
        ]
    ].copy()
    model = DummyCxAImpactModel()
    selected_columns = promotion_summary["governance_summary"]["selected_features"]
    for column in selected_columns:
        if column not in feature_frame.columns:
            feature_frame[column] = 0.0
    matrix = feature_frame[selected_columns]
    predictions["predicted_shot_created_probability"] = model.predict_proba(matrix)[:, 1]
    predictions["diagnostic_cxa"] = predictions["predicted_shot_created_probability"]
    predictions["prediction_source"] = "provisional_promoted_model"
    predictions["model_version"] = "diagnostic_v1"
    predictions["selected_model_candidate"] = "calibrated_gradient_boosting_sigmoid"
    predictions["promotion_status"] = promotion_status
    predictions["promotion_recommendation"] = (
        "provisional_promote" if promotion_status == "provisionally_promoted" else "promote"
    )
    predictions["created_shot_cxg_reference"] = feature_frame["created_shot_cxg"]
    predictions["created_shot_id_reference"] = feature_frame["created_shot_id"]
    predictions.to_parquet(results_dir / "action_predictions.parquet", index=False)

    _write_json(
        results_dir / "model_promotion_summary.json",
        promotion_summary,
    )

    paths = CxAFeatureImpactPaths.from_roots(
        diagnostic_dir=diagnostic_dir,
        results_dir=results_dir,
        output_dir=output_dir,
    )
    return paths, feature_path


def _summary(outputs: dict[str, Path]) -> dict:
    return json.loads(outputs["feature_impact_summary_json"].read_text(encoding="utf-8"))


def test_forbidden_reference_columns_are_excluded():
    grouped = _contract()["selected_feature_candidates"]
    forbidden = forbidden_columns(_contract(), prediction_columns={"diagnostic_cxa"})

    selected = model_impact_features(grouped, forbidden=forbidden)

    assert "shot_created" not in selected
    assert "created_shot_cxg" not in selected
    assert "cxa_value" not in selected
    assert "created_shot_id" not in selected


def test_leakage_excluded_columns_in_contract_block_analysis(tmp_path: Path):
    selected_feature_candidates = {
        "numeric": ["start_x", "length", "minute", "actions_until_shot"],
        "binary": [
            "is_pass",
            "under_pressure",
            "is_progressive",
            "enters_final_third",
            "enters_penalty_area",
        ],
        "categorical": ["action_type", "play_pattern"],
    }
    selected_features = [
        "start_x",
        "length",
        "minute",
        "actions_until_shot",
        "is_pass",
        "under_pressure",
        "is_progressive",
        "enters_final_third",
        "enters_penalty_area",
        "action_type",
        "play_pattern",
    ]
    paths, feature_path = _write_artifacts(
        tmp_path,
        leakage_excluded_columns=["actions_until_shot"],
        selected_feature_candidates=selected_feature_candidates,
        selected_features=selected_features,
    )
    frame = pd.read_parquet(feature_path)
    frame["actions_until_shot"] = np.arange(len(frame)) % 4
    frame.to_parquet(feature_path, index=False)

    with pytest.raises(
        ValueError, match="Forbidden/reference columns were selected as model features"
    ):
        analyze_cxa_feature_impact(
            feature_path=feature_path,
            paths=paths,
            sample_size=20,
            n_repeats=1,
            random_state=7,
        )


def test_feature_impact_output_files_are_written(tmp_path: Path):
    paths, feature_path = _write_artifacts(tmp_path)

    outputs = analyze_cxa_feature_impact(
        feature_path=feature_path,
        paths=paths,
        sample_size=20,
        n_repeats=1,
        random_state=7,
        top_n_examples=5,
    )

    assert outputs["feature_impact_summary_csv"].exists()
    assert outputs["feature_group_impact_csv"].exists()
    assert outputs["top_feature_examples_csv"].exists()
    assert outputs["feature_impact_report"].exists()
    assert outputs["feature_impact_summary_json"].exists()


def test_feature_groups_are_summarised(tmp_path: Path):
    paths, feature_path = _write_artifacts(tmp_path)

    outputs = analyze_cxa_feature_impact(
        feature_path=feature_path,
        paths=paths,
        sample_size=20,
        n_repeats=1,
        random_state=7,
    )

    group_impact = pd.read_csv(outputs["feature_group_impact_csv"])
    assert set(
        [
            "numeric",
            "binary",
            "categorical",
            "progression/location",
            "zone-entry",
            "action-type/context",
            "pressure",
            "time/sequence",
        ]
    ).issubset(set(group_impact["feature_group"]))


def test_report_mentions_provisional_promotion(tmp_path: Path):
    paths, feature_path = _write_artifacts(tmp_path)

    outputs = analyze_cxa_feature_impact(
        feature_path=feature_path,
        paths=paths,
        sample_size=20,
        n_repeats=1,
        random_state=7,
    )

    report = outputs["feature_impact_report"].read_text(encoding="utf-8")
    assert "provisionally promoted" in report


def test_report_mentions_created_shot_cxg_and_cxa_value_are_not_model_features(tmp_path: Path):
    paths, feature_path = _write_artifacts(tmp_path)

    outputs = analyze_cxa_feature_impact(
        feature_path=feature_path,
        paths=paths,
        sample_size=20,
        n_repeats=1,
        random_state=7,
    )

    report = outputs["feature_impact_report"].read_text(encoding="utf-8")
    assert "`created_shot_cxg` and `cxa_value` are not model features" in report


@pytest.mark.parametrize("promotion_status", ["promoted", "provisionally_promoted"])
def test_allowed_promotion_statuses_pass(tmp_path: Path, promotion_status: str):
    paths, feature_path = _write_artifacts(
        tmp_path, promotion_status=promotion_status, gate_passed=True
    )

    outputs = analyze_cxa_feature_impact(
        feature_path=feature_path,
        paths=paths,
        sample_size=20,
        n_repeats=1,
        random_state=7,
    )
    assert outputs["feature_impact_summary_json"].exists()


@pytest.mark.parametrize(
    "promotion_status", ["blocked", "needs_revision", "unknown", "not_promoted"]
)
def test_unrecognized_or_non_allowed_promotion_statuses_fail(tmp_path: Path, promotion_status: str):
    paths, feature_path = _write_artifacts(
        tmp_path, promotion_status=promotion_status, gate_passed=True
    )
    with pytest.raises(ValueError, match="requires promotion_status in"):
        analyze_cxa_feature_impact(
            feature_path=feature_path,
            paths=paths,
            sample_size=20,
            n_repeats=1,
            random_state=7,
        )


def test_missing_promotion_status_fails(tmp_path: Path):
    paths, feature_path = _write_artifacts(
        tmp_path,
        promotion_status="provisionally_promoted",
        gate_passed=True,
        include_promotion_status=False,
    )
    with pytest.raises(ValueError, match="requires promotion_status in"):
        analyze_cxa_feature_impact(
            feature_path=feature_path,
            paths=paths,
            sample_size=20,
            n_repeats=1,
            random_state=7,
        )


def test_false_promotion_gate_fails_even_for_provisional(tmp_path: Path):
    paths, feature_path = _write_artifacts(
        tmp_path, promotion_status="provisionally_promoted", gate_passed=False
    )
    with pytest.raises(ValueError, match="requires promotion_status in"):
        analyze_cxa_feature_impact(
            feature_path=feature_path,
            paths=paths,
            sample_size=20,
            n_repeats=1,
            random_state=7,
        )


def test_missing_optional_columns_do_not_crash(tmp_path: Path):
    paths, feature_path = _write_artifacts(tmp_path, include_optional_columns=False)

    outputs = analyze_cxa_feature_impact(
        feature_path=feature_path,
        paths=paths,
        sample_size=20,
        n_repeats=1,
        random_state=7,
        top_n_examples=4,
    )

    examples = pd.read_csv(outputs["top_feature_examples_csv"])
    summary = _summary(outputs)
    assert len(examples) > 0
    assert summary["selected_feature_count"] == 10


def test_deterministic_output_with_fixed_seed(tmp_path: Path):
    paths, feature_path = _write_artifacts(tmp_path)

    first_outputs = analyze_cxa_feature_impact(
        feature_path=feature_path,
        paths=paths,
        sample_size=20,
        n_repeats=2,
        random_state=11,
    )

    second_paths = CxAFeatureImpactPaths(
        selected_model=paths.selected_model,
        selected_model_metadata=paths.selected_model_metadata,
        feature_contract=paths.feature_contract,
        action_predictions=paths.action_predictions,
        model_promotion_summary=paths.model_promotion_summary,
        output_dir=tmp_path
        / "outputs"
        / "modeling"
        / "cxa"
        / "diagnostic_v1"
        / "feature_impact_again",
    )
    second_outputs = analyze_cxa_feature_impact(
        feature_path=feature_path,
        paths=second_paths,
        sample_size=20,
        n_repeats=2,
        random_state=11,
    )

    first_summary = pd.read_csv(first_outputs["feature_impact_summary_csv"])
    second_summary = pd.read_csv(second_outputs["feature_impact_summary_csv"])
    pdt.assert_frame_equal(first_summary, second_summary)
