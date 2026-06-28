import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from opponent_adjusted.db.models import (
    ActionFeature,
    ActionPrediction,
    AggregatesPlayer,
    AggregatesSequence,
    AggregatesTeam,
    EvaluationMetric,
    ModelRegistry,
)
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.features.cxa import action_features as cxa_feature_builder
from opponent_adjusted.features.cxa.action_features import (
    build_action_features,
    save_action_features,
)
from scripts.check_cxg_outputs import assert_git_ignored
from scripts.report_ingestion_status import build_report
from scripts.run_cxa_end_to_end import run_end_to_end


def _synthetic_events(
    one_class: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    pass_details = []
    shots = []
    event_id = 1
    for match_id in range(1, 5):
        for sequence in range(3):
            team_id = 10 + (sequence % 2)
            possession = (match_id * 10) + sequence
            for pos in range(4):
                action_type = "Pass" if pos % 2 == 0 else "Carry"
                rows.append(
                    {
                        "event_id": event_id,
                        "raw_event_id": event_id,
                        "match_id": match_id,
                        "team_id": team_id,
                        "player_id": 100 + pos,
                        "action_type": action_type,
                        "period": 1,
                        "minute": sequence * 5,
                        "second": pos * 3,
                        "possession": possession,
                        "start_x": 42.0 + (pos * 10),
                        "start_y": 25.0 + (pos * 4),
                        "under_pressure": pos == 1,
                        "event_outcome": None,
                    }
                )
                if action_type == "Pass":
                    pass_details.append(
                        {
                            "event_id": event_id,
                            "end_x": 52.0 + (pos * 10),
                            "end_y": 28.0 + (pos * 4),
                            "length": 11.0,
                            "angle": 0.2,
                            "pass_height": "Ground Pass",
                            "pass_type": None,
                            "body_part": "Right Foot",
                            "is_cross": pos == 2,
                            "is_through_ball": pos == 0,
                        }
                    )
                event_id += 1

            if not one_class and sequence < 2:
                rows.append(
                    {
                        "event_id": event_id,
                        "raw_event_id": event_id,
                        "match_id": match_id,
                        "team_id": team_id,
                        "player_id": 900,
                        "action_type": "Shot",
                        "period": 1,
                        "minute": sequence * 5,
                        "second": 13,
                        "possession": possession,
                        "start_x": 104.0,
                        "start_y": 40.0,
                        "under_pressure": False,
                        "event_outcome": "Goal" if sequence == 0 else "Saved",
                    }
                )
                shots.append(
                    {
                        "shot_id": 5000 + event_id,
                        "event_id": event_id,
                        "statsbomb_xg": 0.25 + (sequence * 0.03),
                    }
                )
                event_id += 1

    return (
        pd.DataFrame(rows),
        pd.DataFrame(shots),
        {"Pass": pd.DataFrame(pass_details), "Carry": pd.DataFrame()},
    )


def test_cxa_feature_table_is_generated_under_contract_path(tmp_path: Path):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)
    output_path = save_action_features(features, tmp_path / "feature_store" / "cxa")

    assert output_path == tmp_path / "feature_store" / "cxa" / "action_features.parquet"
    assert output_path.exists()
    written = pd.read_parquet(output_path)
    assert not written.empty
    assert {"action_id", "shot_created", "created_shot_cxg"}.issubset(written.columns)
    assert written["shot_created"].isin([0, 1]).all()
    assert written["created_shot_cxg"].between(0, 1).all()


def test_cxa_feature_builder_respects_max_actions_and_logs_progress(caplog):
    events, shots, details = _synthetic_events()

    with caplog.at_level(logging.INFO):
        features = build_action_features(events, shots, details, max_actions=5)

    assert len(features) == 5
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "CxA action features: matches=" in messages
    assert "CxA action features: scanning" in messages
    assert "CxA action features: candidate actions=5" in messages
    assert "CxA action features: built 5 rows" in messages


def test_cxa_smoke_pipeline_uses_limited_builder_path(tmp_path: Path, monkeypatch):
    events, shots, details = _synthetic_events()
    captured: dict[str, int | None] = {}

    def fake_build_from_database(
        competition_id: int | None = None,
        max_matches: int | None = None,
        max_actions: int | None = None,
    ) -> pd.DataFrame:
        captured["competition_id"] = competition_id
        captured["max_matches"] = max_matches
        captured["max_actions"] = max_actions
        return build_action_features(events, shots, details, max_actions=max_actions)

    monkeypatch.setattr(
        cxa_feature_builder,
        "build_action_features_from_database",
        fake_build_from_database,
    )

    outputs = cxa_feature_builder.run_pipeline(
        output_dir=tmp_path,
        smoke=True,
        max_matches=3,
        max_actions=4,
    )

    assert captured == {"competition_id": None, "max_matches": 3, "max_actions": 4}
    assert outputs["action_features"].exists()
    written = pd.read_parquet(outputs["action_features"])
    metadata = json.loads(outputs["metadata"].read_text(encoding="utf-8"))
    assert len(written) == 4
    assert metadata["smoke"] is True
    assert metadata["max_matches"] == 3
    assert metadata["max_actions"] == 4


def test_cxa_feature_pipeline_persists_action_features_to_database(
    e2e_test_env, tmp_path: Path, monkeypatch
):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)

    def fake_build_from_database(
        competition_id: int | None = None,
        max_matches: int | None = None,
        max_actions: int | None = None,
    ) -> pd.DataFrame:
        return features

    monkeypatch.setattr(
        cxa_feature_builder,
        "build_action_features_from_database",
        fake_build_from_database,
    )

    for _ in range(2):
        outputs = cxa_feature_builder.run_pipeline(
            output_dir=tmp_path / "feature_store" / "cxa",
            persist_db=True,
            feature_version="test-cxa-features-v1",
        )

    assert outputs["action_features"].exists()
    written = pd.read_parquet(outputs["action_features"])
    assert len(written) == len(features)

    with session_scope() as session:
        rows = (
            session.query(ActionFeature)
            .filter_by(feature_family="cxa", version_tag="test-cxa-features-v1")
            .all()
        )
        assert len(rows) == len(features)
        assert rows[0].target_shot_created in {True, False}

    report = build_report()
    assert report["table_counts"]["action_features"] == len(features)
    assert report["readiness"]["has_action_features"] is True


def test_cxa_feature_builder_source_avoids_old_rowwise_nested_scans():
    text = Path("src/opponent_adjusted/features/cxa/action_features.py").read_text(encoding="utf-8")

    assert ".iterrows(" not in text
    assert "for _, action" not in text
    assert "ordered[(" not in text


def test_cxa_pipeline_scripts_import_shared_package_module():
    build_script = Path("scripts/build_cxa_action_features.py").read_text(encoding="utf-8")
    pipeline_script = Path("scripts/run_cxa_pipeline.py").read_text(encoding="utf-8")

    assert "opponent_adjusted.features.cxa.action_features" in build_script
    assert "opponent_adjusted.features.cxa.action_features" in pipeline_script
    assert "from scripts." not in build_script
    assert "from scripts." not in pipeline_script


def test_cxa_end_to_end_emits_model_predictions_and_aggregates(tmp_path: Path):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)
    input_path = tmp_path / "action_features.parquet"
    features.to_parquet(input_path, index=False)

    outputs = run_end_to_end(
        input_path=input_path, output_dir=tmp_path / "cxa", model_version="cxa-test-v1"
    )

    assert outputs.model_path.exists()
    assert outputs.metadata_path.exists()
    assert outputs.metrics_path.exists()
    assert outputs.predictions_path.exists()
    assert outputs.player_aggregates_path.exists()
    assert outputs.team_aggregates_path.exists()
    assert outputs.sequence_aggregates_path.exists()
    assert outputs.attribution_summary_path.exists()
    assert hasattr(joblib.load(outputs.model_path), "predict_proba")

    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))
    summary = json.loads(outputs.attribution_summary_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(outputs.predictions_path)
    player_aggregates = pd.read_parquet(outputs.player_aggregates_path)
    team_aggregates = pd.read_parquet(outputs.team_aggregates_path)
    sequence_aggregates = pd.read_parquet(outputs.sequence_aggregates_path)
    assert metadata["model_version"] == "cxa-test-v1"
    assert metadata["target"] == "shot_created"
    assert metadata["value_column"] == "created_shot_cxg"
    assert metadata["leakage_guardrails"]["forbidden_training_features_excluded"] is True
    assert "sequence_aggregates" in metadata["outputs"]
    assert "attribution_summary" in metadata["outputs"]
    assert metrics["row_count"] == len(features)
    assert metrics["log_loss_status"] == "computed"
    assert {
        "predicted_cxa",
        "baseline_cxa",
        "cxa_above_baseline",
        "cxa_raw",
        "cxa_value",
        "cxa_share",
        "possession_cxa",
        "sequence_cxa",
        "downstream_shot_value",
        "attribution_method",
    }.issubset(predictions.columns)
    assert not player_aggregates.empty
    assert not team_aggregates.empty
    assert not sequence_aggregates.empty
    assert {"total_cxa", "mean_cxa", "cxa_per_action", "high_value_actions"}.issubset(
        player_aggregates.columns
    )
    assert {"total_cxa", "possession_count", "sequence_count"}.issubset(team_aggregates.columns)
    assert {"total_cxa", "max_action_cxa", "led_to_shot"}.issubset(sequence_aggregates.columns)
    assert abs(player_aggregates["total_cxa"].sum() - predictions["cxa_value"].sum()) < 1e-9
    assert abs(team_aggregates["total_cxa"].sum() - predictions["cxa_value"].sum()) < 1e-9
    assert abs(sequence_aggregates["total_cxa"].sum() - predictions["cxa_value"].sum()) < 1e-9
    assert abs(summary["total_attributed_cxa"] - predictions["cxa_value"].sum()) < 1e-9
    assert summary["attribution"]["method"] == "simple_action_level_baseline_attribution"


def test_cxa_end_to_end_persists_outputs_to_database(e2e_test_env, tmp_path: Path):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)
    input_path = tmp_path / "action_features.parquet"
    features.to_parquet(input_path, index=False)

    run_end_to_end(
        input_path=input_path,
        output_dir=tmp_path / "cxa",
        model_version="cxa-db-v1",
        persist_db=True,
    )

    with session_scope() as session:
        registry = (
            session.query(ModelRegistry).filter_by(model_name="cxa", version="cxa-db-v1").one()
        )
        assert session.query(ActionPrediction).filter_by(model_id=registry.id).count() == len(
            features
        )
        assert session.query(AggregatesPlayer).filter_by(model_id=registry.id).count() > 0
        assert session.query(AggregatesTeam).filter_by(model_id=registry.id).count() > 0
        assert session.query(AggregatesSequence).filter_by(model_id=registry.id).count() > 0
        assert session.query(EvaluationMetric).filter_by(model_id=registry.id).count() > 0

    report = build_report()
    assert report["table_counts"]["action_predictions"] == len(features)
    assert report["table_counts"]["aggregates_sequence"] > 0
    assert report["readiness"]["has_cxa_predictions"] is True
    assert report["readiness"]["has_sequence_aggregates"] is True


def test_cxa_database_persistence_is_idempotent_and_preserves_cxg_rows(
    e2e_test_env, tmp_path: Path
):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)
    input_path = tmp_path / "action_features.parquet"
    features.to_parquet(input_path, index=False)

    with session_scope() as session:
        session.add(
            ModelRegistry(
                model_name="cxg",
                version="existing-cxg",
                algorithm="fixture",
                trained_on_version_tag="existing-cxg",
                artifact_path="fixture.joblib",
            )
        )

    for _ in range(2):
        run_end_to_end(
            input_path=input_path,
            output_dir=tmp_path / "cxa",
            model_version="cxa-idempotent-v1",
            persist_db=True,
        )

    with session_scope() as session:
        cxa_registry = (
            session.query(ModelRegistry)
            .filter_by(model_name="cxa", version="cxa-idempotent-v1")
            .one()
        )
        assert session.query(ModelRegistry).filter_by(model_name="cxg").count() == 1
        assert session.query(ModelRegistry).filter_by(model_name="cxa").count() == 1
        assert session.query(ActionPrediction).filter_by(model_id=cxa_registry.id).count() == len(
            features
        )


def test_cxa_baseline_excludes_forbidden_leakage_features(tmp_path: Path):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)
    features["created_shot_outcome"] = "Goal"
    input_path = tmp_path / "action_features.parquet"
    features.to_parquet(input_path, index=False)

    outputs = run_end_to_end(input_path=input_path, output_dir=tmp_path / "cxa")
    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))
    model_features = set().union(*[set(cols) for cols in metadata["features"].values()])

    assert "created_shot_outcome" not in model_features
    assert "post_shot_xg" not in model_features


def test_cxa_baseline_handles_one_class_data_safely(tmp_path: Path):
    events, shots, details = _synthetic_events(one_class=True)
    features = build_action_features(events, shots.iloc[0:0], details)
    input_path = tmp_path / "action_features.parquet"
    features.to_parquet(input_path, index=False)

    outputs = run_end_to_end(input_path=input_path, output_dir=tmp_path / "cxa")
    metrics = json.loads(outputs.metrics_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(outputs.predictions_path)

    assert metrics["positive_count"] == 0
    assert metrics["log_loss_status"] == "skipped_single_class"
    assert metrics["roc_auc_status"] == "skipped_single_class"
    assert predictions["predicted_cxa"].eq(0.0).all()
    assert predictions["cxa_value"].eq(0.0).all()


def test_cxa_attribution_handles_missing_optional_names(tmp_path: Path):
    events, shots, details = _synthetic_events()
    features = build_action_features(events, shots, details)
    features["player_id"] = pd.NA
    features = features.drop(
        columns=[col for col in ("player_name", "team_name") if col in features]
    )
    input_path = tmp_path / "action_features.parquet"
    features.to_parquet(input_path, index=False)

    outputs = run_end_to_end(input_path=input_path, output_dir=tmp_path / "cxa")
    predictions = pd.read_parquet(outputs.predictions_path)
    player_aggregates = pd.read_parquet(outputs.player_aggregates_path)
    team_aggregates = pd.read_parquet(outputs.team_aggregates_path)

    assert not player_aggregates.empty
    assert not team_aggregates.empty
    assert "player_name" not in player_aggregates.columns
    assert "team_name" not in team_aggregates.columns
    assert abs(player_aggregates["total_cxa"].sum() - predictions["cxa_value"].sum()) < 1e-9


def test_cxa_generated_paths_are_git_ignored():
    assert_git_ignored(
        (
            Path("feature_store/cxa/action_features.parquet"),
            Path("outputs/modeling/cxa/models/baseline_model.joblib"),
            Path("outputs/modeling/cxa/models/baseline_model.json"),
            Path("outputs/modeling/cxa/reports/metrics.json"),
            Path("outputs/modeling/cxa/reports/attribution_summary.json"),
            Path("outputs/modeling/cxa/predictions/action_predictions.parquet"),
            Path("outputs/modeling/cxa/aggregates/player_cxa.parquet"),
            Path("outputs/modeling/cxa/aggregates/team_cxa.parquet"),
            Path("outputs/modeling/cxa/aggregates/sequence_cxa.parquet"),
        ),
        Path.cwd(),
    )
