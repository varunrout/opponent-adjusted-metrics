import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from opponent_adjusted.db.models import (
    AggregatesPlayer,
    AggregatesTeam,
    Competition,
    EvaluationMetric,
    Event,
    Match,
    ModelRegistry,
    Player,
    RawEvent,
    Shot,
    ShotPrediction,
    Team,
)
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.api import cxg_inference
from opponent_adjusted.api.service import app
from scripts.report_ingestion_status import build_report
from scripts.check_cxg_outputs import (
    CxGOutputContract,
    assert_git_ignored,
    validate_cxg_outputs,
)
from scripts.run_cxg_end_to_end import run_end_to_end
from scripts.run_cxg_end_to_end import _aggregate
from scripts.run_cxg_end_to_end import DEFAULT_OUTPUT_DIR


def _synthetic_cxg_frame() -> pd.DataFrame:
    rows = []
    for i in range(60):
        is_goal = int(i % 5 == 0 or (i % 7 == 0 and i % 2 == 0))
        rows.append(
            {
                "shot_id": i,
                "match_id": i // 10,
                "team_id": 10 + (i % 3),
                "team_name": f"Team {i % 3}",
                "player_id": 100 + (i % 6),
                "player_name": f"Player {i % 6}",
                "opponent_team_id": 20 + (i % 4),
                "is_goal": is_goal,
                "shot_distance": 7.0 + (i % 20),
                "shot_angle": 0.15 + ((i % 8) * 0.06),
                "statsbomb_xg": 0.05 + (is_goal * 0.25) + ((i % 4) * 0.01),
                "score_diff_at_shot": (i % 3) - 1,
                "minute": 5 + i,
                "time_gap_seconds": float(i % 12),
                "is_leading": (i % 3) == 2,
                "is_trailing": (i % 3) == 0,
                "is_drawing": (i % 3) == 1,
                "possession_match": float(i % 2),
                "chain_label": "fast" if i % 2 else "slow",
                "pass_style": "cutback" if i % 4 == 0 else "none",
                "score_state": "leading" if (i % 3) == 2 else "drawing",
                "simple_state": "leading" if (i % 3) == 2 else "drawing",
                "minute_bucket_label": "46-60" if i >= 45 else "31-45",
                "assist_category": "pass" if i % 4 == 0 else "none",
                "pressure_state": "under_pressure" if i % 3 == 0 else "no_pressure",
                "set_piece_category": "open_play",
                "set_piece_phase": "none",
                "def_label": "average",
            }
        )
    return pd.DataFrame(rows)


def _seed_cxg_database_shots(frame: pd.DataFrame) -> pd.DataFrame:
    seeded = frame.copy()
    with session_scope() as session:
        competition = Competition(
            statsbomb_competition_id=99,
            name="CxG Fixture League",
            season="2026",
        )
        teams = [Team(statsbomb_team_id=1000 + i, name=f"Team {i}") for i in range(8)]
        players = [Player(statsbomb_player_id=2000 + i, name=f"Player {i}") for i in range(6)]
        session.add(competition)
        session.add_all(teams + players)
        session.flush()

        matches = []
        for i in range(int(seeded["match_id"].nunique())):
            match = Match(
                statsbomb_match_id=3000 + i,
                competition_id=competition.id,
                home_team_id=teams[0].id,
                away_team_id=teams[1].id,
                season="2026",
            )
            matches.append(match)
        session.add_all(matches)
        session.flush()

        team_ids = [team.id for team in teams[:3]]
        opponent_ids = [team.id for team in teams[3:7]]
        player_ids = [player.id for player in players]
        shot_ids = []
        for i, record in seeded.iterrows():
            raw_event = RawEvent(
                match_id=matches[int(record["match_id"])].id,
                statsbomb_event_id=f"shot-{i}",
                raw_json={
                    "id": f"shot-{i}",
                    "type": {"name": "Shot"},
                    "possession": int(i) + 1,
                    "team": {"id": 1000 + (i % 3), "name": f"Team {i % 3}"},
                },
                type="Shot",
                period=1,
                minute=int(record["minute"]),
                second=0,
            )
            session.add(raw_event)
            session.flush()

            event = Event(
                raw_event_id=raw_event.id,
                match_id=raw_event.match_id,
                team_id=team_ids[i % len(team_ids)],
                player_id=player_ids[i % len(player_ids)],
                type="Shot",
                period=1,
                minute=int(record["minute"]),
                second=0,
                possession=int(i) + 1,
            )
            session.add(event)
            session.flush()

            shot = Shot(
                event_id=event.id,
                match_id=event.match_id,
                team_id=event.team_id,
                player_id=event.player_id,
                opponent_team_id=opponent_ids[i % len(opponent_ids)],
                statsbomb_xg=float(record["statsbomb_xg"]),
                outcome="Goal" if int(record["is_goal"]) else "Saved",
                first_time=False,
                is_blocked=False,
            )
            session.add(shot)
            session.flush()
            shot_ids.append(shot.id)

    seeded["shot_id"] = shot_ids
    seeded["team_id"] = [team_ids[i % len(team_ids)] for i in range(len(seeded))]
    seeded["player_id"] = [player_ids[i % len(player_ids)] for i in range(len(seeded))]
    seeded["opponent_team_id"] = [opponent_ids[i % len(opponent_ids)] for i in range(len(seeded))]
    return seeded


def test_cxg_end_to_end_runner_emits_artifact_metadata_and_outputs(tmp_path: Path):
    input_path = tmp_path / "shot_features.parquet"
    _synthetic_cxg_frame().to_parquet(input_path, index=False)

    baseline_dir = tmp_path / "cxg" / "baseline"
    outputs = run_end_to_end(
        input_path=input_path, output_dir=baseline_dir, model_version="test-v1"
    )

    assert DEFAULT_OUTPUT_DIR == Path("outputs") / "modeling" / "cxg" / "baseline"
    assert outputs.model_path == baseline_dir / "models" / "contextual_model.joblib"
    assert outputs.model_path.exists()
    assert outputs.metadata_path.exists()
    assert outputs.metrics_path.exists()
    assert outputs.scored_predictions_path.exists()
    assert outputs.player_aggregates_path.exists()
    assert outputs.team_aggregates_path.exists()
    assert outputs.model_card_path.exists()
    assert hasattr(joblib.load(outputs.model_path), "predict_proba")

    scored = pd.read_parquet(outputs.scored_predictions_path)
    assert {"cxg_raw", "cxg_neutral", "cxg_opp_adjusted_diff"}.issubset(scored.columns)
    assert not pd.read_parquet(outputs.player_aggregates_path).empty
    assert not pd.read_parquet(outputs.team_aggregates_path).empty


def test_cxg_output_contract_validation_accepts_temp_outputs(tmp_path: Path):
    feature_store_dir = tmp_path / "feature_store" / "cxg"
    modeling_dir = tmp_path / "outputs" / "modeling" / "cxg"
    feature_store_dir.mkdir(parents=True)
    input_path = feature_store_dir / "shot_features.parquet"
    _synthetic_cxg_frame().to_parquet(input_path, index=False)

    run_end_to_end(
        input_path=input_path,
        output_dir=modeling_dir / "baseline",
        model_version="contract-v1",
    )

    summary = validate_cxg_outputs(
        CxGOutputContract.from_roots(feature_store_dir, modeling_dir),
        check_git_ignore=False,
    )

    assert summary["feature_store_dir"] == str(feature_store_dir)
    assert summary["model_path"] == str(
        modeling_dir / "baseline" / "models" / "contextual_model.joblib"
    )
    assert summary["predictions_path"] == str(
        modeling_dir / "baseline" / "predictions" / "shot_predictions.parquet"
    )


def test_cxg_generated_roots_are_git_ignored():
    assert_git_ignored(
        (
            Path("feature_store/cxg/shot_features.parquet"),
            Path("outputs/modeling/cxg/baseline/models/contextual_model.joblib"),
            Path("outputs/modeling/cxg/baseline/models/contextual_model.json"),
            Path("outputs/modeling/cxg/baseline/reports/metrics.json"),
            Path("outputs/modeling/cxg/baseline/reports/validation_summary.json"),
            Path("outputs/modeling/cxg/baseline/reports/calibration_table.csv"),
            Path("outputs/modeling/cxg/baseline/reports/slice_metrics.csv"),
            Path("outputs/modeling/cxg/baseline/predictions/shot_predictions.parquet"),
            Path("outputs/modeling/cxg/baseline/aggregates/player_cxg.parquet"),
            Path("outputs/modeling/cxg/baseline/aggregates/team_cxg.parquet"),
        ),
        Path.cwd(),
    )


def test_cxg_aggregate_does_not_collide_when_name_falls_back_to_id():
    scored = pd.DataFrame(
        [
            {
                "shot_id": 1,
                "player_id": 10,
                "team_id": 100,
                "is_goal": 1,
                "cxg_raw": 0.2,
                "cxg_neutral": 0.18,
                "cxg_opp_adjusted_diff": 0.02,
            },
            {
                "shot_id": 2,
                "player_id": 10,
                "team_id": 100,
                "is_goal": 0,
                "cxg_raw": 0.1,
                "cxg_neutral": 0.12,
                "cxg_opp_adjusted_diff": -0.02,
            },
        ]
    )

    player_aggregate = _aggregate(scored, "player_id", "player_name")
    team_aggregate = _aggregate(scored, "team_id", "team_name")

    assert list(player_aggregate["player_id"]) == [10]
    assert list(team_aggregate["team_id"]) == [100]
    assert player_aggregate.loc[0, "shots_count"] == 2
    assert team_aggregate.loc[0, "summed_cxg"] == 0.30000000000000004


def test_cxg_metadata_schema_contains_api_loader_fields(tmp_path: Path):
    input_path = tmp_path / "shot_features.parquet"
    _synthetic_cxg_frame().to_parquet(input_path, index=False)

    outputs = run_end_to_end(
        input_path=input_path, output_dir=tmp_path / "cxg", model_version="test-v2"
    )
    metadata = json.loads(outputs.metadata_path.read_text(encoding="utf-8"))

    assert metadata["artifact_path"] == str(outputs.model_path)
    assert metadata["model_version"] == "test-v2"
    assert metadata["version"] == "test-v2"
    assert metadata["target"] == "is_goal"
    assert metadata["generated_at"]
    assert metadata["trained_at"]
    assert "cxg_raw" in metadata["prediction_columns"]
    assert metadata["created_at"]
    assert metadata["features"]["numeric"]
    assert set(metadata["features"]).issuperset({"numeric", "binary", "categorical"})
    assert "statsbomb_xg" not in metadata["features"]["numeric"]


def test_cxg_api_positive_path_with_fixture_model(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "shot_features.parquet"
    _synthetic_cxg_frame().to_parquet(input_path, index=False)
    outputs = run_end_to_end(
        input_path=input_path, output_dir=tmp_path / "cxg", model_version="api-v1"
    )

    monkeypatch.setattr(
        cxg_inference, "_candidate_model_paths", lambda registry_path=None: [outputs.model_path]
    )

    client = TestClient(app)
    response = client.post(
        "/predict/cxg",
        json={
            "location_x": 102.0,
            "location_y": 40.0,
            "body_part": "Right Foot",
            "technique": "Normal",
            "shot_type": "Open Play",
            "first_time": False,
            "minute": 55,
            "score_diff": 0,
            "under_pressure": False,
            "opponent_team_id": 1,
            "possession_duration": 8.5,
            "possession_length": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["raw_probability"] <= 1.0
    assert 0.0 <= body["neutral_probability"] <= 1.0
    assert body["model_version"] == "api-v1"
    assert body["model_path"] == str(outputs.model_path)


def test_cxg_end_to_end_persists_outputs_to_database(e2e_test_env, tmp_path: Path):
    input_path = tmp_path / "shot_features.parquet"
    frame = _seed_cxg_database_shots(_synthetic_cxg_frame())
    frame.to_parquet(input_path, index=False)

    run_end_to_end(
        input_path=input_path,
        output_dir=tmp_path / "cxg",
        model_version="db-v1",
        persist_db=True,
    )

    with session_scope() as session:
        registry = session.query(ModelRegistry).filter_by(model_name="cxg", version="db-v1").one()
        assert registry.artifact_path.endswith("contextual_model.joblib")
        assert session.query(ShotPrediction).filter_by(model_id=registry.id).count() == len(frame)
        assert session.query(AggregatesPlayer).filter_by(model_id=registry.id).count() > 0
        assert session.query(AggregatesTeam).filter_by(model_id=registry.id).count() > 0
        assert session.query(EvaluationMetric).filter_by(model_id=registry.id).count() > 0

    report = build_report()
    assert report["table_counts"]["model_registry"] == 1
    assert report["table_counts"]["shot_predictions"] == len(frame)
    assert report["table_counts"]["aggregates_player"] > 0
    assert report["table_counts"]["aggregates_team"] > 0
    assert report["table_counts"]["evaluation_metrics"] > 0
    assert report["readiness"]["has_predictions"] is True


def test_cxg_database_persistence_is_idempotent(e2e_test_env, tmp_path: Path):
    input_path = tmp_path / "shot_features.parquet"
    frame = _seed_cxg_database_shots(_synthetic_cxg_frame())
    frame.to_parquet(input_path, index=False)

    for _ in range(2):
        run_end_to_end(
            input_path=input_path,
            output_dir=tmp_path / "cxg",
            model_version="db-idempotent-v1",
            persist_db=True,
        )

    with session_scope() as session:
        registry = (
            session.query(ModelRegistry)
            .filter_by(model_name="cxg", version="db-idempotent-v1")
            .one()
        )
        assert session.query(ModelRegistry).count() == 1
        assert session.query(ShotPrediction).filter_by(model_id=registry.id).count() == len(frame)
        assert session.query(AggregatesPlayer).filter_by(model_id=registry.id).count() > 0
        assert session.query(AggregatesTeam).filter_by(model_id=registry.id).count() > 0
        assert session.query(EvaluationMetric).filter_by(model_id=registry.id).count() > 0
