import json
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from opponent_adjusted.db.base import Base
from opponent_adjusted.db.bootstrap import ensure_sqlite_database_parent
from opponent_adjusted.db.models import Competition
from scripts.fetch_statsbomb_subset import fetch_subset


MIGRATION_PATH = Path("alembic/versions/2b70afefcdee_drop_unique_competition_id.py")
INITIAL_SCHEMA_PATH = Path("alembic/versions/001_initial_schema.py")
EVENT_TYPE_SCHEMA_PATH = Path("alembic/versions/14f289cc51b0_add_event_type_tables.py")
ACTION_OUTPUT_SCHEMA_PATH = Path("alembic/versions/d4f1a2b3c4d5_add_action_model_output_tables.py")
ACTION_FEATURE_SCHEMA_PATH = Path("alembic/versions/e5f6a7b8c9d0_add_action_features_table.py")
MAKEFILE_PATH = Path("Makefile")
CXA_FEATURE_BUILDER_PATH = Path("src/opponent_adjusted/features/cxa/action_features.py")


def test_sqlite_constraint_migration_uses_batch_mode_and_skips_sqlite_fk_drop():
    text = MIGRATION_PATH.read_text(encoding="utf-8")

    assert 'op.batch_alter_table("competitions")' in text
    assert "SQLite does not support ALTER TABLE DROP CONSTRAINT" in text
    assert 'bind.dialect.name != "sqlite"' in text
    assert "batch_op.drop_constraint" in text
    assert "batch_op.create_unique_constraint" in text


def test_sqlite_schema_sources_do_not_use_postgres_now_default():
    for path in (
        INITIAL_SCHEMA_PATH,
        EVENT_TYPE_SCHEMA_PATH,
        ACTION_OUTPUT_SCHEMA_PATH,
        ACTION_FEATURE_SCHEMA_PATH,
    ):
        text = path.read_text(encoding="utf-8")

        assert "sa.text('now()')" not in text
        assert "CURRENT_TIMESTAMP" in text


def test_action_model_output_migration_declares_new_tables():
    text = ACTION_OUTPUT_SCHEMA_PATH.read_text(encoding="utf-8")

    for table_name in (
        "action_predictions",
        "action_threat_predictions",
        "aggregates_sequence",
    ):
        assert f'"{table_name}"' in text


def test_action_feature_migration_declares_engineered_feature_table():
    text = ACTION_FEATURE_SCHEMA_PATH.read_text(encoding="utf-8")

    assert '"action_features"' in text
    assert '"feature_family"' in text
    assert '"target_shot_created"' in text
    assert "uq_action_feature_version_action" in text
    assert "CURRENT_TIMESTAMP" in text


def test_makefile_clean_run_targets_migrate_before_ingestion():
    text = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert (
        "data-smoke: migrate-up fetch-data ingest-all normalize-events "
        "build-possessions ingestion-report"
    ) in text
    assert "pipeline: migrate-up ingest-all normalize-events build-possessions" in text
    assert (
        "clean-rebuild: migrate-up fetch-data ingest-all normalize-events "
        "build-possessions ingestion-report"
    ) in text
    assert "reproduce: clean-rebuild" in text
    assert (
        "reproduce-v1: migrate-up fetch-data ingest-all normalize-events build-possessions "
        "build-features build-profiles build-cxa-action-features run-cxg-pipeline "
        "run-cxg-end-to-end run-cxa-pipeline run-cxa-end-to-end run-cxt-pipeline "
        "ingestion-report"
    ) in text
    assert "build-cxa-action-features:" in text
    assert "\tpoetry run python scripts/build_cxa_action_features.py" in text
    assert "cxa-action-features-smoke:" in text
    assert (
        "\tpoetry run python scripts/build_cxa_action_features.py --smoke --max-matches 20"
    ) in text


def test_sqlite_database_parent_directory_is_created(tmp_path: Path):
    database_path = tmp_path / "missing" / "nested" / "opponent_adjusted.db"

    ensure_sqlite_database_parent(f"sqlite:///{database_path.as_posix()}")

    assert database_path.parent.is_dir()


def test_sqlite_database_parent_directory_ignores_memory_and_non_sqlite(tmp_path: Path):
    postgres_like_path = tmp_path / "postgres-should-not-exist"

    ensure_sqlite_database_parent("sqlite:///:memory:")
    ensure_sqlite_database_parent(
        f"postgresql+psycopg://user:pass@localhost:5432/{postgres_like_path.name}"
    )

    assert not postgres_like_path.exists()


def test_sqlite_competition_insert_uses_cross_dialect_timestamp_defaults(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'timestamp-defaults.db'}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    with Session() as session:
        competition = Competition(
            statsbomb_competition_id=1,
            name="Test Competition",
            season="2026",
        )
        session.add(competition)
        session.commit()

        assert competition.id is not None
        assert competition.created_at is not None
        assert competition.updated_at is not None

    with engine.connect() as connection:
        schema_sql = "\n".join(
            row[0]
            for row in connection.execute(
                text("SELECT sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL")
            )
        )

    assert "DEFAULT now()" not in schema_sql
    assert "DEFAULT CURRENT_TIMESTAMP" in schema_sql


def test_cxa_pipeline_source_has_progress_logging():
    text = CXA_FEATURE_BUILDER_PATH.read_text(encoding="utf-8")

    for message in (
        "CxA pipeline: starting baseline feature build",
        "CxA pipeline: loading normalized events from database",
        "CxA pipeline: loaded %s eligible/shot event rows",
        "CxA pipeline: loaded %s matches",
        "CxA pipeline: loaded %s possessions/sequences",
        "CxA pipeline: loaded %s shot detail rows",
        "CxA action features: scanning %d shot-containing match/team/possession groups",
        "CxA action features: candidate actions=%d",
        "CxA pipeline: wrote %s action feature rows to %s",
        "CxA pipeline: wrote metadata to %s",
        "CxA pipeline: complete",
    ):
        assert message in text


def test_fetch_subset_records_missing_events_and_continues(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "subset.json"
    config_path.write_text(
        json.dumps(
            {
                "competitions": [
                    {
                        "competition_id": 1,
                        "season_id": 2,
                        "include_events": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def fake_fetch(url: str):
        if url.endswith("/competitions.json"):
            return [{"competition_id": 1, "season_id": 2}]
        if url.endswith("/matches/1/2.json"):
            return [{"match_id": 111}, {"match_id": 222}]
        if url.endswith("/events/111.json"):
            return [{"id": "event-1"}]
        if url.endswith("/events/222.json"):
            return None
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("scripts.fetch_statsbomb_subset._fetch_with_retries", fake_fetch)

    summary = fetch_subset(
        config_path=config_path,
        output_dir=tmp_path / "data" / "statsbomb",
        include_events=True,
        force=False,
    )

    assert summary["events_written"] == 1
    assert summary["missing"] == [{"scope": "events", "match_id": 222}]
