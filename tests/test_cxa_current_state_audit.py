import json
import sqlite3
from pathlib import Path

import pandas as pd

from scripts.audit_cxa_current_state import run_audit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _seed_contract(root: Path) -> None:
    _write_json(
        root / "configs" / "dashboard" / "v1_dashboard_contract.json",
        {
            "metrics": {
                "cxa": {
                    "inputs": {
                        "predictions": {
                            "path": "outputs/modeling/cxa/predictions/action_predictions.parquet"
                        },
                        "player_aggregates": {
                            "path": "outputs/modeling/cxa/aggregates/player_cxa.parquet"
                        },
                        "team_aggregates": {
                            "path": "outputs/modeling/cxa/aggregates/team_cxa.parquet"
                        },
                        "sequence_aggregates": {
                            "path": "outputs/modeling/cxa/aggregates/sequence_cxa.parquet"
                        },
                        "metrics": {"path": "outputs/modeling/cxa/reports/metrics.json"},
                        "attribution_summary": {
                            "path": "outputs/modeling/cxa/reports/attribution_summary.json"
                        },
                    }
                }
            }
        },
    )


def _seed_sqlite(root: Path) -> None:
    sqlite_path = root / "data" / "opponent_adjusted.db"
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            CREATE TABLE action_predictions (
                action_id TEXT,
                event_id TEXT,
                match_id INTEGER,
                team_id INTEGER,
                player_id INTEGER,
                sequence_id TEXT,
                cxa_value REAL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO action_predictions
            VALUES ('a1', 'e1', 1, 10, 100, 's1', 0.25),
                   ('a2', NULL, 1, 10, 101, NULL, 0.35)
            """
        )


def _seed_cxa_outputs(root: Path) -> None:
    _seed_contract(root)
    _seed_sqlite(root)

    _write_parquet(
        root / "feature_store" / "cxa" / "action_features.parquet",
        pd.DataFrame(
            [
                {
                    "action_id": "a1",
                    "event_id": "e1",
                    "match_id": 1,
                    "team_id": 10,
                    "player_id": 100,
                    "shot_id": None,
                    "shot_created": 1,
                    "created_shot_cxg": 0.4,
                    "predicted_cxa": 0.25,
                    "statsbomb_xg": 0.3,
                    "post_action_result": "goal",
                },
                {
                    "action_id": "a2",
                    "event_id": None,
                    "match_id": 1,
                    "team_id": 10,
                    "player_id": 101,
                    "shot_id": None,
                    "shot_created": 0,
                    "created_shot_cxg": 0.0,
                    "predicted_cxa": 0.35,
                    "statsbomb_xg": 0.15,
                    "post_action_result": "none",
                },
            ]
        ),
    )

    _write_parquet(
        root / "outputs" / "modeling" / "cxa" / "predictions" / "action_predictions.parquet",
        pd.DataFrame(
            [
                {
                    "action_id": "a1",
                    "event_id": "e1",
                    "match_id": 1,
                    "team_id": 10,
                    "player_id": 100,
                    "sequence_id": "s1",
                    "possession_id": "p1",
                    "shot_created": 1,
                    "created_shot_cxg": 0.4,
                    "predicted_cxa": 0.25,
                    "cxa_value": 0.25,
                    "cxa_share": 0.4167,
                },
                {
                    "action_id": None,
                    "event_id": "e2",
                    "match_id": 1,
                    "team_id": 10,
                    "player_id": 101,
                    "sequence_id": None,
                    "possession_id": "p1",
                    "shot_created": 1,
                    "created_shot_cxg": 0.35,
                    "predicted_cxa": 0.35,
                    "cxa_value": 0.35,
                    "cxa_share": 0.5833,
                },
            ]
        ),
    )

    _write_parquet(
        root / "outputs" / "modeling" / "cxa" / "aggregates" / "player_cxa.parquet",
        pd.DataFrame(
            [
                {"player_id": 100, "action_count": 1, "total_cxa": 0.25, "mean_cxa": 0.25},
                {"player_id": 101, "action_count": 1, "total_cxa": 0.35, "mean_cxa": 0.35},
            ]
        ),
    )
    _write_parquet(
        root / "outputs" / "modeling" / "cxa" / "aggregates" / "team_cxa.parquet",
        pd.DataFrame([{"team_id": 10, "action_count": 2, "total_cxa": 0.6, "mean_cxa": 0.3}]),
    )
    _write_parquet(
        root / "outputs" / "modeling" / "cxa" / "aggregates" / "sequence_cxa.parquet",
        pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "team_id": 10,
                    "sequence_id": "s1",
                    "action_count": 2,
                    "total_cxa": 0.6,
                }
            ]
        ),
    )
    _write_json(
        root / "outputs" / "modeling" / "cxa" / "reports" / "metrics.json",
        {"row_count": 2, "brier_score": 0.2, "log_loss": 0.5, "roc_auc": 0.7, "n_splits": 2},
    )
    _write_json(
        root / "outputs" / "modeling" / "cxa" / "reports" / "attribution_summary.json",
        {"attribution_method": "baseline", "total_cxa": 0.6},
    )
    _write_json(root / "outputs" / "modeling" / "cxa" / "reports" / "extra.json", {"a": 1, "b": 2})
    (root / "outputs" / "modeling" / "cxa" / "reports").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"col_a": 1, "col_b": 2}]).to_csv(
        root / "outputs" / "modeling" / "cxa" / "reports" / "extra.csv",
        index=False,
    )


def test_missing_files_are_reported_without_crashing(tmp_path: Path):
    summary = run_audit(repo_root=tmp_path, output_dir=tmp_path / "outputs" / "audits" / "cxa")
    inventory = pd.read_csv(tmp_path / "outputs" / "audits" / "cxa" / "cxa_output_inventory.csv")
    assert summary["files_missing"] > 0
    assert (inventory["exists"] == False).any()  # noqa: E712


def test_output_inventory_handles_csv_parquet_and_json(tmp_path: Path):
    _seed_cxa_outputs(tmp_path)
    run_audit(repo_root=tmp_path, output_dir=tmp_path / "outputs" / "audits" / "cxa")
    inventory = pd.read_csv(tmp_path / "outputs" / "audits" / "cxa" / "cxa_output_inventory.csv")
    assert (inventory["file_type"] == "parquet").any()
    assert (inventory["file_type"] == "csv").any()
    assert (inventory["file_type"] == "json").any()
    assert inventory["row_count"].notna().any()


def test_id_quality_statuses_are_detected(tmp_path: Path):
    _seed_cxa_outputs(tmp_path)
    run_audit(repo_root=tmp_path, output_dir=tmp_path / "outputs" / "audits" / "cxa")
    id_quality = pd.read_csv(tmp_path / "outputs" / "audits" / "cxa" / "cxa_id_quality.csv")
    statuses = set(id_quality["status"])
    assert {
        "passed",
        "partially_missing",
        "all_missing",
        "column_missing",
        "optional_missing",
    }.issubset(statuses)


def test_feature_inventory_classifies_key_column_types(tmp_path: Path):
    _seed_cxa_outputs(tmp_path)
    run_audit(repo_root=tmp_path, output_dir=tmp_path / "outputs" / "audits" / "cxa")
    inventory = pd.read_csv(tmp_path / "outputs" / "audits" / "cxa" / "cxa_feature_inventory.csv")
    mapped = dict(zip(inventory["column"], inventory["classification"]))
    assert mapped["action_id"] == "identifier"
    assert mapped["shot_created"] == "target"
    assert mapped["predicted_cxa"] == "prediction"
    assert mapped["statsbomb_xg"] == "reference_only"
    assert mapped["post_action_result"] == "leakage_risk"


def test_target_audit_handles_binary_and_numeric_targets(tmp_path: Path):
    _seed_cxa_outputs(tmp_path)
    run_audit(repo_root=tmp_path, output_dir=tmp_path / "outputs" / "audits" / "cxa")
    target_audit = pd.read_csv(tmp_path / "outputs" / "audits" / "cxa" / "cxa_target_audit.csv")
    shot_created = target_audit[target_audit["target_column"] == "shot_created"].iloc[0]
    created_shot_cxg = target_audit[target_audit["target_column"] == "created_shot_cxg"].iloc[0]
    assert shot_created["positive_count"] >= 1
    assert 0.0 <= created_shot_cxg["mean"] <= 1.0


def test_prediction_audit_handles_numeric_prediction_columns(tmp_path: Path):
    _seed_cxa_outputs(tmp_path)
    run_audit(repo_root=tmp_path, output_dir=tmp_path / "outputs" / "audits" / "cxa")
    prediction_audit = pd.read_csv(
        tmp_path / "outputs" / "audits" / "cxa" / "cxa_prediction_audit.csv"
    )
    predicted = prediction_audit[prediction_audit["prediction_column"] == "predicted_cxa"].iloc[0]
    assert predicted["min"] >= 0.0
    assert predicted["max"] <= 1.0


def test_aggregate_audit_tracks_basic_reconciliation(tmp_path: Path):
    _seed_cxa_outputs(tmp_path)
    run_audit(repo_root=tmp_path, output_dir=tmp_path / "outputs" / "audits" / "cxa")
    aggregate_audit = pd.read_csv(
        tmp_path / "outputs" / "audits" / "cxa" / "cxa_aggregate_audit.csv"
    )
    player_row = aggregate_audit[aggregate_audit["aggregation_level"] == "player"].iloc[0]
    assert "reconciles_to_action_total_delta=" in player_row["notes"]


def test_risk_register_flags_target_columns_in_feature_inputs(tmp_path: Path):
    _seed_cxa_outputs(tmp_path)
    run_audit(repo_root=tmp_path, output_dir=tmp_path / "outputs" / "audits" / "cxa")
    risk = pd.read_csv(tmp_path / "outputs" / "audits" / "cxa" / "cxa_risk_register.csv")
    assert ((risk["severity"] == "high") & (risk["category"] == "target_definition")).any()
    assert ((risk["severity"] == "high") & (risk["category"] == "leakage")).any()


def test_json_and_markdown_outputs_are_written(tmp_path: Path):
    _seed_cxa_outputs(tmp_path)
    run_audit(repo_root=tmp_path, output_dir=tmp_path / "outputs" / "audits" / "cxa")
    assert (tmp_path / "outputs" / "audits" / "cxa" / "cxa_current_state_audit.json").exists()
    assert (tmp_path / "outputs" / "audits" / "cxa" / "cxa_current_state_audit.md").exists()


def test_makefile_contains_audit_target():
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "audit-cxa-current-state:" in makefile
    assert "poetry run python scripts/audit_cxa_current_state.py" in makefile
