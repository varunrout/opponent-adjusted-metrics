#!/usr/bin/env python
"""Audit current-state CxA outputs, inputs, IDs, and leakage risks."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs") / "audits" / "cxa"
DEFAULT_DASHBOARD_CONTRACT = Path("configs") / "dashboard" / "v1_dashboard_contract.json"
DEFAULT_SQLITE_PATH = Path("data") / "opponent_adjusted.db"
FEATURE_DIR = Path("feature_store") / "cxa"
MODELING_DIR = Path("outputs") / "modeling" / "cxa"

EXPECTED_CXA_PATHS = (
    FEATURE_DIR / "action_features.parquet",
    MODELING_DIR / "predictions" / "action_predictions.parquet",
    MODELING_DIR / "aggregates" / "player_cxa.parquet",
    MODELING_DIR / "aggregates" / "team_cxa.parquet",
    MODELING_DIR / "aggregates" / "sequence_cxa.parquet",
    MODELING_DIR / "reports" / "metrics.json",
    MODELING_DIR / "reports" / "attribution_summary.json",
    DEFAULT_DASHBOARD_CONTRACT,
    DEFAULT_SQLITE_PATH,
)

TABULAR_SUFFIXES = {".csv", ".parquet"}
ID_COLUMNS = (
    "action_id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "possession_id",
    "sequence_id",
    "target_created_shot_id",
    "created_shot_id",
    "shot_id",
)
OPTIONAL_ID_COLUMNS = {
    "possession_id",
    "sequence_id",
    "target_created_shot_id",
    "created_shot_id",
    "shot_id",
}
TARGET_COLUMNS = {
    "shot_created",
    "created_shot_cxg",
    "cxa_value",
    "target_created_shot_id",
    "led_to_shot",
}
PREDICTION_COLUMNS = {
    "predicted_cxa",
    "cxa_value",
    "cxa_share",
    "sequence_cxa",
    "possession_cxa",
    "predicted_shot_created_probability",
}
SQLITE_RELEVANT_TABLES = (
    "action_features",
    "action_predictions",
    "action_threat_predictions",
    "aggregates_player",
    "aggregates_team",
    "aggregates_sequence",
    "events",
    "possessions",
    "passes",
    "shots",
    "model_registry",
    "evaluation_metrics",
)


@dataclass(frozen=True)
class TableSource:
    source: str
    role_guess: str
    frame: pd.DataFrame


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, TypeError, ValueError):
            return value
    return value


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _role_guess(path: Path) -> str:
    relative = path.as_posix().lower()
    if "feature_store/cxa" in relative:
        return "features"
    if "predictions" in relative:
        return "predictions"
    if "aggregates" in relative:
        return "aggregates"
    if "reports" in relative:
        return "reports"
    if "audit" in relative:
        return "diagnostics"
    return "unknown"


def _file_type(path: Path) -> str:
    if path.is_dir():
        return "directory"
    suffix = path.suffix.lower()
    if not suffix:
        return "unknown"
    return suffix.lstrip(".")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_table(path: Path) -> tuple[pd.DataFrame | None, str]:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path), ""
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path), ""
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return pd.DataFrame(payload), ""
            if isinstance(payload, dict):
                return pd.json_normalize(payload), ""
        return None, "non_tabular_or_unsupported"
    except Exception as exc:  # noqa: BLE001 - report and continue audit
        return None, f"read_error:{exc}"


def _collect_paths(repo_root: Path) -> list[Path]:
    candidates: set[Path] = {repo_root / relative for relative in EXPECTED_CXA_PATHS}
    for root in (repo_root / FEATURE_DIR, repo_root / MODELING_DIR):
        if root.exists():
            candidates.update(path for path in root.rglob("*") if path.is_file())
    scripts_dir = repo_root / "scripts"
    if scripts_dir.exists():
        candidates.update(path for path in scripts_dir.rglob("*cxa*.py") if path.is_file())
    src_dir = repo_root / "src" / "opponent_adjusted"
    if src_dir.exists():
        candidates.update(path for path in src_dir.rglob("*cxa*") if path.is_file())
    contract_path = repo_root / DEFAULT_DASHBOARD_CONTRACT
    if contract_path.exists():
        try:
            payload = _read_json(contract_path)
            cxa_inputs = payload.get("metrics", {}).get("cxa", {}).get("inputs", {})
            if isinstance(cxa_inputs, dict):
                for item in cxa_inputs.values():
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        candidates.add(repo_root / Path(item["path"]))
        except Exception:  # noqa: BLE001 - contract parse is best effort
            pass
    return sorted(candidates)


def _is_binary_series(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_bool_dtype(non_null):
        return True
    unique = (
        set(non_null.astype(float).round(8).unique().tolist())
        if pd.api.types.is_numeric_dtype(non_null)
        else set()
    )
    return bool(unique) and unique.issubset({0.0, 1.0})


def _classify_column(column: str, series: pd.Series) -> tuple[str, str]:
    name = column.lower()
    if column in ID_COLUMNS or name.endswith("_id"):
        return "identifier", "identifier naming"
    if column in TARGET_COLUMNS or ("target" in name and "feature" not in name):
        return "target", "target-like naming"
    if column in PREDICTION_COLUMNS or name.startswith("predicted_") or "probability" in name:
        return "prediction", "prediction-like naming"
    if any(token in name for token in ("statsbomb", "provider_", "baseline_", "reference")):
        return "reference_only", "provider/reference signal"
    if any(
        token in name for token in ("outcome", "result", "post_", "future", "final_shot", "goal_")
    ):
        return "leakage_risk", "future/outcome signal"
    if _is_binary_series(series):
        return "binary_feature_candidate", "binary values"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric_feature_candidate", "numeric dtype"
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return "categorical_feature_candidate", "categorical/object dtype"
    return "excluded_or_unknown", "unclassified"


def _inventory_row(path: Path, repo_root: Path) -> tuple[dict[str, Any], TableSource | None]:
    relative = _repo_relative(path, repo_root)
    role_guess = _role_guess(path)
    exists = path.exists()
    row: dict[str, Any] = {
        "path": relative,
        "exists": exists,
        "file_type": _file_type(path) if exists else "missing",
        "row_count": None,
        "column_count": None,
        "size_bytes": int(path.stat().st_size) if exists and path.is_file() else None,
        "modified_time": (
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
            if exists and path.is_file()
            else None
        ),
        "role_guess": role_guess,
        "notes": "",
    }
    if not exists:
        row["notes"] = "missing"
        return row, None
    if path.suffix.lower() in TABULAR_SUFFIXES or path.suffix.lower() == ".json":
        frame, note = _read_table(path)
        if frame is not None:
            row["row_count"] = int(len(frame))
            row["column_count"] = int(len(frame.columns))
            if path.suffix.lower() == ".json" and row["file_type"] == "json":
                row["notes"] = "json_loaded_as_table"
            return row, TableSource(source=relative, role_guess=role_guess, frame=frame)
        if note:
            row["notes"] = note
    return row, None


def _id_status(column: str, row_count: int, missing_count: int) -> str:
    if row_count == 0:
        return "all_missing"
    if missing_count == 0:
        return "passed"
    if missing_count == row_count:
        return "all_missing"
    return "partially_missing"


def _id_quality_for_frame(source: TableSource) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = source.frame
    row_count = int(len(frame))
    for column in ID_COLUMNS:
        if column not in frame.columns:
            status = "optional_missing" if column in OPTIONAL_ID_COLUMNS else "column_missing"
            rows.append(
                {
                    "source": source.source,
                    "column": column,
                    "row_count": row_count,
                    "missing_count": row_count,
                    "missing_pct": 100.0 if row_count else 0.0,
                    "distinct_non_missing_count": 0,
                    "duplicate_non_missing_count": 0,
                    "status": status,
                    "notes": "column_not_present",
                }
            )
            continue
        series = frame[column]
        missing_count = int(series.isna().sum())
        non_null = series.dropna()
        distinct_count = int(non_null.nunique(dropna=True))
        duplicate_count = int(len(non_null) - distinct_count)
        rows.append(
            {
                "source": source.source,
                "column": column,
                "row_count": row_count,
                "missing_count": missing_count,
                "missing_pct": round((missing_count / row_count) * 100.0, 4) if row_count else 0.0,
                "distinct_non_missing_count": distinct_count,
                "duplicate_non_missing_count": duplicate_count,
                "status": _id_status(column, row_count, missing_count),
                "notes": "",
            }
        )
    return rows


def _sqlite_existing_tables(connection: sqlite3.Connection) -> set[str]:
    query = "SELECT name FROM sqlite_master WHERE type='table'"
    return {str(row[0]) for row in connection.execute(query).fetchall()}


def _sqlite_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row[1]) for row in rows}


def _sqlite_count(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
    return int(row[0]) if row else 0


def _sqlite_id_quality(
    sqlite_path: Path, repo_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    id_rows: list[dict[str, Any]] = []
    table_inventory: list[dict[str, Any]] = []
    if not sqlite_path.exists():
        return id_rows, table_inventory
    with sqlite3.connect(sqlite_path) as connection:
        existing = _sqlite_existing_tables(connection)
        for table in SQLITE_RELEVANT_TABLES:
            source = f"{_repo_relative(sqlite_path, repo_root)}::{table}"
            if table not in existing:
                table_inventory.append(
                    {"source": source, "row_count": 0, "columns": "", "notes": "table_missing"}
                )
                for column in ID_COLUMNS:
                    id_rows.append(
                        {
                            "source": source,
                            "column": column,
                            "row_count": 0,
                            "missing_count": 0,
                            "missing_pct": 0.0,
                            "distinct_non_missing_count": 0,
                            "duplicate_non_missing_count": 0,
                            "status": "column_missing",
                            "notes": "table_missing",
                        }
                    )
                continue
            columns = _sqlite_columns(connection, table)
            row_count = _sqlite_count(connection, table)
            table_inventory.append(
                {
                    "source": source,
                    "row_count": row_count,
                    "columns": ",".join(sorted(columns)),
                    "notes": "",
                }
            )
            for column in ID_COLUMNS:
                if column not in columns:
                    status = (
                        "optional_missing" if column in OPTIONAL_ID_COLUMNS else "column_missing"
                    )
                    id_rows.append(
                        {
                            "source": source,
                            "column": column,
                            "row_count": row_count,
                            "missing_count": row_count,
                            "missing_pct": 100.0 if row_count else 0.0,
                            "distinct_non_missing_count": 0,
                            "duplicate_non_missing_count": 0,
                            "status": status,
                            "notes": "column_not_present",
                        }
                    )
                    continue
                query = (
                    f"SELECT COUNT(*), "
                    f'SUM(CASE WHEN "{column}" IS NULL THEN 1 ELSE 0 END), '
                    f'COUNT(DISTINCT "{column}") '
                    f'FROM "{table}"'
                )
                total, missing, distinct_count = connection.execute(query).fetchone()
                total_int = int(total or 0)
                missing_int = int(missing or 0)
                distinct_int = int(distinct_count or 0)
                duplicate_count = int(max(total_int - missing_int - distinct_int, 0))
                id_rows.append(
                    {
                        "source": source,
                        "column": column,
                        "row_count": total_int,
                        "missing_count": missing_int,
                        "missing_pct": (
                            round((missing_int / total_int) * 100.0, 4) if total_int else 0.0
                        ),
                        "distinct_non_missing_count": distinct_int,
                        "duplicate_non_missing_count": duplicate_count,
                        "status": _id_status(column, total_int, missing_int),
                        "notes": "",
                    }
                )
    return id_rows, table_inventory


def _feature_inventory(sources: list[TableSource]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sources:
        if source.role_guess not in {"features", "predictions", "unknown"}:
            continue
        for column in source.frame.columns:
            classification, reason = _classify_column(column, source.frame[column])
            rows.append(
                {
                    "source": source.source,
                    "column": column,
                    "classification": classification,
                    "notes": reason,
                }
            )
    return pd.DataFrame(rows, columns=["source", "column", "classification", "notes"])


def _target_like_columns(columns: list[str]) -> list[str]:
    matches: list[str] = []
    for column in columns:
        name = column.lower()
        if (
            column in TARGET_COLUMNS
            or "target" in name
            or "shot_created" in name
            or "created_shot" in name
        ):
            matches.append(column)
    return sorted(set(matches))


def _prediction_like_columns(columns: list[str]) -> list[str]:
    matches: list[str] = []
    for column in columns:
        name = column.lower()
        if column in PREDICTION_COLUMNS or name.startswith("predicted_") or "probability" in name:
            matches.append(column)
    return sorted(set(matches))


def _target_audit(sources: list[TableSource]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for column in _target_like_columns(source.frame.columns.tolist()):
            series = source.frame[column]
            row_count = int(len(series))
            null_count = int(series.isna().sum())
            numeric = pd.to_numeric(series, errors="coerce")
            binary = _is_binary_series(series)
            row = {
                "source": source.source,
                "target_column": column,
                "row_count": row_count,
                "positive_count": int((numeric.fillna(0) > 0).sum()) if binary else None,
                "positive_rate": (
                    float((numeric.fillna(0) > 0).mean()) if binary and row_count else None
                ),
                "null_count": null_count,
                "min": float(numeric.min()) if numeric.notna().any() else None,
                "mean": float(numeric.mean()) if numeric.notna().any() else None,
                "max": float(numeric.max()) if numeric.notna().any() else None,
                "interpretation": "binary_target" if binary else "numeric_target_or_reference",
                "risk_notes": (
                    "target_column_present_in_feature_like_source"
                    if source.role_guess == "features"
                    else ""
                ),
            }
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "target_column",
            "row_count",
            "positive_count",
            "positive_rate",
            "null_count",
            "min",
            "mean",
            "max",
            "interpretation",
            "risk_notes",
        ],
    )


def _expected_zero_to_one(column: str) -> bool:
    name = column.lower()
    return any(token in name for token in ("predicted", "prob", "_share", "cxa_value", "cxa"))


def _prediction_audit(sources: list[TableSource], metrics_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sources:
        for column in _prediction_like_columns(source.frame.columns.tolist()):
            numeric = pd.to_numeric(source.frame[column], errors="coerce")
            if numeric.notna().sum() == 0:
                continue
            outside_count = None
            if _expected_zero_to_one(column):
                outside_count = int(((numeric < 0) | (numeric > 1)).sum())
            rows.append(
                {
                    "source": source.source,
                    "prediction_column": column,
                    "row_count": int(len(source.frame)),
                    "null_count": int(numeric.isna().sum()),
                    "min": float(numeric.min()),
                    "mean": float(numeric.mean()),
                    "max": float(numeric.max()),
                    "outside_expected_range_count": outside_count,
                    "notes": "",
                }
            )
    if metrics_path.exists():
        metrics = _read_json(metrics_path)
        rows.append(
            {
                "source": _repo_relative(metrics_path, Path.cwd()),
                "prediction_column": "metrics_summary",
                "row_count": int(metrics.get("row_count") or metrics.get("n_rows") or 0),
                "null_count": None,
                "min": None,
                "mean": None,
                "max": None,
                "outside_expected_range_count": None,
                "notes": json.dumps(
                    {
                        "brier_score": metrics.get("brier_score", metrics.get("brier")),
                        "log_loss": metrics.get("log_loss", metrics.get("log_loss_mean")),
                        "roc_auc": metrics.get("roc_auc", metrics.get("auc_mean")),
                        "calibration": metrics.get("calibration"),
                        "split_metadata": {
                            "n_splits": metrics.get("n_splits"),
                            "fold_count": (
                                len(metrics.get("folds", []))
                                if isinstance(metrics.get("folds"), list)
                                else None
                            ),
                        },
                    }
                ),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "source",
            "prediction_column",
            "row_count",
            "null_count",
            "min",
            "mean",
            "max",
            "outside_expected_range_count",
            "notes",
        ],
    )


def _aggregation_level(path: str) -> str:
    lower = path.lower()
    if "player" in lower:
        return "player"
    if "team" in lower:
        return "team"
    if "sequence" in lower:
        return "sequence"
    if "possession" in lower:
        return "possession"
    return "unknown"


def _aggregate_audit(sources: list[TableSource]) -> tuple[pd.DataFrame, dict[str, float]]:
    rows: list[dict[str, Any]] = []
    consistency: dict[str, float] = {}
    action_total = None
    for source in sources:
        if source.role_guess == "predictions":
            if "cxa_value" in source.frame.columns:
                action_total = float(
                    pd.to_numeric(source.frame["cxa_value"], errors="coerce").fillna(0).sum()
                )
            elif "predicted_cxa" in source.frame.columns:
                action_total = float(
                    pd.to_numeric(source.frame["predicted_cxa"], errors="coerce").fillna(0).sum()
                )
    for source in sources:
        if source.role_guess != "aggregates":
            continue
        frame = source.frame
        key_columns = [
            column
            for column in frame.columns
            if column.endswith("_id") or column in {"match_id", "possession", "sequence_id"}
        ]
        value_columns = [
            column
            for column in frame.columns
            if column not in key_columns and pd.api.types.is_numeric_dtype(frame[column])
        ]
        missing_key_count = int(frame[key_columns].isna().any(axis=1).sum()) if key_columns else 0
        total_cxa_sum = None
        if "total_cxa" in frame.columns:
            total_cxa_sum = float(
                pd.to_numeric(frame["total_cxa"], errors="coerce").fillna(0).sum()
            )
        elif "cxa_value" in frame.columns:
            total_cxa_sum = float(
                pd.to_numeric(frame["cxa_value"], errors="coerce").fillna(0).sum()
            )
        action_count_sum = (
            float(pd.to_numeric(frame["action_count"], errors="coerce").fillna(0).sum())
            if "action_count" in frame.columns
            else None
        )
        notes = ""
        level = _aggregation_level(source.source)
        if (
            action_total is not None
            and total_cxa_sum is not None
            and level in {"player", "team", "sequence"}
        ):
            diff = total_cxa_sum - action_total
            consistency[f"{level}_minus_action"] = diff
            notes = f"reconciles_to_action_total_delta={diff:.8f}"
        rows.append(
            {
                "source": source.source,
                "aggregation_level": level,
                "row_count": int(len(frame)),
                "key_columns": ",".join(key_columns),
                "value_columns": ",".join(value_columns),
                "missing_key_count": missing_key_count,
                "total_cxa_sum": total_cxa_sum,
                "action_count_sum": action_count_sum,
                "notes": notes,
            }
        )
    return (
        pd.DataFrame(
            rows,
            columns=[
                "source",
                "aggregation_level",
                "row_count",
                "key_columns",
                "value_columns",
                "missing_key_count",
                "total_cxa_sum",
                "action_count_sum",
                "notes",
            ],
        ),
        consistency,
    )


def _build_risk_register(
    inventory: pd.DataFrame,
    id_quality: pd.DataFrame,
    feature_inventory: pd.DataFrame,
    aggregate_consistency: dict[str, float],
    metrics_exists: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_risk(
        *,
        severity: str,
        category: str,
        finding: str,
        evidence: str,
        recommendation: str,
        suggested_next_pr: str,
    ) -> None:
        rows.append(
            {
                "risk_id": f"R{len(rows) + 1:03d}",
                "severity": severity,
                "category": category,
                "finding": finding,
                "evidence": evidence,
                "recommendation": recommendation,
                "suggested_next_pr": suggested_next_pr,
            }
        )

    missing_contract_paths = inventory[
        (~inventory["exists"]) & (inventory["path"].str.contains("outputs/modeling/cxa"))
    ]
    if not missing_contract_paths.empty:
        add_risk(
            severity="high",
            category="dashboard_contract",
            finding="Dashboard contract references missing CxA generated files.",
            evidence=", ".join(missing_contract_paths["path"].head(5).tolist()),
            recommendation="Backfill or regenerate missing baseline output paths before CxA promotion.",
            suggested_next_pr="cxa-contract-alignment",
        )

    required_id_issues = id_quality[
        (~id_quality["column"].isin(OPTIONAL_ID_COLUMNS))
        & (id_quality["status"].isin({"column_missing", "all_missing", "partially_missing"}))
    ]
    if not required_id_issues.empty:
        add_risk(
            severity="high",
            category="ID_quality",
            finding="Required action-level identifiers are incomplete in current CxA artifacts.",
            evidence=", ".join(
                required_id_issues.head(8)
                .apply(lambda row: f"{row['source']}:{row['column']}={row['status']}", axis=1)
                .tolist()
            ),
            recommendation="Harden ID lineage for action/event/match/team/player keys before diagnostic CxA modeling.",
            suggested_next_pr="cxa-id-lineage-hardening",
        )

    leakage_rows = feature_inventory[feature_inventory["classification"] == "leakage_risk"]
    if not leakage_rows.empty:
        add_risk(
            severity="high",
            category="leakage",
            finding="Columns with post-action or outcome-like signals appear in CxA feature surfaces.",
            evidence=", ".join(leakage_rows["column"].drop_duplicates().head(8).tolist()),
            recommendation="Exclude leakage-risk columns via explicit feature contract and training allowlist.",
            suggested_next_pr="cxa-leakage-exclusion-contract",
        )

    target_in_feature_rows = feature_inventory[
        (feature_inventory["classification"] == "target")
        & (feature_inventory["source"].str.contains("feature_store/cxa"))
    ]
    if not target_in_feature_rows.empty:
        add_risk(
            severity="high",
            category="target_definition",
            finding="Target-like columns are present in feature-store CxA artifacts and can leak into training inputs.",
            evidence=", ".join(target_in_feature_rows["column"].drop_duplicates().head(6).tolist()),
            recommendation="Split target/reference fields from feature inputs and enforce exclusions in training code.",
            suggested_next_pr="cxa-target-feature-separation",
        )

    for key, delta in aggregate_consistency.items():
        if abs(delta) > 1e-6:
            add_risk(
                severity="medium",
                category="output_consistency",
                finding="Aggregate totals do not fully reconcile with action-level totals.",
                evidence=f"{key}={delta:.8f}",
                recommendation="Review aggregation grouping keys and row filters for CxA exports.",
                suggested_next_pr="cxa-aggregate-reconciliation",
            )

    if not metrics_exists:
        add_risk(
            severity="medium",
            category="model_validation",
            finding="CxA metrics report is missing; baseline quality is not auditable from generated files.",
            evidence=str(MODELING_DIR / "reports" / "metrics.json"),
            recommendation="Regenerate and persist baseline metrics with stable schema.",
            suggested_next_pr="cxa-metrics-schema-hardening",
        )

    if not rows:
        add_risk(
            severity="low",
            category="documentation",
            finding="No major CxA current-state issues were auto-detected from available artifacts.",
            evidence="audit_run_complete",
            recommendation="Proceed to diagnostic CxA candidate modeling with existing controls.",
            suggested_next_pr="cxa-diagnostic-candidate-training",
        )
    return pd.DataFrame(rows)


def _write_markdown_report(
    path: Path,
    *,
    summary: dict[str, Any],
    inventory: pd.DataFrame,
    id_quality: pd.DataFrame,
    target_audit: pd.DataFrame,
    prediction_audit: pd.DataFrame,
    aggregate_audit: pd.DataFrame,
    feature_inventory: pd.DataFrame,
    risk_register: pd.DataFrame,
    sqlite_tables: pd.DataFrame,
) -> None:
    high = int((risk_register["severity"] == "high").sum()) if not risk_register.empty else 0
    medium = int((risk_register["severity"] == "medium").sum()) if not risk_register.empty else 0
    low = int((risk_register["severity"] == "low").sum()) if not risk_register.empty else 0
    lines = [
        "# CxA Current-State Audit",
        "",
        "## Executive summary",
        f"- Files checked: {summary['files_checked']} | found: {summary['files_found']} | missing: {summary['files_missing']}",
        f"- Risks: high={high}, medium={medium}, low={low}",
        "",
        "## Current output inventory",
        f"- Contract/output files present: {int(inventory['exists'].sum())}/{len(inventory)}",
        "",
        "## Current CxA pipeline interpretation",
        "- Baseline CxA artifacts are treated as current-state references and not promoted diagnostics.",
        "",
        "## Available feature/target/prediction files",
        f"- Feature inventory rows: {len(feature_inventory)}",
        f"- Target audit rows: {len(target_audit)}",
        f"- Prediction audit rows: {len(prediction_audit)}",
        "",
        "## ID quality summary",
        f"- ID audit rows: {len(id_quality)}",
        f"- Passed: {int((id_quality['status'] == 'passed').sum()) if not id_quality.empty else 0}",
        f"- Non-passed: {int((id_quality['status'] != 'passed').sum()) if not id_quality.empty else 0}",
        "",
        "## Target definition audit",
        f"- Target-like columns found: {int(target_audit['target_column'].nunique()) if not target_audit.empty else 0}",
        "",
        "## Prediction and metrics audit",
        f"- Prediction columns audited: {int(prediction_audit['prediction_column'].nunique()) if not prediction_audit.empty else 0}",
        "",
        "## Aggregate consistency audit",
        f"- Aggregate artifacts audited: {len(aggregate_audit)}",
        "",
        "## Leakage/reference-risk findings",
        f"- Leakage-risk feature columns: {int((feature_inventory['classification'] == 'leakage_risk').sum()) if not feature_inventory.empty else 0}",
        "",
        "## Dashboard contract status",
        f"- Dashboard contract path: {DEFAULT_DASHBOARD_CONTRACT.as_posix()}",
        "",
        "## SQLite persistence status",
        f"- Audited SQLite table surfaces: {len(sqlite_tables)}",
        "",
        "## Risk register summary",
        f"- High: {high}, Medium: {medium}, Low: {low}",
        "",
        "## Recommended next PRs",
        f"- Primary recommendation: {summary['recommended_next_pr']}",
        "",
        "1. fix critical ID/lineage issues if any",
        "2. define CxA feature contract and leakage exclusions",
        "3. build diagnostic CxA training candidates",
        "4. validate diagnostic CxA against fair baseline",
        "5. generate promoted CxA result outputs",
        "6. add CxA feature impact and portfolio summary",
        "7. wire CxA portfolio into dashboard later",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    *,
    repo_root: Path = Path.cwd(),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = _collect_paths(repo_root)
    inventory_rows: list[dict[str, Any]] = []
    table_sources: list[TableSource] = []
    for path in candidate_paths:
        row, source = _inventory_row(path, repo_root)
        inventory_rows.append(row)
        if source is not None and source.frame is not None:
            table_sources.append(source)

    inventory = pd.DataFrame(inventory_rows).sort_values("path").reset_index(drop=True)

    id_quality_rows: list[dict[str, Any]] = []
    for source in table_sources:
        id_quality_rows.extend(_id_quality_for_frame(source))

    sqlite_path = repo_root / DEFAULT_SQLITE_PATH
    sqlite_id_rows, sqlite_table_rows = _sqlite_id_quality(sqlite_path, repo_root)
    id_quality_rows.extend(sqlite_id_rows)
    id_quality = pd.DataFrame(
        id_quality_rows,
        columns=[
            "source",
            "column",
            "row_count",
            "missing_count",
            "missing_pct",
            "distinct_non_missing_count",
            "duplicate_non_missing_count",
            "status",
            "notes",
        ],
    )

    feature_inventory = _feature_inventory(table_sources)
    target_audit = _target_audit(table_sources)
    prediction_audit = _prediction_audit(
        table_sources, repo_root / MODELING_DIR / "reports" / "metrics.json"
    )
    aggregate_audit, consistency = _aggregate_audit(table_sources)
    sqlite_tables = pd.DataFrame(sqlite_table_rows)

    risk_register = _build_risk_register(
        inventory=inventory,
        id_quality=id_quality,
        feature_inventory=feature_inventory,
        aggregate_consistency=consistency,
        metrics_exists=(repo_root / MODELING_DIR / "reports" / "metrics.json").exists(),
    )

    outputs = {
        "json": output_dir / "cxa_current_state_audit.json",
        "markdown": output_dir / "cxa_current_state_audit.md",
        "inventory_csv": output_dir / "cxa_output_inventory.csv",
        "id_quality_csv": output_dir / "cxa_id_quality.csv",
        "feature_inventory_csv": output_dir / "cxa_feature_inventory.csv",
        "target_audit_csv": output_dir / "cxa_target_audit.csv",
        "prediction_audit_csv": output_dir / "cxa_prediction_audit.csv",
        "aggregate_audit_csv": output_dir / "cxa_aggregate_audit.csv",
        "risk_register_csv": output_dir / "cxa_risk_register.csv",
    }

    inventory.to_csv(outputs["inventory_csv"], index=False)
    id_quality.to_csv(outputs["id_quality_csv"], index=False)
    feature_inventory.to_csv(outputs["feature_inventory_csv"], index=False)
    target_audit.to_csv(outputs["target_audit_csv"], index=False)
    prediction_audit.to_csv(outputs["prediction_audit_csv"], index=False)
    aggregate_audit.to_csv(outputs["aggregate_audit_csv"], index=False)
    risk_register.to_csv(outputs["risk_register_csv"], index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files_checked": int(len(inventory)),
        "files_found": int(inventory["exists"].sum()),
        "files_missing": int((~inventory["exists"]).sum()),
        "total_rows_by_artifact": {
            row["path"]: int(row["row_count"])
            for row in inventory.to_dict(orient="records")
            if row.get("row_count") is not None and pd.notna(row["row_count"])
        },
        "id_quality_summary": (
            id_quality.groupby("status").size().to_dict() if not id_quality.empty else {}
        ),
        "target_summary": (
            target_audit.groupby("target_column")["row_count"].max().to_dict()
            if not target_audit.empty
            else {}
        ),
        "prediction_summary": (
            prediction_audit.groupby("prediction_column")["row_count"].max().to_dict()
            if not prediction_audit.empty
            else {}
        ),
        "aggregate_consistency_summary": consistency,
        "high_risk_findings_count": (
            int((risk_register["severity"] == "high").sum()) if not risk_register.empty else 0
        ),
        "medium_risk_findings_count": (
            int((risk_register["severity"] == "medium").sum()) if not risk_register.empty else 0
        ),
        "low_risk_findings_count": (
            int((risk_register["severity"] == "low").sum()) if not risk_register.empty else 0
        ),
        "recommended_next_pr": (
            str(risk_register.iloc[0]["suggested_next_pr"])
            if not risk_register.empty
            else "cxa-diagnostic-candidate-training"
        ),
        "audit_outputs": {key: str(value) for key, value in outputs.items()},
    }

    _write_markdown_report(
        outputs["markdown"],
        summary=summary,
        inventory=inventory,
        id_quality=id_quality,
        target_audit=target_audit,
        prediction_audit=prediction_audit,
        aggregate_audit=aggregate_audit,
        feature_inventory=feature_inventory,
        risk_register=risk_register,
        sqlite_tables=sqlite_tables,
    )
    outputs["json"].write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit current-state CxA outputs and contracts.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = run_audit(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
