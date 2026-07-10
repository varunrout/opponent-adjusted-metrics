#!/usr/bin/env python
"""Build a governed CxA+ diagnostic feature matrix (no model training)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ACTION_FEATURES_PATH = Path("feature_store/cxa/action_features.parquet")
DEFAULT_TARGETS_PATH = Path("feature_store/cxa_plus/cxa_plus_action_targets.parquet")
DEFAULT_CONTRACT_PATH = Path(
    "outputs/modeling/cxa_plus/diagnostic_v1/contracts/feature_contract.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/modeling/cxa_plus/diagnostic_v1")

PRIMARY_TARGET = "shot_within_next_5_actions"
JOIN_KEYS = [
    "action_id",
    "event_id",
    "match_id",
    "possession",
    "sequence_id",
    "action_position",
]
AUDIT_COLUMNS = [
    "action_id",
    "event_id",
    "match_id",
    "possession",
    "sequence_id",
    "action_position",
    "team_id",
    "player_id",
    PRIMARY_TARGET,
]
EXPLICIT_FORBIDDEN_COLUMNS = {
    "shot_created",
    "created_shot_id",
    "created_shot_cxg",
    "shot_within_next_1_action",
    "shot_within_next_3_actions",
    "shot_within_next_5_actions",
    "shot_later_in_possession",
    "max_created_shot_cxg_within_next_5_actions",
    "sum_created_shot_cxg_rest_of_possession",
    "discounted_downstream_shot_value",
}
LEAKAGE_TOKENS = (
    "future",
    "downstream",
    "outcome",
    "result",
    "predicted",
    "model_",
    "probability",
    "score",
    "within_next",
    "rest_of_possession",
)
CREATED_SHOT_PREFIX = "created_shot_"


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format for {path}: {path.suffix}")


def _read_contract(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required CxA+ feature contract not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _required_columns_exist(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns for join: {missing}")


def _ensure_no_duplicate_ids(frame: pd.DataFrame, *, id_column: str, label: str) -> None:
    if id_column not in frame.columns:
        return
    duplicate_count = int(frame[id_column].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"{label} has duplicate {id_column} values: duplicate_count={duplicate_count}"
        )


def _ensure_unique_join_keys(frame: pd.DataFrame, *, label: str) -> None:
    duplicate_count = int(frame.duplicated(subset=JOIN_KEYS).sum())
    if duplicate_count:
        raise ValueError(
            f"{label} has duplicate join keys; expected one row per action join key. "
            f"duplicate_join_key_count={duplicate_count}"
        )


def _merge_targets_with_features(
    targets: pd.DataFrame, action_features: pd.DataFrame
) -> pd.DataFrame:
    merged = targets.merge(
        action_features,
        on=JOIN_KEYS,
        how="left",
        validate="one_to_one",
        suffixes=("_target", "_feature"),
        indicator=True,
    )
    unmatched = merged.loc[merged["_merge"] != "both"]
    if not unmatched.empty:
        sample = unmatched[JOIN_KEYS].head(5).to_dict(orient="records")
        raise ValueError(
            "CxA+ feature matrix build failed: some target rows did not match action features. "
            f"unmatched_row_count={len(unmatched)} sample={sample}"
        )
    if len(merged) != len(targets):
        raise ValueError(
            "CxA+ feature matrix build changed row count unexpectedly. "
            f"target_rows={len(targets)} merged_rows={len(merged)}"
        )
    return merged.drop(columns=["_merge"])


def _forbidden_feature_columns(contract: dict[str, Any]) -> set[str]:
    forbidden = set(EXPLICIT_FORBIDDEN_COLUMNS)
    for key in (
        "identifier_columns",
        "target_columns",
        "reference_only_columns",
        "leakage_excluded_columns",
        "model_output_columns",
        "requires_review_columns",
    ):
        values = contract.get(key, [])
        if isinstance(values, list):
            forbidden.update(values)
    excluded_columns = contract.get("excluded_columns", {})
    if isinstance(excluded_columns, dict):
        for values in excluded_columns.values():
            if isinstance(values, list):
                forbidden.update(values)
    return forbidden


def _is_leakage_like(column: str) -> bool:
    name = column.lower()
    if name.startswith(CREATED_SHOT_PREFIX):
        return True
    if any(token in name for token in LEAKAGE_TOKENS):
        return True
    return False


def _safe_feature_columns(
    merged: pd.DataFrame,
    *,
    contract_forbidden: set[str],
    target_columns: set[str],
) -> tuple[list[str], list[str]]:
    safe_columns: list[str] = []
    dropped_all_null_columns: list[str] = []
    for column in merged.columns:
        if column in AUDIT_COLUMNS:
            continue
        if column in JOIN_KEYS:
            continue
        if column in target_columns:
            continue
        if column in contract_forbidden:
            continue
        if column.endswith("_target"):
            continue
        if _is_leakage_like(column):
            continue
        if not merged[column].notna().any():
            dropped_all_null_columns.append(column)
            continue
        safe_columns.append(column)
    return sorted(dict.fromkeys(safe_columns)), sorted(dict.fromkeys(dropped_all_null_columns))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if pd.isna(value) and not isinstance(value, (bool, str)):
        return None
    return value


def _quality_checks(
    *,
    target_rows: int,
    matrix: pd.DataFrame,
    safe_features: list[str],
    dropped_leakage_like_columns: list[str],
    dropped_all_null_candidate_columns: list[str],
) -> pd.DataFrame:
    rows = [
        {
            "check_name": "target_row_count",
            "value": target_rows,
            "status": "passed",
            "severity": "info",
            "notes": "Rows in governed CxA+ target table.",
        },
        {
            "check_name": "matrix_row_count",
            "value": len(matrix),
            "status": "passed" if len(matrix) == target_rows else "failed",
            "severity": "blocker",
            "notes": "Feature matrix rows must match target rows exactly.",
        },
        {
            "check_name": "primary_target_present",
            "value": PRIMARY_TARGET in matrix.columns,
            "status": "passed" if PRIMARY_TARGET in matrix.columns else "failed",
            "severity": "blocker",
            "notes": "Primary target required for downstream diagnostic modelling.",
        },
        {
            "check_name": "eligible_feature_count",
            "value": len(safe_features),
            "status": "passed" if safe_features else "failed",
            "severity": "blocker",
            "notes": "Safe model-eligible feature columns retained from CxA action features.",
        },
        {
            "check_name": "dropped_leakage_like_column_count",
            "value": len(dropped_leakage_like_columns),
            "status": "passed",
            "severity": "info",
            "notes": "Columns excluded by leakage/reference name guards.",
        },
        {
            "check_name": "dropped_all_null_candidate_column_count",
            "value": len(dropped_all_null_candidate_columns),
            "status": "passed",
            "severity": "info",
            "notes": "Safe-looking candidate columns excluded because every value is null.",
        },
    ]
    failed_checks = [row["check_name"] for row in rows if row["status"] == "failed"]
    rows.append(
        {
            "check_name": "failed_checks",
            "value": ";".join(failed_checks),
            "status": "failed" if failed_checks else "passed",
            "severity": "blocker" if failed_checks else "info",
            "notes": "Empty when all blocker checks pass.",
        }
    )
    return pd.DataFrame(rows)


def _build_summary(
    *,
    action_features_path: Path,
    targets_path: Path,
    contract_path: Path,
    output_dir: Path,
    matrix: pd.DataFrame,
    safe_features: list[str],
    dropped_leakage_like_columns: list[str],
    dropped_all_null_candidate_columns: list[str],
    quality: pd.DataFrame,
    outputs: dict[str, Path],
) -> dict[str, Any]:
    failed_checks = quality.loc[quality["status"] == "failed", "check_name"].tolist()
    return {
        "metric": "cxa_plus",
        "model_version": "diagnostic_v1",
        "build_type": "feature_matrix_only",
        "primary_target": PRIMARY_TARGET,
        "join_keys": JOIN_KEYS,
        "input_paths": {
            "action_features": action_features_path.as_posix(),
            "targets": targets_path.as_posix(),
            "feature_contract": contract_path.as_posix(),
        },
        "row_count": len(matrix),
        "column_count": len(matrix.columns),
        "eligible_model_features": safe_features,
        "eligible_feature_count": len(safe_features),
        "dropped_leakage_like_columns": dropped_leakage_like_columns,
        "dropped_all_null_candidate_columns": dropped_all_null_candidate_columns,
        "quality_status": "failed" if failed_checks else "passed",
        "failed_checks": failed_checks,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "output_dir": output_dir.as_posix(),
        "outputs": {name: path.as_posix() for name, path in outputs.items()},
    }


def _markdown_table(frame: pd.DataFrame, max_rows: int = 25) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy().fillna("")
    columns = [str(column) for column in display.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(row[column]).replace("|", "\\|") for column in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _build_report(summary: dict[str, Any], quality: pd.DataFrame) -> str:
    feature_preview = summary["eligible_model_features"][:40]
    feature_list = "\n".join(f"- `{column}`" for column in feature_preview) or "- _None_"
    if len(summary["eligible_model_features"]) > len(feature_preview):
        feature_list += "\n- _... additional features omitted for brevity_"
    dropped_all_null_preview = summary["dropped_all_null_candidate_columns"][:40]
    dropped_all_null_list = (
        "\n".join(f"- `{column}`" for column in dropped_all_null_preview)
        if dropped_all_null_preview
        else "- _None_"
    )
    if len(summary["dropped_all_null_candidate_columns"]) > len(dropped_all_null_preview):
        dropped_all_null_list += "\n- _... additional columns omitted for brevity_"

    return f"""# CxA+ Diagnostic Feature Matrix Report

## Executive summary

This step builds a model-ready diagnostic CxA+ feature matrix by joining the
governed CxA+ targets with the original CxA action feature store. It does not
train, score, validate, or promote any model.

## Inputs and join contract

- CxA action features: `{summary["input_paths"]["action_features"]}`
- Governed CxA+ targets: `{summary["input_paths"]["targets"]}`
- Existing CxA+ contract: `{summary["input_paths"]["feature_contract"]}`
- Join keys: `{", ".join(summary["join_keys"])}`

The join enforces one row per target action. The builder fails if join-key
duplicates appear or if any target row cannot be matched back to source action
features.

## Matrix shape

- Rows: {summary["row_count"]}
- Columns: {summary["column_count"]}
- Primary target column: `{summary["primary_target"]}`
- Eligible model feature count: {summary["eligible_feature_count"]}

## Leakage and reference controls

The matrix excludes `shot_created`, `created_shot_id`, `created_shot_cxg`,
every `created_shot_*` field, all CxA+ target/value leakage columns, and
prediction/model-output style fields from eligible model features.

## Eligible feature preview

{feature_list}

## Dropped all-null candidate columns

The builder excludes otherwise safe-looking columns when every value is null.

{dropped_all_null_list}

## Quality checks

{_markdown_table(quality)}

## Outputs

- `outputs/modeling/cxa_plus/diagnostic_v1/datasets/feature_matrix.parquet`
- `outputs/modeling/cxa_plus/diagnostic_v1/datasets/feature_matrix_summary.json`
- `outputs/modeling/cxa_plus/diagnostic_v1/diagnostics/feature_matrix_quality.csv`
- `outputs/modeling/cxa_plus/diagnostic_v1/reports/feature_matrix_report.md`
"""


def build_cxa_plus_feature_matrix(
    *,
    action_features_path: Path = DEFAULT_ACTION_FEATURES_PATH,
    targets_path: Path = DEFAULT_TARGETS_PATH,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Build and write CxA+ diagnostic feature-matrix artifacts."""

    action_features = _read_table(action_features_path)
    targets = _read_table(targets_path)
    contract = _read_contract(contract_path)

    _required_columns_exist(action_features, JOIN_KEYS, "action_features")
    _required_columns_exist(targets, JOIN_KEYS + [PRIMARY_TARGET], "targets")
    _ensure_no_duplicate_ids(action_features, id_column="action_id", label="action_features")
    _ensure_no_duplicate_ids(targets, id_column="action_id", label="targets")
    _ensure_unique_join_keys(action_features, label="action_features")
    _ensure_unique_join_keys(targets, label="targets")

    target_join_projection = targets[JOIN_KEYS + [PRIMARY_TARGET]].copy()
    merged = _merge_targets_with_features(target_join_projection, action_features)

    contract_forbidden = _forbidden_feature_columns(contract)
    target_columns = set(target_join_projection.columns)
    safe_features, dropped_all_null_candidate_columns = _safe_feature_columns(
        merged,
        contract_forbidden=contract_forbidden,
        target_columns=target_columns,
    )
    dropped_leakage_like_columns = sorted(
        column
        for column in action_features.columns
        if column not in JOIN_KEYS and (_is_leakage_like(column) or column in contract_forbidden)
    )

    audit_columns = [column for column in AUDIT_COLUMNS if column in merged.columns]
    matrix_columns = audit_columns + [
        column for column in safe_features if column not in audit_columns
    ]
    matrix = merged[matrix_columns].copy()

    datasets_dir = output_dir / "datasets"
    diagnostics_dir = output_dir / "diagnostics"
    reports_dir = output_dir / "reports"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "feature_matrix": datasets_dir / "feature_matrix.parquet",
        "feature_matrix_summary": datasets_dir / "feature_matrix_summary.json",
        "feature_matrix_quality": diagnostics_dir / "feature_matrix_quality.csv",
        "feature_matrix_report": reports_dir / "feature_matrix_report.md",
    }

    quality = _quality_checks(
        target_rows=len(targets),
        matrix=matrix,
        safe_features=safe_features,
        dropped_leakage_like_columns=dropped_leakage_like_columns,
        dropped_all_null_candidate_columns=dropped_all_null_candidate_columns,
    )
    summary = _build_summary(
        action_features_path=action_features_path,
        targets_path=targets_path,
        contract_path=contract_path,
        output_dir=output_dir,
        matrix=matrix,
        safe_features=safe_features,
        dropped_leakage_like_columns=dropped_leakage_like_columns,
        dropped_all_null_candidate_columns=dropped_all_null_candidate_columns,
        quality=quality,
        outputs=output_paths,
    )

    matrix.to_parquet(output_paths["feature_matrix"], index=False)
    output_paths["feature_matrix_summary"].write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    quality.to_csv(output_paths["feature_matrix_quality"], index=False)
    output_paths["feature_matrix_report"].write_text(
        _build_report(summary, quality),
        encoding="utf-8",
    )
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action-features", type=Path, default=DEFAULT_ACTION_FEATURES_PATH)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_cxa_plus_feature_matrix(
        action_features_path=args.action_features,
        targets_path=args.targets,
        contract_path=args.contract,
        output_dir=args.output_dir,
    )
    print(json.dumps({name: path.as_posix() for name, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
