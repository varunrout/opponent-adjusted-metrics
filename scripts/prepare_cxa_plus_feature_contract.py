#!/usr/bin/env python
"""Prepare the first governed CxA+ modelling feature contract."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT_PATH = Path("feature_store") / "cxa_plus" / "cxa_plus_action_targets.parquet"
DEFAULT_OUTPUT_DIR = Path("outputs") / "modeling" / "cxa_plus" / "diagnostic_v1"
MODEL_VERSION = "diagnostic_v1"
METRIC = "cxa_plus"
PRIMARY_TARGET = "shot_within_next_5_actions"
SPLIT_GROUP_COLUMN = "match_id"

IDENTIFIER_COLUMNS = {
    "action_id",
    "event_id",
    "match_id",
    "possession",
    "sequence_id",
    "team_id",
    "player_id",
}
TARGET_COLUMNS = {
    "shot_within_next_1_action",
    "shot_within_next_3_actions",
    "shot_within_next_5_actions",
    "shot_later_in_possession",
}
REFERENCE_ONLY_COLUMNS = {
    "shot_created",
    "created_shot_id",
    "created_shot_cxg",
}
LEAKAGE_EXCLUDED_COLUMNS = {
    "max_created_shot_cxg_within_next_5_actions",
    "sum_created_shot_cxg_rest_of_possession",
    "discounted_downstream_shot_value",
}
OUTPUT_MODEL_COLUMNS = {
    "predicted_cxa_plus",
    "predicted_shot_within_next_5_actions",
    "prediction_source",
    "model_version",
}
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
LEAKAGE_TOKENS = ("future", "downstream", "outcome", "result", "rest_of_possession")
MODEL_OUTPUT_TOKENS = ("predicted", "model_", "probability", "score")
CREATED_SHOT_PREFIX = "created_shot_"


@dataclass(frozen=True)
class ColumnDecision:
    column: str
    classification: str
    feature_group: str
    reason: str
    severity: str
    can_appear_in_outputs: bool


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


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported CxA+ target input format: {path.suffix}")


def _is_binary_series(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    if pd.api.types.is_bool_dtype(non_null):
        return True
    if not pd.api.types.is_numeric_dtype(non_null):
        return False
    values = set(pd.to_numeric(non_null, errors="coerce").dropna().astype(float).unique())
    return bool(values) and values.issubset({0.0, 1.0})


def _feature_group_for_allowed(column: str) -> str:
    name = column.lower()
    if name == "action_type":
        return "action_type_context"
    if name in {"start_x", "start_y", "end_x", "end_y"}:
        return "location_context"
    if any(token in name for token in ("minute", "second", "action_position")):
        return "temporal_context"
    return "action_context"


def classify_column(column: str, series: pd.Series) -> ColumnDecision:
    """Classify a CxA+ target-dataset column for the first modelling contract."""

    name = column.lower()
    if column in TARGET_COLUMNS:
        return ColumnDecision(
            column,
            "target",
            "target_columns",
            "CxA+ target column excluded from model features.",
            "high",
            True,
        )
    if column in REFERENCE_ONLY_COLUMNS:
        return ColumnDecision(
            column,
            "reference_only",
            "reference_only_columns",
            "Reference-only current-label field excluded from model features.",
            "high",
            True,
        )
    if name.startswith(CREATED_SHOT_PREFIX):
        return ColumnDecision(
            column,
            "reference_only",
            "reference_only_columns",
            "Created-shot reference field excluded from model features.",
            "high",
            True,
        )
    if column in LEAKAGE_EXCLUDED_COLUMNS or any(token in name for token in LEAKAGE_TOKENS):
        return ColumnDecision(
            column,
            "leakage_excluded",
            "leakage_excluded_columns",
            "Future/downstream/outcome-like leakage field excluded from model features.",
            "high",
            True,
        )
    if column in OUTPUT_MODEL_COLUMNS or any(token in name for token in MODEL_OUTPUT_TOKENS):
        return ColumnDecision(
            column,
            "model_output",
            "model_output_columns",
            "Model output or prediction-like column excluded from model features.",
            "high",
            True,
        )
    if column in IDENTIFIER_COLUMNS or name.endswith("_id"):
        return ColumnDecision(
            column,
            "identifier",
            "identifier_columns",
            "Identifier retained for joins/splits/audit only, not model features.",
            "info",
            True,
        )
    if _is_binary_series(series):
        return ColumnDecision(
            column,
            "eligible_binary",
            _feature_group_for_allowed(column),
            "Eligible binary pre-action modelling feature.",
            "info",
            True,
        )
    if pd.api.types.is_numeric_dtype(series) and series.notna().any():
        return ColumnDecision(
            column,
            "eligible_numeric",
            _feature_group_for_allowed(column),
            "Eligible numeric pre-action modelling feature.",
            "info",
            True,
        )
    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        return ColumnDecision(
            column,
            "eligible_categorical",
            _feature_group_for_allowed(column),
            "Eligible categorical pre-action modelling feature.",
            "info",
            True,
        )
    return ColumnDecision(
        column,
        "requires_review",
        "requires_review_columns",
        "Column type/content requires manual review before model use.",
        "medium",
        True,
    )


def _selected_candidates(decisions: list[ColumnDecision]) -> dict[str, list[str]]:
    return {
        "numeric": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "eligible_numeric"
        ),
        "binary": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "eligible_binary"
        ),
        "categorical": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "eligible_categorical"
        ),
    }


def _excluded_columns(decisions: list[ColumnDecision]) -> dict[str, list[str]]:
    return {
        "identifier_columns": sorted(
            decision.column for decision in decisions if decision.classification == "identifier"
        ),
        "target_columns": sorted(
            decision.column for decision in decisions if decision.classification == "target"
        ),
        "reference_only_columns": sorted(
            decision.column for decision in decisions if decision.classification == "reference_only"
        ),
        "leakage_excluded_columns": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "leakage_excluded"
        ),
        "model_output_columns": sorted(
            decision.column for decision in decisions if decision.classification == "model_output"
        ),
        "requires_review_columns": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "requires_review"
        ),
    }


def _target_summary(df: pd.DataFrame) -> dict[str, Any]:
    if PRIMARY_TARGET not in df.columns:
        return {
            "row_count": int(len(df)),
            "positive_count": None,
            "positive_rate": None,
            "null_count": int(len(df)),
        }
    target = pd.to_numeric(df[PRIMARY_TARGET], errors="coerce")
    non_null = target.dropna()
    positive_count = int((non_null > 0).sum())
    return {
        "row_count": int(len(df)),
        "positive_count": positive_count,
        "positive_rate": float(positive_count / len(non_null)) if len(non_null) else None,
        "null_count": int(target.isna().sum()),
    }


def _feature_group_summary(df: pd.DataFrame, decisions: list[ColumnDecision]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    row_count = len(df)
    for decision in decisions:
        series = df[decision.column]
        missing = int(series.isna().sum())
        rows.append(
            {
                "feature_group": decision.feature_group,
                "column": decision.column,
                "classification": decision.classification,
                "dtype": str(series.dtype),
                "missing_count": missing,
                "missing_pct": float((missing / row_count) * 100.0) if row_count else 0.0,
                "distinct_count": int(series.nunique(dropna=True)),
                "notes": decision.reason,
            }
        )
    return pd.DataFrame(rows)


def _excluded_columns_frame(decisions: list[ColumnDecision]) -> pd.DataFrame:
    rows = [
        {
            "column": decision.column,
            "exclusion_type": decision.classification,
            "reason": decision.reason,
            "severity": decision.severity,
            "can_appear_in_outputs": decision.can_appear_in_outputs,
        }
        for decision in decisions
        if decision.classification
        not in {"eligible_numeric", "eligible_binary", "eligible_categorical"}
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "column",
            "exclusion_type",
            "reason",
            "severity",
            "can_appear_in_outputs",
        ],
    )


def _feature_groups(decisions: list[ColumnDecision]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for decision in decisions:
        groups.setdefault(decision.feature_group, []).append(decision.column)
    return {group: sorted(columns) for group, columns in sorted(groups.items())}


def _report_markdown(
    *,
    contract: dict[str, Any],
    resolved: dict[str, Any],
    excluded_frame: pd.DataFrame,
    group_summary: pd.DataFrame,
) -> str:
    selected = contract["eligible_feature_candidates"]
    selected_count = sum(len(values) for values in selected.values())
    return "\n".join(
        [
            "# CxA+ Feature Contract Report",
            "",
            "## Executive summary",
            "- This PR does not train a model.",
            f"- Input rows: {contract['row_count']} | columns: {contract['column_count']}",
            f"- Eligible CxA+ modelling feature candidates: {selected_count}",
            "",
            "## Primary target definition",
            "- Primary target: `shot_within_next_5_actions`.",
            "- Target means a future shot-created action occurs within the next five actions in the same match and possession. Same-team consistency is audited separately and is not enforced by this first governed target.",
            f"- Split/group column for first model: `{contract['split_group_column']}`.",
            "",
            "## Allowed feature families",
            f"- Numeric: {len(selected['numeric'])}",
            f"- Binary: {len(selected['binary'])}",
            f"- Categorical: {len(selected['categorical'])}",
            "",
            "## Leakage and exclusion controls",
            "- Excludes current/future leakage targets and downstream value labels.",
            "- Excludes identifiers as reference-only.",
            "- Excludes any prediction/model-output/outcome/future/downstream columns.",
            f"- Excluded rows recorded: {len(excluded_frame)}",
            "",
            "## Contract separation summary",
            "- Identifier columns, target columns, reference-only columns, leakage exclusions, model-output columns, and review-required columns are explicitly separated in JSON.",
            "",
            "## Target quality snapshot",
            f"- Positive target count: {resolved['target_summary']['positive_count']}",
            f"- Positive target rate: {resolved['target_summary']['positive_rate']}",
            "",
            "## Next modelling path (future PRs)",
            "- Train first diagnostic CxA+ model using only this contract allowlist.",
            "- Validate against governed splits using `match_id` grouping.",
            "- Keep CxG, diagnostic CxA, and current CxA+ target outputs unchanged in this step.",
            "",
        ]
    )


def prepare_cxa_plus_feature_contract(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Create first governed CxA+ feature-contract artifacts from target dataset."""

    df = _read_table(input_path)
    if PRIMARY_TARGET not in df.columns:
        raise ValueError(f"CxA+ target dataset is missing primary target column: {PRIMARY_TARGET}")
    if SPLIT_GROUP_COLUMN not in df.columns:
        raise ValueError(f"CxA+ target dataset is missing split group column: {SPLIT_GROUP_COLUMN}")

    output_dir.mkdir(parents=True, exist_ok=True)
    contracts_dir = output_dir / "contracts"
    diagnostics_dir = output_dir / "diagnostics"
    reports_dir = output_dir / "reports"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    decisions = [classify_column(column, df[column]) for column in df.columns]
    selected = _selected_candidates(decisions)
    excluded = _excluded_columns(decisions)

    selected_flat = set().union(*[set(values) for values in selected.values()])
    forbidden_flat = set(EXPLICIT_FORBIDDEN_COLUMNS) | set(
        excluded["identifier_columns"]
        + excluded["target_columns"]
        + excluded["reference_only_columns"]
        + excluded["leakage_excluded_columns"]
        + excluded["model_output_columns"]
        + excluded["requires_review_columns"]
    )
    overlap = sorted(selected_flat.intersection(forbidden_flat))
    if overlap:
        raise ValueError(
            f"CxA+ feature contract leakage guard failed; forbidden selected features: {overlap}"
        )
    if not selected_flat:
        raise ValueError("CxA+ feature contract produced no eligible modelling features")

    now = datetime.now(timezone.utc).isoformat()
    contract = {
        "model_version": MODEL_VERSION,
        "metric": METRIC,
        "primary_target": PRIMARY_TARGET,
        "split_group_column": SPLIT_GROUP_COLUMN,
        "eligible_feature_candidates": selected,
        "identifier_columns": excluded["identifier_columns"],
        "target_columns": excluded["target_columns"],
        "reference_only_columns": excluded["reference_only_columns"],
        "leakage_excluded_columns": excluded["leakage_excluded_columns"],
        "model_output_columns": excluded["model_output_columns"],
        "requires_review_columns": excluded["requires_review_columns"],
        "feature_groups": _feature_groups(decisions),
        "generated_at": now,
        "input_path": input_path.as_posix(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }

    resolved = {
        "available_columns": sorted(df.columns.tolist()),
        "primary_target": PRIMARY_TARGET,
        "split_group_column": SPLIT_GROUP_COLUMN,
        "eligible_feature_candidates": selected,
        "excluded_columns": excluded,
        "target_summary": _target_summary(df),
        "leakage_guard_overlap": overlap,
    }

    excluded_frame = _excluded_columns_frame(decisions)
    group_summary = _feature_group_summary(df, decisions)

    contract_path = contracts_dir / "feature_contract.json"
    resolved_path = diagnostics_dir / "resolved_features.json"
    excluded_path = diagnostics_dir / "excluded_columns.csv"
    group_path = diagnostics_dir / "feature_group_summary.csv"
    report_path = reports_dir / "feature_contract_report.md"

    contract_path.write_text(json.dumps(_json_safe(contract), indent=2), encoding="utf-8")
    resolved_path.write_text(json.dumps(_json_safe(resolved), indent=2), encoding="utf-8")
    excluded_frame.to_csv(excluded_path, index=False)
    group_summary.to_csv(group_path, index=False)
    report_path.write_text(
        _report_markdown(
            contract=contract,
            resolved=resolved,
            excluded_frame=excluded_frame,
            group_summary=group_summary,
        ),
        encoding="utf-8",
    )
    return {
        "feature_contract": contract_path,
        "resolved_features": resolved_path,
        "excluded_columns": excluded_path,
        "feature_group_summary": group_path,
        "feature_contract_report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare first governed CxA+ feature contract.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    outputs = prepare_cxa_plus_feature_contract(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: value.as_posix() for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
