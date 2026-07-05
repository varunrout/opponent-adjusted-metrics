#!/usr/bin/env python
"""Prepare the diagnostic CxA feature contract and leakage exclusions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT_PATH = Path("feature_store") / "cxa" / "action_features.parquet"
DEFAULT_OUTPUT_DIR = Path("outputs") / "modeling" / "cxa" / "diagnostic_v1"
MODEL_VERSION = "diagnostic_v1"
METRIC = "cxa"
PRIMARY_TARGET = "shot_created"
ATTRIBUTION_REFERENCE = "created_shot_cxg"
VALUE_OUTPUT = "cxa_value"

IDENTIFIER_COLUMNS = {
    "action_id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "possession_id",
    "possession",
    "sequence_id",
    "target_created_shot_id",
    "created_shot_id",
    "shot_id",
}
TARGET_COLUMNS = {PRIMARY_TARGET, "target", "attribution.target"}
REFERENCE_ONLY_COLUMNS = {
    ATTRIBUTION_REFERENCE,
    "created_shot_id",
    "created_shot_distance",
    "created_shot_angle",
    "positive_mean_created_shot_cxg",
    "statsbomb_xg",
    "provider_xg",
    "reference_xg",
}
OUTPUT_PREDICTION_COLUMNS = {
    VALUE_OUTPUT,
    "predicted_cxa",
    "predicted_shot_created_probability",
    "predicted_chance_action",
    "predicted_chance_actions",
    "cxa_share",
    "sequence_cxa",
    "possession_cxa",
    "led_to_shot",
    "baseline_probability",
    "mean_predicted_probability",
}
REVIEWED_SPATIAL_CONTEXT_COLUMNS = {
    "distance_to_goal_before",
    "distance_to_goal_after",
    "angle_to_goal_before",
    "angle_to_goal_after",
}
FUTURE_LEAKAGE_EXACT_COLUMNS = {
    "shot_outcome",
    "goal_outcome",
    "final_shot",
    "future_shot",
    "actions_until_shot",
    "next_action_is_shot",
    "total_possession_length",
}
FUTURE_LEAKAGE_PREFIXES = (
    "future_",
    "post_",
    "outcome_",
)
FUTURE_LEAKAGE_SUFFIXES = ("_outcome",)
FUTURE_LEAKAGE_CONTAINS = (
    "result",
    "final_shot",
    "future_shot",
    "actions_until_shot",
    "next_action_is_shot",
)
PROVIDER_REFERENCE_TOKENS = ("provider", "reference", "statsbomb")


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
    raise ValueError(f"Unsupported CxA feature input format: {path.suffix}")


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
    if name in {"action_type", "play_pattern"}:
        return "event_type_context"
    if name.startswith("start_") or name in {"start_x", "start_y", "start_zone", "start_third"}:
        return "location_start_context"
    if name.startswith("end_") or name in {"end_x", "end_y", "end_zone", "end_third"}:
        return "location_end_context"
    if any(token in name for token in ("progression", "length", "angle", "carry", "dribble")):
        return "movement_context"
    if any(token in name for token in ("sequence", "possession", "prior_action")):
        return "possession_sequence_context"
    if "pressure" in name:
        return "pressure_context"
    if any(token in name for token in ("pass", "cross", "cutback", "through_ball", "body_part")):
        return "pass_context"
    if any(token in name for token in ("minute", "second", "time", "seconds_since")):
        return "temporal_context"
    if name in IDENTIFIER_COLUMNS or name.endswith("_id"):
        return "team_player_identifiers_only"
    return "action_identity_context"


def _is_future_leakage_name(name: str) -> bool:
    return (
        name in FUTURE_LEAKAGE_EXACT_COLUMNS
        or name.startswith(FUTURE_LEAKAGE_PREFIXES)
        or name.endswith(FUTURE_LEAKAGE_SUFFIXES)
        or any(pattern in name for pattern in FUTURE_LEAKAGE_CONTAINS)
    )


def classify_column(column: str, series: pd.Series) -> ColumnDecision:
    """Classify a CxA feature-store column for diagnostic modelling."""

    name = column.lower()
    if column in IDENTIFIER_COLUMNS or name.endswith("_id"):
        return ColumnDecision(
            column,
            "identifier",
            "team_player_identifiers_only",
            "Identifier retained for audit, joins, and aggregation only.",
            "info",
            True,
        )
    if column in TARGET_COLUMNS:
        return ColumnDecision(
            column,
            "target",
            "target_reference_outputs",
            "Primary or alias target column excluded from model inputs.",
            "high",
            True,
        )
    if column in REVIEWED_SPATIAL_CONTEXT_COLUMNS:
        return ColumnDecision(
            column,
            "requires_review",
            "reviewed_spatial_context",
            (
                "Before/after spatial wording requires football review; it is not treated "
                "as ordinary allowed signal in this contract."
            ),
            "medium",
            True,
        )
    if column in REFERENCE_ONLY_COLUMNS:
        return ColumnDecision(
            column,
            "reference_only",
            "target_reference_outputs",
            "Attribution, provider, or created-shot reference column excluded from model inputs.",
            "high",
            True,
        )
    if (
        column in OUTPUT_PREDICTION_COLUMNS
        or name.startswith("predicted_")
        or "probability" in name
    ):
        return ColumnDecision(
            column,
            "output_prediction",
            "target_reference_outputs",
            "Model output, probability, or attribution output excluded from model inputs.",
            "high",
            True,
        )
    if any(token in name for token in PROVIDER_REFERENCE_TOKENS):
        return ColumnDecision(
            column,
            "reference_only",
            "target_reference_outputs",
            "Provider/reference column excluded from diagnostic model inputs.",
            "high",
            True,
        )
    if _is_future_leakage_name(name):
        return ColumnDecision(
            column,
            "leakage_excluded",
            "target_reference_outputs",
            "Future, outcome, result, or post-action signal excluded for leakage control.",
            "high",
            True,
        )
    if _is_binary_series(series):
        return ColumnDecision(
            column,
            "allowed_binary",
            _feature_group_for_allowed(column),
            "Binary pre-action candidate feature.",
            "info",
            True,
        )
    if pd.api.types.is_numeric_dtype(series) and series.notna().any():
        return ColumnDecision(
            column,
            "allowed_numeric",
            _feature_group_for_allowed(column),
            "Numeric pre-action candidate feature.",
            "info",
            True,
        )
    if (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    ):
        return ColumnDecision(
            column,
            "allowed_categorical",
            _feature_group_for_allowed(column),
            "Categorical pre-action candidate feature.",
            "info",
            True,
        )
    return ColumnDecision(
        column,
        "excluded_unknown",
        "requires_review",
        "Column type or content is not safely classifiable as a model feature.",
        "medium",
        True,
    )


def _selected_candidates(decisions: list[ColumnDecision]) -> dict[str, list[str]]:
    return {
        "numeric": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "allowed_numeric"
        ),
        "binary": sorted(
            decision.column for decision in decisions if decision.classification == "allowed_binary"
        ),
        "categorical": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "allowed_categorical"
        ),
    }


def _excluded_columns(decisions: list[ColumnDecision]) -> dict[str, list[str]]:
    return {
        "target_columns": sorted(
            decision.column for decision in decisions if decision.classification == "target"
        ),
        "reference_only_columns": sorted(
            decision.column for decision in decisions if decision.classification == "reference_only"
        ),
        "output_prediction_columns": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "output_prediction"
        ),
        "leakage_excluded_columns": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "leakage_excluded"
        ),
        "identifier_columns": sorted(
            decision.column for decision in decisions if decision.classification == "identifier"
        ),
        "requires_review_columns": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "requires_review"
        ),
        "excluded_unknown_columns": sorted(
            decision.column
            for decision in decisions
            if decision.classification == "excluded_unknown"
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


def _identifier_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column in sorted(IDENTIFIER_COLUMNS):
        if column not in df.columns:
            summary[column] = {"present": False, "missing": None, "missing_pct": None}
            continue
        missing = int(df[column].isna().sum())
        summary[column] = {
            "present": True,
            "missing": missing,
            "missing_pct": float((missing / len(df)) * 100.0) if len(df) else 0.0,
        }
    return summary


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
        not in {"allowed_numeric", "allowed_binary", "allowed_categorical"}
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
    selected = contract["selected_feature_candidates"]
    selected_count = sum(len(values) for values in selected.values())
    requires_review = contract["excluded_columns"]["requires_review_columns"]
    created_shot_id_summary = resolved["identifier_summary"].get("created_shot_id")
    created_shot_note = (
        "- `created_shot_id` sparsity is expected because only shot-creating actions have one."
    )
    if created_shot_id_summary:
        created_shot_note = (
            f"- `created_shot_id` sparsity is expected; missing count is "
            f"{created_shot_id_summary.get('missing')}."
        )
    return "\n".join(
        [
            "# CxA Diagnostic Feature Contract Report",
            "",
            "## Executive summary",
            "- This PR does not train a model.",
            f"- Input rows: {contract['row_count']} | columns: {contract['column_count']}",
            f"- Selected diagnostic feature candidates: {selected_count}",
            "- Diagnostic CxA training should use the selected feature candidates only.",
            "",
            "## Primary target definition",
            "- `shot_created` is the primary binary diagnostic target.",
            "- The positive class means an eligible downstream shot was created inside the CxA window.",
            "",
            "## Target/reference/output separation",
            "- `created_shot_cxg` is an attribution/reference value, not a classification feature.",
            "- `cxa_value` is an output/attribution value, not a model feature.",
            created_shot_note,
            "",
            "## Allowed feature candidates",
            f"- Numeric: {len(selected['numeric'])}",
            f"- Binary: {len(selected['binary'])}",
            f"- Categorical: {len(selected['categorical'])}",
            "",
            "## Excluded columns",
            f"- Excluded rows recorded: {len(excluded_frame)}",
            "- Target, reference, prediction, output, provider, and identifier columns are excluded from model inputs.",
            "",
            "## Requires-review columns",
            (
                "- Reviewed spatial context columns: "
                + (", ".join(requires_review) if requires_review else "none present")
            ),
            "- Before/after spatial columns are not blindly treated as leakage; they require football review before training use.",
            "",
            "## ID quality summary for action-level table",
            f"- Identifier fields tracked: {len(resolved['identifier_summary'])}",
            "- Missing IDs in config or metadata JSON are not treated as modelling blockers here.",
            "",
            "## Feature group summary",
            f"- Feature group rows: {len(group_summary)}",
            "",
            "## Leakage-risk notes",
            "- Any `predicted_*` column and any probability/output column is excluded from candidates.",
            "- Future, post-action, final, outcome, and result columns are leakage-excluded.",
            "",
            "## Next recommended PR",
            "- Train diagnostic CxA candidates using this contract and selected feature candidates.",
            "",
        ]
    )


def prepare_cxa_diagnostic_feature_contract(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Create diagnostic CxA contract artifacts from an action feature table."""

    df = _read_table(input_path)
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
    now = datetime.now(timezone.utc).isoformat()
    missing_required = [
        column
        for column in ("action_id", "match_id", "team_id", PRIMARY_TARGET)
        if column not in df
    ]
    review_notes = [
        "Focus ID quality on action-level feature/prediction parquet tables and DB action tables.",
        "created_shot_id sparsity is expected for non-shot-creating actions.",
        "Before/after spatial context is held for football review, not silently allowed.",
    ]

    contract = {
        "model_version": MODEL_VERSION,
        "metric": METRIC,
        "primary_target": PRIMARY_TARGET,
        "attribution_reference": ATTRIBUTION_REFERENCE,
        "value_output": VALUE_OUTPUT,
        "selected_feature_candidates": selected,
        "excluded_columns": excluded,
        "feature_groups": _feature_groups(decisions),
        "generated_at": now,
        "input_path": input_path.as_posix(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
    }
    resolved = {
        "available_columns": sorted(df.columns.tolist()),
        "selected_feature_candidates": selected,
        "excluded_columns": excluded,
        "missing_required_columns": missing_required,
        "target_summary": _target_summary(df),
        "identifier_summary": _identifier_summary(df),
        "review_notes": review_notes,
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
    parser = argparse.ArgumentParser(
        description="Prepare diagnostic CxA feature contract and leakage exclusions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    outputs = prepare_cxa_diagnostic_feature_contract(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: value.as_posix() for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
