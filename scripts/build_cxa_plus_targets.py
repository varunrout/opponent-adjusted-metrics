#!/usr/bin/env python
"""Build governed CxA+ possession-window target datasets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FEATURE_PATH = Path("feature_store/cxa/action_features.parquet")
DEFAULT_DESIGN_SUMMARY_PATH = Path("outputs/analysis/cxa_plus/design/cxa_plus_design_summary.json")
DEFAULT_OUTPUT_DIR = Path("feature_store/cxa_plus")

PRIMARY_TARGET = "shot_within_next_5_actions"
REQUIRED_ORDER_FIELDS = ["match_id", "possession", "action_position", "sequence_id", "action_id"]
ORDER_COLUMNS = ["match_id", "possession", "action_position", "sequence_id", "action_id"]
IDENTIFIER_COLUMNS = [
    "action_id",
    "event_id",
    "match_id",
    "possession",
    "sequence_id",
    "action_position",
    "team_id",
    "player_id",
]
CONTEXT_COLUMNS = [
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "action_type",
    "period",
    "minute",
    "second",
]
TARGET_COLUMNS = [
    "shot_within_next_1_action",
    "shot_within_next_3_actions",
    "shot_within_next_5_actions",
    "shot_later_in_possession",
    "max_created_shot_cxg_within_next_5_actions",
    "sum_created_shot_cxg_rest_of_possession",
    "discounted_downstream_shot_value",
]
REFERENCE_COLUMNS = ["shot_created", "created_shot_id", "created_shot_cxg"]
BASE_LEAKAGE_EXCLUSIONS = {
    "shot_created": "Current diagnostic CxA target/reference; not a CxA+ model input.",
    "created_shot_id": "Future created-shot reference/output; target construction only.",
    "created_shot_cxg": "Future shot-value reference used to construct CxA+ value targets.",
    "created_shot_cxg_reference": "Created-shot value reference; not a model input.",
    "diagnostic_cxa": "Diagnostic model output; exclude from future CxA+ feature matrices.",
    "predicted_shot_created_probability": "Prediction output; not a source feature.",
    "cxa_value": "Existing attribution/output value; not a source feature.",
}
LEAKAGE_NAME_TOKENS = (
    "future",
    "downstream",
    "within_next",
    "window_label",
    "outcome",
    "result",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) or np.isinf(value) else float(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if pd.isna(value) and not isinstance(value, (bool, str)):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_action_features(path: Path) -> pd.DataFrame:
    """Load the CxA action feature table."""

    if not path.exists():
        raise FileNotFoundError(f"Required CxA action feature table does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported action feature format: {path.suffix}")


def validate_required_columns(frame: pd.DataFrame) -> list[str]:
    """Validate required ordering fields and fail clearly when they are missing."""

    missing = [column for column in REQUIRED_ORDER_FIELDS if column not in frame.columns]
    if missing:
        raise ValueError(
            "CxA+ target building requires deterministic ordering fields; "
            f"missing_required_order_fields={missing}"
        )
    return missing


def _sorted_actions(frame: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(frame)
    working = frame.copy()
    for column in ("match_id", "possession", "action_position"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    return working.sort_values(ORDER_COLUMNS, kind="mergesort").reset_index(drop=True)


def _possession_slices(frame: pd.DataFrame) -> list[tuple[int, int]]:
    if frame.empty:
        return []
    boundaries = (
        frame["match_id"].ne(frame["match_id"].shift())
        | frame["possession"].ne(frame["possession"].shift())
    ).to_numpy()
    starts = np.flatnonzero(boundaries)
    ends = np.r_[starts[1:], len(frame)]
    return list(zip(starts, ends, strict=False))


def build_cxa_plus_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Build future-only CxA+ possession-window targets."""

    working = _sorted_actions(frame)
    target_arrays: dict[str, np.ndarray] = {
        column: np.zeros(len(working), dtype=float) for column in TARGET_COLUMNS
    }
    for start, end in _possession_slices(working):
        group = working.iloc[start:end]
        group_targets = _compute_group_targets(group)
        for column, values in group_targets.items():
            target_arrays[column][start:end] = values
    for column, values in target_arrays.items():
        if column.startswith("shot_"):
            working[column] = values.astype(int)
        else:
            working[column] = values
    return working[_output_columns(working)].copy()


def _output_columns(frame: pd.DataFrame) -> list[str]:
    columns = [
        *IDENTIFIER_COLUMNS,
        *CONTEXT_COLUMNS,
        *TARGET_COLUMNS,
        *REFERENCE_COLUMNS,
    ]
    return [column for column in columns if column in frame.columns]


def _compute_group_targets(group: pd.DataFrame) -> dict[str, np.ndarray]:
    shot = pd.to_numeric(group.get("shot_created", 0), errors="coerce").fillna(0).gt(0).to_numpy()
    created_values = pd.to_numeric(group.get("created_shot_cxg", 0.0), errors="coerce").to_numpy()
    shot_ids = (
        group["created_shot_id"].to_numpy(dtype=object)
        if "created_shot_id" in group.columns
        else np.array([None] * len(group), dtype=object)
    )
    n_rows = len(group)
    next_1 = np.zeros(n_rows, dtype=int)
    next_3 = np.zeros(n_rows, dtype=int)
    next_5 = np.zeros(n_rows, dtype=int)
    rest = np.zeros(n_rows, dtype=int)
    max_next_5 = np.zeros(n_rows, dtype=float)
    sum_rest = np.zeros(n_rows, dtype=float)
    discounted = np.zeros(n_rows, dtype=float)

    for current_position in range(n_rows):
        future_1 = _dedup_future_shot_positions(
            current_position,
            range(current_position + 1, min(n_rows, current_position + 2)),
            shot,
            shot_ids,
        )
        future_3 = _dedup_future_shot_positions(
            current_position,
            range(current_position + 1, min(n_rows, current_position + 4)),
            shot,
            shot_ids,
        )
        future_5 = _dedup_future_shot_positions(
            current_position,
            range(current_position + 1, min(n_rows, current_position + 6)),
            shot,
            shot_ids,
        )
        next_1[current_position] = int(bool(future_1))
        next_3[current_position] = int(bool(future_3))
        next_5[current_position] = int(bool(future_5))
        max_next_5[current_position] = _max_value(created_values, future_5)

    rest, sum_rest, discounted = _rest_of_possession_values(shot, shot_ids, created_values)
    return {
        "shot_within_next_1_action": next_1,
        "shot_within_next_3_actions": next_3,
        "shot_within_next_5_actions": next_5,
        "shot_later_in_possession": rest,
        "max_created_shot_cxg_within_next_5_actions": max_next_5,
        "sum_created_shot_cxg_rest_of_possession": sum_rest,
        "discounted_downstream_shot_value": discounted,
    }


def _rest_of_possession_values(
    shot: np.ndarray, shot_ids: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(shot)
    rest = np.zeros(n_rows, dtype=int)
    sum_rest = np.zeros(n_rows, dtype=float)
    discounted = np.zeros(n_rows, dtype=float)
    future_by_id: dict[str, tuple[int, float]] = {}
    future_missing_id: list[tuple[int, float]] = []
    future_id_value_sum = 0.0
    future_missing_value_sum = 0.0

    for current_position in range(n_rows - 1, -1, -1):
        current_shot_id = shot_ids[current_position] if current_position < len(shot_ids) else None
        excluded_key = (
            _shot_id_key(current_shot_id)
            if bool(shot[current_position]) and _has_shot_id(current_shot_id)
            else None
        )
        excluded = future_by_id.get(excluded_key) if excluded_key is not None else None
        future_count = len(future_by_id) + len(future_missing_id) - (1 if excluded else 0)
        rest[current_position] = int(future_count > 0)
        excluded_value = _safe_value(excluded[1]) if excluded else 0.0
        sum_rest[current_position] = future_id_value_sum + future_missing_value_sum - excluded_value
        discounted[current_position] = _discounted_from_future_state(
            current_position,
            future_by_id,
            future_missing_id,
            excluded_key,
        )

        if shot[current_position]:
            current_value = values[current_position]
            if _has_shot_id(current_shot_id):
                shot_key = _shot_id_key(current_shot_id)
                previous = future_by_id.get(shot_key)
                if previous is not None:
                    future_id_value_sum -= _safe_value(previous[1])
                future_by_id[shot_key] = (current_position, current_value)
                future_id_value_sum += _safe_value(current_value)
            else:
                future_missing_id.append((current_position, current_value))
                future_missing_value_sum += _safe_value(current_value)
    return rest, sum_rest, discounted


def _discounted_from_future_state(
    current_position: int,
    future_by_id: dict[str, tuple[int, float]],
    future_missing_id: list[tuple[int, float]],
    excluded_key: str | None,
) -> float:
    total = 0.0
    for shot_key, (future_position, value) in future_by_id.items():
        if shot_key == excluded_key:
            continue
        total += _safe_value(value) / (future_position - current_position)
    for future_position, value in future_missing_id:
        total += _safe_value(value) / (future_position - current_position)
    return float(total)


def _safe_value(value: Any) -> float:
    return 0.0 if pd.isna(value) else float(value)


def _dedup_future_shot_positions(
    current_position: int,
    candidate_positions: range,
    shot: np.ndarray,
    shot_ids: np.ndarray,
) -> list[int]:
    current_shot_id = shot_ids[current_position] if current_position < len(shot_ids) else None
    exclude_current_shot_id = bool(shot[current_position]) and _has_shot_id(current_shot_id)
    seen_shot_ids: set[str] = set()
    selected: list[int] = []
    for future_position in candidate_positions:
        if not shot[future_position]:
            continue
        future_shot_id = shot_ids[future_position] if future_position < len(shot_ids) else None
        if (
            exclude_current_shot_id
            and _has_shot_id(future_shot_id)
            and _shot_id_key(future_shot_id) == _shot_id_key(current_shot_id)
        ):
            continue
        if _has_shot_id(future_shot_id):
            shot_key = _shot_id_key(future_shot_id)
            if shot_key in seen_shot_ids:
                continue
            seen_shot_ids.add(shot_key)
        selected.append(future_position)
    return selected


def _has_shot_id(value: Any) -> bool:
    return not pd.isna(value) and str(value) != ""


def _shot_id_key(value: Any) -> str:
    return str(value)


def _max_value(values: np.ndarray, positions: list[int]) -> float:
    valid_values = [
        float(values[position]) for position in positions if not pd.isna(values[position])
    ]
    return max(valid_values) if valid_values else 0.0


def compute_quality_checks(
    targets: pd.DataFrame, *, missing_order_fields: list[str]
) -> pd.DataFrame:
    """Compute target dataset quality checks as a check table."""

    row_count = len(targets)
    current_leakage = _current_action_label_leakage_count(targets)
    cross_possession = _cross_possession_window_leakage_count(targets)
    missing_future_values = _missing_created_shot_cxg_for_future_shots(targets)
    checks = [
        _quality_row("row_count", row_count, "passed", "info", "Rows written to target dataset."),
        _quality_row(
            "action_id_missing_count",
            _missing_count(targets, "action_id"),
            "failed" if _missing_count(targets, "action_id") else "passed",
            "blocker",
            "Action IDs are required for downstream joins.",
        ),
        _quality_row(
            "action_id_duplicate_count",
            _duplicate_count(targets, "action_id"),
            "failed" if _duplicate_count(targets, "action_id") else "passed",
            "blocker",
            "Action IDs should be unique in the governed target dataset.",
        ),
        _quality_row(
            "missing_required_order_fields",
            ";".join(missing_order_fields),
            "failed" if missing_order_fields else "passed",
            "blocker",
            "Required ordering fields: match_id, possession, action_position, sequence_id, action_id.",
        ),
        _quality_row(
            "primary_target", PRIMARY_TARGET, "passed", "info", "First governed CxA+ target."
        ),
        _quality_row(
            "primary_target_positive_count",
            _positive_count(targets, PRIMARY_TARGET),
            "passed",
            "info",
            "Actions with a shot in the next five same-possession actions.",
        ),
        _quality_row(
            "primary_target_positive_rate",
            _positive_rate(targets, PRIMARY_TARGET),
            "passed",
            "info",
            "Share of actions with a shot in the next five same-possession actions.",
        ),
        _quality_row("next_1_positive_rate", _positive_rate(targets, "shot_within_next_1_action")),
        _quality_row("next_3_positive_rate", _positive_rate(targets, "shot_within_next_3_actions")),
        _quality_row("next_5_positive_rate", _positive_rate(targets, PRIMARY_TARGET)),
        _quality_row(
            "rest_of_possession_positive_rate",
            _positive_rate(targets, "shot_later_in_possession"),
        ),
        _quality_row(
            "max_created_shot_cxg_nonzero_rate",
            _positive_rate(targets, "max_created_shot_cxg_within_next_5_actions"),
        ),
        _quality_row(
            "discounted_downstream_shot_value_nonzero_rate",
            _positive_rate(targets, "discounted_downstream_shot_value"),
        ),
        _quality_row(
            "same_possession_window_check",
            cross_possession,
            "failed" if cross_possession else "passed",
            "blocker",
            "Positive labels must be explainable by future shots in the same match and possession.",
        ),
        _quality_row(
            "no_cross_possession_leakage_check",
            cross_possession,
            "failed" if cross_possession else "passed",
            "blocker",
            "Lookahead windows must never cross match or possession boundaries.",
        ),
        _quality_row(
            "no_current_action_label_leakage_check",
            current_leakage,
            "failed" if current_leakage else "passed",
            "blocker",
            "Current action shot_created cannot count as future evidence.",
        ),
        _quality_row(
            "missing_created_shot_cxg_for_future_shots",
            missing_future_values,
            "warning" if missing_future_values else "passed",
            "warning",
            "Future shot rows with missing created_shot_cxg are tracked for value-target audits.",
        ),
    ]
    failed = [row["check_name"] for row in checks if row["status"] == "failed"]
    checks.append(
        _quality_row(
            "failed_checks",
            ";".join(failed),
            "failed" if failed else "passed",
            "blocker" if failed else "info",
            "Empty when all blocker checks pass.",
        )
    )
    return pd.DataFrame(checks)


def _quality_row(
    check_name: str,
    value: Any,
    status: str = "passed",
    severity: str = "info",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "value": value,
        "status": status,
        "severity": severity,
        "notes": notes,
    }


def _missing_count(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].isna().sum()) if column in frame.columns else len(frame)


def _duplicate_count(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].duplicated().sum()) if column in frame.columns else 0


def _positive_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).gt(0).sum())


def _positive_rate(frame: pd.DataFrame, column: str) -> float:
    return _positive_count(frame, column) / len(frame) if len(frame) else np.nan


def _current_action_label_leakage_count(targets: pd.DataFrame) -> int:
    if not {"shot_created", PRIMARY_TARGET}.issubset(targets.columns):
        return 0
    shot = pd.to_numeric(targets["shot_created"], errors="coerce").fillna(0).gt(0)
    target = pd.to_numeric(targets[PRIMARY_TARGET], errors="coerce").fillna(0).gt(0)
    suspicious = shot & target & ~_has_future_shot_within(targets, window=5)
    return int(suspicious.sum())


def _cross_possession_window_leakage_count(targets: pd.DataFrame) -> int:
    if PRIMARY_TARGET not in targets.columns:
        return len(targets)
    positives = pd.to_numeric(targets[PRIMARY_TARGET], errors="coerce").fillna(0).gt(0)
    explainable = _has_future_shot_within(targets, window=5)
    return int((positives & ~explainable).sum())


def _has_future_shot_within(targets: pd.DataFrame, *, window: int) -> pd.Series:
    result = pd.Series(False, index=targets.index)
    if "shot_created" not in targets.columns:
        return result
    indexed_targets = targets.copy()
    indexed_targets["__original_index"] = targets.index
    working = _sorted_actions(indexed_targets)
    for _, group in working.groupby(["match_id", "possession"], sort=False, dropna=False):
        shot = pd.to_numeric(group["shot_created"], errors="coerce").fillna(0).gt(0).to_numpy()
        shot_ids = (
            group["created_shot_id"].to_numpy(dtype=object)
            if "created_shot_id" in group.columns
            else np.array([None] * len(group), dtype=object)
        )
        group_result = np.zeros(len(group), dtype=bool)
        for current_position in range(len(group)):
            future_positions = _dedup_future_shot_positions(
                current_position,
                range(current_position + 1, min(len(group), current_position + window + 1)),
                shot,
                shot_ids,
            )
            group_result[current_position] = bool(future_positions)
        result.loc[group["__original_index"].to_numpy()] = group_result
    return result.reindex(targets.index).fillna(False)


def _missing_created_shot_cxg_for_future_shots(targets: pd.DataFrame) -> int:
    if not {"shot_created", "created_shot_cxg"}.issubset(targets.columns):
        return 0
    missing_hits = 0
    working = _sorted_actions(targets)
    for _, group in working.groupby(["match_id", "possession"], sort=False, dropna=False):
        shot = pd.to_numeric(group["shot_created"], errors="coerce").fillna(0).gt(0).to_numpy()
        values = pd.to_numeric(group["created_shot_cxg"], errors="coerce").to_numpy()
        for shot_position in np.flatnonzero(shot & np.isnan(values)):
            missing_hits += min(5, shot_position)
    return int(missing_hits)


def build_leakage_exclusions(frame: pd.DataFrame) -> pd.DataFrame:
    """Build future CxA+ model feature exclusions."""

    exclusions: dict[str, str] = dict(BASE_LEAKAGE_EXCLUSIONS)
    for column in TARGET_COLUMNS:
        exclusions[column] = "CxA+ target label; never a model feature."
    for column in frame.columns:
        lower = column.lower()
        if column in exclusions:
            continue
        if any(token in lower for token in LEAKAGE_NAME_TOKENS):
            exclusions[column] = "Future/downstream/outcome/result field; target or reporting only."
    rows = [
        {
            "column": column,
            "present": column in frame.columns,
            "exclusion_type": _exclusion_type(column),
            "reason": reason,
            "severity": "blocker",
            "can_appear_in_outputs": True,
        }
        for column, reason in sorted(exclusions.items())
    ]
    return pd.DataFrame(rows)


def _exclusion_type(column: str) -> str:
    if column in TARGET_COLUMNS or "within_next" in column or "downstream" in column:
        return "target_or_future_label"
    if column.startswith("predicted_") or column in {"diagnostic_cxa", "cxa_value"}:
        return "model_output"
    if column in {
        "shot_created",
        "created_shot_id",
        "created_shot_cxg",
        "created_shot_cxg_reference",
    }:
        return "target_reference"
    return "leakage_risk"


def build_summary(
    *,
    targets: pd.DataFrame,
    quality: pd.DataFrame,
    input_path: Path,
    output_dir: Path,
    outputs: dict[str, Path],
    design_summary: dict[str, Any],
) -> dict[str, Any]:
    failed_checks = quality.loc[quality["status"] == "failed", "check_name"].tolist()
    return {
        "metric": "cxa_plus",
        "primary_target": PRIMARY_TARGET,
        "input_path": input_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "row_count": len(targets),
        "positive_counts": {
            "next_1": _positive_count(targets, "shot_within_next_1_action"),
            "next_3": _positive_count(targets, "shot_within_next_3_actions"),
            "next_5": _positive_count(targets, PRIMARY_TARGET),
            "rest_of_possession": _positive_count(targets, "shot_later_in_possession"),
        },
        "positive_rates": {
            "next_1": _positive_rate(targets, "shot_within_next_1_action"),
            "next_3": _positive_rate(targets, "shot_within_next_3_actions"),
            "next_5": _positive_rate(targets, PRIMARY_TARGET),
            "rest_of_possession": _positive_rate(targets, "shot_later_in_possession"),
        },
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "quality_status": "failed" if failed_checks else "passed",
        "failed_checks": failed_checks,
        "design_reference": design_summary or None,
        "outputs": {name: path.as_posix() for name, path in outputs.items()},
    }


def build_report(summary: dict[str, Any], quality: pd.DataFrame, leakage: pd.DataFrame) -> str:
    failed = summary.get("failed_checks", [])
    failed_text = ", ".join(failed) if failed else "None"
    quality_table = _markdown_table(quality)
    leakage_table = _markdown_table(leakage.head(20))
    rates = summary["positive_rates"]
    return f"""# CxA+ Target Builder Report

## Executive summary

This PR builds the first governed CxA+ possession-window target dataset. No
model is trained, existing diagnostic CxA outputs are not modified, and all
CxA+ targets are written separately under `feature_store/cxa_plus/`.

## What CxA+ is

Diagnostic CxA estimates whether an action directly creates a shot via
`shot_created`. CxA+ is the next layer: it estimates downstream chance creation
inside the same possession window. Advanced CxA state-value deltas are separate
future work and are not implemented here.

## Primary target definition

Primary target: `{PRIMARY_TARGET}`.

For each action, the label is 1 when a future action within the next five
actions in the same match and same possession has `shot_created = 1`; otherwise
it is 0. The current action's own `shot_created` value is never counted.

## Why next-5 is the first governed target

Next-5 is short enough to avoid broad rest-of-possession attribution while still
capturing downstream chance creation beyond direct assists. It is binary,
interpretable, and appropriate for the first CxA+ model-design step.

## Target-building rules

Actions are sorted by `match_id`, `possession`, `action_position`,
`sequence_id`, and `action_id`. Lookahead windows stay inside the same match and
possession. The target builder also writes next-1, next-3,
rest-of-possession, max future shot value, summed future shot value, and
discounted downstream shot value as diagnostic/reference targets.

## Target rates

- next 1 positive rate: {rates["next_1"]:.6f}
- next 3 positive rate: {rates["next_3"]:.6f}
- next 5 positive rate: {rates["next_5"]:.6f}
- rest-of-possession positive rate: {rates["rest_of_possession"]:.6f}

## Leakage controls

`shot_created`, `created_shot_id`, `created_shot_cxg`, model outputs, future
window labels, downstream value labels, and outcome/result fields are written to
the leakage exclusion artifact for future modelling. They may appear in target
or audit outputs but must not be used as CxA+ model features.

## Quality results

Quality status: `{summary["quality_status"]}`.

Failed checks: {failed_text}

{quality_table}

## Leakage exclusions preview

{leakage_table}

## Generated outputs

- `feature_store/cxa_plus/cxa_plus_action_targets.parquet`
- `feature_store/cxa_plus/cxa_plus_target_summary.json`
- `feature_store/cxa_plus/cxa_plus_target_quality.csv`
- `feature_store/cxa_plus/cxa_plus_leakage_exclusions.csv`
- `feature_store/cxa_plus/cxa_plus_target_report.md`

## No model training

This script constructs governed target labels only. It does not train, validate,
score, promote, or dashboard any CxA+ model.
"""


def _markdown_table(frame: pd.DataFrame, max_rows: int = 25) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.head(max_rows).copy()
    display = display.fillna("")
    columns = [str(column) for column in display.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(_markdown_cell(row[column]) for column in display.columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_outputs(
    *,
    targets: pd.DataFrame,
    quality: pd.DataFrame,
    leakage: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write the governed CxA+ target output pack."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "targets": output_dir / "cxa_plus_action_targets.parquet",
        "summary": output_dir / "cxa_plus_target_summary.json",
        "quality": output_dir / "cxa_plus_target_quality.csv",
        "leakage_exclusions": output_dir / "cxa_plus_leakage_exclusions.csv",
        "report": output_dir / "cxa_plus_target_report.md",
    }
    targets.to_parquet(outputs["targets"], index=False)
    quality.to_csv(outputs["quality"], index=False)
    leakage.to_csv(outputs["leakage_exclusions"], index=False)
    _write_json(outputs["summary"], summary)
    outputs["report"].write_text(build_report(summary, quality, leakage), encoding="utf-8")
    return outputs


def build_cxa_plus_target_dataset(
    *,
    feature_path: Path = DEFAULT_FEATURE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    design_summary_path: Path = DEFAULT_DESIGN_SUMMARY_PATH,
) -> dict[str, Path]:
    """Build and write CxA+ target artifacts."""

    frame = load_action_features(feature_path)
    missing_order_fields = validate_required_columns(frame)
    targets = build_cxa_plus_targets(frame)
    quality = compute_quality_checks(targets, missing_order_fields=missing_order_fields)
    leakage_columns = list(dict.fromkeys([*frame.columns.tolist(), *targets.columns.tolist()]))
    leakage = build_leakage_exclusions(pd.DataFrame(columns=leakage_columns))
    output_paths = {
        "targets": output_dir / "cxa_plus_action_targets.parquet",
        "summary": output_dir / "cxa_plus_target_summary.json",
        "quality": output_dir / "cxa_plus_target_quality.csv",
        "leakage_exclusions": output_dir / "cxa_plus_leakage_exclusions.csv",
        "report": output_dir / "cxa_plus_target_report.md",
    }
    summary = build_summary(
        targets=targets,
        quality=quality,
        input_path=feature_path,
        output_dir=output_dir,
        outputs=output_paths,
        design_summary=_read_optional_json(design_summary_path),
    )
    return write_outputs(
        targets=targets,
        quality=quality,
        leakage=leakage,
        summary=summary,
        output_dir=output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--design-summary", type=Path, default=DEFAULT_DESIGN_SUMMARY_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_cxa_plus_target_dataset(
        feature_path=args.input,
        output_dir=args.output_dir,
        design_summary_path=args.design_summary,
    )
    print("Built CxA+ target artifacts:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
