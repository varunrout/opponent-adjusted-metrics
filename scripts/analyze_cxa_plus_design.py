#!/usr/bin/env python
"""Design and audit CxA+ possession-window attribution targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_FEATURE_PATH = Path("feature_store/cxa/action_features.parquet")
DEFAULT_DIAGNOSTIC_RESULTS_PATH = Path(
    "outputs/results/cxa/diagnostic_v1/action_predictions.parquet"
)
DEFAULT_OUTPUT_DIR = Path("outputs/analysis/cxa_plus/design")
DEFAULT_MAX_WINDOW_ACTIONS = 100_000

WINDOWS = {
    "next_1_action": 1,
    "next_3_actions": 3,
    "next_5_actions": 5,
    "rest_of_possession": None,
}
REQUIRED_FIELDS = {
    "action_id": "identifier",
    "match_id": "ordering",
    "possession": "ordering",
    "team_id": "context",
    "player_id": "context",
    "action_position": "ordering",
    "sequence_id": "sequence",
    "shot_created": "current diagnostic target",
    "created_shot_id": "created-shot reference",
    "created_shot_cxg": "created-shot reference",
    "diagnostic_cxa": "diagnostic result reference",
    "start_x": "location",
    "start_y": "location",
    "end_x": "location",
    "end_y": "location",
    "action_type": "action context",
}
OPTIONAL_ORDER_FIELDS = ("period", "minute", "second", "timestamp", "sequence_length_so_far")
LEAKAGE_FIELDS = {
    "created_shot_id": "Created-shot reference/output; never a model feature.",
    "created_shot_cxg": "Downstream shot value reference and candidate target component.",
    "cxa_value": "Existing attribution/output value, not a CxA+ model input.",
    "diagnostic_cxa": "Current model output; exclude when target uses future value.",
    "predicted_shot_created_probability": "Prediction output from diagnostic CxA.",
    "future_window_label": "Future-window labels are targets only.",
    "shot_within_next_5_actions": "Candidate CxA+ target, not a feature.",
    "discounted_downstream_shot_value": "Candidate CxA+ target, not a feature.",
    "outcome": "Post-action outcome field.",
    "result": "Post-action result field.",
}
TARGET_COLUMNS = [
    "shot_within_next_1_action",
    "shot_within_next_3_actions",
    "shot_within_next_5_actions",
    "shot_later_in_possession",
    "max_created_shot_cxg_within_next_5_actions",
    "sum_created_shot_cxg_rest_of_possession",
    "discounted_downstream_shot_value",
    "possession_value_after_action",
]


@dataclass(frozen=True)
class CxAPlusDesignPaths:
    """Input and output paths for CxA+ design analysis."""

    feature_path: Path
    diagnostic_results_path: Path
    output_dir: Path


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


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


def load_cxa_plus_inputs(feature_path: Path, diagnostic_results_path: Path | None) -> pd.DataFrame:
    """Load action features and optional diagnostic result references."""

    frame = _read_table(feature_path).copy()
    if diagnostic_results_path and diagnostic_results_path.exists():
        diagnostic = _read_table(diagnostic_results_path)
        merge_columns = [
            column
            for column in (
                "action_id",
                "diagnostic_cxa",
                "predicted_shot_created_probability",
                "created_shot_cxg_reference",
                "created_shot_id_reference",
            )
            if column in diagnostic.columns
        ]
        if "action_id" in frame.columns and "action_id" in merge_columns:
            diagnostic = diagnostic[merge_columns].drop_duplicates("action_id")
            for column in merge_columns:
                if column != "action_id" and column in frame.columns:
                    diagnostic = diagnostic.drop(columns=[column])
            frame = frame.merge(diagnostic, on="action_id", how="left")
    return frame


def audit_required_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Audit fields needed for CxA+ window attribution design."""

    rows = []
    total = len(frame)
    for field, role in REQUIRED_FIELDS.items():
        present = field in frame.columns
        missing = int(frame[field].isna().sum()) if present else total
        severity = "required" if field in {"action_id", "match_id", "possession"} else "recommended"
        status = "present" if present else "missing"
        if present and missing > 0:
            status = "partial"
        rows.append(
            {
                "field": field,
                "role": role,
                "present": present,
                "missing_count": missing,
                "missing_pct": (missing / total) if total else np.nan,
                "status": status,
                "severity": severity,
                "notes": _field_note(field, present),
            }
        )
    for field in OPTIONAL_ORDER_FIELDS:
        if field in REQUIRED_FIELDS:
            continue
        present = field in frame.columns
        missing = int(frame[field].isna().sum()) if present else total
        rows.append(
            {
                "field": field,
                "role": "optional ordering",
                "present": present,
                "missing_count": missing,
                "missing_pct": (missing / total) if total else np.nan,
                "status": "present" if present else "missing",
                "severity": "optional",
                "notes": "Useful as a secondary ordering or audit field.",
            }
        )
    return pd.DataFrame(rows)


def _field_note(field: str, present: bool) -> str:
    if not present:
        return (
            "Missing; downstream design should either derive it upstream or block target building."
        )
    if field in {"created_shot_id", "created_shot_cxg", "diagnostic_cxa"}:
        return "Reference/target-audit field only, not a model feature."
    if field in {"action_position", "minute", "second", "sequence_length_so_far"}:
        return "Supports action ordering inside possession windows."
    return "Available for CxA+ design audit."


def resolve_order_columns(frame: pd.DataFrame) -> list[str]:
    """Resolve deterministic ordering columns or fail clearly."""

    missing_group = [column for column in ("match_id", "possession") if column not in frame.columns]
    if missing_group:
        raise ValueError(
            "CxA+ design requires match_id and possession to order possession windows; "
            f"missing={missing_group}"
        )

    if "action_position" in frame.columns:
        return ["match_id", "possession", "action_position", *_tie_breakers(frame)]
    if {"period", "minute", "second"}.issubset(frame.columns):
        return ["match_id", "possession", "period", "minute", "second", *_tie_breakers(frame)]
    if "sequence_length_so_far" in frame.columns:
        return ["match_id", "possession", "sequence_length_so_far", *_tie_breakers(frame)]
    raise ValueError(
        "CxA+ design requires a within-possession ordering field such as action_position, "
        "period/minute/second, or sequence_length_so_far."
    )


def _tie_breakers(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("sequence_id", "action_id") if column in frame.columns]


def sorted_action_frame(frame: pd.DataFrame) -> pd.DataFrame:
    order_columns = resolve_order_columns(frame)
    sorted_frame = frame.copy()
    for column in order_columns:
        if column in {"action_position", "period", "minute", "second", "sequence_length_so_far"}:
            sorted_frame[column] = pd.to_numeric(sorted_frame[column], errors="coerce")
    return sorted_frame.sort_values(order_columns, kind="mergesort").reset_index(drop=True)


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


def possession_prefix_sample(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Return a deterministic possession-preserving prefix for window diagnostics."""

    sorted_frame = sorted_action_frame(frame)
    if not max_rows or max_rows <= 0 or len(sorted_frame) <= max_rows:
        return sorted_frame

    selected_end = 0
    for _, end in _possession_slices(sorted_frame):
        if end > max_rows and selected_end > 0:
            break
        selected_end = end
        if selected_end >= max_rows:
            break
    return sorted_frame.iloc[:selected_end].copy()


def add_downstream_window_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Add candidate CxA+ target columns without changing source artifacts."""

    working = sorted_action_frame(frame)
    target_arrays: dict[str, np.ndarray] = {
        column: np.full(len(working), np.nan) for column in TARGET_COLUMNS
    }
    for start, end in _possession_slices(working):
        group = working.iloc[start:end]
        targets = _compute_group_targets(group)
        for column, values in targets.items():
            target_arrays[column][start:end] = values
    for column, values in target_arrays.items():
        working[column] = values
    return working


def _compute_group_targets(group: pd.DataFrame) -> dict[str, np.ndarray]:
    shot = pd.to_numeric(group.get("shot_created", 0), errors="coerce").fillna(0).gt(0).to_numpy()
    raw_values = pd.to_numeric(group.get("created_shot_cxg", 0.0), errors="coerce").to_numpy()
    values = np.nan_to_num(raw_values, nan=0.0)
    shot_values = np.where(shot, values, 0.0)
    n_rows = len(group)

    return {
        "shot_within_next_1_action": _future_any(shot, 1).astype(int),
        "shot_within_next_3_actions": _future_any(shot, 3).astype(int),
        "shot_within_next_5_actions": _future_any(shot, 5).astype(int),
        "shot_later_in_possession": _future_rest_any(shot).astype(int),
        "max_created_shot_cxg_within_next_5_actions": _future_max(shot_values, 5),
        "sum_created_shot_cxg_rest_of_possession": _future_rest_sum(shot_values),
        "discounted_downstream_shot_value": _future_discounted_sum(shot, values),
        "possession_value_after_action": np.full(n_rows, np.nan),
    }


def _future_any(values: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros(len(values), dtype=bool)
    for offset in range(1, window + 1):
        if offset >= len(values):
            break
        result[:-offset] |= values[offset:]
    return result


def _future_rest_any(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(bool)
    future_counts = np.cumsum(values[::-1])[::-1] - values.astype(int)
    return future_counts > 0


def _future_max(values: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)
    for offset in range(1, window + 1):
        if offset >= len(values):
            break
        result[:-offset] = np.maximum(result[:-offset], values[offset:])
    return result


def _future_rest_sum(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(float)
    return np.cumsum(values[::-1])[::-1] - values


def _future_discounted_sum(shot: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=float)
    shot_positions = np.flatnonzero(shot & ~np.isnan(values))
    for shot_position in shot_positions:
        if shot_position == 0:
            continue
        previous_positions = np.arange(shot_position)
        result[previous_positions] += values[shot_position] / (shot_position - previous_positions)
    return result


def possession_window_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise action availability for each possession window."""

    working = sorted_action_frame(frame)
    rows = []
    for window_name, size in WINDOWS.items():
        eligible_count = 0
        future_count_sum = 0
        for start, end in _possession_slices(working):
            n_rows = end - start
            if n_rows <= 1:
                continue
            remaining = np.arange(n_rows - 1, -1, -1)
            window_counts = remaining if size is None else np.minimum(size, remaining)
            eligible_count += int(np.count_nonzero(window_counts > 0))
            future_count_sum += int(window_counts.sum())
        rows.append(
            {
                "window": window_name,
                "window_size_actions": "rest_of_possession" if size is None else size,
                "total_actions": len(working),
                "eligible_actions": eligible_count,
                "eligible_rate": eligible_count / len(working) if len(working) else np.nan,
                "mean_future_actions_in_window": (
                    future_count_sum / len(working) if len(working) else np.nan
                ),
                "ordering_columns": ", ".join(resolve_order_columns(working)),
            }
        )
    return pd.DataFrame(rows)


def downstream_shot_window_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute downstream shot/value rates for possession windows."""

    working = (
        sorted_action_frame(frame)
        if set(TARGET_COLUMNS).issubset(frame.columns)
        else add_downstream_window_targets(frame)
    )
    rows = []
    for window_name, size in WINDOWS.items():
        eligible = 0
        downstream = 0
        value_sum = 0.0
        value_count = 0
        missing_values = 0
        same_team_hits = 0
        downstream_hits = 0
        target_column = _target_column_for_window(size)
        for start, end in _possession_slices(working):
            group = working.iloc[start:end]
            group_summary = _window_group_rate_summary(group, size, target_column)
            eligible += group_summary["eligible"]
            downstream += group_summary["downstream"]
            value_sum += group_summary["value_sum"]
            value_count += group_summary["value_count"]
            missing_values += group_summary["missing_values"]
            same_team_hits += group_summary["same_team_hits"]
            downstream_hits += group_summary["downstream_hits"]
        rows.append(
            {
                "window": window_name,
                "window_size_actions": "rest_of_possession" if size is None else size,
                "eligible_actions": eligible,
                "actions_with_downstream_shot": downstream,
                "downstream_shot_rate": downstream / eligible if eligible else np.nan,
                "mean_downstream_created_shot_cxg": (
                    value_sum / value_count if value_count else np.nan
                ),
                "missing_downstream_shot_value_rate": (
                    missing_values / downstream_hits if downstream_hits else np.nan
                ),
                "same_team_consistency_rate": (
                    same_team_hits / downstream_hits if downstream_hits else np.nan
                ),
                "same_possession_consistency_rate": 1.0 if downstream_hits else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _window_group_rate_summary(
    group: pd.DataFrame, size: int | None, target_column: str
) -> dict[str, Any]:
    n_rows = len(group)
    if n_rows <= 1:
        return {
            "eligible": 0,
            "downstream": 0,
            "value_sum": 0.0,
            "value_count": 0,
            "missing_values": 0,
            "same_team_hits": 0,
            "downstream_hits": 0,
        }

    eligible = n_rows - 1
    if size is not None and size < n_rows:
        eligible = n_rows - 1
    downstream = int(pd.to_numeric(group[target_column], errors="coerce").fillna(0).gt(0).sum())
    shot = pd.to_numeric(group.get("shot_created", 0), errors="coerce").fillna(0).gt(0).to_numpy()
    values = pd.to_numeric(group.get("created_shot_cxg"), errors="coerce").to_numpy()
    teams = group["team_id"].to_numpy() if "team_id" in group.columns else np.array([])

    value_sum = 0.0
    value_count = 0
    missing_values = 0
    same_team_hits = 0
    downstream_hits = 0
    for shot_position in np.flatnonzero(shot):
        previous_start = 0 if size is None else max(0, shot_position - size)
        previous_positions = np.arange(previous_start, shot_position)
        if len(previous_positions) == 0:
            continue
        downstream_hits += len(previous_positions)
        value = values[shot_position]
        if pd.isna(value):
            missing_values += len(previous_positions)
        else:
            value_sum += float(value) * len(previous_positions)
            value_count += len(previous_positions)
        if len(teams):
            same_team_hits += int(np.sum(teams[previous_positions] == teams[shot_position]))

    return {
        "eligible": eligible,
        "downstream": downstream,
        "value_sum": value_sum,
        "value_count": value_count,
        "missing_values": missing_values,
        "same_team_hits": same_team_hits,
        "downstream_hits": downstream_hits,
    }


def _target_column_for_window(size: int | None) -> str:
    if size is None:
        return "shot_later_in_possession"
    if size == 1:
        return "shot_within_next_1_action"
    return f"shot_within_next_{size}_actions"


def candidate_target_audit(frame_with_targets: pd.DataFrame) -> pd.DataFrame:
    """Evaluate candidate CxA+ targets."""

    specs = {
        "shot_within_next_1_action": (
            "binary",
            "low",
            "Very direct but sparse; useful diagnostic, likely too narrow first target.",
            False,
        ),
        "shot_within_next_3_actions": (
            "binary",
            "low",
            "Short-window downstream chance creation with clear football meaning.",
            False,
        ),
        "shot_within_next_5_actions": (
            "binary",
            "low",
            "Recommended first target: enough downstream horizon without full value attribution.",
            True,
        ),
        "shot_later_in_possession": (
            "binary",
            "medium",
            "Interpretable but can reward actions far from the shot inside long possessions.",
            False,
        ),
        "max_created_shot_cxg_within_next_5_actions": (
            "regression",
            "medium",
            "Good value target after binary feasibility is established.",
            False,
        ),
        "sum_created_shot_cxg_rest_of_possession": (
            "regression",
            "medium",
            "Captures possession value but can leak long downstream possession dynamics.",
            False,
        ),
        "discounted_downstream_shot_value": (
            "regression",
            "medium",
            "Promising CxA+ value target after first binary model.",
            False,
        ),
        "possession_value_after_action": (
            "regression",
            "high",
            "Belongs closer to Advanced CxA state-value modelling.",
            False,
        ),
    }
    rows = []
    for target, (target_type, leakage_risk, reason, recommended) in specs.items():
        series = frame_with_targets[target] if target in frame_with_targets.columns else pd.Series()
        numeric = pd.to_numeric(series, errors="coerce")
        if target_type == "binary":
            rate = float(numeric.fillna(0).gt(0).mean()) if len(numeric) else np.nan
            metric_name = "positive_rate"
        else:
            rate = float(numeric.fillna(0).gt(0).mean()) if len(numeric) else np.nan
            metric_name = "nonzero_rate"
        rows.append(
            {
                "target_name": target,
                "target_type": target_type,
                metric_name: rate,
                "positive_rate": rate if target_type == "binary" else np.nan,
                "nonzero_rate": rate if target_type == "regression" else np.nan,
                "missing_rate": float(numeric.isna().mean()) if len(numeric) else np.nan,
                "leakage_risk": leakage_risk,
                "interpretability": _interpretability(target),
                "recommended_for_first_model": recommended,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _interpretability(target: str) -> str:
    if target.startswith("shot_within"):
        return "high"
    if target == "shot_later_in_possession":
        return "medium_high"
    if "created_shot_cxg" in target or "downstream_shot_value" in target:
        return "medium"
    return "requires_state_value_design"


def leakage_risk_register(frame: pd.DataFrame) -> pd.DataFrame:
    """List CxA+ fields that must remain targets/references, not features."""

    rows = []
    for field, reason in LEAKAGE_FIELDS.items():
        rows.append(
            {
                "field": field,
                "present": field in frame.columns,
                "risk_type": "future_or_output_reference",
                "leakage_risk": "high",
                "model_feature_allowed": False,
                "can_appear_in_outputs": True,
                "reason": reason,
            }
        )
    future_like = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in ("future", "downstream", "within_next"))
    ]
    for field in sorted(set(future_like).difference(LEAKAGE_FIELDS)):
        rows.append(
            {
                "field": field,
                "present": True,
                "risk_type": "future_window_label",
                "leakage_risk": "high",
                "model_feature_allowed": False,
                "can_appear_in_outputs": True,
                "reason": "Future-window label or downstream value; target/reporting only.",
            }
        )
    return pd.DataFrame(rows)


def sequence_window_examples(frame_with_targets: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """Return readable examples of possession windows with downstream shots."""

    columns = [
        column
        for column in (
            "action_id",
            "match_id",
            "possession",
            "sequence_id",
            "action_position",
            "team_id",
            "player_id",
            "action_type",
            "shot_created",
            "created_shot_cxg",
            "shot_within_next_5_actions",
            "shot_later_in_possession",
            "discounted_downstream_shot_value",
        )
        if column in frame_with_targets.columns
    ]
    if not columns:
        return pd.DataFrame()
    examples = frame_with_targets.loc[
        pd.to_numeric(frame_with_targets.get("shot_within_next_5_actions", 0), errors="coerce")
        .fillna(0)
        .gt(0),
        columns,
    ]
    if examples.empty:
        examples = frame_with_targets[columns].head(limit)
    return examples.head(limit).copy()


def build_design_summary(
    *,
    full_frame: pd.DataFrame,
    window_frame: pd.DataFrame,
    required_fields: pd.DataFrame,
    coverage: pd.DataFrame,
    window_rates: pd.DataFrame,
    candidate_targets: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    recommended = candidate_targets[candidate_targets["recommended_for_first_model"]]
    required_missing = required_fields[
        (required_fields["severity"] == "required") & (~required_fields["present"])
    ]["field"].tolist()
    return {
        "analysis": "cxa_plus_possession_window_design",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(full_frame),
        "column_count": len(full_frame.columns),
        "window_analysis_row_count": len(window_frame),
        "window_analysis_is_sampled": len(window_frame) < len(full_frame),
        "ordering_columns": resolve_order_columns(full_frame),
        "missing_required_fields": required_missing,
        "window_count": len(coverage),
        "candidate_target_count": len(candidate_targets),
        "recommended_first_target": (
            recommended.iloc[0]["target_name"] if not recommended.empty else None
        ),
        "recommended_next_pr": "modeling: add first CxA+ possession-window target builder",
        "outputs": {
            "cxa_plus_design_report": (output_dir / "cxa_plus_design_report.md").as_posix(),
            "cxa_plus_design_summary": (output_dir / "cxa_plus_design_summary.json").as_posix(),
            "possession_window_coverage": (
                output_dir / "possession_window_coverage.csv"
            ).as_posix(),
            "downstream_shot_window_rates": (
                output_dir / "downstream_shot_window_rates.csv"
            ).as_posix(),
            "candidate_targets": (output_dir / "candidate_targets.csv").as_posix(),
            "leakage_risk_register": (output_dir / "leakage_risk_register.csv").as_posix(),
            "required_fields_audit": (output_dir / "required_fields_audit.csv").as_posix(),
            "sequence_window_examples": (output_dir / "sequence_window_examples.csv").as_posix(),
        },
        "headline_window_rates": window_rates.to_dict(orient="records"),
    }


def build_design_report(
    *,
    summary: dict[str, Any],
    required_fields: pd.DataFrame,
    coverage: pd.DataFrame,
    window_rates: pd.DataFrame,
    candidate_targets: pd.DataFrame,
    leakage: pd.DataFrame,
) -> str:
    recommended = summary.get("recommended_first_target") or "No target recommended"
    missing_required = summary.get("missing_required_fields", [])
    coverage_table = _markdown_table(coverage)
    rates_table = _markdown_table(window_rates)
    targets_table = _markdown_table(candidate_targets)
    missing_text = ", ".join(missing_required) if missing_required else "None"
    leakage_fields = ", ".join(leakage["field"].head(10).astype(str).tolist())
    available_required = required_fields[required_fields["present"]]["field"].tolist()
    return f"""# CxA+ Possession-Window Design Report

## Executive summary

This analysis designs the next CxA+ layer without training a model or changing
current CxA results. The current diagnostic CxA model estimates whether an
action directly creates a shot. CxA+ should instead estimate downstream chance
creation inside a possession window. The recommended first modelling target is
`{recommended}` because it is binary, interpretable, possession-window-aware,
and safer than jumping directly to value attribution.

## Difference between Diagnostic CxA, CxA+, and Advanced CxA

- Diagnostic CxA predicts `shot_created`: whether an action creates a shot.
- CxA+ should predict downstream chance creation or value within the next few
  actions or the rest of the possession.
- Advanced CxA should later estimate state-value delta:
  attacking_state_value_after_action - attacking_state_value_before_action.

## Available data assessment

Rows audited: {summary["row_count"]:,}

Columns audited: {summary["column_count"]:,}

Rows used for possession-window diagnostics: {summary["window_analysis_row_count"]:,}

Window diagnostics sampled: {summary["window_analysis_is_sampled"]}

Ordering columns: {", ".join(summary["ordering_columns"])}

Missing required fields: {missing_text}

Available required fields include: {", ".join(available_required[:20])}

## Possession-window feasibility

The audit builds downstream windows for the next 1 action, next 3 actions, next
5 actions, and the rest of the possession. These are design diagnostics only;
they are not persisted back into the feature store.

{coverage_table}

## Downstream shot/value rates

{rates_table}

## Candidate target comparison

{targets_table}

## Leakage risks

Fields such as {leakage_fields} must remain target/reference/reporting fields,
not model inputs. Created-shot IDs, created-shot xG, diagnostic predictions,
future-window labels, and post-action outcomes are leakage risks for CxA+
modelling.

## Recommended first CxA+ modelling path

Start with a binary possession-window target: `{recommended}`. This gives the
next modelling PR a clear supervised target that is downstream-aware without
requiring a full state-value model. After validating that target builder, add a
regression/value target such as `discounted_downstream_shot_value`.

## Next recommended PR

modeling: add first CxA+ possession-window target builder
"""


def _markdown_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows available._"
    display = frame.head(max_rows).copy()
    columns = [str(column) for column in display.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in display.iterrows():
        values = [_format_markdown_value(row[column]) for column in display.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def analyze_cxa_plus_design(
    *,
    feature_path: Path = DEFAULT_FEATURE_PATH,
    diagnostic_results_path: Path = DEFAULT_DIAGNOSTIC_RESULTS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_window_actions: int | None = DEFAULT_MAX_WINDOW_ACTIONS,
) -> dict[str, Path]:
    frame = load_cxa_plus_inputs(feature_path, diagnostic_results_path)
    required_fields = audit_required_fields(frame)
    window_frame = possession_prefix_sample(frame, max_window_actions)
    frame_with_targets = add_downstream_window_targets(window_frame)
    coverage = possession_window_coverage(window_frame)
    window_rates = downstream_shot_window_rates(frame_with_targets)
    candidates = candidate_target_audit(frame_with_targets)
    leakage = leakage_risk_register(frame_with_targets)
    examples = sequence_window_examples(frame_with_targets)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "cxa_plus_design_report": output_dir / "cxa_plus_design_report.md",
        "cxa_plus_design_summary": output_dir / "cxa_plus_design_summary.json",
        "possession_window_coverage": output_dir / "possession_window_coverage.csv",
        "downstream_shot_window_rates": output_dir / "downstream_shot_window_rates.csv",
        "candidate_targets": output_dir / "candidate_targets.csv",
        "leakage_risk_register": output_dir / "leakage_risk_register.csv",
        "required_fields_audit": output_dir / "required_fields_audit.csv",
        "sequence_window_examples": output_dir / "sequence_window_examples.csv",
    }
    required_fields.to_csv(outputs["required_fields_audit"], index=False)
    coverage.to_csv(outputs["possession_window_coverage"], index=False)
    window_rates.to_csv(outputs["downstream_shot_window_rates"], index=False)
    candidates.to_csv(outputs["candidate_targets"], index=False)
    leakage.to_csv(outputs["leakage_risk_register"], index=False)
    examples.to_csv(outputs["sequence_window_examples"], index=False)

    summary = build_design_summary(
        full_frame=frame,
        window_frame=window_frame,
        required_fields=required_fields,
        coverage=coverage,
        window_rates=window_rates,
        candidate_targets=candidates,
        output_dir=output_dir,
    )
    _write_json(outputs["cxa_plus_design_summary"], summary)
    outputs["cxa_plus_design_report"].write_text(
        build_design_report(
            summary=summary,
            required_fields=required_fields,
            coverage=coverage,
            window_rates=window_rates,
            candidate_targets=candidates,
            leakage=leakage,
        ),
        encoding="utf-8",
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument(
        "--diagnostic-results-path", type=Path, default=DEFAULT_DIAGNOSTIC_RESULTS_PATH
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max-window-actions",
        type=int,
        default=DEFAULT_MAX_WINDOW_ACTIONS,
        help=(
            "Maximum deterministic possession-prefix rows for window diagnostics. "
            "Use 0 to audit all rows."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = analyze_cxa_plus_design(
        feature_path=args.feature_path,
        diagnostic_results_path=args.diagnostic_results_path,
        output_dir=args.output_dir,
        max_window_actions=args.max_window_actions,
    )
    print("CxA+ design analysis written:")
    for path in outputs.values():
        print(f"- {path}")


if __name__ == "__main__":
    main()
