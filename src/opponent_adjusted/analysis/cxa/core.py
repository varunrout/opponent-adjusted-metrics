"""Pre-model CxA target and action-feature diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from sqlalchemy import Select, select
from sqlalchemy.orm import Session

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opponent_adjusted.db.models import ActionFeature

DEFAULT_OUTPUT_DIR = Path("outputs/analysis/cxa")
TARGET_CANDIDATES = (
    "shot_created",
    "created_shot",
    "creates_shot",
    "downstream_shot_created",
    "target_shot_created",
    "target",
    "label",
)
VALUE_CANDIDATES = (
    "created_shot_cxg",
    "target_created_shot_cxg",
    "created_shot_value",
    "target_created_shot_value",
)
REQUIRED_SECTIONS = (
    "Question",
    "Calculation",
    "Visual/Table",
    "Interpretation",
    "Modelling implication",
    "Limitation",
)
ID_COLUMNS = {
    "action_id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "possession_id",
    "possession_number",
    "sequence_id",
    "created_shot_id",
    "created_shot_event_id",
    "target_created_shot_id",
    "competition_id",
}
DOWNSTREAM_REFERENCE_COLUMNS = {
    "seconds_until_shot",
    "time_until_shot",
    "actions_until_shot",
    "downstream_action_count",
    "created_shot_id",
    "created_shot_event_id",
    "target_created_shot_id",
    "same_possession",
    "same_team",
}
POST_MODEL_PATTERNS = ("prediction", "cxa", "model", "registry", "aggregate", "leaderboard")
FEATURE_FOCUS = (
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "dx",
    "dy",
    "distance_progressed",
    "goal_distance_start",
    "goal_distance_end",
    "goal_distance_reduction",
    "progressive_distance",
    "action_duration",
    "sequence_position",
    "time_since_possession_start",
    "is_progressive",
    "final_third_entry",
    "box_entry",
    "zone14_entry",
    "cross",
    "through_ball",
    "cutback",
    "switch",
    "carry",
    "dribble",
    "under_pressure",
    "carry_under_pressure",
)
REDUNDANCY_PAIRS = (
    ("start_x", "end_x"),
    ("start_y", "end_y"),
    ("start_x", "goal_distance_start"),
    ("end_x", "goal_distance_end"),
    ("distance_progressed", "goal_distance_reduction"),
    ("final_third_entry", "box_entry"),
    ("progressive_distance", "is_progressive"),
    ("sequence_position", "time_since_possession_start"),
)
WINDOW_FIELDS = (
    "downstream_action_count",
    "actions_until_shot",
    "seconds_until_shot",
    "time_until_shot",
    "sequence_position",
    "possession_id",
    "sequence_id",
    "created_shot_id",
    "created_shot_event_id",
    "same_possession",
    "same_team",
)


@dataclass(frozen=True)
class CxAAnalysisResult:
    """Paths and summary metrics emitted by the pre-model CxA analysis."""

    output_dir: Path
    report_path: Path
    row_count: int
    target_column: str
    target_rate: float
    candidate_feature_count: int
    leakage_risk_count: int


def load_action_feature_dataset(
    session: Session,
    *,
    feature_family: str | None = "cxa",
    version_tag: str | None = None,
) -> pd.DataFrame:
    """Load pre-model CxA rows from `action_features` only."""

    stmt: Select[tuple[ActionFeature]] = select(ActionFeature)
    if feature_family is not None:
        stmt = stmt.where(ActionFeature.feature_family == feature_family)
    if version_tag is not None:
        stmt = stmt.where(ActionFeature.version_tag == version_tag)

    records = []
    for row in session.execute(stmt).scalars().all():
        records.append(
            {
                "feature_family": row.feature_family,
                "version_tag": row.version_tag,
                "action_id": row.action_id,
                "event_id": row.event_id,
                "match_id": row.match_id,
                "team_id": row.team_id,
                "player_id": row.player_id,
                "possession_id": row.possession_id,
                "possession_number": row.possession_number,
                "sequence_id": row.sequence_id,
                "action_type": row.action_type,
                "start_x": row.start_x,
                "start_y": row.start_y,
                "end_x": row.end_x,
                "end_y": row.end_y,
                "length": row.length,
                "angle": row.angle,
                "x_progression": row.x_progression,
                "y_progression": row.y_progression,
                "distance_to_goal_before": row.distance_to_goal_before,
                "distance_to_goal_after": row.distance_to_goal_after,
                "angle_to_goal_before": row.angle_to_goal_before,
                "angle_to_goal_after": row.angle_to_goal_after,
                "start_zone": row.start_zone,
                "end_zone": row.end_zone,
                "is_progressive": row.is_progressive,
                "enters_final_third": row.enters_final_third,
                "enters_penalty_area": row.enters_penalty_area,
                "target_shot_created": row.target_shot_created,
                "target_created_shot_cxg": row.target_created_shot_cxg,
                "target_created_shot_id": row.target_created_shot_id,
            }
        )
    return pd.DataFrame.from_records(records)


def detect_cxa_target_column(frame: pd.DataFrame) -> str:
    """Detect the binary CxA shot-created target column."""

    for column in TARGET_CANDIDATES:
        if column in frame.columns:
            return column
    expected = ", ".join(TARGET_CANDIDATES)
    raise ValueError(f"CxA analysis requires one target column. Expected one of: {expected}")


def build_pre_model_cxa_analysis(
    action_features: pd.DataFrame,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    min_slice_size: int = 30,
) -> CxAAnalysisResult:
    """Generate the full pre-model CxA target and action-feature study."""

    output_path = Path(output_dir)
    folders = _create_output_folders(output_path)
    frame = _prepare_dataset(action_features)
    target_column = detect_cxa_target_column(frame)
    value_column = _detect_value_column(frame)
    _apply_matplotlib_style()

    target = _target_analysis(frame, target_column, value_column, folders["00_target"])
    coverage = _action_coverage(frame, folders["01_action_coverage"], min_slice_size)
    distributions = _feature_distributions(
        frame, target_column, folders["02_feature_distributions"]
    )
    relationships = _feature_target_relationships(
        frame, target_column, folders["03_feature_target_relationships"], min_slice_size
    )
    correlations = _feature_correlations(frame, target_column, folders["04_feature_correlations"])
    windows = _sequence_window_stability(
        frame, target_column, folders["05_sequence_window_stability"], min_slice_size
    )
    stability = _slice_stability(
        frame, target_column, folders["06_slice_stability"], min_slice_size
    )
    quality = _data_quality(frame, target_column, folders["07_data_quality"], min_slice_size)
    leakage = _leakage_checks(frame, target_column, value_column, folders["08_leakage_checks"])

    report_path = output_path / "report.md"
    report_path.write_text(
        _render_report(
            target=target,
            coverage=coverage,
            distributions=distributions,
            relationships=relationships,
            correlations=correlations,
            windows=windows,
            stability=stability,
            quality=quality,
            leakage=leakage,
            min_slice_size=min_slice_size,
        ),
        encoding="utf-8",
    )

    return CxAAnalysisResult(
        output_dir=output_path,
        report_path=report_path,
        row_count=int(target["row_count"]),
        target_column=target_column,
        target_rate=float(target["target_rate"]),
        candidate_feature_count=int(distributions["feature_count"]),
        leakage_risk_count=int(leakage["risk_count"]),
    )


def run_pre_model_cxa_analysis(
    session: Session,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    feature_family: str | None = "cxa",
    version_tag: str | None = None,
    min_slice_size: int = 30,
) -> CxAAnalysisResult:
    """Load DB action features and write the pre-model CxA analysis report."""

    frame = load_action_feature_dataset(
        session,
        feature_family=feature_family,
        version_tag=version_tag,
    )
    return build_pre_model_cxa_analysis(
        frame,
        output_dir=output_dir,
        min_slice_size=min_slice_size,
    )


def _create_output_folders(output_dir: Path) -> dict[str, Path]:
    names = (
        "00_target",
        "01_action_coverage",
        "02_feature_distributions",
        "03_feature_target_relationships",
        "04_feature_correlations",
        "05_sequence_window_stability",
        "06_slice_stability",
        "07_data_quality",
        "08_leakage_checks",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    folders = {name: output_dir / name for name in names}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plots").mkdir(exist_ok=True)
        (folder / "tables").mkdir(exist_ok=True)
    return folders


def _prepare_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    alias_map = {
        "target_shot_created": "shot_created",
        "target_created_shot_cxg": "created_shot_cxg",
        "target_created_shot_id": "created_shot_id",
        "x_progression": "dx",
        "y_progression": "dy",
        "length": "distance_progressed",
        "distance_to_goal_before": "goal_distance_start",
        "distance_to_goal_after": "goal_distance_end",
        "is_cross": "cross",
        "is_through_ball": "through_ball",
        "is_cutback": "cutback",
        "switches_play": "switch",
        "is_carry": "carry",
        "is_dribble": "dribble",
        "enters_final_third": "final_third_entry",
        "enters_penalty_area": "box_entry",
        "enters_zone14": "zone14_entry",
        "action_position": "sequence_position",
        "seconds_since_possession_start": "time_since_possession_start",
    }
    for source, alias in alias_map.items():
        if source in prepared.columns and alias not in prepared.columns:
            prepared[alias] = prepared[source]

    if {"goal_distance_start", "goal_distance_end"}.issubset(prepared.columns):
        prepared["goal_distance_reduction"] = pd.to_numeric(
            prepared["goal_distance_start"], errors="coerce"
        ) - pd.to_numeric(prepared["goal_distance_end"], errors="coerce")
    if "dx" in prepared.columns and "progressive_distance" not in prepared.columns:
        prepared["progressive_distance"] = prepared["dx"]
    if "event_type" not in prepared.columns and "action_type" in prepared.columns:
        prepared["event_type"] = prepared["action_type"]
    return prepared


def _detect_value_column(frame: pd.DataFrame) -> str | None:
    for column in VALUE_CANDIDATES:
        if column in frame.columns:
            return column
    return None


def _apply_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (11, 7),
            "figure.dpi": 120,
            "axes.facecolor": "#f7f7f4",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
            "savefig.bbox": "tight",
        }
    )


def _save_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    path: Path,
    *,
    title: str,
    subtitle: str,
    caption: str,
) -> None:
    ax.set_title(title, loc="left", pad=26, fontsize=13, fontweight="bold")
    ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, fontsize=10, color="#374151")
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=8.5, color="#4b5563")
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.12, right=0.96)
    fig.savefig(path)
    plt.close(fig)


def _target_series(frame: pd.DataFrame, target_column: str) -> pd.Series:
    series = frame[target_column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).gt(0).astype(int)
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .isin({"1", "true", "yes", "shot", "goal"})
        .astype(int)
    )


def _target_analysis(
    frame: pd.DataFrame,
    target_column: str,
    value_column: str | None,
    folder: Path,
) -> dict[str, object]:
    target = _target_series(frame, target_column)
    row_count = len(frame)
    positives = int(target.sum())
    target_rate = float(target.mean()) if row_count else 0.0
    status = _target_status(row_count, target_rate)
    summary = pd.DataFrame(
        [
            {
                "rows": row_count,
                "target_column": target_column,
                "positives": positives,
                "negatives": row_count - positives,
                "target_rate": target_rate,
                "status": status,
            }
        ]
    )
    summary.to_csv(folder / "tables" / "target_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        ["No downstream shot", "Created shot"],
        [row_count - positives, positives],
        color=["#6b7280", "#0f766e"],
    )
    ax.axhline(max(row_count * 0.01, 1), color="#b45309", linestyle="--", label="1% reference")
    ax.set_ylabel("Action count")
    ax.legend(loc="upper right")
    _save_plot(
        fig,
        ax,
        folder / "plots" / "target_balance.png",
        title="Is the CxA shot-created target usable?",
        subtitle=f"{positives:,} positives from {row_count:,} actions; target rate {target_rate:.2%}.",
        caption="Source: action_features. Caveat: sparsity requires validation beyond this diagnostic.",
    )

    value_summary_path = folder / "tables" / "created_shot_value_summary.csv"
    value_plot_path = folder / "plots" / "created_shot_value_distribution.png"
    if value_column is not None:
        values = pd.to_numeric(frame[value_column], errors="coerce").dropna()
        value_summary = (
            values.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_frame("value").reset_index()
        )
        value_summary.to_csv(value_summary_path, index=False)
        fig, ax = plt.subplots(figsize=(10, 6.5))
        ax.hist(
            values,
            bins=min(30, max(8, int(np.sqrt(max(len(values), 1))))),
            color="#2563eb",
            alpha=0.82,
        )
        ax.axvline(values.median(), color="#b45309", linestyle="--", label="Median")
        ax.set_xlabel(value_column)
        ax.set_ylabel("Action count")
        ax.legend(loc="upper right")
        _save_plot(
            fig,
            ax,
            value_plot_path,
            title="What is the distribution of created-shot value?",
            subtitle=f"Median {values.median():.4f}; p99 {values.quantile(0.99):.4f}.",
            caption="Source: action_features target/reference value. Caveat: use as target/reference only.",
        )
    else:
        pd.DataFrame(
            [{"status": "skipped", "reason": "No created-shot value column found."}]
        ).to_csv(value_summary_path, index=False)
        _write_skipped_plot(value_plot_path, "No created-shot value column found.")

    return {
        "row_count": row_count,
        "positives": positives,
        "target_rate": target_rate,
        "target_column": target_column,
        "value_column": value_column,
        "status": status,
        "table": "00_target/tables/target_summary.csv",
        "visual": "00_target/plots/target_balance.png",
        "value_table": "00_target/tables/created_shot_value_summary.csv",
        "value_visual": "00_target/plots/created_shot_value_distribution.png",
    }


def _action_coverage(frame: pd.DataFrame, folder: Path, min_slice_size: int) -> dict[str, object]:
    type_column = "action_type" if "action_type" in frame.columns else "event_type"
    if type_column in frame.columns:
        table = (
            frame.assign(action_type_value=frame[type_column].fillna("missing").astype(str))
            .groupby("action_type_value", observed=True)
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
        )
        table["row_share"] = table["rows"] / max(len(frame), 1)
        table["rare_action_type"] = table["rows"] < min_slice_size
    else:
        table = pd.DataFrame(columns=["action_type_value", "rows", "row_share", "rare_action_type"])
    table.to_csv(folder / "tables" / "action_type_coverage.csv", index=False)
    if not table.empty:
        plot = table.head(15).sort_values("rows")
        fig, ax = plt.subplots(figsize=(10, max(6, len(plot) * 0.45)))
        ax.barh(plot["action_type_value"], plot["rows"], color="#0f766e")
        ax.axvline(min_slice_size, color="#b45309", linestyle="--", label="Rare threshold")
        ax.set_xlabel("Action rows")
        ax.set_ylabel("Action type")
        ax.legend(loc="lower right")
        _save_plot(
            fig,
            ax,
            folder / "plots" / "action_type_coverage.png",
            title="Are eligible CxA action types represented?",
            subtitle=f"{len(frame):,} eligible actions across {len(table):,} action types.",
            caption="Source: action_features. Caveat: rare action types may need pooling or exclusion.",
        )
    else:
        _write_skipped_plot(
            folder / "plots" / "action_type_coverage.png", "No action type column found."
        )

    id_columns = [
        c
        for c in (
            "action_id",
            "event_id",
            "match_id",
            "team_id",
            "player_id",
            "possession_id",
            "sequence_id",
        )
        if c in frame.columns
    ]
    id_table = pd.DataFrame(
        [
            {
                "column": column,
                "rows": len(frame),
                "populated_rate": float(frame[column].notna().mean()),
                "unique_values": int(frame[column].nunique(dropna=True)),
            }
            for column in id_columns
        ]
    )
    id_table.to_csv(folder / "tables" / "id_coverage.csv", index=False)

    loc_columns = [c for c in ("start_x", "start_y", "end_x", "end_y") if c in frame.columns]
    loc_table = pd.DataFrame(
        [
            {
                "column": column,
                "rows": len(frame),
                "populated_rate": float(frame[column].notna().mean()),
                "min": float(pd.to_numeric(frame[column], errors="coerce").min()),
                "max": float(pd.to_numeric(frame[column], errors="coerce").max()),
            }
            for column in loc_columns
        ]
    )
    loc_table.to_csv(folder / "tables" / "location_coverage.csv", index=False)
    represented = set(table["action_type_value"]) if not table.empty else set()
    return {
        "action_type_count": len(table),
        "rows": len(frame),
        "represented_core_types": sorted(represented.intersection({"Pass", "Carry", "Dribble"})),
        "rare_count": int(table["rare_action_type"].sum()) if not table.empty else 0,
        "table": "01_action_coverage/tables/action_type_coverage.csv",
        "visual": "01_action_coverage/plots/action_type_coverage.png",
        "id_table": "01_action_coverage/tables/id_coverage.csv",
        "location_table": "01_action_coverage/tables/location_coverage.csv",
    }


def _feature_distributions(
    frame: pd.DataFrame,
    target_column: str,
    folder: Path,
) -> dict[str, object]:
    features = _candidate_feature_columns(frame, target_column)
    focus = [c for c in FEATURE_FOCUS if c in frame.columns and c in features]
    remaining = [c for c in features if c not in focus]
    features = focus + remaining
    numeric = _numeric_features(frame, features)
    categorical = _categorical_features(frame, features)

    numeric_profile = pd.DataFrame([_numeric_profile_row(frame[c], c) for c in numeric])
    numeric_profile.to_csv(folder / "tables" / "numeric_feature_profiles.csv", index=False)
    for column in numeric:
        _plot_numeric_distribution(
            frame, column, folder / "plots" / f"{_slug(column)}_distribution.png"
        )

    categorical_profile = pd.DataFrame([_categorical_profile_row(frame[c], c) for c in categorical])
    categorical_profile.to_csv(folder / "tables" / "categorical_feature_profiles.csv", index=False)
    for column in categorical:
        _plot_categorical_levels(frame, column, folder / "plots" / f"{_slug(column)}_levels.png")

    return {
        "feature_count": len(features),
        "numeric_count": len(numeric),
        "categorical_count": len(categorical),
        "numeric_table": "02_feature_distributions/tables/numeric_feature_profiles.csv",
        "categorical_table": "02_feature_distributions/tables/categorical_feature_profiles.csv",
    }


def _feature_target_relationships(
    frame: pd.DataFrame,
    target_column: str,
    folder: Path,
    min_slice_size: int,
) -> dict[str, object]:
    target = _target_series(frame, target_column)
    global_rate = float(target.mean()) if len(target) else 0.0
    outputs = {}

    action_col = "action_type" if "action_type" in frame.columns else "event_type"
    if action_col in frame.columns:
        outputs["action_type"] = _categorical_target_output(
            frame,
            target_column,
            action_col,
            folder,
            "action_type_target_rate",
            "Which action types most often create downstream shots?",
            min_slice_size,
        )
    else:
        _write_empty_table(folder / "tables" / "action_type_target_rate.csv")
        _write_skipped_plot(
            folder / "plots" / "action_type_target_rate.png", "No action type column found."
        )

    if "end_zone" in frame.columns:
        outputs["end_zone"] = _categorical_target_output(
            frame,
            target_column,
            "end_zone",
            folder,
            "end_zone_target_rate",
            "Do actions ending in advanced zones create shots more often?",
            min_slice_size,
        )
    else:
        _write_empty_table(folder / "tables" / "end_zone_target_rate.csv")
        _write_skipped_plot(
            folder / "plots" / "end_zone_target_rate.png", "No end_zone column found."
        )

    progression_columns = [
        c
        for c in (
            "is_progressive",
            "final_third_entry",
            "box_entry",
            "zone14_entry",
            "through_ball",
            "cutback",
            "cross",
            "carry",
            "dribble",
        )
        if c in frame.columns
    ]
    progression_rows = []
    for column in progression_columns:
        table = _categorical_target_rate(frame, target_column, column, min_slice_size=1)
        for _, row in table.iterrows():
            progression_rows.append({"feature": column, **row.to_dict()})
    progression_table = pd.DataFrame(progression_rows)
    progression_table.to_csv(
        folder / "tables" / "progression_feature_target_rates.csv", index=False
    )
    if not progression_table.empty:
        plot = progression_table[progression_table["value"].isin(["True", "1", "true"])].copy()
        if plot.empty:
            plot = progression_table.sort_values("target_rate", ascending=False).head(12)
        fig, ax = plt.subplots(figsize=(11, max(6, len(plot) * 0.45)))
        labels = plot["feature"] + "=" + plot["value"].astype(str)
        ax.barh(labels, plot["target_rate"], color="#7c3aed")
        ax.axvline(global_rate, color="#b45309", linestyle="--", label="Global")
        ax.set_xlabel("Target rate")
        ax.set_ylabel("Progression feature")
        ax.legend(loc="lower right")
        _save_plot(
            fig,
            ax,
            folder / "plots" / "progression_feature_target_rates.png",
            title="Do progressive and chance-creation flags show target signal?",
            subtitle=f"{len(progression_columns)} progression/context flags were analysed.",
            caption="Source: action_features. Caveat: binary flags can be sparse and correlated.",
        )
    else:
        _write_skipped_plot(
            folder / "plots" / "progression_feature_target_rates.png",
            "No progression/context flags found.",
        )

    seq_col = _first_existing(frame, ("sequence_position", "action_position"))
    if seq_col:
        outputs["sequence_position"] = _binned_target_output(
            frame,
            target_column,
            seq_col,
            folder,
            "sequence_position_target_rate",
            "Does sequence position matter for shot creation?",
        )
    else:
        _write_empty_table(folder / "tables" / "sequence_position_target_rate.csv")
        _write_skipped_plot(
            folder / "plots" / "sequence_position_target_rate.png",
            "No sequence position column found.",
        )

    numeric_rows = []
    for column in _numeric_features(frame, _candidate_feature_columns(frame, target_column)):
        table = _binned_target_rate(frame, target_column, column)
        if table.empty:
            continue
        table.insert(0, "feature", column)
        numeric_rows.append(table)
    numeric_table = (
        pd.concat(numeric_rows, ignore_index=True)
        if numeric_rows
        else pd.DataFrame(
            columns=[
                "feature",
                "bin",
                "rows",
                "positives",
                "target_rate",
                "global_target_rate",
                "lift",
                "sample_size_warning",
                "modelling_implication",
            ]
        )
    )
    numeric_table.to_csv(folder / "tables" / "numeric_target_relationships.csv", index=False)
    return {
        "action_type": outputs.get("action_type", {}),
        "end_zone": outputs.get("end_zone", {}),
        "progression_feature_count": len(progression_columns),
        "sequence_position": outputs.get("sequence_position", {}),
        "numeric_table": "03_feature_target_relationships/tables/numeric_target_relationships.csv",
    }


def _feature_correlations(
    frame: pd.DataFrame,
    target_column: str,
    folder: Path,
) -> dict[str, object]:
    numeric = _numeric_features(frame, _candidate_feature_columns(frame, target_column))
    corr = frame[numeric].corr(numeric_only=True) if len(numeric) >= 2 else pd.DataFrame()
    corr.to_csv(folder / "tables" / "numeric_correlations.csv")
    high_rows = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= 0.8:
                high_rows.append(
                    {"feature_a": left, "feature_b": right, "correlation": float(value)}
                )
    high = pd.DataFrame(high_rows, columns=["feature_a", "feature_b", "correlation"])
    if not high.empty:
        high = high.sort_values("correlation", key=lambda s: s.abs(), ascending=False)
    high.to_csv(folder / "tables" / "high_correlations.csv", index=False)

    targeted_rows = []
    for left, right in REDUNDANCY_PAIRS:
        if left in frame.columns and right in frame.columns:
            valid = frame[[left, right]].dropna()
            corr_value = (
                valid[left].astype(float).corr(valid[right].astype(float))
                if len(valid) and valid[left].nunique() > 1
                else np.nan
            )
            targeted_rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "rows": len(valid),
                    "correlation": float(corr_value) if pd.notna(corr_value) else np.nan,
                    "redundancy_flag": bool(pd.notna(corr_value) and abs(float(corr_value)) >= 0.8),
                    "modelling_implication": _redundancy_implication(corr_value, left, right),
                }
            )
    targeted = pd.DataFrame(
        targeted_rows,
        columns=[
            "feature_a",
            "feature_b",
            "rows",
            "correlation",
            "redundancy_flag",
            "modelling_implication",
        ],
    )
    targeted.to_csv(folder / "tables" / "targeted_redundancy_checks.csv", index=False)

    if not corr.empty:
        fig, ax = plt.subplots(figsize=(max(9, len(corr) * 0.72), max(7, len(corr) * 0.58)))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index)
        fig.colorbar(image, ax=ax, label="Pearson correlation")
        _save_plot(
            fig,
            ax,
            folder / "plots" / "correlation_heatmap.png",
            title="Which movement and location features are redundant?",
            subtitle=f"{len(high):,} pairs exceed absolute correlation 0.80.",
            caption="Source: action_features. Caveat: correlation is linear and does not prove interchangeability.",
        )
    else:
        _write_skipped_plot(
            folder / "plots" / "correlation_heatmap.png",
            "Not enough numeric features for correlations.",
        )
    return {
        "high_pairs": len(high),
        "targeted_pairs": len(targeted),
        "table": "04_feature_correlations/tables/high_correlations.csv",
        "targeted_table": "04_feature_correlations/tables/targeted_redundancy_checks.csv",
        "visual": "04_feature_correlations/plots/correlation_heatmap.png",
    }


def _sequence_window_stability(
    frame: pd.DataFrame,
    target_column: str,
    folder: Path,
    min_slice_size: int,
) -> dict[str, object]:
    missing = [column for column in WINDOW_FIELDS if column not in frame.columns]
    pd.DataFrame(
        {
            "missing_field": missing,
            "recommendation": "Add to action_features for defensible CxA window diagnostics.",
        }
    ).to_csv(folder / "tables" / "missing_window_fields.csv", index=False)
    target = _target_series(frame, target_column)
    positives = frame.loc[target.eq(1)].copy()
    rows = []
    if "actions_until_shot" in frame.columns:
        values = pd.to_numeric(positives["actions_until_shot"], errors="coerce")
        for threshold in (1, 2, 3, 4, 5):
            count = int(values.le(threshold).sum())
            rows.append(
                {
                    "window": f"actions<={threshold}",
                    "positives": count,
                    "positive_share": count / max(int(target.sum()), 1),
                }
            )
    if "downstream_action_count" in frame.columns:
        values = pd.to_numeric(positives["downstream_action_count"], errors="coerce")
        for threshold in (1, 2, 3, 4, 5):
            count = int(values.le(threshold).sum())
            rows.append(
                {
                    "window": f"downstream_actions<={threshold}",
                    "positives": count,
                    "positive_share": count / max(int(target.sum()), 1),
                }
            )
    time_col = _first_existing(frame, ("seconds_until_shot", "time_until_shot"))
    if time_col:
        values = pd.to_numeric(positives[time_col], errors="coerce")
        for threshold in (5, 10, 15):
            count = int(values.le(threshold).sum())
            rows.append(
                {
                    "window": f"seconds<={threshold}",
                    "positives": count,
                    "positive_share": count / max(int(target.sum()), 1),
                }
            )
    window_table = pd.DataFrame(rows, columns=["window", "positives", "positive_share"])
    window_table.to_csv(folder / "tables" / "window_coverage.csv", index=False)
    if not window_table.empty:
        fig, ax = plt.subplots(figsize=(11, max(6, len(window_table) * 0.4)))
        plot = window_table.sort_values("positive_share")
        ax.barh(plot["window"], plot["positive_share"], color="#0f766e")
        ax.set_xlabel("Share of positive actions")
        ax.set_ylabel("Window")
        _save_plot(
            fig,
            ax,
            folder / "plots" / "window_coverage.png",
            title="Is the downstream shot window defensible?",
            subtitle=f"{int(target.sum()):,} positive actions are evaluated against available window fields.",
            caption="Source: action_features. Caveat: absent window fields limit causal attribution checks.",
        )
    else:
        _write_skipped_plot(
            folder / "plots" / "window_coverage.png",
            "No downstream action/time window fields found.",
        )

    seq_col = _first_existing(frame, ("sequence_position", "action_position"))
    if seq_col:
        seq_table = _binned_target_rate(frame, target_column, seq_col)
    else:
        seq_table = pd.DataFrame(
            columns=[
                "bin",
                "rows",
                "positives",
                "target_rate",
                "global_target_rate",
                "lift",
                "sample_size_warning",
                "modelling_implication",
            ]
        )
    seq_table.to_csv(folder / "tables" / "sequence_position_positive_rate.csv", index=False)
    if not seq_table.empty:
        _plot_binned_rate(
            seq_table,
            folder / "plots" / "sequence_position_positive_rate.png",
            title="Does the target over-credit early buildup or final actions?",
            xlabel="Sequence position bin",
            caption=f"Source: action_features. Minimum slice size reference {min_slice_size}.",
        )
    else:
        _write_skipped_plot(
            folder / "plots" / "sequence_position_positive_rate.png",
            "No sequence position column found.",
        )
    return {
        "missing_fields": missing,
        "window_rows": len(window_table),
        "table": "05_sequence_window_stability/tables/window_coverage.csv",
        "visual": "05_sequence_window_stability/plots/window_coverage.png",
        "missing_table": "05_sequence_window_stability/tables/missing_window_fields.csv",
    }


def _slice_stability(
    frame: pd.DataFrame,
    target_column: str,
    folder: Path,
    min_slice_size: int,
) -> dict[str, object]:
    target = _target_series(frame, target_column)
    global_rate = float(target.mean()) if len(target) else 0.0
    frame_with_target = frame.assign(_target=target)
    slice_columns = [
        c
        for c in (
            "action_type",
            "event_type",
            "team_id",
            "competition_id",
            "competition_name",
            "start_zone",
            "end_zone",
            "under_pressure",
            "pressure",
            "sequence_position_bucket",
            "possession_phase",
            "play_pattern",
        )
        if c in frame.columns
    ]
    rows = []
    for column in slice_columns:
        grouped = (
            frame_with_target.assign(_value=frame_with_target[column].fillna("missing").astype(str))
            .groupby("_value", observed=True)["_target"]
            .agg(rows="count", positives="sum", target_rate="mean")
            .reset_index()
        )
        grouped = grouped[grouped["rows"] >= min_slice_size]
        for _, row in grouped.iterrows():
            delta = float(row["target_rate"] - global_rate)
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": row["_value"],
                    "rows": int(row["rows"]),
                    "positives": int(row["positives"]),
                    "target_rate": float(row["target_rate"]),
                    "global_target_rate": global_rate,
                    "target_rate_delta": delta,
                    "sample_size_warning": int(row["rows"]) < max(min_slice_size * 2, 100),
                    "modelling_implication": _slice_implication(delta),
                }
            )
    table = pd.DataFrame(
        rows,
        columns=[
            "slice_column",
            "slice_value",
            "rows",
            "positives",
            "target_rate",
            "global_target_rate",
            "target_rate_delta",
            "sample_size_warning",
            "modelling_implication",
        ],
    )
    table.to_csv(folder / "tables" / "slice_stability.csv", index=False)
    if not table.empty:
        plot = table.reindex(
            table["target_rate_delta"].abs().sort_values(ascending=False).index
        ).head(16)
        fig, ax = plt.subplots(figsize=(11, max(7, len(plot) * 0.42)))
        ax.barh(
            plot["slice_column"] + "=" + plot["slice_value"],
            plot["target_rate_delta"],
            color="#7c3aed",
        )
        ax.axvline(0, color="#111827", linewidth=1)
        ax.axvline(0.01, color="#b45309", linestyle="--")
        ax.axvline(-0.01, color="#b45309", linestyle="--", label="1pp reference")
        ax.set_xlabel("Target-rate delta from global rate")
        ax.set_ylabel("Slice")
        ax.legend(loc="lower right")
        _save_plot(
            fig,
            ax,
            folder / "plots" / "slice_stability.png",
            title="Does CxA target behaviour remain stable across slices?",
            subtitle=f"{int((table['target_rate_delta'].abs() >= 0.01).sum())} slices exceed a 1pp delta.",
            caption="Source: action_features. Caveat: CxA target sparsity makes small-slice deltas noisy.",
        )
    else:
        _write_skipped_plot(
            folder / "plots" / "slice_stability.png", "No slices met the minimum sample size."
        )
    return {
        "slice_columns": slice_columns,
        "unstable_slices": (
            int((table["target_rate_delta"].abs() >= 0.01).sum()) if not table.empty else 0
        ),
        "table": "06_slice_stability/tables/slice_stability.csv",
        "visual": "06_slice_stability/plots/slice_stability.png",
    }


def _data_quality(
    frame: pd.DataFrame,
    target_column: str,
    folder: Path,
    min_slice_size: int,
) -> dict[str, object]:
    features = _candidate_feature_columns(frame, target_column)
    quality = pd.DataFrame([_quality_row(frame[c], c) for c in features])
    quality.to_csv(folder / "tables" / "feature_quality.csv", index=False)
    value_checks = _football_value_checks(frame, target_column, min_slice_size)
    value_checks.to_csv(folder / "tables" / "football_value_checks.csv", index=False)
    recommendations = _cleaning_recommendations(quality, value_checks)
    recommendations.to_csv(folder / "tables" / "cleaning_recommendations.csv", index=False)
    return {
        "action_count": len(recommendations),
        "quality_table": "07_data_quality/tables/feature_quality.csv",
        "football_checks": "07_data_quality/tables/football_value_checks.csv",
        "recommendations": "07_data_quality/tables/cleaning_recommendations.csv",
    }


def _football_value_checks(
    frame: pd.DataFrame,
    target_column: str,
    min_slice_size: int,
) -> pd.DataFrame:
    rows = []
    range_checks = {
        "start_x": lambda s: (s < 0) | (s > 120),
        "end_x": lambda s: (s < 0) | (s > 120),
        "start_y": lambda s: (s < 0) | (s > 80),
        "end_y": lambda s: (s < 0) | (s > 80),
        "distance_progressed": lambda s: s < 0,
        "goal_distance_reduction": lambda s: s.abs() > 120,
        "action_duration": lambda s: s < 0,
        "seconds_until_shot": lambda s: s < 0,
        "time_until_shot": lambda s: s < 0,
        "sequence_position": lambda s: s < 0,
    }
    for column, predicate in range_checks.items():
        if column not in frame.columns:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        invalid = predicate(numeric)
        rows.append(_check_row(column, "football_value_range", int(invalid.sum()), len(numeric)))

    for column in _candidate_feature_columns(frame, target_column):
        series = frame[column]
        if series.isna().mean() >= 0.25:
            rows.append(
                _check_row(column, "high_missingness", int(series.isna().sum()), len(series))
            )
        unique = series.nunique(dropna=True)
        if unique <= 1:
            rows.append(_check_row(column, "constant_or_near_constant", len(series), len(series)))
        if pd.api.types.is_numeric_dtype(series) and unique > 2:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                q1, q3 = numeric.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outliers = (numeric < q1 - 3 * iqr) | (numeric > q3 + 3 * iqr)
                    if outliers.any():
                        rows.append(
                            _check_row(
                                column,
                                "extreme_numeric_outliers",
                                int(outliers.sum()),
                                len(numeric),
                            )
                        )
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            counts = series.fillna("missing").astype(str).value_counts()
            rare = int((counts < min_slice_size).sum())
            if rare:
                rows.append(_check_row(column, "rare_categorical_levels", rare, len(counts)))
    return pd.DataFrame(
        rows, columns=["column", "check", "issue_count", "issue_rate", "modelling_implication"]
    )


def _leakage_checks(
    frame: pd.DataFrame,
    target_column: str,
    value_column: str | None,
    folder: Path,
) -> dict[str, object]:
    rows = []
    for column in frame.columns:
        group = _eligibility_group(column, target_column, value_column)
        eligibility = (
            "eligible_candidate_feature"
            if group == "candidate_feature"
            else "exclude_or_reference_only"
        )
        rows.append(
            {"column": column, "classification": group, "training_eligibility": eligibility}
        )
    table = pd.DataFrame(rows)
    table.to_csv(folder / "tables" / "feature_training_eligibility.csv", index=False)
    table.to_csv(folder / "tables" / "leakage_checks.csv", index=False)
    risk_count = int((table["training_eligibility"] != "eligible_candidate_feature").sum())
    return {
        "risk_count": risk_count,
        "table": "08_leakage_checks/tables/feature_training_eligibility.csv",
        "leakage_table": "08_leakage_checks/tables/leakage_checks.csv",
        "excluded_columns": table.loc[
            table["training_eligibility"] != "eligible_candidate_feature", "column"
        ].tolist(),
    }


def _eligibility_group(column: str, target_column: str, value_column: str | None) -> str:
    lower = column.lower()
    if column == target_column or column in TARGET_CANDIDATES:
        return "target"
    if value_column is not None and column == value_column:
        return "target_reference"
    if column in VALUE_CANDIDATES:
        return "target_reference"
    if column in DOWNSTREAM_REFERENCE_COLUMNS:
        return "downstream_reference"
    if any(pattern in lower for pattern in POST_MODEL_PATTERNS):
        return "post_model_output"
    if column in ID_COLUMNS or lower.endswith("_id"):
        return "identifier"
    if "future" in lower or "after_shot" in lower:
        return "review_required"
    return "candidate_feature"


def _candidate_feature_columns(frame: pd.DataFrame, target_column: str) -> list[str]:
    value_column = _detect_value_column(frame)
    return [
        c
        for c in frame.columns
        if _eligibility_group(c, target_column, value_column) == "candidate_feature"
    ]


def _numeric_features(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [
        c
        for c in columns
        if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
    ]


def _categorical_features(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [
        c
        for c in columns
        if not pd.api.types.is_numeric_dtype(frame[c]) or pd.api.types.is_bool_dtype(frame[c])
    ]


def _numeric_profile_row(series: pd.Series, column: str) -> dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "feature": column,
        "rows": int(len(series)),
        "missing_rate": float(series.isna().mean()),
        "unique_values": int(series.nunique(dropna=True)),
        "min": float(numeric.min()) if not numeric.empty else np.nan,
        "max": float(numeric.max()) if not numeric.empty else np.nan,
        "mean": float(numeric.mean()) if not numeric.empty else np.nan,
        "median": float(numeric.median()) if not numeric.empty else np.nan,
        "p95": float(numeric.quantile(0.95)) if not numeric.empty else np.nan,
        "p99": float(numeric.quantile(0.99)) if not numeric.empty else np.nan,
        "skew": float(numeric.skew()) if len(numeric) > 2 else np.nan,
        "recommendation": _feature_recommendation(series),
    }


def _categorical_profile_row(series: pd.Series, column: str) -> dict[str, object]:
    counts = series.fillna("missing").astype(str).value_counts()
    return {
        "feature": column,
        "rows": int(len(series)),
        "missing_rate": float(series.isna().mean()),
        "unique_values": int(series.nunique(dropna=True)),
        "top_level": str(counts.index[0]) if not counts.empty else "",
        "top_level_share": (
            float(counts.iloc[0] / len(series)) if len(series) and not counts.empty else 0.0
        ),
        "recommendation": _feature_recommendation(series),
    }


def _quality_row(series: pd.Series, column: str) -> dict[str, object]:
    return {
        "feature": column,
        "rows": int(len(series)),
        "missing_rate": float(series.isna().mean()),
        "unique_values": int(series.nunique(dropna=True)),
        "recommendation": _feature_recommendation(series),
    }


def _feature_recommendation(series: pd.Series) -> str:
    missing = float(series.isna().mean())
    unique = int(series.nunique(dropna=True))
    if missing >= 0.5 or unique <= 1:
        return "drop"
    if missing >= 0.2:
        return "review"
    if pd.api.types.is_bool_dtype(series):
        return "keep"
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) > 2 and abs(float(numeric.skew())) >= 1.5:
            return "transform"
        q1, q3 = numeric.quantile([0.25, 0.75]) if len(numeric) else (0, 0)
        if len(numeric) and q3 > q1:
            outliers = ((numeric < q1 - 3 * (q3 - q1)) | (numeric > q3 + 3 * (q3 - q1))).any()
            if outliers:
                return "cap"
        return "keep"
    if unique > 20:
        return "encode"
    return "keep"


def _plot_numeric_distribution(frame: pd.DataFrame, column: str, path: Path) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if values.empty:
        ax.text(0.5, 0.5, "No non-missing numeric values", ha="center", va="center")
    else:
        ax.hist(
            values, bins=min(30, max(8, int(np.sqrt(len(values))))), color="#2563eb", alpha=0.82
        )
        ax.axvline(values.median(), color="#b45309", linestyle="--", label="Median")
        ax.set_xlabel(column)
        ax.set_ylabel("Action count")
        ax.legend(loc="upper right")
    _save_plot(
        fig,
        ax,
        path,
        title=f"Does {column} need transformation, binning, or capping?",
        subtitle=_distribution_takeaway(values),
        caption=f"Source: action_features. n={len(values):,}. Caveat: distribution shape is not target signal.{_outlier_note(values)}",
    )


def _plot_categorical_levels(frame: pd.DataFrame, column: str, path: Path) -> None:
    counts = frame[column].fillna("missing").astype(str).value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.42)))
    if counts.empty:
        ax.text(0.5, 0.5, "No levels", ha="center", va="center")
    else:
        ax.barh(counts.index, counts.values, color="#0f766e")
        ax.set_xlabel("Rows")
        ax.set_ylabel(column)
    _save_plot(
        fig,
        ax,
        path,
        title=f"Does {column} need encoding or rare-level pooling?",
        subtitle=f"{frame[column].nunique(dropna=True):,} observed levels.",
        caption="Source: action_features. Caveat: rare levels may be hidden outside top 20.",
    )


def _categorical_target_output(
    frame: pd.DataFrame,
    target_column: str,
    column: str,
    folder: Path,
    stem: str,
    question: str,
    min_slice_size: int,
) -> dict[str, object]:
    table = _categorical_target_rate(frame, target_column, column, min_slice_size=min_slice_size)
    table.to_csv(folder / "tables" / f"{stem}.csv", index=False)
    if table.empty:
        _write_skipped_plot(
            folder / "plots" / f"{stem}.png", f"No {column} levels met sample size."
        )
    else:
        plot = table.sort_values(["target_rate", "rows"], ascending=[True, False]).tail(15)
        fig, ax = plt.subplots(figsize=(11, max(6, len(plot) * 0.45)))
        ax.barh(plot["value"].astype(str), plot["target_rate"], color="#2563eb")
        ax.axvline(
            table["global_target_rate"].iloc[0], color="#b45309", linestyle="--", label="Global"
        )
        ax.set_xlabel("Target rate")
        ax.set_ylabel(column)
        ax.legend(loc="lower right")
        _save_plot(
            fig,
            ax,
            folder / "plots" / f"{stem}.png",
            title=question,
            subtitle=f"{len(table):,} levels met the sample-size threshold.",
            caption="Source: action_features. Caveat: categorical effects are descriptive and unadjusted.",
        )
    return {
        "table": f"03_feature_target_relationships/tables/{stem}.csv",
        "visual": f"03_feature_target_relationships/plots/{stem}.png",
        "interpretation": _target_table_takeaway(table, column),
    }


def _binned_target_output(
    frame: pd.DataFrame,
    target_column: str,
    column: str,
    folder: Path,
    stem: str,
    question: str,
) -> dict[str, object]:
    table = _binned_target_rate(frame, target_column, column)
    table.to_csv(folder / "tables" / f"{stem}.csv", index=False)
    if table.empty:
        _write_skipped_plot(folder / "plots" / f"{stem}.png", f"No usable {column} values.")
    else:
        _plot_binned_rate(
            table,
            folder / "plots" / f"{stem}.png",
            title=question,
            xlabel=f"{column} bin",
            caption="Source: action_features. Caveat: binned target rates are unadjusted.",
        )
    return {
        "table": f"03_feature_target_relationships/tables/{stem}.csv",
        "visual": f"03_feature_target_relationships/plots/{stem}.png",
        "interpretation": _binned_takeaway(table, column),
    }


def _categorical_target_rate(
    frame: pd.DataFrame,
    target_column: str,
    column: str,
    *,
    min_slice_size: int,
) -> pd.DataFrame:
    target = _target_series(frame, target_column)
    global_rate = float(target.mean()) if len(target) else 0.0
    table = (
        frame.assign(_target=target, _value=frame[column].fillna("missing").astype(str))
        .groupby("_value", observed=True)["_target"]
        .agg(rows="count", positives="sum", target_rate="mean")
        .reset_index()
        .rename(columns={"_value": "value"})
    )
    table = table[table["rows"] >= min_slice_size].copy()
    table["global_target_rate"] = global_rate
    table["lift"] = table["target_rate"] / global_rate if global_rate else np.nan
    table["sample_size_warning"] = table["rows"] < max(min_slice_size * 2, 100)
    table["modelling_implication"] = table["target_rate"].apply(
        lambda rate: "candidate_signal" if abs(rate - global_rate) >= 0.01 else "monitor"
    )
    return table.sort_values(["target_rate", "rows"], ascending=[False, False])


def _binned_target_rate(
    frame: pd.DataFrame, target_column: str, column: str, bins: int = 5
) -> pd.DataFrame:
    target = _target_series(frame, target_column)
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = pd.DataFrame({column: values, "_target": target}).dropna()
    global_rate = float(target.mean()) if len(target) else 0.0
    if valid.empty or valid[column].nunique() < 2:
        return pd.DataFrame(
            columns=[
                "bin",
                "rows",
                "positives",
                "target_rate",
                "global_target_rate",
                "lift",
                "sample_size_warning",
                "modelling_implication",
            ]
        )
    valid["_bin"] = pd.qcut(valid[column], q=min(bins, valid[column].nunique()), duplicates="drop")
    table = (
        valid.groupby("_bin", observed=True)["_target"]
        .agg(rows="count", positives="sum", target_rate="mean")
        .reset_index()
    )
    table["bin"] = table["_bin"].astype(str)
    table["global_target_rate"] = global_rate
    table["lift"] = table["target_rate"] / global_rate if global_rate else np.nan
    table["sample_size_warning"] = table["rows"] < 100
    table["modelling_implication"] = table["target_rate"].apply(
        lambda rate: "candidate_signal" if abs(rate - global_rate) >= 0.01 else "monitor"
    )
    return table[
        [
            "bin",
            "rows",
            "positives",
            "target_rate",
            "global_target_rate",
            "lift",
            "sample_size_warning",
            "modelling_implication",
        ]
    ]


def _plot_binned_rate(
    table: pd.DataFrame, path: Path, *, title: str, xlabel: str, caption: str
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.plot(table["bin"], table["target_rate"], marker="o", color="#0f766e")
    ax.axhline(table["global_target_rate"].iloc[0], color="#b45309", linestyle="--", label="Global")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Target rate")
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(loc="upper right")
    _save_plot(
        fig, ax, path, title=title, subtitle=_binned_takeaway(table, "feature"), caption=caption
    )


def _write_empty_table(path: Path) -> None:
    pd.DataFrame().to_csv(path, index=False)


def _write_skipped_plot(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save_plot(
        fig,
        ax,
        path,
        title="Analysis skipped",
        subtitle=message,
        caption="Source: action_features.",
    )


def _check_row(column: str, check: str, issue_count: int, denominator: int) -> dict[str, object]:
    return {
        "column": column,
        "check": check,
        "issue_count": issue_count,
        "issue_rate": issue_count / denominator if denominator else 0.0,
        "modelling_implication": (
            "Clean, transform, or review before modelling." if issue_count else "No issue found."
        ),
    }


def _cleaning_recommendations(quality: pd.DataFrame, value_checks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in quality.iterrows():
        if row["recommendation"] not in {"keep"}:
            rows.append(
                {
                    "feature": row["feature"],
                    "reason": row["recommendation"],
                    "recommendation": row["recommendation"],
                }
            )
    for _, row in value_checks.iterrows():
        if int(row["issue_count"]) > 0:
            rows.append(
                {
                    "feature": row["column"],
                    "reason": row["check"],
                    "recommendation": row["modelling_implication"],
                }
            )
    return pd.DataFrame(rows, columns=["feature", "reason", "recommendation"]).drop_duplicates()


def _target_status(rows: int, target_rate: float) -> str:
    if rows < 100:
        return "sample_size_warning"
    if target_rate < 0.005 or target_rate > 0.4:
        return "imbalance_warning"
    return "usable"


def _distribution_takeaway(values: pd.Series) -> str:
    if values.empty:
        return "No non-missing numeric values."
    return f"Median {values.median():.2f}; p95 {values.quantile(0.95):.2f}; p99 {values.quantile(0.99):.2f}."


def _outlier_note(values: pd.Series) -> str:
    if values.empty:
        return ""
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return ""
    outliers = (values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)
    return f" Outlier note: {int(outliers.sum()):,} 3*IQR values flagged."


def _target_table_takeaway(table: pd.DataFrame, column: str) -> str:
    if table.empty:
        return f"No {column} levels met the sample-size threshold."
    return f"{column} target rates range from {table['target_rate'].min():.2%} to {table['target_rate'].max():.2%}."


def _binned_takeaway(table: pd.DataFrame, column: str) -> str:
    if table.empty:
        return f"{column} could not be binned."
    return f"{column} binned target rates span {table['target_rate'].min():.2%} to {table['target_rate'].max():.2%}."


def _redundancy_implication(corr_value: float | np.floating | None, left: str, right: str) -> str:
    if pd.notna(corr_value) and abs(float(corr_value)) >= 0.8:
        return f"Review whether both {left} and {right} are needed or regularise strongly."
    return f"Keep both {left} and {right} available unless diagnostics say otherwise."


def _slice_implication(delta: float) -> str:
    if abs(delta) >= 0.02:
        return "Use this slice for validation, monitoring, and possible interaction checks."
    if abs(delta) >= 0.01:
        return "Monitor this slice during validation."
    return "No immediate slice-specific modelling action."


def _first_existing(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def _render_report(
    *,
    target: dict[str, object],
    coverage: dict[str, object],
    distributions: dict[str, object],
    relationships: dict[str, object],
    correlations: dict[str, object],
    windows: dict[str, object],
    stability: dict[str, object],
    quality: dict[str, object],
    leakage: dict[str, object],
    min_slice_size: int,
) -> str:
    lines = [
        "# CxA Pre-Model Target and Action-Feature Analysis",
        "",
        "This diagnostic layer sits between CxA action feature engineering and model training. "
        "It reads `action_features`; downstream-shot fields are treated as target/reference fields.",
        "",
        _section(
            "1. Target usability",
            {
                "Question": "Is the CxA shot-created target usable?",
                "Calculation": f"{target['positives']} positives from {target['row_count']} actions using `{target['target_column']}`.",
                "Visual/Table": f"{target['table']} and {target['visual']}.",
                "Interpretation": f"Target status is {target['status']} with rate {float(target['target_rate']):.2%}.",
                "Modelling implication": "Use stratified validation and sparse-target metrics before training.",
                "Limitation": "This does not evaluate model calibration or attribution quality.",
            },
        ),
        _section(
            "2. Target sparsity and imbalance",
            {
                "Question": "Is the shot-created target too sparse for naive modelling?",
                "Calculation": "Positive/negative split and created-shot value distribution are profiled.",
                "Visual/Table": f"{target['value_table']} and {target['value_visual']}.",
                "Interpretation": f"The positive target share is {float(target['target_rate']):.2%}.",
                "Modelling implication": "Avoid accuracy-only evaluation; monitor precision-recall and calibration.",
                "Limitation": "Created-shot value may be absent or reference-only.",
            },
        ),
        _section(
            "3. Action coverage",
            {
                "Question": "Are eligible actions, identifiers, and locations reliably represented?",
                "Calculation": f"{coverage['rows']} rows across {coverage['action_type_count']} action types.",
                "Visual/Table": f"{coverage['table']}, {coverage['id_table']}, and {coverage['location_table']}.",
                "Interpretation": f"Core represented types: {', '.join(coverage['represented_core_types']) or 'none'}; rare types: {coverage['rare_count']}.",
                "Modelling implication": "Pool or exclude sparse action types and validate ID/location coverage.",
                "Limitation": "Coverage does not prove target construction quality.",
            },
        ),
        _section(
            "4. Action-type signal",
            {
                "Question": "Which action types most often create downstream shots?",
                "Calculation": "Action-type target rates are compared with the global target rate.",
                "Visual/Table": relationships.get("action_type", {}).get(
                    "table", "03_feature_target_relationships/tables/action_type_target_rate.csv"
                ),
                "Interpretation": relationships.get("action_type", {}).get(
                    "interpretation", "Action-type signal was skipped."
                ),
                "Modelling implication": "Encode action type and pool rare levels before training.",
                "Limitation": f"Levels below {min_slice_size} rows are excluded.",
            },
        ),
        _section(
            "5. Movement and spatial feature signal",
            {
                "Question": "Do end zones, progression, sequence position, and spatial fields show target signal?",
                "Calculation": "Categorical and binned numeric target-rate tables are written for available fields.",
                "Visual/Table": f"{relationships.get('end_zone', {}).get('table', '')}; {relationships['numeric_table']}.",
                "Interpretation": relationships.get("end_zone", {}).get(
                    "interpretation", "End-zone signal unavailable."
                ),
                "Modelling implication": "Keep spatial/progression candidates that remain stable under validation.",
                "Limitation": "Relationships are descriptive and unadjusted.",
            },
        ),
        _section(
            "6. Sequence/window stability",
            {
                "Question": "Is the downstream shot window defensible?",
                "Calculation": "Available action-count/time windows and missing window fields are reported.",
                "Visual/Table": f"{windows['table']}, {windows['visual']}, and {windows['missing_table']}.",
                "Interpretation": f"{len(windows['missing_fields'])} recommended window fields are missing.",
                "Modelling implication": "Add missing window fields before making strong attribution claims.",
                "Limitation": "Window diagnostics are incomplete when fields are absent.",
            },
        ),
        _section(
            "7. Feature redundancy",
            {
                "Question": "Which movement/location features are redundant?",
                "Calculation": "Numeric correlations and targeted redundancy checks are computed.",
                "Visual/Table": f"{correlations['table']}, {correlations['targeted_table']}, and {correlations['visual']}.",
                "Interpretation": f"{correlations['high_pairs']} high-correlation pairs were flagged.",
                "Modelling implication": "Drop, combine, or regularise redundant movement features.",
                "Limitation": "Correlation is linear and cannot detect all redundancy.",
            },
        ),
        _section(
            "8. Slice stability",
            {
                "Question": "Does target behaviour remain stable across CxA slices?",
                "Calculation": f"Available slices with at least {min_slice_size} rows are compared with the global target rate.",
                "Visual/Table": f"{stability['table']} and {stability['visual']}.",
                "Interpretation": f"{stability['unstable_slices']} slices exceed the reference delta.",
                "Modelling implication": "Use unstable slices in validation and monitoring.",
                "Limitation": "Sparse positives make small-slice rates noisy.",
            },
        ),
        _section(
            "9. Data quality and cleaning recommendations",
            {
                "Question": "Which action features need cleaning before modelling?",
                "Calculation": "Location ranges, progression values, timing fields, rare levels, missingness, constants, and outliers are checked.",
                "Visual/Table": f"{quality['quality_table']}, {quality['football_checks']}, and {quality['recommendations']}.",
                "Interpretation": f"{quality['action_count']} cleaning recommendations were produced.",
                "Modelling implication": "Encode cleaning rules into the modelling feature contract.",
                "Limitation": "Automated checks require domain review.",
            },
        ),
        _section(
            "10. Leakage risks and training eligibility",
            {
                "Question": "Which columns must be excluded before training?",
                "Calculation": "Columns are classified as target, reference, downstream reference, post-model output, identifier, candidate, or review.",
                "Visual/Table": f"{leakage['table']} and {leakage['leakage_table']}.",
                "Interpretation": f"{leakage['risk_count']} columns are excluded or reference-only.",
                "Modelling implication": "Train only on candidate features; keep downstream-shot IDs/values out of inputs.",
                "Limitation": "Name-based classification cannot prove semantic safety for future columns.",
            },
        ),
        _section(
            "11. Modelling recommendations",
            {
                "Question": "What should CxA modelling do next?",
                "Calculation": "Combine target sparsity, coverage, feature signal, window stability, redundancy, quality, and leakage results.",
                "Visual/Table": "Use all numbered artifact folders under `outputs/analysis/cxa/`.",
                "Interpretation": "The action-feature table is ready for modelling only after cleaning and leakage exclusions are encoded.",
                "Modelling implication": "Build a CxA feature contract, sparse-target validation plan, window diagnostics, and slice monitoring before training.",
                "Limitation": "This report does not train, score, attribute, aggregate, or publish CxA predictions.",
            },
        ),
    ]
    return "\n".join(lines) + "\n"


def _section(title: str, values: dict[str, str]) -> str:
    lines = [f"## {title}", ""]
    for key in REQUIRED_SECTIONS:
        lines.append(f"**{key}:** {values[key]}")
        lines.append("")
    return "\n".join(lines)
