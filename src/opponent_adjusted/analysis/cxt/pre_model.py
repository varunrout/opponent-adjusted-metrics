"""Pre-model CxT-style ball progression diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opponent_adjusted.config import settings
from opponent_adjusted.db.models import ActionFeature

DEFAULT_OUTPUT_DIR = Path("outputs/analysis/cxt")
TARGET_PROXY_CANDIDATES = (
    "threat_delta",
    "xt_delta",
    "xT_delta",
    "cxt_delta",
    "future_shot_created",
    "future_goal",
    "future_shot_value",
    "possession_value_change",
    "zone_value_delta",
    "progressive_value",
    "target",
    "label",
)
EXPECTED_TARGET_PROXIES = {
    "threat_delta": "Continuous threat-value change target for supervised CxT modelling.",
    "xt_delta": "Classic expected-threat grid delta between start and end zones.",
    "future_shot_created": "Binary proxy for whether progression leads to a future shot.",
    "future_goal": "Binary proxy for whether progression leads to a future goal.",
    "future_shot_value": "Continuous proxy using downstream shot quality.",
    "possession_value_change": "Possession-level value change after the action.",
    "zone_value_delta": "Zone transition value delta reference.",
}
ID_COLUMNS = {
    "action_id",
    "event_id",
    "match_id",
    "team_id",
    "player_id",
    "possession_id",
    "possession_number",
    "sequence_id",
    "competition_id",
    "created_shot_id",
    "created_shot_event_id",
}
DOWNSTREAM_REFERENCE_PATTERNS = (
    "future_shot",
    "future_goal",
    "shot_created",
    "created_shot",
    "downstream",
)
POST_MODEL_PATTERNS = ("prediction", "model", "registry", "aggregate", "leaderboard")
THREAT_PATTERNS = ("threat_delta", "xt_delta", "xT_delta", "cxt_delta")
REQUIRED_SECTIONS = (
    "Question",
    "Calculation",
    "Visual/Table",
    "Interpretation",
    "Modelling implication",
    "Limitation",
)
FOCUS_FEATURES = (
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "dx",
    "dy",
    "distance_moved",
    "distance_progressed",
    "progressive_distance",
    "goal_distance_start",
    "goal_distance_end",
    "goal_distance_reduction",
    "angle_to_goal_start",
    "angle_to_goal_end",
    "final_third_entry",
    "box_entry",
    "zone14_entry",
    "carry",
    "pass",
    "dribble",
    "cross",
    "switch",
    "through_ball",
    "under_pressure",
    "pressure_flag",
    "sequence_position",
    "time_since_possession_start",
)
REDUNDANCY_PAIRS = (
    ("start_x", "end_x"),
    ("start_y", "end_y"),
    ("start_x", "goal_distance_start"),
    ("end_x", "goal_distance_end"),
    ("distance_moved", "distance_progressed"),
    ("distance_progressed", "goal_distance_reduction"),
    ("progressive_distance", "distance_progressed"),
    ("final_third_entry", "box_entry"),
    ("box_entry", "zone14_entry"),
)


@dataclass(frozen=True)
class CxTAnalysisResult:
    """Summary of generated pre-model CxT diagnostics."""

    output_dir: Path
    report_path: Path
    row_count: int
    data_source: str
    target_proxy_column: str | None
    candidate_feature_count: int
    leakage_risk_count: int


def load_progression_feature_dataset(
    session: Session | None = None,
    *,
    parquet_path: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Load the richest available pre-model progression/action feature dataset."""

    if session is not None:
        rows = session.execute(select(ActionFeature)).scalars().all()
        if rows:
            records = []
            for row in rows:
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
            return pd.DataFrame.from_records(records), "database:action_features"

    candidate_paths = [
        parquet_path,
        settings.feature_store_path / "cxt" / "progressions_featured.parquet",
        settings.feature_store_path / "cxt" / "progressions.parquet",
        settings.feature_store_path / "cxa" / "action_features.parquet",
    ]
    for path in candidate_paths:
        if path is not None and Path(path).exists():
            return pd.read_parquet(path), f"parquet:{path}"
    raise ValueError(
        "No usable pre-model progression/action feature data found. Expected DB "
        "`action_features` or a feature-store progression/action parquet file."
    )


def detect_target_proxy_column(frame: pd.DataFrame) -> str | None:
    """Return the first available constructed CxT target/proxy column, if present."""

    lower_to_original = {column.lower(): column for column in frame.columns}
    for candidate in TARGET_PROXY_CANDIDATES:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def build_pre_model_cxt_analysis(
    features: pd.DataFrame,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    data_source: str = "dataframe",
    min_sample_size: int = 30,
) -> CxTAnalysisResult:
    """Generate the full pre-model CxT-style ball progression feature study."""

    output_path = Path(output_dir)
    folders = _create_output_folders(output_path)
    frame = _prepare_dataset(features)
    _apply_matplotlib_style()
    target_proxy = detect_target_proxy_column(frame)

    coverage = _action_coverage(frame, folders["00_action_coverage"], min_sample_size)
    spatial = _spatial_coverage(frame, folders["01_spatial_coverage"], min_sample_size)
    distributions = _feature_distributions(frame, target_proxy, folders["02_feature_distributions"])
    relationships = _relationships(
        frame,
        target_proxy,
        folders["03_feature_target_relationships"],
        min_sample_size,
    )
    correlations = _feature_correlations(frame, target_proxy, folders["04_feature_correlations"])
    transitions = _transition_stability(frame, folders["05_transition_stability"], min_sample_size)
    slices = _slice_stability(frame, folders["06_slice_stability"], min_sample_size)
    quality = _data_quality(frame, target_proxy, folders["07_data_quality"], min_sample_size)
    leakage = _leakage_checks(frame, target_proxy, folders["08_leakage_checks"])

    report_path = output_path / "report.md"
    report_path.write_text(
        _render_report(
            data_source=data_source,
            row_count=len(frame),
            coverage=coverage,
            spatial=spatial,
            distributions=distributions,
            relationships=relationships,
            correlations=correlations,
            transitions=transitions,
            slices=slices,
            quality=quality,
            leakage=leakage,
            target_proxy=target_proxy,
            min_sample_size=min_sample_size,
        ),
        encoding="utf-8",
    )
    return CxTAnalysisResult(
        output_dir=output_path,
        report_path=report_path,
        row_count=len(frame),
        data_source=data_source,
        target_proxy_column=target_proxy,
        candidate_feature_count=int(distributions["feature_count"]),
        leakage_risk_count=int(leakage["risk_count"]),
    )


def run_pre_model_cxt_analysis(
    session: Session | None = None,
    *,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    parquet_path: Path | None = None,
    min_sample_size: int = 30,
) -> CxTAnalysisResult:
    """Load progression features and write the pre-model CxT diagnostics."""

    frame, source = load_progression_feature_dataset(session, parquet_path=parquet_path)
    return build_pre_model_cxt_analysis(
        frame,
        output_dir=output_dir,
        data_source=source,
        min_sample_size=min_sample_size,
    )


def _create_output_folders(output_dir: Path) -> dict[str, Path]:
    names = (
        "00_action_coverage",
        "01_spatial_coverage",
        "02_feature_distributions",
        "03_feature_target_relationships",
        "04_feature_correlations",
        "05_transition_stability",
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
    aliases = {
        "x_progression": "dx",
        "y_progression": "dy",
        "length": "distance_moved",
        "distance_to_goal_before": "goal_distance_start",
        "distance_to_goal_after": "goal_distance_end",
        "angle_to_goal_before": "angle_to_goal_start",
        "angle_to_goal_after": "angle_to_goal_end",
        "is_pass": "pass",
        "is_carry": "carry",
        "is_dribble": "dribble",
        "is_cross": "cross",
        "is_through_ball": "through_ball",
        "is_cutback": "cutback",
        "switches_play": "switch",
        "enters_final_third": "final_third_entry",
        "enters_penalty_area": "box_entry",
        "enters_zone14": "zone14_entry",
        "action_position": "sequence_position",
        "seconds_since_possession_start": "time_since_possession_start",
    }
    for source, alias in aliases.items():
        if source in prepared.columns and alias not in prepared.columns:
            prepared[alias] = prepared[source]

    if "event_type" not in prepared.columns and "action_type" in prepared.columns:
        prepared["event_type"] = prepared["action_type"]
    if {"start_x", "start_y"}.issubset(prepared.columns) and "start_zone" not in prepared.columns:
        prepared["start_zone"] = _zone_series(prepared["start_x"], prepared["start_y"])
    if {"end_x", "end_y"}.issubset(prepared.columns) and "end_zone" not in prepared.columns:
        prepared["end_zone"] = _zone_series(prepared["end_x"], prepared["end_y"])
    if {"start_x", "end_x"}.issubset(prepared.columns) and "dx" not in prepared.columns:
        prepared["dx"] = pd.to_numeric(prepared["end_x"], errors="coerce") - pd.to_numeric(
            prepared["start_x"], errors="coerce"
        )
    if {"start_y", "end_y"}.issubset(prepared.columns) and "dy" not in prepared.columns:
        prepared["dy"] = pd.to_numeric(prepared["end_y"], errors="coerce") - pd.to_numeric(
            prepared["start_y"], errors="coerce"
        )
    if {"dx", "dy"}.issubset(prepared.columns) and "distance_moved" not in prepared.columns:
        prepared["distance_moved"] = np.hypot(prepared["dx"], prepared["dy"])
    if "dx" in prepared.columns and "distance_progressed" not in prepared.columns:
        prepared["distance_progressed"] = prepared["dx"]
    if "distance_progressed" in prepared.columns and "progressive_distance" not in prepared.columns:
        prepared["progressive_distance"] = prepared["distance_progressed"].clip(lower=0)
    if {"goal_distance_start", "goal_distance_end"}.issubset(prepared.columns):
        prepared["goal_distance_reduction"] = pd.to_numeric(
            prepared["goal_distance_start"], errors="coerce"
        ) - pd.to_numeric(prepared["goal_distance_end"], errors="coerce")
    if {"end_x", "end_y"}.issubset(prepared.columns):
        prepared["final_third_entry"] = prepared.get(
            "final_third_entry", pd.to_numeric(prepared["end_x"], errors="coerce") >= 80
        )
        prepared["box_entry"] = prepared.get(
            "box_entry",
            (pd.to_numeric(prepared["end_x"], errors="coerce") >= 102)
            & pd.to_numeric(prepared["end_y"], errors="coerce").between(18, 62),
        )
        prepared["zone14_entry"] = prepared.get(
            "zone14_entry",
            pd.to_numeric(prepared["end_x"], errors="coerce").between(80, 102)
            & pd.to_numeric(prepared["end_y"], errors="coerce").between(26.67, 53.33),
        )
    for action in ("pass", "carry", "dribble"):
        if action not in prepared.columns and "action_type" in prepared.columns:
            prepared[action] = prepared["action_type"].astype(str).str.lower().eq(action)
    return prepared


def _zone_series(x: pd.Series, y: pd.Series) -> pd.Series:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    third = pd.Series(
        np.select(
            [x_num.isna(), x_num < 40, x_num < 80],
            ["unknown", "defensive", "middle"],
            default="final",
        ),
        index=x.index,
    )
    lane = pd.Series(
        np.select(
            [y_num.isna(), y_num < 26.67, y_num < 53.33],
            ["unknown", "wide_left", "central"],
            default="wide_right",
        ),
        index=x.index,
    )
    box = (x_num >= 102) & y_num.between(18, 62)
    return pd.Series(
        np.where(
            box,
            "box",
            np.where((third == "unknown") | (lane == "unknown"), "unknown", third + "_" + lane),
        ),
        index=x.index,
    )


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
    fig: plt.Figure, ax: plt.Axes, path: Path, *, title: str, subtitle: str, caption: str
) -> None:
    ax.set_title(title, loc="left", pad=26, fontsize=13, fontweight="bold")
    ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, fontsize=10, color="#374151")
    fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=8.5, color="#4b5563")
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.12, right=0.96)
    fig.savefig(path)
    plt.close(fig)


def _action_coverage(frame: pd.DataFrame, folder: Path, min_sample_size: int) -> dict[str, object]:
    action_col = "action_type" if "action_type" in frame.columns else "event_type"
    if action_col in frame.columns:
        table = (
            frame.assign(action_type_value=frame[action_col].fillna("missing").astype(str))
            .groupby("action_type_value", observed=True)
            .size()
            .reset_index(name="actions")
            .sort_values("actions", ascending=False)
        )
        table["share"] = table["actions"] / max(len(frame), 1)
        table["sample_size_warning"] = table["actions"] < min_sample_size
        table["modelling_implication"] = np.where(
            table["sample_size_warning"],
            "Pool or exclude sparse action type.",
            "Usable action type slice.",
        )
    else:
        table = pd.DataFrame(
            columns=[
                "action_type_value",
                "actions",
                "share",
                "sample_size_warning",
                "modelling_implication",
            ]
        )
    table.to_csv(folder / "tables" / "action_type_coverage.csv", index=False)
    _bar_plot(
        table,
        "action_type_value",
        "actions",
        folder / "plots" / "action_type_coverage.png",
        title="Which movement action types are represented?",
        xlabel="Actions",
        ylabel="Action type",
        reference=min_sample_size,
        caption="Source: pre-model progression/action features. Caveat: sparse types need pooling.",
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
    pd.DataFrame(
        [
            {
                "column": column,
                "populated_rate": float(frame[column].notna().mean()),
                "unique_values": int(frame[column].nunique(dropna=True)),
            }
            for column in id_columns
        ]
    ).to_csv(folder / "tables" / "id_coverage.csv", index=False)
    loc_columns = [c for c in ("start_x", "start_y", "end_x", "end_y") if c in frame.columns]
    pd.DataFrame(
        [
            {
                "column": column,
                "populated_rate": float(frame[column].notna().mean()),
                "min": float(pd.to_numeric(frame[column], errors="coerce").min()),
                "max": float(pd.to_numeric(frame[column], errors="coerce").max()),
            }
            for column in loc_columns
        ]
    ).to_csv(folder / "tables" / "location_coverage.csv", index=False)
    represented = set(table["action_type_value"]) if not table.empty else set()
    return {
        "rows": len(frame),
        "action_type_count": len(table),
        "core_types": sorted(
            {v.lower() for v in represented}.intersection({"pass", "carry", "dribble"})
        ),
        "rare_count": int(table["sample_size_warning"].sum()) if not table.empty else 0,
        "table": "00_action_coverage/tables/action_type_coverage.csv",
        "visual": "00_action_coverage/plots/action_type_coverage.png",
    }


def _spatial_coverage(frame: pd.DataFrame, folder: Path, min_sample_size: int) -> dict[str, object]:
    start = _zone_coverage(frame, "start_zone", min_sample_size)
    end = _zone_coverage(frame, "end_zone", min_sample_size)
    transitions = (
        frame.groupby(["start_zone", "end_zone"], dropna=False, observed=True)
        .size()
        .reset_index(name="actions")
        if {"start_zone", "end_zone"}.issubset(frame.columns)
        else pd.DataFrame(columns=["start_zone", "end_zone", "actions"])
    )
    if not transitions.empty:
        transitions["share"] = transitions["actions"] / max(len(frame), 1)
        transitions["sample_size_warning"] = transitions["actions"] < min_sample_size
        transitions["modelling_implication"] = np.where(
            transitions["sample_size_warning"],
            "Sparse transition; coarsen zones or smooth.",
            "Transition has usable support.",
        )
        transitions = transitions.sort_values("actions", ascending=False)
    start.to_csv(folder / "tables" / "start_zone_coverage.csv", index=False)
    end.to_csv(folder / "tables" / "end_zone_coverage.csv", index=False)
    transitions.to_csv(folder / "tables" / "transition_coverage.csv", index=False)
    _bar_plot(
        start,
        "zone",
        "actions",
        folder / "plots" / "start_zone_coverage.png",
        title="Which start zones are represented?",
        xlabel="Actions",
        ylabel="Start zone",
        reference=min_sample_size,
        caption="Source: pre-model progression/action features.",
    )
    _bar_plot(
        end,
        "zone",
        "actions",
        folder / "plots" / "end_zone_coverage.png",
        title="Which end zones are represented?",
        xlabel="Actions",
        ylabel="End zone",
        reference=min_sample_size,
        caption="Source: pre-model progression/action features.",
    )
    top_transitions = transitions.head(20).copy()
    if not top_transitions.empty:
        top_transitions["transition"] = (
            top_transitions["start_zone"].astype(str)
            + " -> "
            + top_transitions["end_zone"].astype(str)
        )
    _bar_plot(
        top_transitions,
        "transition",
        "actions",
        folder / "plots" / "transition_coverage.png",
        title="Which zone transitions have enough samples?",
        xlabel="Actions",
        ylabel="Transition",
        reference=min_sample_size,
        caption="Source: pre-model progression/action features. Caveat: low-count transitions may need coarser zones.",
    )
    return {
        "start_zones": len(start),
        "end_zones": len(end),
        "transitions": len(transitions),
        "sparse_transitions": (
            int(transitions["sample_size_warning"].sum()) if not transitions.empty else 0
        ),
        "transition_table": "01_spatial_coverage/tables/transition_coverage.csv",
    }


def _zone_coverage(frame: pd.DataFrame, column: str, min_sample_size: int) -> pd.DataFrame:
    if column not in frame.columns:
        return pd.DataFrame(
            columns=["zone", "actions", "share", "sample_size_warning", "modelling_implication"]
        )
    table = (
        frame.assign(_zone=frame[column].fillna("missing").astype(str))
        .groupby("_zone", observed=True)
        .size()
        .reset_index(name="actions")
        .rename(columns={"_zone": "zone"})
        .sort_values("actions", ascending=False)
    )
    table["share"] = table["actions"] / max(len(frame), 1)
    table["sample_size_warning"] = table["actions"] < min_sample_size
    table["modelling_implication"] = np.where(
        table["sample_size_warning"], "Sparse zone; coarsen or smooth.", "Zone has usable support."
    )
    return table


def _feature_distributions(
    frame: pd.DataFrame, target_proxy: str | None, folder: Path
) -> dict[str, object]:
    features = _candidate_features(frame, target_proxy)
    focus = [c for c in FOCUS_FEATURES if c in frame.columns and c in features]
    features = focus + [c for c in features if c not in focus]
    numeric = _numeric_features(frame, features)
    categorical = _categorical_features(frame, features)
    pd.DataFrame([_numeric_profile(frame[c], c) for c in numeric]).to_csv(
        folder / "tables" / "numeric_feature_profiles.csv", index=False
    )
    pd.DataFrame([_categorical_profile(frame[c], c) for c in categorical]).to_csv(
        folder / "tables" / "categorical_feature_profiles.csv", index=False
    )
    for column in numeric:
        _numeric_distribution_plot(
            frame, column, folder / "plots" / f"{_slug(column)}_distribution.png"
        )
    for column in categorical:
        _categorical_levels_plot(frame, column, folder / "plots" / f"{_slug(column)}_levels.png")
    return {
        "feature_count": len(features),
        "numeric_count": len(numeric),
        "categorical_count": len(categorical),
        "numeric_table": "02_feature_distributions/tables/numeric_feature_profiles.csv",
        "categorical_table": "02_feature_distributions/tables/categorical_feature_profiles.csv",
    }


def _relationships(
    frame: pd.DataFrame, target_proxy: str | None, folder: Path, min_sample_size: int
) -> dict[str, object]:
    if target_proxy is not None:
        numeric = _numeric_target_relationships(frame, target_proxy, min_sample_size)
        categorical = _categorical_target_relationships(frame, target_proxy, min_sample_size)
        numeric.to_csv(folder / "tables" / "numeric_target_relationships.csv", index=False)
        categorical.to_csv(folder / "tables" / "categorical_target_relationships.csv", index=False)
        _top_relationship_plot(
            numeric,
            folder / "plots" / "top_numeric_target_relationships.png",
            "Which numeric features relate to the CxT target/proxy?",
        )
        _top_relationship_plot(
            categorical,
            folder / "plots" / "top_categorical_target_relationships.png",
            "Which categorical features relate to the CxT target/proxy?",
        )
        missing = pd.DataFrame(
            columns=[
                "expected_target_or_proxy",
                "purpose",
                "available",
                "modelling_implication",
                "recommended_next_step",
            ]
        )
    else:
        numeric = pd.DataFrame()
        categorical = pd.DataFrame()
        missing = pd.DataFrame(
            [
                {
                    "expected_target_or_proxy": name,
                    "purpose": purpose,
                    "available": False,
                    "modelling_implication": "Supervised CxT modelling cannot use this table as-is.",
                    "recommended_next_step": "Construct a pre-model target/proxy before training.",
                }
                for name, purpose in EXPECTED_TARGET_PROXIES.items()
            ]
        )
        missing.to_csv(folder / "tables" / "missing_target_proxy.csv", index=False)
    if target_proxy is not None:
        missing.to_csv(folder / "tables" / "missing_target_proxy.csv", index=False)

    action_progression = _progression_summary(frame, "action_type", min_sample_size)
    zone_progression = _progression_summary(frame, "start_zone", min_sample_size)
    entry_progression = _entry_summary(frame, min_sample_size)
    action_progression.to_csv(
        folder / "tables" / "action_type_progression_summary.csv", index=False
    )
    zone_progression.to_csv(folder / "tables" / "zone_progression_summary.csv", index=False)
    entry_progression.to_csv(folder / "tables" / "final_third_box_entry_summary.csv", index=False)
    _progression_plot(
        action_progression,
        "action_type",
        folder / "plots" / "action_type_progression_summary.png",
        "Which action types progress the ball most?",
    )
    _progression_plot(
        zone_progression,
        "start_zone",
        folder / "plots" / "zone_progression_summary.png",
        "Which zones produce the most forward movement?",
    )
    _progression_plot(
        entry_progression,
        "feature_value",
        folder / "plots" / "final_third_box_entry_summary.png",
        "Are final-third and box entries represented?",
    )
    return {
        "target_proxy": target_proxy,
        "missing_target_table": "03_feature_target_relationships/tables/missing_target_proxy.csv",
        "numeric_table": "03_feature_target_relationships/tables/numeric_target_relationships.csv",
        "progression_table": "03_feature_target_relationships/tables/action_type_progression_summary.csv",
        "action_progression_rows": len(action_progression),
    }


def _progression_column(frame: pd.DataFrame) -> str:
    for column in ("goal_distance_reduction", "distance_progressed", "progressive_distance", "dx"):
        if column in frame.columns:
            return column
    raise ValueError("Progression analysis requires a movement/progression column.")


def _progression_summary(
    frame: pd.DataFrame, group_column: str, min_sample_size: int
) -> pd.DataFrame:
    if group_column not in frame.columns:
        return pd.DataFrame(
            columns=[
                group_column,
                "rows",
                "mean_progression",
                "median_progression",
                "p75_progression",
                "share_positive_progression",
                "share_negative_progression",
                "sample_size_warning",
                "modelling_implication",
            ]
        )
    prog_col = _progression_column(frame)
    work = frame[[group_column, prog_col]].dropna().copy()
    work[prog_col] = pd.to_numeric(work[prog_col], errors="coerce")
    table = (
        work.groupby(group_column, observed=True)[prog_col]
        .agg(
            rows="count",
            mean_progression="mean",
            median_progression="median",
            p75_progression=lambda s: s.quantile(0.75),
            share_positive_progression=lambda s: (s > 0).mean(),
            share_negative_progression=lambda s: (s < 0).mean(),
        )
        .reset_index()
        .sort_values("mean_progression", ascending=False)
    )
    table["sample_size_warning"] = table["rows"] < min_sample_size
    table["modelling_implication"] = np.where(
        table["sample_size_warning"],
        "Sparse slice; pool or coarsen.",
        "Use for progression diagnostics.",
    )
    return table


def _entry_summary(frame: pd.DataFrame, min_sample_size: int) -> pd.DataFrame:
    rows = []
    for column in ("final_third_entry", "box_entry", "zone14_entry"):
        if column in frame.columns:
            table = _progression_summary(
                frame.assign(feature_value=column + "=" + frame[column].astype(str)),
                "feature_value",
                min_sample_size,
            )
            rows.append(table)
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "feature_value",
                "rows",
                "mean_progression",
                "median_progression",
                "p75_progression",
                "share_positive_progression",
                "share_negative_progression",
                "sample_size_warning",
                "modelling_implication",
            ]
        )
    )


def _feature_correlations(
    frame: pd.DataFrame, target_proxy: str | None, folder: Path
) -> dict[str, object]:
    numeric = _numeric_features(frame, _candidate_features(frame, target_proxy))
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
    targeted = _targeted_redundancy(frame)
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
            title="Which movement/location features are redundant?",
            subtitle=f"{len(high)} pairs exceed absolute correlation 0.80.",
            caption="Source: pre-model progression/action features. Caveat: correlation is linear.",
        )
    else:
        _skipped_plot(
            folder / "plots" / "correlation_heatmap.png",
            "Not enough numeric features for correlations.",
        )
    return {
        "high_pairs": len(high),
        "targeted_pairs": len(targeted),
        "table": "04_feature_correlations/tables/high_correlations.csv",
    }


def _targeted_redundancy(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right in REDUNDANCY_PAIRS:
        if left in frame.columns and right in frame.columns:
            valid = frame[[left, right]].dropna()
            corr = (
                valid[left].astype(float).corr(valid[right].astype(float))
                if len(valid) and valid[left].nunique() > 1
                else np.nan
            )
            rows.append(
                {
                    "feature_a": left,
                    "feature_b": right,
                    "rows": len(valid),
                    "correlation": float(corr) if pd.notna(corr) else np.nan,
                    "redundancy_flag": bool(pd.notna(corr) and abs(float(corr)) >= 0.8),
                    "modelling_implication": "Review/drop/combine if highly correlated.",
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "feature_a",
            "feature_b",
            "rows",
            "correlation",
            "redundancy_flag",
            "modelling_implication",
        ],
    )


def _transition_stability(
    frame: pd.DataFrame, folder: Path, min_sample_size: int
) -> dict[str, object]:
    prog_col = _progression_column(frame)
    if {"start_zone", "end_zone"}.issubset(frame.columns):
        work = frame[["start_zone", "end_zone", prog_col]].dropna().copy()
        work[prog_col] = pd.to_numeric(work[prog_col], errors="coerce")
        table = (
            work.groupby(["start_zone", "end_zone"], observed=True)[prog_col]
            .agg(
                actions="count",
                mean_progression="mean",
                positive_progression_rate=lambda s: (s > 0).mean(),
            )
            .reset_index()
            .sort_values("actions", ascending=False)
        )
        table["share"] = table["actions"] / max(len(frame), 1)
        table["sample_size_warning"] = table["actions"] < min_sample_size
        table["stability_status"] = np.where(table["sample_size_warning"], "sparse", "usable")
        table["modelling_implication"] = np.where(
            table["sample_size_warning"],
            "Coarsen zones or smooth transition value.",
            "Transition has usable sample support.",
        )
    else:
        table = pd.DataFrame(
            columns=[
                "start_zone",
                "end_zone",
                "actions",
                "share",
                "mean_progression",
                "positive_progression_rate",
                "sample_size_warning",
                "stability_status",
                "modelling_implication",
            ]
        )
    table.to_csv(folder / "tables" / "transition_stability.csv", index=False)
    sparse = table[table["sample_size_warning"]] if not table.empty else table
    sparse.to_csv(folder / "tables" / "sparse_transitions.csv", index=False)
    pd.DataFrame(
        [
            {
                "recommendation": (
                    "coarsen_zones" if len(sparse) else "current_resolution_supported"
                ),
                "sparse_transition_count": len(sparse),
                "modelling_implication": "Use coarser zones or smoothing when many transitions are sparse.",
            }
        ]
    ).to_csv(folder / "tables" / "zone_resolution_recommendations.csv", index=False)
    plot = table.head(20).copy()
    if not plot.empty:
        plot["transition"] = plot["start_zone"].astype(str) + " -> " + plot["end_zone"].astype(str)
    _bar_plot(
        plot,
        "transition",
        "actions",
        folder / "plots" / "transition_stability.png",
        title="Which transitions have enough samples for stable value estimates?",
        xlabel="Actions",
        ylabel="Transition",
        reference=min_sample_size,
        caption="Source: pre-model progression/action features. Caveat: stability also depends on target/proxy variance.",
    )
    return {
        "transitions": len(table),
        "sparse": len(sparse),
        "table": "05_transition_stability/tables/transition_stability.csv",
        "visual": "05_transition_stability/plots/transition_stability.png",
    }


def _slice_stability(frame: pd.DataFrame, folder: Path, min_sample_size: int) -> dict[str, object]:
    prog_col = _progression_column(frame)
    global_mean = float(pd.to_numeric(frame[prog_col], errors="coerce").mean())
    rows = []
    for column in (
        "action_type",
        "event_type",
        "team_id",
        "competition_id",
        "competition_name",
        "start_zone",
        "end_zone",
        "under_pressure",
        "pressure",
        "possession_phase",
        "play_pattern",
        "final_third_entry",
        "box_entry",
    ):
        if column not in frame.columns:
            continue
        work = frame[[column, prog_col]].dropna().copy()
        work[prog_col] = pd.to_numeric(work[prog_col], errors="coerce")
        grouped = (
            work.groupby(column, observed=True)[prog_col]
            .agg(
                rows="count",
                mean_progression="mean",
                positive_progression_rate=lambda s: (s > 0).mean(),
            )
            .reset_index()
        )
        grouped = grouped[grouped["rows"] >= min_sample_size]
        for _, row in grouped.iterrows():
            delta = float(row["mean_progression"] - global_mean)
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": row[column],
                    "rows": int(row["rows"]),
                    "mean_progression": float(row["mean_progression"]),
                    "positive_progression_rate": float(row["positive_progression_rate"]),
                    "global_mean_progression": global_mean,
                    "delta_from_global": delta,
                    "sample_size_warning": int(row["rows"]) < max(100, min_sample_size * 2),
                    "modelling_implication": _slice_implication(delta),
                }
            )
    table = pd.DataFrame(
        rows,
        columns=[
            "slice_column",
            "slice_value",
            "rows",
            "mean_progression",
            "positive_progression_rate",
            "global_mean_progression",
            "delta_from_global",
            "sample_size_warning",
            "modelling_implication",
        ],
    )
    table.to_csv(folder / "tables" / "slice_stability.csv", index=False)
    plot = (
        table.reindex(table["delta_from_global"].abs().sort_values(ascending=False).index).head(16)
        if not table.empty
        else table
    )
    if not plot.empty:
        plot["label"] = plot["slice_column"].astype(str) + "=" + plot["slice_value"].astype(str)
    _bar_plot(
        plot,
        "label",
        "delta_from_global",
        folder / "plots" / "slice_stability.png",
        title="Does progression behaviour remain stable across slices?",
        xlabel="Delta from global mean progression",
        ylabel="Slice",
        reference=0,
        caption="Source: pre-model progression/action features.",
    )
    return {
        "slices": len(table),
        "unstable": int((table["delta_from_global"].abs() > 5).sum()) if not table.empty else 0,
        "table": "06_slice_stability/tables/slice_stability.csv",
    }


def _data_quality(
    frame: pd.DataFrame, target_proxy: str | None, folder: Path, min_sample_size: int
) -> dict[str, object]:
    features = _candidate_features(frame, target_proxy)
    quality = pd.DataFrame([_quality_row(frame[c], c) for c in features])
    quality.to_csv(folder / "tables" / "feature_quality.csv", index=False)
    checks = _football_value_checks(frame, features, min_sample_size)
    checks.to_csv(folder / "tables" / "football_value_checks.csv", index=False)
    recs = _cleaning_recommendations(quality, checks)
    recs.to_csv(folder / "tables" / "cleaning_recommendations.csv", index=False)
    return {
        "recommendations": len(recs),
        "quality_table": "07_data_quality/tables/feature_quality.csv",
        "checks_table": "07_data_quality/tables/football_value_checks.csv",
    }


def _football_value_checks(
    frame: pd.DataFrame, features: list[str], min_sample_size: int
) -> pd.DataFrame:
    rows = []
    checks = {
        "start_x": lambda s: (s < 0) | (s > 120),
        "end_x": lambda s: (s < 0) | (s > 120),
        "start_y": lambda s: (s < 0) | (s > 80),
        "end_y": lambda s: (s < 0) | (s > 80),
        "distance_moved": lambda s: (s < 0) | (s > 140),
        "distance_progressed": lambda s: s.abs() > 120,
        "goal_distance_reduction": lambda s: s.abs() > 120,
        "action_duration": lambda s: s < 0,
        "time_since_possession_start": lambda s: s < 0,
    }
    for column, predicate in checks.items():
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            invalid = predicate(values)
            rows.append(_check_row(column, "football_value_range", int(invalid.sum()), len(values)))
    for column in ("start_zone", "end_zone"):
        if column in frame.columns:
            missing = int(frame[column].isna().sum())
            rows.append(_check_row(column, "zone_missingness", missing, len(frame)))
    for column in features:
        series = frame[column]
        if series.isna().mean() >= 0.25:
            rows.append(
                _check_row(column, "high_missingness", int(series.isna().sum()), len(series))
            )
        if series.nunique(dropna=True) <= 1:
            rows.append(_check_row(column, "constant_or_near_constant", len(series), len(series)))
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 2:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if not values.empty:
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outliers = (values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)
                    if outliers.any():
                        rows.append(
                            _check_row(
                                column, "extreme_numeric_outliers", int(outliers.sum()), len(values)
                            )
                        )
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            counts = series.fillna("missing").astype(str).value_counts()
            rare = int((counts < min_sample_size).sum())
            if rare:
                rows.append(_check_row(column, "rare_categorical_levels", rare, len(counts)))
    return pd.DataFrame(
        rows, columns=["column", "check", "issue_count", "issue_rate", "modelling_implication"]
    )


def _leakage_checks(
    frame: pd.DataFrame, target_proxy: str | None, folder: Path
) -> dict[str, object]:
    rows = []
    for column in frame.columns:
        classification = _eligibility(column, target_proxy)
        rows.append(
            {
                "column": column,
                "classification": classification,
                "training_eligibility": (
                    "eligible_candidate_feature"
                    if classification == "candidate_feature"
                    else "exclude_or_reference_only"
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(folder / "tables" / "feature_training_eligibility.csv", index=False)
    table.to_csv(folder / "tables" / "leakage_checks.csv", index=False)
    risk = int((table["training_eligibility"] != "eligible_candidate_feature").sum())
    return {
        "risk_count": risk,
        "table": "08_leakage_checks/tables/feature_training_eligibility.csv",
    }


def _eligibility(column: str, target_proxy: str | None) -> str:
    lower = column.lower()
    if target_proxy is not None and column == target_proxy:
        return "target_or_proxy"
    if lower in {c.lower() for c in TARGET_PROXY_CANDIDATES} or any(
        p.lower() in lower for p in THREAT_PATTERNS
    ):
        return "target_or_proxy"
    if (
        any(pattern in lower for pattern in DOWNSTREAM_REFERENCE_PATTERNS)
        or "goal" in lower
        or "outcome" in lower
    ):
        return "downstream_reference"
    if any(pattern in lower for pattern in POST_MODEL_PATTERNS):
        return "post_model_output"
    if column in ID_COLUMNS or lower.endswith("_id"):
        return "identifier"
    if "future" in lower:
        return "review_required"
    return "candidate_feature"


def _candidate_features(frame: pd.DataFrame, target_proxy: str | None) -> list[str]:
    return [
        column
        for column in frame.columns
        if _eligibility(column, target_proxy) == "candidate_feature"
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


def _numeric_profile(series: pd.Series, feature: str) -> dict[str, object]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "feature": feature,
        "type": "numeric",
        "rows": len(series),
        "missing_rate": float(series.isna().mean()),
        "unique_values": int(series.nunique(dropna=True)),
        "min": float(values.min()) if not values.empty else np.nan,
        "max": float(values.max()) if not values.empty else np.nan,
        "mean": float(values.mean()) if not values.empty else np.nan,
        "median": float(values.median()) if not values.empty else np.nan,
        "p95": float(values.quantile(0.95)) if not values.empty else np.nan,
        "p99": float(values.quantile(0.99)) if not values.empty else np.nan,
        "skew": float(values.skew()) if len(values) > 2 else np.nan,
        "recommendation": _recommendation(series),
    }


def _categorical_profile(series: pd.Series, feature: str) -> dict[str, object]:
    return {
        "feature": feature,
        "type": "categorical",
        "rows": len(series),
        "missing_rate": float(series.isna().mean()),
        "unique_values": int(series.nunique(dropna=True)),
        "min": np.nan,
        "max": np.nan,
        "mean": np.nan,
        "median": np.nan,
        "p95": np.nan,
        "p99": np.nan,
        "skew": np.nan,
        "recommendation": _recommendation(series),
    }


def _quality_row(series: pd.Series, feature: str) -> dict[str, object]:
    return {
        "feature": feature,
        "type": "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical",
        "rows": len(series),
        "missing_rate": float(series.isna().mean()),
        "unique_values": int(series.nunique(dropna=True)),
        "recommendation": _recommendation(series),
    }


def _recommendation(series: pd.Series) -> str:
    if series.isna().mean() >= 0.5 or series.nunique(dropna=True) <= 1:
        return "drop"
    if series.isna().mean() >= 0.2:
        return "review"
    if pd.api.types.is_bool_dtype(series):
        return "keep"
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        if len(values) > 2 and abs(float(values.skew())) >= 1.5:
            return "transform"
        return "keep"
    if series.nunique(dropna=True) > 20:
        return "encode"
    return "keep"


def _numeric_distribution_plot(frame: pd.DataFrame, column: str, path: Path) -> None:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if values.empty:
        ax.text(0.5, 0.5, "No numeric values", ha="center", va="center")
    else:
        ax.hist(
            values, bins=min(30, max(8, int(np.sqrt(len(values))))), color="#2563eb", alpha=0.82
        )
        ax.axvline(values.median(), color="#b45309", linestyle="--", label="Median")
        ax.set_xlabel(column)
        ax.set_ylabel("Actions")
        ax.legend(loc="upper right")
    _save_plot(
        fig,
        ax,
        path,
        title=f"Does {column} need transformation, binning, or capping?",
        subtitle=_distribution_takeaway(values),
        caption=f"Source: pre-model progression/action features. n={len(values):,}. Caveat: distribution is not target signal.",
    )


def _categorical_levels_plot(frame: pd.DataFrame, column: str, path: Path) -> None:
    counts = frame[column].fillna("missing").astype(str).value_counts().head(20).sort_values()
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.42)))
    if counts.empty:
        ax.text(0.5, 0.5, "No levels", ha="center", va="center")
    else:
        ax.barh(counts.index, counts.values, color="#0f766e")
        ax.set_xlabel("Actions")
        ax.set_ylabel(column)
    _save_plot(
        fig,
        ax,
        path,
        title=f"Does {column} need encoding or rare-level pooling?",
        subtitle=f"{frame[column].nunique(dropna=True):,} observed levels.",
        caption="Source: pre-model progression/action features. Caveat: plot shows top levels only.",
    )


def _bar_plot(
    table: pd.DataFrame,
    label_col: str,
    value_col: str,
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    reference: float | None,
    caption: str,
) -> None:
    fig, ax = plt.subplots(
        figsize=(11, 6.5 if table.empty else max(6.5, min(12, len(table) * 0.45)))
    )
    if table.empty or label_col not in table.columns:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        plot = table.sort_values(value_col).tail(20)
        ax.barh(plot[label_col].astype(str), plot[value_col], color="#0f766e")
        if reference is not None:
            ax.axvline(reference, color="#b45309", linestyle="--", label="Reference")
            ax.legend(loc="lower right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    _save_plot(
        fig, ax, path, title=title, subtitle="Sorted by available sample/support.", caption=caption
    )


def _progression_plot(table: pd.DataFrame, label_col: str, path: Path, title: str) -> None:
    _bar_plot(
        table,
        label_col,
        "mean_progression",
        path,
        title=title,
        xlabel="Mean progression",
        ylabel=label_col,
        reference=0,
        caption="Source: pre-model progression/action features. Caveat: progression is structural, not target value.",
    )


def _numeric_target_relationships(
    frame: pd.DataFrame, target: str, min_sample_size: int
) -> pd.DataFrame:
    rows = []
    target_values = pd.to_numeric(frame[target], errors="coerce")
    for column in _numeric_features(frame, _candidate_features(frame, target)):
        work = pd.DataFrame(
            {
                "feature_value": pd.to_numeric(frame[column], errors="coerce"),
                "target": target_values,
            }
        ).dropna()
        if work.empty or work["feature_value"].nunique() < 2:
            continue
        work["bin"] = pd.qcut(
            work["feature_value"], q=min(5, work["feature_value"].nunique()), duplicates="drop"
        )
        grouped = (
            work.groupby("bin", observed=True)["target"]
            .agg(rows="count", mean_target="mean")
            .reset_index()
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "feature": column,
                    "bin": str(row["bin"]),
                    "rows": int(row["rows"]),
                    "mean_target": float(row["mean_target"]),
                    "sample_size_warning": int(row["rows"]) < min_sample_size,
                    "modelling_implication": "candidate_signal",
                }
            )
    return pd.DataFrame(rows)


def _categorical_target_relationships(
    frame: pd.DataFrame, target: str, min_sample_size: int
) -> pd.DataFrame:
    rows = []
    target_values = pd.to_numeric(frame[target], errors="coerce")
    for column in _categorical_features(frame, _candidate_features(frame, target)):
        work = pd.DataFrame(
            {"value": frame[column].fillna("missing").astype(str), "target": target_values}
        ).dropna()
        grouped = (
            work.groupby("value", observed=True)["target"]
            .agg(rows="count", mean_target="mean")
            .reset_index()
        )
        grouped = grouped[grouped["rows"] >= min_sample_size]
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "feature": column,
                    "value": row["value"],
                    "rows": int(row["rows"]),
                    "mean_target": float(row["mean_target"]),
                    "sample_size_warning": int(row["rows"]) < max(100, min_sample_size * 2),
                    "modelling_implication": "candidate_signal",
                }
            )
    return pd.DataFrame(rows)


def _top_relationship_plot(table: pd.DataFrame, path: Path, title: str) -> None:
    if table.empty:
        _skipped_plot(path, "No target/proxy relationship data available.")
        return
    summary = (
        table.groupby("feature", observed=True)["mean_target"].agg(["min", "max"]).reset_index()
    )
    summary["spread"] = summary["max"] - summary["min"]
    _bar_plot(
        summary.sort_values("spread", ascending=False).head(15),
        "feature",
        "spread",
        path,
        title=title,
        xlabel="Target/proxy spread",
        ylabel="Feature",
        reference=None,
        caption="Source: pre-model progression/action features. Caveat: univariate relationship only.",
    )


def _check_row(column: str, check: str, issue_count: int, denominator: int) -> dict[str, object]:
    return {
        "column": column,
        "check": check,
        "issue_count": issue_count,
        "issue_rate": issue_count / denominator if denominator else 0.0,
        "modelling_implication": (
            "Clean, transform, coarsen, or review before modelling."
            if issue_count
            else "No issue found."
        ),
    }


def _cleaning_recommendations(quality: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in quality.iterrows():
        if row["recommendation"] != "keep":
            rows.append(
                {
                    "feature": row["feature"],
                    "reason": row["recommendation"],
                    "recommendation": row["recommendation"],
                }
            )
    for _, row in checks.iterrows():
        if int(row["issue_count"]) > 0:
            rows.append(
                {
                    "feature": row["column"],
                    "reason": row["check"],
                    "recommendation": row["modelling_implication"],
                }
            )
    return pd.DataFrame(rows, columns=["feature", "reason", "recommendation"]).drop_duplicates()


def _distribution_takeaway(values: pd.Series) -> str:
    if values.empty:
        return "No non-missing numeric values."
    return f"Median {values.median():.2f}; p95 {values.quantile(0.95):.2f}; p99 {values.quantile(0.99):.2f}."


def _slice_implication(delta: float) -> str:
    if abs(delta) >= 10:
        return "Use this slice for validation and possible interaction checks."
    if abs(delta) >= 5:
        return "Monitor this slice during value construction."
    return "No immediate slice-specific action."


def _skipped_plot(path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    _save_plot(
        fig,
        ax,
        path,
        title="Analysis skipped",
        subtitle=message,
        caption="Source: pre-model progression/action features.",
    )


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def _section(title: str, values: dict[str, str]) -> str:
    lines = [f"## {title}", ""]
    for key in REQUIRED_SECTIONS:
        lines.append(f"**{key}:** {values[key]}")
        lines.append("")
    return "\n".join(lines)


def _render_report(
    *,
    data_source: str,
    row_count: int,
    coverage: dict[str, object],
    spatial: dict[str, object],
    distributions: dict[str, object],
    relationships: dict[str, object],
    correlations: dict[str, object],
    transitions: dict[str, object],
    slices: dict[str, object],
    quality: dict[str, object],
    leakage: dict[str, object],
    target_proxy: str | None,
    min_sample_size: int,
) -> str:
    target_text = (
        f"A supervised target/proxy is available: `{target_proxy}`."
        if target_proxy
        else "No supervised CxT target/proxy is currently available; construct one before supervised modelling."
    )
    sections = [
        "# CxT Pre-Model Ball Progression Feature Analysis",
        "",
        "This diagnostic layer sits between progression feature engineering and CxT modelling. "
        "It studies pre-model progression/action features only.",
        "",
        _section(
            "1. Action/progression table usability",
            {
                "Question": "Is there usable pre-model progression/action data?",
                "Calculation": f"{row_count:,} rows loaded from {data_source}.",
                "Visual/Table": "00_action_coverage/tables/action_type_coverage.csv.",
                "Interpretation": "A progression table was found and analysed.",
                "Modelling implication": "Proceed to feature/transition diagnostics before value construction.",
                "Limitation": "This does not train or evaluate a CxT model.",
            },
        ),
        _section(
            "2. Action coverage",
            {
                "Question": "Are passes, carries, dribbles, identifiers, and locations covered?",
                "Calculation": f"{coverage['action_type_count']} action types; core types: {', '.join(coverage['core_types']) or 'none'}.",
                "Visual/Table": f"{coverage['table']} and {coverage['visual']}.",
                "Interpretation": f"{coverage['rare_count']} action types are below the sample-size reference.",
                "Modelling implication": "Pool rare actions and validate ID/location coverage.",
                "Limitation": "Coverage is not target/value quality.",
            },
        ),
        _section(
            "3. Spatial coverage",
            {
                "Question": "Is spatial coverage sufficient across the pitch?",
                "Calculation": f"{spatial['start_zones']} start zones, {spatial['end_zones']} end zones, {spatial['transitions']} transitions.",
                "Visual/Table": "01_spatial_coverage/tables/transition_coverage.csv.",
                "Interpretation": f"{spatial['sparse_transitions']} transitions are sparse.",
                "Modelling implication": "Coarsen zones or smooth sparse transitions before grid value construction.",
                "Limitation": "Spatial support alone does not prove value stability.",
            },
        ),
        _section(
            "4. Progression feature distributions",
            {
                "Question": "Which progression features need transformation, binning, encoding, capping, or review?",
                "Calculation": f"{distributions['numeric_count']} numeric and {distributions['categorical_count']} categorical features profiled.",
                "Visual/Table": f"{distributions['numeric_table']} and {distributions['categorical_table']}.",
                "Interpretation": "One chart per feature was written under 02_feature_distributions/plots/.",
                "Modelling implication": "Encode preprocessing decisions before CxT construction.",
                "Limitation": "Distribution shape is not target/proxy signal.",
            },
        ),
        _section(
            "5. Target/proxy availability",
            {
                "Question": "Is a supervised CxT target/proxy available?",
                "Calculation": "Known target/proxy names were detected against the feature table.",
                "Visual/Table": (
                    relationships["missing_target_table"]
                    if not target_proxy
                    else relationships["numeric_table"]
                ),
                "Interpretation": target_text,
                "Modelling implication": (
                    "Construct threat_delta/xt_delta/future-value proxy before supervised CxT modelling."
                    if not target_proxy
                    else "Use the target/proxy as reference, not as ordinary feature input."
                ),
                "Limitation": "A proxy may be incomplete even when named columns exist.",
            },
        ),
        _section(
            "6. Progression signal without target/proxy",
            {
                "Question": "What progression structure exists before target construction?",
                "Calculation": "Action-type, zone, and entry summaries compare progression distributions.",
                "Visual/Table": relationships["progression_table"],
                "Interpretation": f"{relationships['action_progression_rows']} action progression rows were produced.",
                "Modelling implication": "Use structural progression diagnostics to assess feature readiness.",
                "Limitation": "Progression is not equivalent to threat value.",
            },
        ),
        _section(
            "7. Feature redundancy",
            {
                "Question": "Which movement/location features are redundant?",
                "Calculation": "Numeric correlations and targeted redundancy checks are computed.",
                "Visual/Table": correlations["table"],
                "Interpretation": f"{correlations['high_pairs']} high-correlation pairs were flagged.",
                "Modelling implication": "Drop/combine/regularise redundant features.",
                "Limitation": "Correlation is linear.",
            },
        ),
        _section(
            "8. Transition stability",
            {
                "Question": "Which transitions have enough samples to estimate stable value?",
                "Calculation": f"{transitions['transitions']} transitions analysed with minimum sample reference {min_sample_size}.",
                "Visual/Table": f"{transitions['table']} and {transitions['visual']}.",
                "Interpretation": f"{transitions['sparse']} transitions are sparse.",
                "Modelling implication": "Coarsen zone resolution or smooth sparse transition values.",
                "Limitation": "Target/proxy variance is not measured when unavailable.",
            },
        ),
        _section(
            "9. Slice stability",
            {
                "Question": "Does progression behaviour remain stable by action/team/zone/pressure slices?",
                "Calculation": f"{slices['slices']} slice rows were compared to global mean progression.",
                "Visual/Table": slices["table"],
                "Interpretation": f"{slices['unstable']} slices show large progression deltas.",
                "Modelling implication": "Use unstable slices for validation and monitoring.",
                "Limitation": "Slice progression may reflect team/style mix.",
            },
        ),
        _section(
            "10. Data quality and cleaning recommendations",
            {
                "Question": "Which features need cleaning before CxT modelling?",
                "Calculation": "Coordinate ranges, progression values, missingness, constants, rare levels, and outliers are checked.",
                "Visual/Table": f"{quality['quality_table']} and {quality['checks_table']}.",
                "Interpretation": f"{quality['recommendations']} cleaning recommendations were produced.",
                "Modelling implication": "Encode cleaning rules in the modelling feature contract.",
                "Limitation": "Automated checks require domain review.",
            },
        ),
        _section(
            "11. Leakage risks and training eligibility",
            {
                "Question": "Which columns must be excluded before training?",
                "Calculation": "Columns are classified as target/proxy, reference, downstream reference, post-model output, identifier, candidate, or review.",
                "Visual/Table": leakage["table"],
                "Interpretation": f"{leakage['risk_count']} columns are excluded or reference-only.",
                "Modelling implication": "Train only on candidate features; keep future/reference/post-model columns out of inputs.",
                "Limitation": "Name-based checks cannot prove semantic safety for every future column.",
            },
        ),
        _section(
            "12. Modelling recommendations",
            {
                "Question": "What should CxT modelling do next?",
                "Calculation": "Combine coverage, spatial support, distributions, target/proxy availability, progression structure, transition stability, quality, and leakage results.",
                "Visual/Table": "Use all numbered folders under outputs/analysis/cxt/.",
                "Interpretation": target_text,
                "Modelling implication": "Build or validate a target/proxy, choose zone resolution, clean candidate features, and monitor sparse transitions before threat-value construction.",
                "Limitation": "This report does not produce CxT predictions, aggregates, leaderboards, or dashboard output.",
            },
        ),
    ]
    return "\n".join(sections) + "\n"
