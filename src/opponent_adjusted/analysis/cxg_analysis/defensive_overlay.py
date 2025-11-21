"""CxG exploratory analysis: opponent defensive overlay.

This module links opponent defensive actions (pressures, tackles, interceptions,
blocks, ball recoveries, duels) to the shots they quickly generate after those
wins. It surfaces how disruptive actions convert into xG conceded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from sqlalchemy import select
from sqlalchemy.orm import aliased

from opponent_adjusted.analysis.plot_utilities import (
    STYLE,
    configure_matplotlib,
    get_analysis_output_dir,
    save_figure,
)
from opponent_adjusted.config import settings
from opponent_adjusted.db.models import (
    BlockEvent,
    DuelEvent,
    Event,
    InterceptionEvent,
    PressureEvent,
    Shot,
    ShotFeature,
    Team,
)
from opponent_adjusted.db.session import SessionLocal

DEFENSIVE_EVENT_TYPES = {
    "Pressure",
    "Interception",
    "Ball Recovery",
    "Tackle",
    "Block",
    "Duel",
}
BALL_PROGRESS_EVENT_TYPES = {"Carry", "Dribble"}
TRIGGER_EVENT_TYPES = DEFENSIVE_EVENT_TYPES | BALL_PROGRESS_EVENT_TYPES
MAX_EVENT_TIME_GAP_SECONDS = 5
MIN_SEQUENCE_SHOTS = 10
PITCH_BINS = (30, 20)
NO_IMMEDIATE_LABEL = "No immediate defensive trigger"
NO_POSSESSION_LABEL = "No possession defensive trigger"
pitch = Pitch(
    pitch_type="statsbomb",
    pitch_length=settings.pitch_length,
    pitch_width=settings.pitch_width,
    line_zorder=2,
    linewidth=1.0,
)


@dataclass
class ShotRecord:
    shot_id: int
    match_id: int
    team_id: int
    team_name: str
    opponent_team_id: int
    opponent_team_name: str
    statsbomb_xg: float | None
    is_goal: bool
    shot_distance: float | None
    shot_angle: float | None
    shot_event_id: int
    shot_possession: Optional[int]
    shot_minute: Optional[int]
    shot_second: Optional[int]
    location_x: float | None
    location_y: float | None


@dataclass
class DefensiveEventRecord:
    event_id: int
    match_id: int
    team_id: int
    possession: Optional[int]
    event_type: str
    minute: Optional[int]
    second: Optional[int]
    location_x: float | None
    location_y: float | None
    under_pressure: bool
    outcome: Optional[str]
    duel_type: Optional[str]
    duel_outcome: Optional[str]
    block_type: Optional[str]
    interception_outcome: Optional[str]
    pressure_duration: Optional[float]


def load_shot_data(feature_version: str = "v1") -> pd.DataFrame:
    team_alias = aliased(Team)
    opponent_alias = aliased(Team)
    with SessionLocal() as session:
        stmt = (
            select(
                Shot.id.label("shot_id"),
                Shot.match_id,
                Shot.team_id,
                team_alias.name.label("team_name"),
                Shot.opponent_team_id,
                opponent_alias.name.label("opponent_team_name"),
                Shot.statsbomb_xg,
                Shot.outcome,
                ShotFeature.shot_distance,
                ShotFeature.shot_angle,
                Event.id.label("event_id"),
                Event.possession,
                Event.minute,
                Event.second,
                Event.location_x,
                Event.location_y,
            )
            .join(ShotFeature, ShotFeature.shot_id == Shot.id)
            .join(Event, Event.id == Shot.event_id)
            .join(team_alias, team_alias.id == Shot.team_id)
            .join(opponent_alias, opponent_alias.id == Shot.opponent_team_id)
            .where(ShotFeature.version_tag == feature_version)
        )
        rows = session.execute(stmt).all()

    records: List[ShotRecord] = []
    for row in rows:
        is_goal = (row.outcome or "").lower() == "goal"
        records.append(
            ShotRecord(
                shot_id=row.shot_id,
                match_id=row.match_id,
                team_id=row.team_id,
                team_name=row.team_name,
                opponent_team_id=row.opponent_team_id,
                opponent_team_name=row.opponent_team_name,
                statsbomb_xg=float(row.statsbomb_xg)
                if row.statsbomb_xg is not None
                else None,
                is_goal=is_goal,
                shot_distance=float(row.shot_distance)
                if row.shot_distance is not None
                else None,
                shot_angle=float(row.shot_angle) if row.shot_angle is not None else None,
                shot_event_id=row.event_id,
                shot_possession=row.possession,
                shot_minute=int(row.minute) if row.minute is not None else None,
                shot_second=int(row.second) if row.second is not None else None,
                location_x=float(row.location_x) if row.location_x is not None else None,
                location_y=float(row.location_y) if row.location_y is not None else None,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        return df
    df["def_team_id"] = df["team_id"]
    return df


def load_defensive_events(match_ids: Iterable[int], team_ids: Iterable[int]) -> pd.DataFrame:
    match_ids = list({mid for mid in match_ids if mid is not None})
    team_ids = list({tid for tid in team_ids if tid is not None})
    if not match_ids or not team_ids:
        return pd.DataFrame()

    with SessionLocal() as session:
        stmt = (
            select(
                Event.id.label("event_id"),
                Event.match_id,
                Event.team_id,
                Event.possession,
                Event.type,
                Event.minute,
                Event.second,
                Event.location_x,
                Event.location_y,
                Event.under_pressure,
                Event.outcome,
                DuelEvent.duel_type,
                DuelEvent.outcome.label("duel_outcome"),
                BlockEvent.block_type,
                InterceptionEvent.outcome.label("interception_outcome"),
                PressureEvent.duration_seconds.label("pressure_duration"),
            )
            .outerjoin(DuelEvent, DuelEvent.event_id == Event.id)
            .outerjoin(BlockEvent, BlockEvent.event_id == Event.id)
            .outerjoin(InterceptionEvent, InterceptionEvent.event_id == Event.id)
            .outerjoin(PressureEvent, PressureEvent.event_id == Event.id)
            .where(Event.type.in_(TRIGGER_EVENT_TYPES))
            .where(Event.match_id.in_(match_ids))
            .where(Event.team_id.in_(team_ids))
        )
        rows = session.execute(stmt).all()

    records: List[DefensiveEventRecord] = []
    for row in rows:
        records.append(
            DefensiveEventRecord(
                event_id=row.event_id,
                match_id=row.match_id,
                team_id=row.team_id,
                possession=row.possession,
                event_type=row.type,
                minute=int(row.minute) if row.minute is not None else None,
                second=int(row.second) if row.second is not None else None,
                location_x=float(row.location_x) if row.location_x is not None else None,
                location_y=float(row.location_y) if row.location_y is not None else None,
                under_pressure=bool(row.under_pressure),
                outcome=row.outcome,
                duel_type=row.duel_type,
                duel_outcome=row.duel_outcome,
                block_type=row.block_type,
                interception_outcome=row.interception_outcome,
                pressure_duration=row.pressure_duration,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in records])
    if df.empty:
        return df
    return df.rename(
        columns={
            "event_id": "def_event_id",
            "team_id": "def_team_id",
            "possession": "def_possession",
            "event_type": "def_event_type",
            "minute": "def_minute",
            "second": "def_second",
            "location_x": "def_location_x",
            "location_y": "def_location_y",
            "under_pressure": "def_under_pressure",
            "outcome": "def_event_outcome",
            "duel_type": "def_duel_type",
            "duel_outcome": "def_duel_outcome",
            "block_type": "def_block_type",
            "interception_outcome": "def_interception_outcome",
            "pressure_duration": "def_pressure_duration",
        }
    )


def _resolve_event_role(
    event_team_id: Optional[float], shot_team_id: int, opponent_team_id: int
) -> Optional[str]:
    if pd.isna(event_team_id):
        return None
    try:
        team_id = int(event_team_id)
    except (TypeError, ValueError):
        return None
    if team_id == int(shot_team_id):
        return "Own"
    if team_id == int(opponent_team_id):
        return "Opponent"
    return "Other"


def _compose_trigger_label(
    event_type: Optional[str],
    role: Optional[str],
    duel_type: Optional[str],
    block_type: Optional[str],
    interception_outcome: Optional[str],
    event_outcome: Optional[str],
    *,
    missing_label: str,
) -> str:
    if pd.isna(event_type) or not event_type:
        return missing_label

    descriptor = event_type
    if role == "Opponent":
        descriptor = f"Opponent {descriptor}"
    elif role == "Other":
        descriptor = f"Other {descriptor}"

    subtype = duel_type or block_type or interception_outcome or event_outcome
    if isinstance(subtype, str):
        subtype = subtype.strip()
    if subtype:
        return f"{descriptor} – {subtype}"
    return descriptor


def classify_sequences(shots: pd.DataFrame, defensive_events: pd.DataFrame) -> pd.DataFrame:
    df = shots.copy()
    df["shot_seconds"] = df["shot_minute"].fillna(0) * 60 + df["shot_second"].fillna(0)
    df["shot_possession"] = pd.to_numeric(df["shot_possession"], errors="coerce")

    if defensive_events.empty:
        df["def_event_type"] = np.nan
        df["def_event_role"] = np.nan
        df["seconds_since_def_action"] = np.nan
        df["def_label"] = NO_IMMEDIATE_LABEL
        df["def_event_id"] = np.nan
        df["def_possession"] = np.nan
        df["def_event_seconds"] = np.nan
        df["pos_event_type"] = np.nan
        df["pos_event_role"] = np.nan
        df["seconds_since_possession_event"] = np.nan
        df["possession_label"] = NO_POSSESSION_LABEL
        df["pos_event_seconds"] = np.nan
        df["pos_event_id"] = np.nan
        df["pos_possession"] = np.nan
        df["pos_team_id"] = np.nan
        return df

    defensive_events = defensive_events.copy()
    defensive_events["def_event_seconds"] = (
        defensive_events["def_minute"].fillna(0) * 60 + defensive_events["def_second"].fillna(0)
    )
    defensive_events = defensive_events.sort_values(
        ["match_id", "def_event_seconds", "def_event_id"]
    )

    event_cols = [col for col in defensive_events.columns if col != "match_id"]
    pos_event_cols = [
        col.replace("def_", "pos_", 1)
        for col in event_cols
        if col.startswith("def_") or col == "def_event_seconds"
    ]

    merged_groups: List[pd.DataFrame] = []
    for match_id, shot_group in df.groupby("match_id", sort=False):
        match_events = defensive_events[defensive_events["match_id"] == match_id]
        shot_sorted = shot_group.sort_values(["shot_seconds", "shot_event_id"])
        if match_events.empty:
            shot_filled = shot_sorted.copy()
            for col in event_cols:
                if col not in shot_filled.columns:
                    shot_filled[col] = np.nan
            for col in pos_event_cols:
                if col not in shot_filled.columns:
                    shot_filled[col] = np.nan
            merged_groups.append(shot_filled)
            continue

        base_events = match_events.drop(columns=["match_id"], errors="ignore")
        immediate = pd.merge_asof(
            shot_sorted,
            base_events,
            left_on="shot_seconds",
            right_on="def_event_seconds",
            direction="backward",
        )

        pos_events = base_events.dropna(subset=["def_possession"]).copy()
        if not pos_events.empty:
            pos_events["possession_key"] = pos_events["def_possession"].astype(float)
            pos_events = pos_events.sort_values(["def_event_seconds", "def_event_id"])
            rename_map = {
                col: col.replace("def_", "pos_", 1)
                for col in pos_events.columns
                if col.startswith("def_")
            }
            pos_events = pos_events.rename(columns=rename_map)
            pos_events = pos_events.rename(columns={"pos_event_seconds": "pos_event_seconds"})
            shot_tmp = immediate.copy()
            shot_tmp["possession_key"] = shot_tmp["shot_possession"].astype(float)
            merged = pd.merge_asof(
                shot_tmp.sort_values(["shot_seconds", "shot_event_id"]),
                pos_events,
                left_on="shot_seconds",
                right_on="pos_event_seconds",
                by="possession_key",
                direction="backward",
            )
            merged = merged.drop(columns=["possession_key"], errors="ignore")
        else:
            merged = immediate
            for col in pos_event_cols:
                if col not in merged.columns:
                    merged[col] = np.nan

        merged_groups.append(merged)

    merged = pd.concat(merged_groups, axis=0).sort_index()

    for bool_col in ("def_under_pressure", "pos_under_pressure"):
        if bool_col in merged.columns:
            merged[bool_col] = merged[bool_col].astype("boolean")

    merged["seconds_since_def_action"] = (
        merged["shot_seconds"] - merged["def_event_seconds"]
    )
    merged.loc[merged["seconds_since_def_action"] < 0, "seconds_since_def_action"] = np.nan

    if "pos_event_seconds" not in merged.columns:
        merged["pos_event_seconds"] = np.nan

    merged["seconds_since_possession_event"] = (
        merged["shot_seconds"] - merged["pos_event_seconds"]
    )
    merged.loc[
        merged["seconds_since_possession_event"] < 0, "seconds_since_possession_event"
    ] = np.nan

    mask_valid = (
        (~merged["def_event_id"].isna())
        & (merged["seconds_since_def_action"] <= MAX_EVENT_TIME_GAP_SECONDS)
    )
    immediate_cols = [
        "def_event_id",
        "def_event_type",
        "def_location_x",
        "def_location_y",
        "def_under_pressure",
        "def_event_outcome",
        "def_duel_type",
        "def_duel_outcome",
        "def_block_type",
        "def_interception_outcome",
        "def_pressure_duration",
        "def_possession",
        "def_minute",
        "def_second",
        "def_event_seconds",
        "def_team_id",
    ]
    immediate_cols = [col for col in immediate_cols if col in merged.columns]
    merged.loc[~mask_valid, immediate_cols] = np.nan

    merged["def_event_role"] = merged.apply(
        lambda row: _resolve_event_role(
            row.get("def_team_id"), row["team_id"], row["opponent_team_id"]
        ),
        axis=1,
    )
    merged["def_label"] = merged.apply(
        lambda row: _compose_trigger_label(
            row.get("def_event_type"),
            row.get("def_event_role"),
            row.get("def_duel_type"),
            row.get("def_block_type"),
            row.get("def_interception_outcome"),
            row.get("def_event_outcome"),
            missing_label=NO_IMMEDIATE_LABEL,
        ),
        axis=1,
    )

    merged["pos_event_role"] = merged.apply(
        lambda row: _resolve_event_role(
            row.get("pos_team_id"), row["team_id"], row["opponent_team_id"]
        ),
        axis=1,
    )
    merged["possession_label"] = merged.apply(
        lambda row: _compose_trigger_label(
            row.get("pos_event_type"),
            row.get("pos_event_role"),
            row.get("pos_duel_type"),
            row.get("pos_block_type"),
            row.get("pos_interception_outcome"),
            row.get("pos_event_outcome"),
            missing_label=NO_POSSESSION_LABEL,
        ),
        axis=1,
    )

    merged["possession_alignment"] = (
        merged["shot_possession"].notna()
        & merged["def_possession"].notna()
        & (merged["shot_possession"] == merged["def_possession"])
    )

    return merged


def summarize_sequences(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("def_label", observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
            avg_gap=("seconds_since_def_action", "mean"),
            possession_match=("possession_alignment", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["mean_xg"].fillna(0)
    summary = summary.sort_values("shots", ascending=False)
    return summary


def summarize_possession_sequences(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("possession_label", observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
            avg_gap=("seconds_since_possession_event", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["mean_xg"].fillna(0)
    summary = summary.sort_values("shots", ascending=False)
    return summary


def summarize_by_opponent(df: pd.DataFrame) -> pd.DataFrame:
    opponent_summary = (
        df.groupby("opponent_team_name", observed=False)
        .agg(
            shots=("shot_id", "count"),
            xg_total=("statsbomb_xg", "sum"),
            goals=("is_goal", "sum"),
            avg_gap=("seconds_since_def_action", "mean"),
        )
        .reset_index()
    )
    opponent_summary["goal_rate"] = opponent_summary["goals"] / opponent_summary["shots"].clip(lower=1)
    opponent_summary = opponent_summary.sort_values("xg_total", ascending=False)
    return opponent_summary


def export_tables(
    summary: pd.DataFrame, opponent_summary: pd.DataFrame, possession_summary: pd.DataFrame
) -> None:
    out_dir = get_analysis_output_dir("cxg", subdir="csv")
    summary.to_csv(out_dir / "defensive_overlay_summary.csv", index=False)
    opponent_summary.to_csv(out_dir / "defensive_overlay_by_opponent.csv", index=False)
    possession_summary.to_csv(out_dir / "defensive_overlay_possession_summary.csv", index=False)


def plot_trigger_goal_rates(summary: pd.DataFrame) -> None:
    plot_df = summary[summary["shots"] >= MIN_SEQUENCE_SHOTS]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
    idx = np.arange(len(plot_df))
    width = 0.4
    ax.bar(
        idx - width / 2,
        plot_df["goal_rate"],
        width=width,
        color=STYLE.primary_color,
        label="Goal rate",
    )
    ax.bar(
        idx + width / 2,
        plot_df["mean_xg"],
        width=width,
        color=STYLE.secondary_color,
        label="Mean StatsBomb xG",
    )
    ax.set_xticks(idx)
    ax.set_xticklabels(plot_df["def_label"], rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Shots generated after defensive triggers")
    ax.legend()
    save_figure(fig, analysis_type="cxg", name="defensive_overlay_goal_rates")


def plot_trigger_counts(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
    ax.barh(summary["def_label"], summary["shots"], color=STYLE.primary_color)
    ax.set_xlabel("Shots")
    ax.set_title("Shot volume by defensive trigger")
    save_figure(fig, analysis_type="cxg", name="defensive_overlay_counts")


def plot_trigger_lift(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
    max_shots = summary["shots"].max()
    sizes = np.clip(summary["shots"] / max_shots, 0.2, 1.0) * 400
    colors = [
        STYLE.secondary_color if label.startswith("No ") else STYLE.primary_color
        for label in summary["def_label"]
    ]
    ax.scatter(summary["shots"], summary["lift_vs_xg"], s=sizes, c=colors, alpha=0.75)
    ax.axhline(0, color=STYLE.grid_color, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Shot volume")
    ax.set_ylabel("Goal rate lift vs xG")
    ax.set_title("Defensive trigger efficiency vs usage")
    save_figure(fig, analysis_type="cxg", name="defensive_overlay_lift_scatter")


def plot_possession_goal_rates(summary: pd.DataFrame) -> None:
    plot_df = summary[summary["shots"] >= MIN_SEQUENCE_SHOTS]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
    idx = np.arange(len(plot_df))
    width = 0.4
    ax.bar(
        idx - width / 2,
        plot_df["goal_rate"],
        width=width,
        color=STYLE.primary_color,
        label="Goal rate",
    )
    ax.bar(
        idx + width / 2,
        plot_df["mean_xg"],
        width=width,
        color=STYLE.secondary_color,
        label="Mean StatsBomb xG",
    )
    ax.set_xticks(idx)
    ax.set_xticklabels(plot_df["possession_label"], rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Shots by last possession defensive action")
    ax.legend()
    save_figure(fig, analysis_type="cxg", name="defensive_overlay_possession_goal_rates")


def plot_possession_counts(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
    ax.barh(summary["possession_label"], summary["shots"], color=STYLE.primary_color)
    ax.set_xlabel("Shots")
    ax.set_title("Shot volume by possession defensive action")
    save_figure(fig, analysis_type="cxg", name="defensive_overlay_possession_counts")


def plot_possession_lift(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
    max_shots = summary["shots"].max()
    sizes = np.clip(summary["shots"] / max_shots, 0.2, 1.0) * 400
    colors = [
        STYLE.secondary_color if label.startswith("No ") else STYLE.primary_color
        for label in summary["possession_label"]
    ]
    ax.scatter(summary["shots"], summary["lift_vs_xg"], s=sizes, c=colors, alpha=0.75)
    ax.axhline(0, color=STYLE.grid_color, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Shot volume")
    ax.set_ylabel("Goal rate lift vs xG")
    ax.set_title("Possession-level trigger efficiency")
    save_figure(fig, analysis_type="cxg", name="defensive_overlay_possession_lift_scatter")


def _top_labels(summary: pd.DataFrame) -> List[str]:
    labels = summary[summary["def_label"] != "No immediate defensive trigger"].head(4)[
        "def_label"
    ].tolist()
    if (
        "No immediate defensive trigger" in summary["def_label"].values
        and "No immediate defensive trigger" not in labels
    ):
        labels.append("No immediate defensive trigger")
    return labels


def plot_defensive_heatmaps(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    if df["def_location_x"].isna().all():
        return
    labels = _top_labels(summary)
    for label in labels:
        subset = df[(df["def_label"] == label) & df["def_location_x"].notna()]
        if subset.empty:
            continue
        stat = pitch.bin_statistic(
            subset["def_location_x"].to_numpy(),
            subset["def_location_y"].to_numpy(),
            statistic="count",
            bins=PITCH_BINS,
        )
        if np.nanmax(stat["statistic"]) == 0:
            continue
        normalized = stat.copy()
        normalized["statistic"] = stat["statistic"] / np.nanmax(stat["statistic"])
        fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
        pitch.draw(ax=ax)
        heat = pitch.heatmap(normalized, ax=ax, cmap="inferno", edgecolors=None)
        fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label="Relative density")
        ax.set_title(f"Defensive action origin – {label}")
        slug = label.lower().replace(" ", "-").replace("/", "-")
        save_figure(fig, analysis_type="cxg", name=f"defensive_overlay_heatmap_{slug}")


def plot_opponent_impact(opponent_summary: pd.DataFrame) -> None:
    plot_df = opponent_summary.head(10)
    if plot_df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    idx = np.arange(len(plot_df))
    ax1.bar(idx, plot_df["xg_total"], color=STYLE.primary_color)
    ax1.set_ylabel("xG conceded")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(plot_df["opponent_team_name"], rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(idx, plot_df["goal_rate"], color=STYLE.secondary_color, marker="o")
    ax2.set_ylim(0, max(0.6, plot_df["goal_rate"].max() * 1.1))
    ax2.set_ylabel("Goal rate")
    ax1.set_title("Top opponents by xG from defensive triggers")
    save_figure(fig, analysis_type="cxg", name="defensive_overlay_opponent_impact")


def plot_gap_distribution(df: pd.DataFrame) -> None:
    valid = df["seconds_since_def_action"].dropna()
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
    ax.hist(valid, bins=np.arange(0, MAX_EVENT_TIME_GAP_SECONDS + 2, 2), color=STYLE.primary_color)
    ax.set_xlabel("Seconds from defensive action to shot")
    ax.set_ylabel("Shots")
    ax.set_title("Tempo of transition after defensive triggers")
    save_figure(fig, analysis_type="cxg", name="defensive_overlay_time_gap_distribution")


def build_overlay_dataset(feature_version: str = "v1") -> pd.DataFrame:
    """Return per-shot defensive overlay features for modelling workflows."""

    shots = load_shot_data(feature_version=feature_version)
    if shots.empty:
        return pd.DataFrame()

    defensive = load_defensive_events(shots["match_id"], shots["team_id"])
    enriched = classify_sequences(shots, defensive)
    return enriched


def run_defensive_overlay_analysis(feature_version: str = "v1") -> None:
    configure_matplotlib()
    shots = load_shot_data(feature_version=feature_version)
    if shots.empty:
        print("No shot data available for feature version", feature_version)
        return

    defensive = load_defensive_events(shots["match_id"], shots["team_id"])
    enriched = classify_sequences(shots, defensive)
    summary = summarize_sequences(enriched)
    possession_summary = summarize_possession_sequences(enriched)
    opponent_summary = summarize_by_opponent(enriched)
    export_tables(summary, opponent_summary, possession_summary)
    plot_trigger_goal_rates(summary)
    plot_trigger_counts(summary)
    plot_trigger_lift(summary)
    plot_possession_goal_rates(possession_summary)
    plot_possession_counts(possession_summary)
    plot_possession_lift(possession_summary)
    plot_defensive_heatmaps(enriched, summary)
    plot_opponent_impact(opponent_summary)
    plot_gap_distribution(enriched)


if __name__ == "__main__":
    run_defensive_overlay_analysis()
