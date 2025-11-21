"""CxG exploratory analysis: finishing outcomes by assist context.

This module surfaces how different assist/pass archetypes impact the
empirical goal rate relative to StatsBomb xG. It produces:

- Aggregated table of goal rates, mean xG, and shot counts per assist category
- Bar chart comparing empirical goal rate vs StatsBomb xG per category
- Distance vs goal-rate curves for high-volume categories
- Pitch heatmaps for the most common assist archetypes

Outputs are written under outputs/analysis/cxg/.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import select
from mplsoccer import Pitch

from opponent_adjusted.analysis.plot_utilities import (
    STYLE,
    configure_matplotlib,
    get_analysis_output_dir,
    save_figure,
)
from opponent_adjusted.analysis.cxg_analysis.geometry_vs_outcome import (
    bin_and_aggregate_distance,
)
from opponent_adjusted.config import settings
from opponent_adjusted.db.models import Event, RawEvent, Shot, ShotFeature
from opponent_adjusted.db.session import SessionLocal

SET_PIECE_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Throw In",
    "From Goal Kick",
}
COUNTER_PATTERNS = {"From Counter"}
CATEGORY_MIN_SHOTS = 40
CURVE_MIN_SHOTS = 200
HEATMAP_MIN_SHOTS = 150
HEATMAP_MAX_PANELS = 3
PITCH_BINS = (30, 20)
pitch = Pitch(
    pitch_type="statsbomb",
    pitch_length=settings.pitch_length,
    pitch_width=settings.pitch_width,
    line_zorder=2,
    linewidth=1.0,
)


def load_shot_and_assist_data(feature_version: str = "v1") -> pd.DataFrame:
    """Load shot geometry plus assist metadata into a DataFrame."""

    with SessionLocal() as session:
        stmt = (
            select(
                Shot.id.label("shot_id"),
                Shot.statsbomb_xg,
                Shot.outcome,
                Shot.is_blocked,
                Shot.first_time,
                Shot.body_part,
                Event.player_id,
                Event.location_x,
                Event.location_y,
                Event.under_pressure,
                RawEvent.raw_json.label("shot_raw_json"),
                ShotFeature.shot_distance,
                ShotFeature.shot_angle,
            )
            .join(Event, Shot.event_id == Event.id)
            .join(RawEvent, Event.raw_event_id == RawEvent.id)
            .join(ShotFeature, ShotFeature.shot_id == Shot.id)
            .where(ShotFeature.version_tag == feature_version)
        )

        rows = session.execute(stmt).all()
        if not rows:
            return pd.DataFrame()

        records: List[Dict[str, Any]] = []
        key_pass_ids: set[str] = set()

        for row in rows:
            shot_json = row.shot_raw_json or {}
            shot_block = shot_json.get("shot") or {}
            play_pattern = (shot_json.get("play_pattern") or {}).get("name")
            raw_key_pass_id = shot_block.get("key_pass_id")
            key_pass_id = str(raw_key_pass_id) if raw_key_pass_id else None
            if key_pass_id:
                key_pass_ids.add(key_pass_id)

            raw_player = (shot_json.get("player") or {}).get("id")
            player_id = raw_player if raw_player is not None else row.player_id
            player_id = int(player_id) if player_id is not None else None

            outcome = (row.outcome or "").lower()
            is_goal = outcome == "goal"

            records.append(
                {
                    "shot_id": row.shot_id,
                    "is_goal": is_goal,
                    "statsbomb_xg": float(row.statsbomb_xg) if row.statsbomb_xg is not None else np.nan,
                    "shot_distance": float(row.shot_distance) if row.shot_distance is not None else np.nan,
                    "shot_angle": float(row.shot_angle) if row.shot_angle is not None else np.nan,
                    "location_x": float(row.location_x) if row.location_x is not None else np.nan,
                    "location_y": float(row.location_y) if row.location_y is not None else np.nan,
                    "is_blocked": bool(row.is_blocked),
                    "first_time": bool(row.first_time),
                    "under_pressure": bool(row.under_pressure),
                    "play_pattern": play_pattern,
                    "key_pass_id": key_pass_id,
                    "shot_body_part": row.body_part or "Unknown",
                    "player_id": player_id,
                }
            )

        pass_meta = _load_pass_metadata(session, key_pass_ids)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    if pass_meta:
        pass_records = [
            {"key_pass_id": pid, **meta}
            for pid, meta in pass_meta.items()
        ]
        pass_df = pd.DataFrame(pass_records)
        df = df.merge(pass_df, how="left", on="key_pass_id")
    pass_value_cols = [
        "pass_height",
        "pass_type",
        "pass_body_part",
        "pass_length",
        "pass_angle",
        "pass_end_x",
        "pass_end_y",
    ]
    for col in pass_value_cols:
        if col not in df.columns:
            df[col] = np.nan

    bool_cols = ["pass_is_cross", "pass_is_through_ball", "pass_cut_back"]
    for col in bool_cols:
        if col not in df.columns:
            df[col] = False
        else:
            df[col] = pd.Series(
                df[col].to_numpy(dtype=bool, na_value=False),
                index=df.index,
            )

    df["assist_category"] = df.apply(_categorize_assist, axis=1)
    df["pressure_state"] = np.where(
        df["under_pressure"], "Under pressure", "Not under pressure"
    )
    foot_pref = _infer_player_strong_foot(df)
    df["foot_finish_group"] = df.apply(
        lambda row: _categorize_foot_finish(row, foot_pref),
        axis=1,
    )
    df["recipient_zone"] = df.apply(_categorize_recipient_zone, axis=1)
    return df


def _load_pass_metadata(session, pass_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    ids = [pid for pid in pass_ids if pid]
    if not ids:
        return metadata

    chunk_size = 800
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        rows = (
            session.query(RawEvent.statsbomb_event_id, RawEvent.raw_json)
            .filter(RawEvent.type == "Pass")
            .filter(RawEvent.statsbomb_event_id.in_(chunk))
            .all()
        )
        for statsbomb_id, raw_json in rows:
            pass_block = raw_json.get("pass") or {}
            end_loc = pass_block.get("end_location") or [None, None]
            metadata[str(statsbomb_id)] = {
                "pass_height": (pass_block.get("height") or {}).get("name"),
                "pass_type": (pass_block.get("type") or {}).get("name"),
                "pass_body_part": (pass_block.get("body_part") or {}).get("name"),
                "pass_length": pass_block.get("length"),
                "pass_angle": pass_block.get("angle"),
                "pass_is_cross": bool(pass_block.get("cross", False)),
                "pass_is_through_ball": bool(pass_block.get("through_ball", False)),
                "pass_cut_back": bool(pass_block.get("cut_back", False)),
                "pass_end_x": end_loc[0] if len(end_loc) > 0 else None,
                "pass_end_y": end_loc[1] if len(end_loc) > 1 else None,
            }
    return metadata


def _categorize_assist(row: pd.Series) -> str:
    pattern = (row.get("play_pattern") or "").strip()
    key_pass_id = row.get("key_pass_id")
    if pattern in SET_PIECE_PATTERNS:
        return "Set Piece"

    if key_pass_id is None:
        if pattern in COUNTER_PATTERNS:
            return "Counter (solo)"
        return "Unassisted"

    if pattern in COUNTER_PATTERNS and row.get("pass_is_through_ball"):
        return "Counter Through Ball"
    if pattern in COUNTER_PATTERNS:
        return "Counter Attack"

    if row.get("pass_cut_back"):
        return "Cutback"
    if row.get("pass_is_cross"):
        return "Cross"
    if row.get("pass_is_through_ball"):
        return "Through Ball"

    height = row.get("pass_height") or ""
    if height == "High Pass":
        return "High Pass"
    if height in {"Low Pass", "Ground Pass"}:
        return "Ground Pass"

    if pd.isna(row.get("pass_height")) and not (
        row.get("pass_is_cross") or row.get("pass_is_through_ball")
    ):
        return "Assist (unspecified)"

    return "Other Pass"


def _infer_player_strong_foot(df: pd.DataFrame) -> Dict[int, str]:
    mask = (
        df["shot_body_part"].isin({"Left Foot", "Right Foot"})
        & df["player_id"].notna()
    )
    foot_df = df[mask].copy()
    if foot_df.empty:
        return {}

    counts = (
        foot_df.groupby(["player_id", "shot_body_part"], observed=False)
        .size()
        .unstack(fill_value=0)
    )

    preferences: Dict[int, str] = {}
    for player_id, row in counts.iterrows():
        left = row.get("Left Foot", 0)
        right = row.get("Right Foot", 0)
        if left > right:
            preferences[int(player_id)] = "Left Foot"
        elif right > left:
            preferences[int(player_id)] = "Right Foot"
    return preferences


def _categorize_foot_finish(row: pd.Series, preferences: Dict[int, str]) -> str:
    body_part = (row.get("shot_body_part") or "Unknown").strip()
    if body_part in {"Left Foot", "Right Foot"}:
        player_id = row.get("player_id")
        pref = preferences.get(int(player_id)) if player_id is not None and not pd.isna(player_id) else None
        if pref is None:
            return "Footed shot (unknown preference)"
        return "Strong Foot" if body_part == pref else "Weak Foot"

    lower = body_part.lower()
    if "head" in lower:
        return "Header"
    if body_part == "Unknown" or not body_part:
        return "Unknown body part"
    return body_part


def _categorize_recipient_zone(row: pd.Series) -> str:
    x = row.get("pass_end_x")
    y = row.get("pass_end_y")
    if pd.isna(x) or pd.isna(y):
        x = row.get("location_x")
        y = row.get("location_y")
    if pd.isna(x) or pd.isna(y):
        return "Unknown zone"

    if x >= settings.pitch_length - 18 and (
        (settings.pitch_width / 2) - 10.5 <= y <= (settings.pitch_width / 2) + 10.5
    ):
        return "Penalty area"

    if x >= 80:
        third = "Final third"
    elif x >= 40:
        third = "Middle third"
    else:
        third = "Build-up third"

    lane_width = settings.pitch_width / 3
    if y < lane_width:
        lane = "Left"
    elif y > 2 * lane_width:
        lane = "Right"
    else:
        lane = "Central"

    return f"{third} - {lane}"


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("assist_category", observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
            avg_distance=("shot_distance", "mean"),
            avg_angle=("shot_angle", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    return summary.sort_values("shots", ascending=False)


def summarize_with_split(df: pd.DataFrame, split_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(["assist_category", split_col], observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    return summary.sort_values(["assist_category", split_col])


def export_summary(summary: pd.DataFrame, filename: str = "assist_context_summary.csv") -> None:
    out_dir = get_analysis_output_dir("cxg", subdir="csv")
    summary.to_csv(out_dir / filename, index=False)


def export_split_summaries(df: pd.DataFrame) -> None:
    export_summary(
        summarize_with_split(df, "pressure_state"),
        filename="assist_context_by_pressure.csv",
    )
    export_summary(
        summarize_with_split(df, "foot_finish_group"),
        filename="assist_context_by_footedness.csv",
    )
    export_summary(
        summarize_with_split(df, "recipient_zone"),
        filename="assist_context_by_recipient_zone.csv",
    )


def plot_category_goal_rates(
    summary: pd.DataFrame,
    *,
    name_suffix: str = "",
    title: str | None = None,
) -> None:
    plot_df = summary[summary["shots"] >= CATEGORY_MIN_SHOTS]
    if plot_df.empty:
        return

    fig, ax = plt.subplots()
    indices = np.arange(len(plot_df))
    bar_width = 0.4

    ax.bar(
        indices - bar_width / 2,
        plot_df["goal_rate"],
        width=bar_width,
        color=STYLE.primary_color,
        label="Empirical goal rate",
    )
    ax.bar(
        indices + bar_width / 2,
        plot_df["mean_xg"],
        width=bar_width,
        color=STYLE.secondary_color,
        label="Mean StatsBomb xG",
    )

    ax.set_xticks(indices)
    ax.set_xticklabels(plot_df["assist_category"], rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title(title or "Goal rate vs StatsBomb xG by assist context")
    ax.legend()

    suffix = f"{name_suffix}" if name_suffix else ""
    save_figure(fig, analysis_type="cxg", name=f"assist_context_goal_rates{suffix}")


def plot_distance_curves(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ordered_categories = summary["assist_category"].tolist()
    selected = []
    for cat in ordered_categories:
        shots = df[df["assist_category"] == cat]
        if len(shots) >= CURVE_MIN_SHOTS:
            selected.append(cat)
        if len(selected) >= 4:
            break

    if not selected:
        return

    fig, ax = plt.subplots()
    for cat in selected:
        subset = df[df["assist_category"] == cat]
        agg = bin_and_aggregate_distance(subset.copy())
        ax.plot(
            agg["distance_mid"],
            agg["goal_rate"],
            marker="o",
            label=cat,
        )

    ax.set_xlabel("Shot distance (StatsBomb units)")
    ax.set_ylabel("Goal rate")
    ax.set_title("Goal rate vs distance by assist type")
    ax.legend()

    save_figure(fig, analysis_type="cxg", name="assist_context_distance_curves")


def plot_pitch_heatmaps(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    ordered = summary.sort_values("shots", ascending=False)
    chosen: List[str] = []
    for _, row in ordered.iterrows():
        cat = row["assist_category"]
        if row["shots"] < HEATMAP_MIN_SHOTS:
            continue
        if cat in {"Unassisted", "Assist (unspecified)"}:
            continue
        chosen.append(cat)
        if len(chosen) >= HEATMAP_MAX_PANELS:
            break

    if not chosen:
        return

    for cat in chosen:
        subset = df[df["assist_category"] == cat]
        if subset.empty:
            continue
        mask = subset["location_x"].notna() & subset["location_y"].notna()
        filtered = subset[mask]
        if filtered.empty:
            continue

        coords_x = filtered["location_x"].to_numpy()
        coords_y = filtered["location_y"].to_numpy()
        goal_values = filtered["is_goal"].astype(float).to_numpy()
        xg_values = filtered["statsbomb_xg"].to_numpy()

        shots_stat = pitch.bin_statistic(
            coords_x,
            coords_y,
            statistic="count",
            bins=PITCH_BINS,
        )
        goals_stat = pitch.bin_statistic(
            coords_x,
            coords_y,
            statistic="sum",
            bins=PITCH_BINS,
            values=goal_values,
        )
        goal_rate_grid = np.divide(
            goals_stat["statistic"],
            shots_stat["statistic"],
            out=np.zeros_like(goals_stat["statistic"]),
            where=shots_stat["statistic"] > 0,
        )
        goal_rate_stat = shots_stat.copy()
        goal_rate_stat["statistic"] = goal_rate_grid

        mean_xg_stat = pitch.bin_statistic(
            coords_x,
            coords_y,
            statistic="mean",
            bins=PITCH_BINS,
            values=xg_values,
        )

        fig, axes = plt.subplots(1, 2, figsize=(STYLE.fig_width * 1.6, STYLE.fig_height))
        cmap_pairs = [
            (goal_rate_stat, "Goal rate", "OrRd"),
            (mean_xg_stat, "Mean StatsBomb xG", "PuBu"),
        ]
        for ax, (stat, label, cmap_name) in zip(axes, cmap_pairs):
            pitch.draw(ax=ax)
            heat = pitch.heatmap(stat, ax=ax, cmap=cmap_name)
            cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(label)
            ax.set_title(f"{label}")

        fig.suptitle(f"Shot maps – {cat}")
        slug = _slugify(cat)
        save_figure(fig, analysis_type="cxg", name=f"assist_context_pitch_{slug}")


def plot_split_goal_rates(summary: pd.DataFrame, split_col: str, slug: str) -> None:
    if summary.empty:
        return
    split_values = summary[split_col].dropna().unique().tolist()
    for value in split_values:
        sub = summary[summary[split_col] == value]
        if sub["shots"].sum() < CATEGORY_MIN_SHOTS:
            continue
        suffix = f"_{slug}_{_slugify(str(value))}"
        title = f"Goal rate vs xG by assist context ({value})"
        plot_category_goal_rates(sub, name_suffix=suffix, title=title)


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "assist"


def run_assist_context_analysis(feature_version: str = "v1") -> None:
    configure_matplotlib()
    df = load_shot_and_assist_data(feature_version=feature_version)
    if df.empty:
        print("No shot data available for feature version", feature_version)
        return

    summary = summarize_by_category(df)
    export_summary(summary)
    export_split_summaries(df)
    plot_category_goal_rates(summary)
    plot_split_goal_rates(
        summarize_with_split(df, "pressure_state"),
        split_col="pressure_state",
        slug="pressure",
    )
    plot_split_goal_rates(
        summarize_with_split(df, "foot_finish_group"),
        split_col="foot_finish_group",
        slug="footedness",
    )
    plot_distance_curves(df, summary)
    plot_pitch_heatmaps(df, summary)


if __name__ == "__main__":
    run_assist_context_analysis()
