"""CxG exploratory analysis: set-piece lens for finishing."""

from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from sqlalchemy import select

from opponent_adjusted.analysis.plot_utilities import (
    STYLE,
    configure_matplotlib,
    get_analysis_output_dir,
    save_figure,
)
from opponent_adjusted.config import settings
from opponent_adjusted.db.models import Event, RawEvent, Shot, ShotFeature, Team
from opponent_adjusted.db.session import SessionLocal

SET_PIECE_PATTERN_MAP = {
    "From Corner": "Corner",
    "From Free Kick": "Indirect Free Kick",
    "From Throw In": "Throw In",
    "From Goal Kick": "Goal Kick",
    "From Kick Off": "Kick Off",
}
DIRECT_TYPE_MAP = {
    "Penalty": "Penalty",
    "Free Kick": "Direct Free Kick",
}
CATEGORY_ORDER = [
    "Penalty",
    "Direct Free Kick",
    "Indirect Free Kick",
    "Corner",
    "Throw In",
    "Goal Kick",
    "Kick Off",
    "Other Restart",
    "Open Play",
]
CATEGORY_GROUP_MAP = {
    "Penalty": "Direct",
    "Direct Free Kick": "Direct",
    "Indirect Free Kick": "Restart",
    "Corner": "Restart",
    "Throw In": "Restart",
    "Goal Kick": "Restart",
    "Kick Off": "Restart",
    "Other Restart": "Restart",
}
HEATMAP_MAX_PANELS = 3
TEAM_ANNOTATION_LIMIT = 6
MIN_CATEGORY_SHOTS = 15
SCORE_STATE_BUCKETS = [(-99, -2, "Trailing 2+"), (-1, -1, "Trailing 1"), (0, 0, "Level"), (1, 1, "Leading 1"), (2, 99, "Leading 2+")]
SCORE_STATE_ORDER = [bucket[2] for bucket in SCORE_STATE_BUCKETS]
MINUTE_BUCKETS = [0, 15, 30, 45, 60, 75, 90, 121]
MINUTE_LABELS = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"]

pitch = Pitch(
    pitch_type="statsbomb",
    pitch_length=settings.pitch_length,
    pitch_width=settings.pitch_width,
    line_zorder=2,
    linewidth=1.0,
)


def _load_team_lookup() -> Dict[int, str]:
    with SessionLocal() as session:
        rows = session.query(Team.id, Team.name).all()
    return {row.id: row.name for row in rows}


def _assign_score_state(diff: float | int | None) -> str:
    if diff is None or pd.isna(diff):
        return "Unknown"
    diff_int = int(diff)
    for low, high, label in SCORE_STATE_BUCKETS:
        if low <= diff_int <= high:
            return label
    return "Unknown"


def _assign_minute_bucket(minute: float | int | None) -> str:
    if minute is None or pd.isna(minute):
        return "Unknown"
    return pd.cut(
        [minute], bins=MINUTE_BUCKETS, labels=MINUTE_LABELS, right=False, include_lowest=True
    )[0]


def _classify_phase(row: pd.Series) -> str:
    if row.get("set_piece_category") in {"Penalty", "Direct Free Kick"}:
        return "Direct"
    if row.get("set_piece_category") == "Open Play":
        return "Open Play"
    if row.get("key_pass_id"):
        return "First Phase"
    return "Second Phase"


def load_shot_data(feature_version: str = "v1") -> pd.DataFrame:
    with SessionLocal() as session:
        stmt = (
            select(
                Shot.id.label("shot_id"),
                Shot.match_id,
                Shot.team_id,
                Shot.opponent_team_id,
                Shot.statsbomb_xg,
                Shot.outcome,
                Shot.shot_type,
                Shot.technique,
                Shot.body_part,
                Shot.first_time,
                ShotFeature.shot_distance,
                ShotFeature.shot_angle,
                ShotFeature.score_diff_at_shot,
                ShotFeature.is_leading,
                ShotFeature.is_trailing,
                ShotFeature.is_drawing,
                Event.id.label("event_id"),
                Event.minute,
                Event.second,
                Event.location_x,
                Event.location_y,
                Event.under_pressure,
                RawEvent.raw_json.label("raw_json"),
            )
            .join(ShotFeature, ShotFeature.shot_id == Shot.id)
            .join(Event, Event.id == Shot.event_id)
            .join(RawEvent, RawEvent.id == Event.raw_event_id)
            .where(ShotFeature.version_tag == feature_version)
        )

        rows = session.execute(stmt).all()

    records: List[Dict[str, object]] = []
    for row in rows:
        raw_json = row.raw_json or {}
        play_pattern = (raw_json.get("play_pattern") or {}).get("name")
        shot_block = raw_json.get("shot") or {}
        key_pass_id = shot_block.get("key_pass_id")
        shot_type = row.shot_type or (shot_block.get("type") or {}).get("name")
        is_goal = (row.outcome or "").lower() == "goal"
        records.append(
            {
                "shot_id": row.shot_id,
                "match_id": row.match_id,
                "team_id": row.team_id,
                "opponent_team_id": row.opponent_team_id,
                "statsbomb_xg": float(row.statsbomb_xg)
                if row.statsbomb_xg is not None
                else np.nan,
                "is_goal": is_goal,
                "shot_distance": float(row.shot_distance)
                if row.shot_distance is not None
                else np.nan,
                "shot_angle": float(row.shot_angle)
                if row.shot_angle is not None
                else np.nan,
                "location_x": float(row.location_x)
                if row.location_x is not None
                else np.nan,
                "location_y": float(row.location_y)
                if row.location_y is not None
                else np.nan,
                "minute": row.minute,
                "second": row.second,
                "under_pressure": bool(row.under_pressure),
                "shot_type": shot_type,
                "technique": row.technique,
                "body_part": row.body_part,
                "play_pattern": play_pattern,
                "key_pass_id": str(key_pass_id) if key_pass_id else None,
                "score_diff_at_shot": row.score_diff_at_shot,
                "is_leading": row.is_leading,
                "is_trailing": row.is_trailing,
                "is_drawing": row.is_drawing,
            }
        )

    return pd.DataFrame(records)


def _classify_set_piece(row: pd.Series) -> str:
    shot_type = (row.get("shot_type") or "").strip()
    pattern = (row.get("play_pattern") or "").strip()
    category = DIRECT_TYPE_MAP.get(shot_type)
    if category:
        return category
    category = SET_PIECE_PATTERN_MAP.get(pattern)
    if category:
        return category
    if pattern and pattern.startswith("From "):
        return "Other Restart"
    return "Open Play"


def prepare_set_piece_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["set_piece_category"] = enriched.apply(_classify_set_piece, axis=1)
    enriched["is_set_piece"] = enriched["set_piece_category"] != "Open Play"
    enriched["set_piece_group"] = enriched["set_piece_category"].map(CATEGORY_GROUP_MAP).fillna(
        "Open Play"
    )
    enriched["set_piece_category"] = pd.Categorical(
        enriched["set_piece_category"], categories=CATEGORY_ORDER, ordered=True
    )
    team_lookup = _load_team_lookup()
    enriched["team_name"] = enriched["team_id"].map(team_lookup)
    enriched["opponent_name"] = enriched["opponent_team_id"].map(team_lookup)
    missing_mask = enriched["team_name"].isna() & enriched["team_id"].notna()
    enriched.loc[missing_mask, "team_name"] = (
        "Team " + enriched.loc[missing_mask, "team_id"].astype(str)
    )
    opp_missing = enriched["opponent_name"].isna() & enriched["opponent_team_id"].notna()
    enriched.loc[opp_missing, "opponent_name"] = (
        "Team " + enriched.loc[opp_missing, "opponent_team_id"].astype(str)
    )
    enriched["score_state"] = enriched["score_diff_at_shot"].apply(_assign_score_state)
    enriched["minute_bucket"] = enriched["minute"].apply(_assign_minute_bucket)
    enriched.loc[
        enriched["minute_bucket"].isna(), "minute_bucket"
    ] = "Unknown"
    enriched["score_state"] = pd.Categorical(
        enriched["score_state"], categories=SCORE_STATE_ORDER + ["Unknown"], ordered=True
    )
    enriched["minute_bucket"] = pd.Categorical(
        enriched["minute_bucket"], categories=MINUTE_LABELS + ["Unknown"], ordered=True
    )
    enriched["set_piece_phase"] = enriched.apply(_classify_phase, axis=1)
    return enriched


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("set_piece_category", observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["set_piece_category"] != "Open Play"]
    summary = summary.sort_values("shots", ascending=False)
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["mean_xg"].fillna(0)
    return summary


def summarize_team_categories(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["set_piece_category"] != "Open Play"]
    summary = (
        subset.groupby(["team_id", "team_name", "set_piece_category"], observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["mean_xg"].fillna(0)
    return summary


def summarize_team_totals(df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        df.groupby(["team_id", "team_name"], observed=False)
        .agg(total_shots=("shot_id", "count"), total_goals=("is_goal", "sum"))
        .reset_index()
    )
    set_pieces = (
        df[df["is_set_piece"]]
        .groupby(["team_id", "team_name"], observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    merged = set_pieces.merge(totals, on=["team_id", "team_name"], how="left")
    merged["goal_rate"] = merged["goals"] / merged["shots"].clip(lower=1)
    merged["overall_goal_rate"] = merged["total_goals"] / merged["total_shots"].clip(lower=1)
    merged["set_piece_share"] = merged["shots"] / merged["total_shots"].clip(lower=1)
    merged["lift_vs_team_avg"] = merged["goal_rate"] - merged["overall_goal_rate"]
    return merged.sort_values("shots", ascending=False)


def summarize_category_score_state(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[(df["set_piece_category"] != "Open Play") & (df["score_state"] != "Unknown")]
    if subset.empty:
        return pd.DataFrame()
    summary = (
        subset.groupby(["set_piece_category", "score_state"], observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["mean_xg"].fillna(0)
    return summary


def summarize_category_phase(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["set_piece_category"].isin(CATEGORY_ORDER) & (df["set_piece_category"] != "Open Play")]
    if subset.empty:
        return pd.DataFrame()
    summary = (
        subset.groupby(["set_piece_category", "set_piece_phase"], observed=False)
        .agg(shots=("shot_id", "count"), goals=("is_goal", "sum"))
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    total = summary.groupby("set_piece_category", observed=False)["shots"].transform("sum")
    summary["shot_share"] = summary["shots"] / total.clip(lower=1)
    return summary


def summarize_opponent_set_pieces(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["set_piece_category"] != "Open Play"]
    if subset.empty:
        return pd.DataFrame()
    summary = (
        subset.groupby(["opponent_team_id", "opponent_name"], observed=False)
        .agg(
            shots_conceded=("shot_id", "count"),
            goals_conceded=("is_goal", "sum"),
            mean_xg_allowed=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals_conceded"] / summary["shots_conceded"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["mean_xg_allowed"].fillna(0)
    return summary.sort_values("shots_conceded", ascending=False)


def summarize_category_minute(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[(df["set_piece_category"] != "Open Play") & (df["minute_bucket"] != "Unknown")]
    if subset.empty:
        return pd.DataFrame()
    summary = (
        subset.groupby(["set_piece_category", "minute_bucket"], observed=False)
        .agg(shots=("shot_id", "count"), goals=("is_goal", "sum"))
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    return summary


def export_tables(
    category_summary: pd.DataFrame,
    team_category_summary: pd.DataFrame,
    team_totals: pd.DataFrame,
    category_score_summary: pd.DataFrame,
    category_minute_summary: pd.DataFrame,
    phase_summary: pd.DataFrame,
    opponent_summary: pd.DataFrame,
) -> None:
    out_dir = get_analysis_output_dir("cxg", subdir="csv")
    category_summary.to_csv(out_dir / "set_piece_category_summary.csv", index=False)
    team_category_summary.to_csv(out_dir / "set_piece_team_category_summary.csv", index=False)
    team_totals.to_csv(out_dir / "set_piece_team_totals.csv", index=False)
    if not category_score_summary.empty:
        category_score_summary.to_csv(out_dir / "set_piece_category_score_summary.csv", index=False)
    if not category_minute_summary.empty:
        category_minute_summary.to_csv(out_dir / "set_piece_category_minute_summary.csv", index=False)
    if not phase_summary.empty:
        phase_summary.to_csv(out_dir / "set_piece_phase_summary.csv", index=False)
    if not opponent_summary.empty:
        opponent_summary.to_csv(out_dir / "set_piece_opponent_summary.csv", index=False)


def plot_category_goal_rates(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.1, STYLE.fig_height))
    idx = np.arange(len(summary))
    width = 0.35
    ax.bar(idx - width / 2, summary["goal_rate"], width=width, label="Goal rate")
    ax.bar(idx + width / 2, summary["mean_xg"], width=width, label="Mean StatsBomb xG")
    ax.set_xticks(idx)
    ax.set_xticklabels(summary["set_piece_category"].astype(str), rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Set-piece goal rate vs xG")
    ax.legend()
    save_figure(fig, analysis_type="cxg", name="set_piece_goal_rates")


def plot_category_volume(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
    ax.barh(summary["set_piece_category"].astype(str), summary["shots"], color=STYLE.primary_color)
    ax.set_xlabel("Shots")
    ax.set_title("Shot volume by set-piece type")
    save_figure(fig, analysis_type="cxg", name="set_piece_shot_volume")


def _slugify(value: str) -> str:
    return (
        value.lower()
        .replace("+", " plus ")
        .replace("/", "-")
        .replace(" ", "-")
    )


def plot_set_piece_heatmaps(df: pd.DataFrame) -> None:
    counts = (
        df[df["set_piece_category"] != "Open Play"]
        .groupby("set_piece_category", observed=False)
        ["shot_id"]
        .count()
        .sort_values(ascending=False)
    )
    top_categories = counts.head(HEATMAP_MAX_PANELS).index
    for category in top_categories:
        subset = df[(df["set_piece_category"] == category) & df["location_x"].notna() & df["location_y"].notna()]
        if subset.empty or counts.loc[category] < MIN_CATEGORY_SHOTS:
            continue
        stat = pitch.bin_statistic(
            subset["location_x"].to_numpy(),
            subset["location_y"].to_numpy(),
            statistic="count",
            bins=(30, 20),
        )
        if np.nanmax(stat["statistic"]) == 0:
            continue
        normalized = stat.copy()
        normalized["statistic"] = stat["statistic"] / np.nanmax(stat["statistic"])
        fig, ax = plt.subplots(figsize=(STYLE.fig_width, STYLE.fig_height))
        pitch.draw(ax=ax)
        heat = pitch.heatmap(normalized, ax=ax, cmap="magma", edgecolors=None)
        cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative shot density")
        ax.set_title(f"Shot map – {category}")
        save_figure(fig, analysis_type="cxg", name=f"set_piece_pitch_{_slugify(category)}")


def plot_team_share_vs_goal_rate(team_totals: pd.DataFrame) -> None:
    if team_totals.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    ax.scatter(
        team_totals["set_piece_share"],
        team_totals["goal_rate"],
        color=STYLE.primary_color,
        alpha=0.75,
    )
    ax.set_xlabel("Share of shots from set pieces")
    ax.set_ylabel("Set-piece goal rate")
    ax.set_title("Team reliance & efficiency from set pieces")
    mean_share = team_totals["set_piece_share"].mean()
    mean_goal_rate = team_totals["goal_rate"].mean()
    ax.axvline(mean_share, color=STYLE.grid_color, linestyle="--", linewidth=0.8)
    ax.axhline(mean_goal_rate, color=STYLE.grid_color, linestyle="--", linewidth=0.8)
    annotate_candidates = team_totals.sort_values("shots", ascending=False).head(TEAM_ANNOTATION_LIMIT)
    for _, row in annotate_candidates.iterrows():
        ax.annotate(
            row["team_name"],
            (row["set_piece_share"], row["goal_rate"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=STYLE.font_size_small,
        )
    save_figure(fig, analysis_type="cxg", name="set_piece_team_scatter")


def plot_category_score_heatmap(category_score_summary: pd.DataFrame) -> None:
    if category_score_summary.empty:
        return
    pivot = category_score_summary.pivot(
        index="set_piece_category", columns="score_state", values="goal_rate"
    )
    valid_states = [state for state in SCORE_STATE_ORDER if state in pivot.columns]
    pivot = pivot.reindex(columns=valid_states)
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    im = ax.imshow(pivot.values, cmap="magma", aspect="auto", vmin=0, vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_title("Set-piece goal rate by score state")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Goal rate")
    save_figure(fig, analysis_type="cxg", name="set_piece_score_heatmap")


def plot_category_minute_heatmap(category_minute_summary: pd.DataFrame) -> None:
    if category_minute_summary.empty:
        return
    pivot = category_minute_summary.pivot(
        index="set_piece_category", columns="minute_bucket", values="goal_rate"
    )
    valid_minutes = [label for label in MINUTE_LABELS if label in pivot.columns]
    pivot = pivot.reindex(columns=valid_minutes)
    pivot = pivot.dropna(how="all")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_title("Set-piece goal rate by minute bucket")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Goal rate")
    save_figure(fig, analysis_type="cxg", name="set_piece_minute_heatmap")


def plot_phase_distribution(phase_summary: pd.DataFrame) -> None:
    if phase_summary.empty:
        return
    pivot = phase_summary.pivot(
        index="set_piece_category", columns="set_piece_phase", values="shot_share"
    ).fillna(0)
    pivot = pivot.reindex([cat for cat in CATEGORY_ORDER if cat in pivot.index])
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    bottom = np.zeros(len(pivot))
    for phase in ["Direct", "First Phase", "Second Phase"]:
        if phase not in pivot.columns:
            continue
        ax.barh(
            pivot.index.astype(str),
            pivot[phase].values,
            left=bottom,
            label=phase,
        )
        bottom += pivot[phase].values
    ax.set_xlabel("Share of set-piece shots")
    ax.set_title("Phase mix by set-piece type")
    ax.legend()
    save_figure(fig, analysis_type="cxg", name="set_piece_phase_distribution")


def plot_opponent_concession_scatter(opponent_summary: pd.DataFrame) -> None:
    if opponent_summary.empty:
        return
    top = opponent_summary.head(TEAM_ANNOTATION_LIMIT)
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    ax.scatter(
        opponent_summary["shots_conceded"],
        opponent_summary["goal_rate"],
        color=STYLE.secondary_color,
        alpha=0.6,
    )
    ax.set_xlabel("Set-piece shots conceded")
    ax.set_ylabel("Goal rate conceded")
    ax.set_title("Defensive outlook vs set pieces")
    for _, row in top.iterrows():
        ax.annotate(
            row["opponent_name"],
            (row["shots_conceded"], row["goal_rate"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=STYLE.font_size_small,
        )
    save_figure(fig, analysis_type="cxg", name="set_piece_opponent_scatter")


def run_set_piece_analysis(feature_version: str = "v1") -> None:
    configure_matplotlib()
    shots = load_shot_data(feature_version=feature_version)
    if shots.empty:
        print("No shot data available for feature version", feature_version)
        return

    enriched = prepare_set_piece_features(shots)
    set_piece_df = enriched[enriched["is_set_piece"]]
    if set_piece_df.empty:
        print("No set-piece shots found for feature version", feature_version)
        return

    category_summary = summarize_by_category(enriched)
    team_category_summary = summarize_team_categories(enriched)
    team_totals = summarize_team_totals(enriched)
    category_score_summary = summarize_category_score_state(enriched)
    category_minute_summary = summarize_category_minute(enriched)
    phase_summary = summarize_category_phase(enriched)
    opponent_summary = summarize_opponent_set_pieces(enriched)
    export_tables(
        category_summary,
        team_category_summary,
        team_totals,
        category_score_summary,
        category_minute_summary,
        phase_summary,
        opponent_summary,
    )
    plot_category_goal_rates(category_summary)
    plot_category_volume(category_summary)
    plot_set_piece_heatmaps(enriched)
    plot_team_share_vs_goal_rate(team_totals)
    plot_category_score_heatmap(category_score_summary)
    plot_category_minute_heatmap(category_minute_summary)
    plot_phase_distribution(phase_summary)
    plot_opponent_concession_scatter(opponent_summary)


if __name__ == "__main__":
    run_set_piece_analysis()
