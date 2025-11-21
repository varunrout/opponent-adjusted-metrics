"""CxG exploratory analysis: pass value chains (proto CxA).

This module inspects assisted shots, traces their key pass back to the
preceding action (carry/dribble/duel), and measures how different
chains affect finishing. Outputs include a summary CSV and comparison
plots under outputs/analysis/cxg/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

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
from opponent_adjusted.config import settings
from opponent_adjusted.db.models import Event, RawEvent, Shot, ShotFeature
from opponent_adjusted.db.session import SessionLocal

CHAIN_MIN_SHOTS = 30
PRECEDING_TYPES = {"Carry", "Dribble", "Duel"}
TOP_CHAIN_COUNT = 4
PITCH_BINS = (30, 20)
MINUTE_BUCKETS = [0, 15, 30, 45, 60, 75, 90, 120]
OUTCOME_MAP = {
    "Goal": "Goal",
    "Saved": "On Target",
    "Blocked": "Blocked",
    "Off T": "Off Target",
    "Wayward": "Off Target",
    "Post": "On Target",
    "Saved To Post": "On Target",
}
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
    opponent_team_id: int
    statsbomb_xg: float | None
    is_goal: bool
    shot_distance: float | None
    shot_angle: float | None
    key_pass_id: Optional[str]
    shot_event_id: int
    shot_possession: Optional[int]
    location_x: float | None
    location_y: float | None
    under_pressure: bool
    minute: Optional[int]
    shot_body_part: Optional[str]
    shot_outcome: Optional[str]
    score_diff_at_shot: Optional[int]
    is_leading: Optional[bool]
    is_trailing: Optional[bool]
    is_drawing: Optional[bool]
    feature_minute_bucket: Optional[str]


@dataclass
class PassEventRecord:
    statsbomb_event_id: str
    event_id: int
    match_id: int
    team_id: int
    possession: Optional[int]
    raw_json: dict


@dataclass
class SimpleEventRecord:
    event_id: int
    match_id: int
    team_id: int
    possession: Optional[int]
    type: str


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
                Shot.body_part,
                ShotFeature.shot_distance,
                ShotFeature.shot_angle,
                ShotFeature.score_diff_at_shot,
                ShotFeature.is_leading,
                ShotFeature.is_trailing,
                ShotFeature.is_drawing,
                ShotFeature.minute_bucket,
                Event.id.label("event_id"),
                Event.possession,
                Event.location_x,
                Event.location_y,
                Event.under_pressure,
                Event.minute,
                RawEvent.raw_json.label("shot_raw_json"),
            )
            .join(ShotFeature, ShotFeature.shot_id == Shot.id)
            .join(Event, Event.id == Shot.event_id)
            .join(RawEvent, RawEvent.id == Event.raw_event_id)
            .where(ShotFeature.version_tag == feature_version)
        )

        rows = session.execute(stmt).all()

    records: List[ShotRecord] = []
    for row in rows:
        shot_json = row.shot_raw_json or {}
        shot_block = shot_json.get("shot") or {}
        key_pass_id = shot_block.get("key_pass_id")
        outcome = (row.outcome or "").lower()
        is_goal = outcome == "goal"
        records.append(
            ShotRecord(
                shot_id=row.shot_id,
                match_id=row.match_id,
                team_id=row.team_id,
                opponent_team_id=row.opponent_team_id,
                statsbomb_xg=float(row.statsbomb_xg) if row.statsbomb_xg is not None else None,
                is_goal=is_goal,
                shot_distance=float(row.shot_distance) if row.shot_distance is not None else None,
                shot_angle=float(row.shot_angle) if row.shot_angle is not None else None,
                key_pass_id=str(key_pass_id) if key_pass_id else None,
                shot_event_id=row.event_id,
                shot_possession=row.possession,
                location_x=float(row.location_x) if row.location_x is not None else None,
                location_y=float(row.location_y) if row.location_y is not None else None,
                under_pressure=bool(row.under_pressure),
                minute=int(row.minute) if row.minute is not None else None,
                shot_body_part=row.body_part,
                shot_outcome=row.outcome,
                score_diff_at_shot=int(row.score_diff_at_shot)
                if row.score_diff_at_shot is not None
                else None,
                is_leading=bool(row.is_leading) if row.is_leading is not None else None,
                is_trailing=bool(row.is_trailing) if row.is_trailing is not None else None,
                is_drawing=bool(row.is_drawing) if row.is_drawing is not None else None,
                feature_minute_bucket=row.minute_bucket,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in records])
    return df


def _load_pass_events(pass_ids: Iterable[str]) -> pd.DataFrame:
    ids = [pid for pid in pass_ids if pid]
    if not ids:
        return pd.DataFrame()

    with SessionLocal() as session:
        stmt = (
            select(
                Event.id.label("event_id"),
                Event.match_id,
                Event.team_id,
                Event.possession,
                RawEvent.statsbomb_event_id,
                RawEvent.raw_json,
            )
            .join(RawEvent, RawEvent.id == Event.raw_event_id)
            .where(RawEvent.statsbomb_event_id.in_(ids))
        )
        rows = session.execute(stmt).all()

    pass_records: List[PassEventRecord] = []
    for row in rows:
        pass_records.append(
            PassEventRecord(
                statsbomb_event_id=str(row.statsbomb_event_id),
                event_id=row.event_id,
                match_id=row.match_id,
                team_id=row.team_id,
                possession=row.possession,
                raw_json=row.raw_json or {},
            )
        )

    return pd.DataFrame([r.__dict__ for r in pass_records])


def _load_preceding_events(match_ids: Iterable[int]) -> pd.DataFrame:
    match_ids = list({mid for mid in match_ids if mid is not None})
    if not match_ids:
        return pd.DataFrame()

    with SessionLocal() as session:
        stmt = (
            select(
                Event.id.label("event_id"),
                Event.match_id,
                Event.team_id,
                Event.possession,
                Event.type,
            )
            .where(Event.match_id.in_(match_ids))
            .where(Event.type.in_(PRECEDING_TYPES))
        )
        rows = session.execute(stmt).all()

    simple_records = [
        SimpleEventRecord(
            event_id=row.event_id,
            match_id=row.match_id,
            team_id=row.team_id,
            possession=row.possession,
            type=row.type,
        )
        for row in rows
    ]
    return pd.DataFrame([r.__dict__ for r in simple_records])


def _categorize_pass_style(pass_json: dict) -> str:
    block = pass_json.get("pass") or {}
    if not block:
        return "Assist Unknown"
    if block.get("cut_back"):
        return "Cutback"
    if block.get("through_ball"):
        return "Through Ball"
    if block.get("switch"):
        return "Switch"
    if block.get("cross"):
        return "Cross"
    height = (block.get("height") or {}).get("name")
    if height == "High Pass":
        return "High Pass"
    if height in {"Ground Pass", "Low Pass"}:
        return "Ground Pass"
    pass_type = (block.get("type") or {}).get("name")
    if pass_type:
        return pass_type
    return "Other Pass"


def _classify_chain(row: pd.Series) -> str:
    if pd.isna(row.get("key_pass_id")) or not row.get("key_pass_id"):
        return "Unassisted"

    pass_style = row.get("pass_style") or "Assist Unknown"
    prev_type = row.get("prev_type")

    if pd.isna(prev_type) or not prev_type:
        return f"Direct + {pass_style}"

    if prev_type == "Duel":
        return f"Duel Win + {pass_style}"
    return f"{prev_type} + {pass_style}"


def classify_chains(shots: pd.DataFrame) -> pd.DataFrame:
    shots = shots.copy()
    pass_ids = shots["key_pass_id"].dropna().unique().tolist()
    pass_df = _load_pass_events(pass_ids)
    if not pass_df.empty:
        pass_df["pass_style"] = pass_df["raw_json"].apply(_categorize_pass_style)
        pass_df = pass_df.drop(columns=["raw_json"])

    merged = shots.merge(
        pass_df,
        how="left",
        left_on="key_pass_id",
        right_on="statsbomb_event_id",
        suffixes=("", "_pass"),
    )

    merged["chain_match_id"] = pd.to_numeric(
        merged["match_id_pass"].fillna(merged["match_id"]), errors="coerce"
    ).astype("Int64")
    merged["chain_team_id"] = pd.to_numeric(
        merged["team_id_pass"].fillna(merged["team_id"]), errors="coerce"
    ).astype("Int64")
    merged["chain_possession"] = merged["possession"].fillna(merged["shot_possession"])
    merged["chain_possession"] = pd.to_numeric(merged["chain_possession"], errors="coerce").astype("Int64")

    match_ids = merged["chain_match_id"].dropna().unique()
    prev_df = _load_preceding_events(match_ids)

    if not prev_df.empty:
        prev_df = prev_df.rename(
            columns={"match_id": "chain_match_id", "team_id": "chain_team_id"}
        )
        prev_df["chain_match_id"] = pd.to_numeric(
            prev_df["chain_match_id"], errors="coerce"
        ).astype("Int64")
        prev_df["chain_team_id"] = pd.to_numeric(
            prev_df["chain_team_id"], errors="coerce"
        ).astype("Int64")
        prev_df["chain_possession"] = pd.to_numeric(
            prev_df["possession"], errors="coerce"
        ).astype("Int64")
        prev_df = prev_df.dropna(
            subset=["chain_match_id", "chain_team_id", "chain_possession"]
        )
        prev_df["chain_match_id"] = prev_df["chain_match_id"].astype(int)
        prev_df["chain_team_id"] = prev_df["chain_team_id"].astype(int)
        prev_df["chain_possession"] = prev_df["chain_possession"].astype(int)
        prev_df = prev_df.dropna(subset=["event_id"])
        prev_df["event_id"] = prev_df["event_id"].astype(int)
        prev_df = prev_df.sort_values("event_id")
        pass_subset = merged[~merged["event_id"].isna()].copy()
        pass_subset = pass_subset.dropna(
            subset=["chain_match_id", "chain_possession", "chain_team_id"]
        )
        if not pass_subset.empty:
            pass_subset["event_id"] = pass_subset["event_id"].astype(int)
            pass_subset["chain_match_id"] = pass_subset["chain_match_id"].astype(int)
            pass_subset["chain_team_id"] = pass_subset["chain_team_id"].astype(int)
            pass_subset["chain_possession"] = pass_subset["chain_possession"].astype(int)
            pass_subset = pass_subset.sort_values("event_id")
            aligned = pd.merge_asof(
                pass_subset,
                prev_df,
                left_on="event_id",
                right_on="event_id",
                by=["chain_match_id", "chain_possession", "chain_team_id"],
                direction="backward",
                suffixes=("", "_prev"),
            )
            merged.loc[pass_subset.index, "prev_type"] = aligned["type"]
    else:
        merged["prev_type"] = np.nan

    merged["chain_label"] = merged.apply(_classify_chain, axis=1)
    return merged


def summarize_chains(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("chain_label", observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["lift"] = summary["goal_rate"] - summary["mean_xg"].fillna(0)
    summary = summary.sort_values("shots", ascending=False)
    return summary


def export_summary(summary: pd.DataFrame) -> None:
    out_dir = get_analysis_output_dir("cxg", subdir="csv")
    summary.to_csv(out_dir / "pass_value_chain_summary.csv", index=False)


def plot_chain_goal_rates(summary: pd.DataFrame) -> None:
    plot_df = summary[summary["shots"] >= CHAIN_MIN_SHOTS]
    if plot_df.empty:
        return

    fig, ax = plt.subplots()
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
    ax.set_xticklabels(plot_df["chain_label"], rotation=30, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Pass value chain impact")
    ax.legend()

    save_figure(fig, analysis_type="cxg", name="pass_value_chain_goal_rates")


def plot_chain_counts(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.barh(summary["chain_label"], summary["shots"], color=STYLE.primary_color)
    ax.set_xlabel("Shots")
    ax.set_title("Shot volume by pass chain")
    save_figure(fig, analysis_type="cxg", name="pass_value_chain_counts")


def plot_chain_lift_scatter(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    fig, ax = plt.subplots()
    max_shots = summary["shots"].max()
    sizes = np.clip(summary["shots"] / max_shots, 0.2, 1.0) * 400
    colors = [
        STYLE.secondary_color if label == "Unassisted" else STYLE.primary_color
        for label in summary["chain_label"]
    ]
    ax.scatter(summary["shots"], summary["lift"], s=sizes, c=colors, alpha=0.75)
    top_labels = summary.head(7)
    for _, row in top_labels.iterrows():
        ax.annotate(
            row["chain_label"],
            (row["shots"], row["lift"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=STYLE.font_size_small,
        )
    ax.axhline(0, color=STYLE.grid_color, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Shot volume")
    ax.set_ylabel("Goal rate lift vs StatsBomb xG")
    ax.set_title("Chain efficiency vs volume")
    save_figure(fig, analysis_type="cxg", name="pass_value_chain_lift_scatter")


def _top_chain_labels(summary: pd.DataFrame, include_unassisted: bool = True) -> List[str]:
    labels = (
        summary[summary["chain_label"] != "Unassisted"]
        .head(TOP_CHAIN_COUNT)
        .loc[:, "chain_label"]
        .tolist()
    )
    if include_unassisted and "Unassisted" not in labels and "Unassisted" in summary["chain_label"].values:
        labels.append("Unassisted")
    return labels


def _slugify(value: str) -> str:
    return "-".join(value.lower().split()).replace("+", "plus").replace("/", "-")


def plot_chain_pitch_heatmaps(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    if df["location_x"].isna().all() or df["location_y"].isna().all():
        return
    labels = _top_chain_labels(summary)
    for label in labels:
        subset = df[(df["chain_label"] == label) & df["location_x"].notna() & df["location_y"].notna()]
        if subset.empty:
            continue
        stat = pitch.bin_statistic(
            subset["location_x"].to_numpy(),
            subset["location_y"].to_numpy(),
            statistic="count",
            bins=PITCH_BINS,
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
        ax.set_title(f"Shot map – {label}")
        save_figure(fig, analysis_type="cxg", name=f"pass_value_chain_pitch_{_slugify(label)}")


def plot_unassisted_breakdowns(df: pd.DataFrame) -> None:
    unassisted = df[df["chain_label"] == "Unassisted"]
    assisted = df[df["chain_label"] != "Unassisted"]
    if unassisted.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(STYLE.fig_width * 1.5, STYLE.fig_height * 1.4))

    distance_bins = np.linspace(0, 40, 21)
    axes[0, 0].hist(
        unassisted["shot_distance"].dropna(),
        bins=distance_bins,
        density=True,
        alpha=0.7,
        label="Unassisted",
    )
    axes[0, 0].hist(
        assisted["shot_distance"].dropna(),
        bins=distance_bins,
        density=True,
        alpha=0.5,
        label="Assisted",
    )
    axes[0, 0].set_xlabel("Shot distance")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Distance distribution")
    axes[0, 0].legend()

    angle_bins = np.linspace(0, np.pi, 21)
    axes[0, 1].hist(
        unassisted["shot_angle"].dropna(),
        bins=angle_bins,
        density=True,
        alpha=0.7,
        label="Unassisted",
    )
    axes[0, 1].hist(
        assisted["shot_angle"].dropna(),
        bins=angle_bins,
        density=True,
        alpha=0.5,
        label="Assisted",
    )
    axes[0, 1].set_xlabel("Shot angle (radians)")
    axes[0, 1].set_title("Angle distribution")

    body_counts = (
        unassisted["shot_body_part"].fillna("Unknown")
        .value_counts()
        .head(5)
        .sort_values(ascending=False)
    )
    axes[1, 0].bar(body_counts.index, body_counts.values, color=STYLE.primary_color)
    axes[1, 0].set_title("Unassisted body parts")
    axes[1, 0].set_ylabel("Shots")
    axes[1, 0].tick_params(axis="x", rotation=20)

    pressure_counts = unassisted["under_pressure"].value_counts(normalize=True)
    axes[1, 1].bar(
        ["Under pressure", "Not under pressure"],
        [pressure_counts.get(True, 0), pressure_counts.get(False, 0)],
        color=[STYLE.secondary_color, STYLE.primary_color],
    )
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_ylabel("Share of unassisted shots")
    axes[1, 1].set_title("Pressure context")

    fig.suptitle("Unassisted shot deep dive")
    save_figure(fig, analysis_type="cxg", name="pass_value_chain_unassisted_breakdown")


def plot_xg_distribution(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    labels = _top_chain_labels(summary)
    data = []
    final_labels: List[str] = []
    for label in labels:
        values = df.loc[(df["chain_label"] == label) & df["statsbomb_xg"].notna(), "statsbomb_xg"].values
        if len(values) < 5:
            continue
        data.append(values)
        final_labels.append(label)
    if not data:
        return

    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    vp = ax.violinplot(data, showmeans=True, showmedians=False, widths=0.8)
    for body in vp["bodies"]:
        body.set_facecolor(STYLE.primary_color)
        body.set_alpha(0.6)
    vp["cmeans"].set_color(STYLE.secondary_color)
    ax.set_xticks(np.arange(1, len(final_labels) + 1))
    ax.set_xticklabels(final_labels, rotation=30, ha="right")
    ax.set_ylabel("StatsBomb xG")
    ax.set_title("xG distribution by chain")
    save_figure(fig, analysis_type="cxg", name="pass_value_chain_xg_distribution")


def plot_chain_timeline(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    df = df.dropna(subset=["minute"]).copy()
    if df.empty:
        return
    labels = _top_chain_labels(summary)
    df["minute_bucket"] = pd.cut(
        df["minute"],
        bins=MINUTE_BUCKETS,
        right=False,
        include_lowest=True,
        labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"],
    )
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.3, STYLE.fig_height))
    for label in labels:
        subset = df[df["chain_label"] == label]
        if subset.empty:
            continue
        counts = subset.groupby("minute_bucket", observed=False).size()
        counts = counts.reindex(df["minute_bucket"].cat.categories, fill_value=0)
        ax.plot(counts.index, counts.values, marker="o", label=label)
    ax.set_xlabel("Minute bucket")
    ax.set_ylabel("Shot count")
    ax.set_title("Chain usage over match time")
    ax.legend()
    save_figure(fig, analysis_type="cxg", name="pass_value_chain_timeline")


def plot_outcome_mix(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    labels = _top_chain_labels(summary)
    df = df.copy()
    df["outcome_group"] = df["shot_outcome"].map(OUTCOME_MAP).fillna("Other")
    pivot = (
        df.groupby(["chain_label", "outcome_group"], observed=False)
        .size()
        .unstack(fill_value=0)
    )
    pivot = pivot.reindex(labels).dropna(how="all")
    if pivot.empty:
        return
    percent = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    im = ax.imshow(percent.values, aspect="auto", cmap="Blues", vmin=0, vmax=percent.values.max())
    ax.set_xticks(range(percent.shape[1]))
    ax.set_xticklabels(percent.columns, rotation=30, ha="right")
    ax.set_yticks(range(percent.shape[0]))
    ax.set_yticklabels(percent.index)
    ax.set_title("Shot outcome mix by chain")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Share")
    save_figure(fig, analysis_type="cxg", name="pass_value_chain_outcome_mix")


def run_pass_value_chain_analysis(feature_version: str = "v1") -> None:
    configure_matplotlib()
    shots = load_shot_data(feature_version=feature_version)
    if shots.empty:
        print("No shot data available for feature version", feature_version)
        return

    enriched = classify_chains(shots)
    summary = summarize_chains(enriched)
    export_summary(summary)
    plot_chain_goal_rates(summary)
    plot_chain_counts(summary)
    plot_chain_lift_scatter(summary)
    plot_chain_pitch_heatmaps(enriched, summary)
    plot_unassisted_breakdowns(enriched)
    plot_xg_distribution(enriched, summary)
    plot_chain_timeline(enriched, summary)
    plot_outcome_mix(enriched, summary)


if __name__ == "__main__":
    run_pass_value_chain_analysis()
