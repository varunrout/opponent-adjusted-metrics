"""CxG exploratory analysis: game-state lens for pass value chains."""

from __future__ import annotations

from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opponent_adjusted.analysis.cxg_analysis import pass_value_chain as pvc
from opponent_adjusted.analysis.plot_utilities import (
    STYLE,
    configure_matplotlib,
    get_analysis_output_dir,
    save_figure,
)
from opponent_adjusted.db.models import Team
from opponent_adjusted.db.session import SessionLocal

SCORE_STATE_BUCKETS = [
    (-99, -2, "Trailing 2+"),
    (-1, -1, "Trailing 1"),
    (0, 0, "Level"),
    (1, 1, "Leading 1"),
    (2, 99, "Leading 2+"),
]
SCORE_STATE_ORDER = [bucket[2] for bucket in SCORE_STATE_BUCKETS]
MINUTE_BUCKETS = [0, 15, 30, 45, 60, 75, 90, 121]
MINUTE_LABELS = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"]
TOP_CHAIN_COUNT = 6
MIN_STATE_SHOTS = 10


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


def _assign_simple_state(row: pd.Series) -> str:
    if bool(row.get("is_leading")):
        return "Leading"
    if bool(row.get("is_trailing")):
        return "Trailing"
    if bool(row.get("is_drawing")):
        return "Drawing"
    return "Unknown"


def _assign_minute_bucket(minute: float | int | None) -> str:
    if minute is None or pd.isna(minute):
        return "Unknown"
    return pd.cut(
        [minute],
        bins=MINUTE_BUCKETS,
        labels=MINUTE_LABELS,
        right=False,
        include_lowest=True,
    )[0]


def prepare_game_state_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["score_state"] = enriched["score_diff_at_shot"].apply(_assign_score_state)
    enriched["simple_state"] = enriched.apply(_assign_simple_state, axis=1)
    enriched["minute_bucket_label"] = enriched["minute"].apply(_assign_minute_bucket)
    enriched.loc[
        enriched["minute_bucket_label"].isna(), "minute_bucket_label"
    ] = enriched.loc[
        enriched["minute_bucket_label"].isna(), "feature_minute_bucket"
    ].fillna("Unknown")
    enriched["score_state"] = pd.Categorical(
        enriched["score_state"],
        categories=SCORE_STATE_ORDER + ["Unknown"],
        ordered=True,
    )
    enriched["minute_bucket_label"] = pd.Categorical(
        enriched["minute_bucket_label"],
        categories=MINUTE_LABELS + ["Unknown"],
        ordered=True,
    )
    return enriched


def summarize_by_score_state(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["chain_label", "score_state"], observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["mean_xg"].fillna(0)
    state_totals = summary.groupby("score_state", observed=False)["shots"].transform("sum")
    summary["state_share"] = summary["shots"] / state_totals.clip(lower=1)
    chain_totals = summary.groupby("chain_label", observed=False)["shots"].transform("sum")
    summary["chain_share"] = summary["shots"] / chain_totals.clip(lower=1)
    return summary


def summarize_by_minute_bucket(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["chain_label", "minute_bucket_label"], observed=False)
        .agg(
            shots=("shot_id", "count"),
            goals=("is_goal", "sum"),
            mean_xg=("statsbomb_xg", "mean"),
        )
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    summary["lift_vs_xg"] = summary["goal_rate"] - summary["mean_xg"].fillna(0)
    bucket_totals = summary.groupby("minute_bucket_label", observed=False)["shots"].transform("sum")
    summary["bucket_share"] = summary["shots"] / bucket_totals.clip(lower=1)
    return summary


def export_tables(
    score_summary: pd.DataFrame,
    minute_summary: pd.DataFrame,
    team_summary: pd.DataFrame | None = None,
) -> None:
    out_dir = get_analysis_output_dir("cxg", subdir="csv")
    score_summary.to_csv(out_dir / "game_state_score_summary.csv", index=False)
    minute_summary.to_csv(out_dir / "game_state_minute_summary.csv", index=False)
    if team_summary is not None:
        team_summary.to_csv(out_dir / "game_state_team_summary.csv", index=False)


def _top_chain_labels(summary: pd.DataFrame) -> List[str]:
    totals = (
        summary.groupby("chain_label", observed=False)["shots"].sum().sort_values(ascending=False)
    )
    return totals.head(TOP_CHAIN_COUNT).index.tolist()


def summarize_team_simple_state(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["team_name"].notna()].copy()
    summary = (
        subset.groupby(["team_id", "team_name", "simple_state"], observed=False)
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


def plot_score_state_goal_rates(score_summary: pd.DataFrame) -> None:
    labels = _top_chain_labels(score_summary)
    plot_df = score_summary[score_summary["chain_label"].isin(labels)]
    plot_df = plot_df[plot_df["score_state"] != "Unknown"]
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    for label in labels:
        series = plot_df[plot_df["chain_label"] == label]
        series = series.sort_values("score_state")
        ax.plot(
            series["score_state"].astype(str),
            series["goal_rate"],
            marker="o",
            label=label,
        )
    ax.set_ylabel("Goal rate")
    ax.set_xlabel("Score state")
    ax.set_title("Chain efficiency across score states")
    ax.legend()
    save_figure(fig, analysis_type="cxg", name="game_state_score_goal_rates")


def plot_score_state_heatmap(score_summary: pd.DataFrame) -> None:
    labels = _top_chain_labels(score_summary)
    filtered = score_summary[
        (score_summary["chain_label"].isin(labels))
        & (score_summary["score_state"] != "Unknown")
    ]
    pivot = filtered.pivot(
        index="chain_label", columns="score_state", values="lift_vs_xg"
    )
    if pivot.empty:
        return
    valid_states = [state for state in SCORE_STATE_ORDER if state in pivot.columns]
    pivot = pivot.reindex(index=labels, columns=valid_states)
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    im = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto", vmin=-0.15, vmax=0.15)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Goal-rate lift vs score state")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Goal rate - xG")
    save_figure(fig, analysis_type="cxg", name="game_state_score_heatmap")


def plot_minute_heatmap(minute_summary: pd.DataFrame) -> None:
    labels = _top_chain_labels(minute_summary)
    filtered = minute_summary[
        (minute_summary["chain_label"].isin(labels))
        & (minute_summary["minute_bucket_label"] != "Unknown")
    ]
    pivot = filtered.pivot(
        index="chain_label",
        columns="minute_bucket_label",
        values="goal_rate",
    )
    if pivot.empty:
        return
    minute_columns = [label for label in MINUTE_LABELS if label in pivot.columns]
    pivot = pivot.reindex(index=labels, columns=minute_columns)
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto", vmin=0, vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Goal rate by minute bucket")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Goal rate")
    save_figure(fig, analysis_type="cxg", name="game_state_minute_heatmap")


def plot_state_distribution(score_summary: pd.DataFrame) -> None:
    filtered = score_summary[score_summary["score_state"] != "Unknown"]
    pivot = filtered.pivot_table(
        index="score_state",
        columns="chain_label",
        values="state_share",
        fill_value=0,
        observed=False,
    )
    if pivot.empty:
        return
    pivot = pivot.reindex([state for state in SCORE_STATE_ORDER if state in pivot.index])
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.1, STYLE.fig_height))
    bottom = np.zeros(len(pivot))
    for label in pivot.columns:
        ax.bar(
            pivot.index.astype(str),
            pivot[label].values,
            bottom=bottom,
            label=label,
            width=0.8,
        )
        bottom += pivot[label].values
    ax.set_ylabel("Share of shots in state")
    ax.set_title("Chain mix by score state")
    ax.legend(ncol=2, fontsize=STYLE.font_size_small)
    save_figure(fig, analysis_type="cxg", name="game_state_state_distribution")


def plot_score_minute_grid(df: pd.DataFrame) -> None:
    filtered = df[
        (df["score_state"] != "Unknown") & (df["minute_bucket_label"] != "Unknown")
    ]
    if filtered.empty:
        return
    summary = (
        filtered.groupby(["score_state", "minute_bucket_label"], observed=False)
        .agg(shots=("shot_id", "count"), goals=("is_goal", "sum"))
        .reset_index()
    )
    summary["goal_rate"] = summary["goals"] / summary["shots"].clip(lower=1)
    pivot = summary.pivot(
        index="score_state", columns="minute_bucket_label", values="goal_rate"
    )
    pivot = pivot.reindex([s for s in SCORE_STATE_ORDER if s in pivot.index])
    pivot = pivot.reindex(columns=[m for m in MINUTE_LABELS if m in pivot.columns])
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    im = ax.imshow(pivot.values, cmap="magma", aspect="auto", vmin=0, vmax=np.nanmax(pivot.values))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Goal rate by score + minute state")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Goal rate")
    save_figure(fig, analysis_type="cxg", name="game_state_score_minute_grid")


def plot_team_state_scatter(team_summary: pd.DataFrame) -> None:
    if team_summary.empty:
        return
    pivot = team_summary.pivot(
        index="team_name", columns="simple_state", values="goal_rate"
    )
    required = {"Leading", "Trailing"}
    if not required.issubset(pivot.columns):
        return
    subset = pivot.dropna(subset=["Leading", "Trailing"])
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    ax.scatter(subset["Leading"], subset["Trailing"], alpha=0.7, color=STYLE.primary_color)
    min_val = min(subset.min().min(), 0)
    max_val = subset.max().max()
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=STYLE.grid_color)
    ax.set_xlabel("Goal rate when leading")
    ax.set_ylabel("Goal rate when trailing")
    ax.set_title("Team finishing by game state")
    delta = (subset["Trailing"] - subset["Leading"]).abs().sort_values(ascending=False)
    for team in delta.head(6).index:
        ax.annotate(team, (subset.loc[team, "Leading"], subset.loc[team, "Trailing"]), textcoords="offset points", xytext=(4, 4), fontsize=STYLE.font_size_small)
    save_figure(fig, analysis_type="cxg", name="game_state_team_scatter")


def plot_team_state_delta(team_summary: pd.DataFrame) -> None:
    if team_summary.empty:
        return
    pivot = team_summary.pivot(
        index="team_name", columns="simple_state", values="goal_rate"
    )
    if not {"Leading", "Trailing"}.issubset(pivot.columns):
        return
    pivot = pivot.dropna(subset=["Leading", "Trailing"])
    if pivot.empty:
        return
    pivot["delta"] = pivot["Leading"] - pivot["Trailing"]
    top = pivot.reindex(pivot["delta"].abs().sort_values(ascending=False).head(10).index)
    fig, ax = plt.subplots(figsize=(STYLE.fig_width * 1.2, STYLE.fig_height))
    ax.barh(top.index, top["delta"], color=np.where(top["delta"] >= 0, STYLE.primary_color, STYLE.secondary_color))
    ax.axvline(0, color=STYLE.grid_color, linestyle="--", linewidth=0.8)
    ax.set_xlabel("Goal rate delta (leading - trailing)")
    ax.set_title("Teams: finishing edge leading vs trailing")
    save_figure(fig, analysis_type="cxg", name="game_state_team_delta")


def run_game_state_analysis(feature_version: str = "v1") -> None:
    configure_matplotlib()
    shots = pvc.load_shot_data(feature_version=feature_version)
    if shots.empty:
        print("No shot data available for feature version", feature_version)
        return

    enriched = pvc.classify_chains(shots)
    enriched = prepare_game_state_features(enriched)
    team_lookup = _load_team_lookup()
    enriched["team_name"] = enriched["team_id"].map(team_lookup)
    missing_mask = enriched["team_name"].isna() & enriched["team_id"].notna()
    enriched.loc[missing_mask, "team_name"] = (
        "Team " + enriched.loc[missing_mask, "team_id"].astype(str)
    )
    score_summary = summarize_by_score_state(enriched)
    minute_summary = summarize_by_minute_bucket(enriched)
    team_summary = summarize_team_simple_state(enriched)
    export_tables(score_summary, minute_summary, team_summary)
    plot_score_state_goal_rates(score_summary)
    plot_score_state_heatmap(score_summary)
    plot_state_distribution(score_summary)
    plot_minute_heatmap(minute_summary)
    plot_score_minute_grid(enriched)
    plot_team_state_scatter(team_summary)
    plot_team_state_delta(team_summary)


if __name__ == "__main__":
    run_game_state_analysis()
