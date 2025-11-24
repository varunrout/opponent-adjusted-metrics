"""Build context-driven finishing and concession priors without team IDs.

This replaces the previous team-encoded submodel with neutral priors derived
from rolling form, style archetypes, and player-driven finishing history.
The resulting artifacts are keyed by match/side so downstream models never
see a club identifier.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from opponent_adjusted.config import settings
from opponent_adjusted.db.models import Event, Possession
from opponent_adjusted.db.session import SessionLocal
from opponent_adjusted.modeling.modeling_utilities import (
    get_modeling_output_dir,
    load_cxg_modeling_dataset,
)

LOGGER = logging.getLogger(__name__)

ROLLING_WINDOWS: tuple[int, ...] = (3, 5, 8)
SHOTS_PER_MATCH_BASE = 6.0
PLAYER_WINDOW = 35
PLAYER_TOP_N = 3
STYLE_CLUSTER_COUNT = 6
STYLE_MIN_MATCHES = 4
MATCH_MINUTES = 95.0
MAX_LOGIT = 3.0
EPS = 1e-4

INCLUDE_PL1516_IN_PRIORS = os.getenv("CXG_INCLUDE_PL1516_PRIORS", "").lower() in {"1", "true", "yes"}

DEFENSIVE_EVENT_TYPES = ("Pressure", "Duel", "Tackle", "Interception", "Block", "Clearance")
PL_COMP_STATS_BOMB_ID = 2

PITCH_LENGTH = float(getattr(settings, "pitch_length", 120.0) or 120.0)


def _logit(prob: pd.Series | np.ndarray | float) -> pd.Series:
    clipped = np.clip(prob, EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def _safe_ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    denom = denom.replace(0.0, np.nan)
    return num / denom


def _normalize_season(season: str | None) -> str:
    if season is None:
        return ""
    return season.strip().lower().replace(" ", "").replace("-", "/")


def _is_pl1516_row(comp_sb_id: int | float, season: str | None) -> bool:
    if int(comp_sb_id) != PL_COMP_STATS_BOMB_ID:
        return False
    normalized = _normalize_season(season)
    return normalized in {"2015/2016", "2015/16"}


def _chunked(items: Iterable[int], chunk_size: int = 500) -> Iterable[list[int]]:
    buffer: list[int] = []
    for item in items:
        buffer.append(int(item))
        if len(buffer) >= chunk_size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def _fetch_possessions(match_ids: Sequence[int]) -> pd.DataFrame:
    unique_ids = sorted({int(mid) for mid in match_ids if pd.notna(mid)})
    if not unique_ids:
        return pd.DataFrame(columns=["match_id", "team_id", "duration_seconds"])

    rows: list[tuple[int, int, float | None]] = []
    with SessionLocal() as session:
        base_stmt = select(Possession.match_id, Possession.team_id, Possession.duration_seconds)
        for chunk in _chunked(unique_ids):
            stmt = base_stmt.where(Possession.match_id.in_(chunk))
            rows.extend(session.execute(stmt).all())
    return pd.DataFrame(rows, columns=["match_id", "team_id", "duration_seconds"])


def _fetch_defensive_events(match_ids: Sequence[int]) -> pd.DataFrame:
    unique_ids = sorted({int(mid) for mid in match_ids if pd.notna(mid)})
    if not unique_ids:
        return pd.DataFrame(columns=["match_id", "team_id", "type", "location_x"])

    rows: list[tuple[int, int, str, float | None]] = []
    with SessionLocal() as session:
        base_stmt = select(Event.match_id, Event.team_id, Event.type, Event.location_x).where(
            Event.type.in_(DEFENSIVE_EVENT_TYPES)
        )
        for chunk in _chunked(unique_ids):
            stmt = base_stmt.where(Event.match_id.in_(chunk))
            rows.extend(session.execute(stmt).all())
    return pd.DataFrame(rows, columns=["match_id", "team_id", "type", "location_x"])


def _build_match_team_summary(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "match_id",
        "team_id",
        "opponent_team_id",
        "match_date",
        "is_home",
        "competition_id",
        "competition_season",
        "competition_statsbomb_id",
        "statsbomb_xg",
        "is_goal",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CxG dataset missing columns for priors: {sorted(missing)}")

    group_cols = [
        "match_id",
        "team_id",
        "opponent_team_id",
        "match_date",
        "is_home",
        "competition_id",
        "competition_season",
        "competition_statsbomb_id",
    ]

    agg_dict: dict[str, tuple[str, Any]] = {
        "shots_for": ("shot_id", "count"),
        "goals_for": ("is_goal", "sum"),
        "xg_for": ("statsbomb_xg", "sum"),
        "avg_shot_distance": ("shot_distance", "mean"),
    }
    if "pressure_state" in df.columns:
        agg_dict["under_pressure_rate"] = (
            "pressure_state",
            lambda s: float((s == "Under pressure").mean()) if len(s) else np.nan,
        )
    if "set_piece_category" in df.columns:
        agg_dict["set_piece_share"] = (
            "set_piece_category",
            lambda s: float((s.fillna("Open Play") != "Open Play").mean()),
        )

    summary = (
        df.groupby(group_cols, dropna=False, observed=False)
        .agg(**agg_dict)
        .reset_index()
    )

    summary["match_date"] = pd.to_datetime(summary["match_date"], errors="coerce")
    summary["is_home"] = summary["is_home"].fillna(False).astype(bool)
    summary["competition_season"] = summary["competition_season"].fillna("Unknown")
    summary["competition_statsbomb_id"] = summary["competition_statsbomb_id"].fillna(-1).astype(int)
    summary["competition_season_key"] = (
        summary["competition_statsbomb_id"].astype(str)
        + "-"
        + summary["competition_season"].astype(str)
    )

    summary = summary.sort_values(
        ["competition_season_key", "team_id", "match_date", "match_id"], na_position="last"
    )
    summary["match_number"] = summary.groupby(["competition_season_key", "team_id"], observed=False).cumcount()
    summary["is_pl1516"] = summary.apply(
        lambda row: _is_pl1516_row(row["competition_statsbomb_id"], row["competition_season"]), axis=1
    )

    opponent_stats = summary[
        ["match_id", "team_id", "shots_for", "goals_for", "xg_for"]
    ].rename(
        columns={
            "team_id": "opponent_team_id",
            "shots_for": "shots_against",
            "goals_for": "goals_against",
            "xg_for": "xg_against",
        }
    )
    summary = summary.merge(opponent_stats, on=["match_id", "opponent_team_id"], how="left")
    return summary


def _attach_rolling_components(summary: pd.DataFrame) -> pd.DataFrame:
    key = ["competition_season_key", "team_id"]
    ordered = summary.sort_values(["competition_season_key", "team_id", "match_date", "match_id"])

    for window in ROLLING_WINDOWS:
        goals_col = f"att_goals_window_{window}"
        shots_col = f"att_shots_window_{window}"
        xg_col = f"att_xg_window_{window}"
        ordered[goals_col] = ordered.groupby(key, observed=False)["goals_for"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        ordered[shots_col] = ordered.groupby(key, observed=False)["shots_for"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        ordered[xg_col] = ordered.groupby(key, observed=False)["xg_for"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )

        def_goals_col = f"def_goals_window_{window}"
        def_shots_col = f"def_shots_window_{window}"
        def_xg_col = f"def_xg_window_{window}"
        ordered[def_goals_col] = ordered.groupby(key, observed=False)["goals_against"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        ordered[def_shots_col] = ordered.groupby(key, observed=False)["shots_against"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )
        ordered[def_xg_col] = ordered.groupby(key, observed=False)["xg_against"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum()
        )

        ordered[f"att_delta_{window}"] = _logit(
            _safe_ratio(ordered[goals_col], ordered[shots_col]).fillna(0.0)
        ) - _logit(_safe_ratio(ordered[xg_col], ordered[shots_col]).fillna(0.0))
        ordered[f"att_weight_{window}"] = np.clip(
            ordered[shots_col] / (window * SHOTS_PER_MATCH_BASE), 0.0, 1.0
        ).fillna(0.0)

        ordered[f"def_delta_{window}"] = _logit(
            _safe_ratio(ordered[def_goals_col], ordered[def_shots_col]).fillna(0.0)
        ) - _logit(_safe_ratio(ordered[def_xg_col], ordered[def_shots_col]).fillna(0.0))
        ordered[f"def_weight_{window}"] = np.clip(
            ordered[def_shots_col] / (window * SHOTS_PER_MATCH_BASE), 0.0, 1.0
        ).fillna(0.0)

    def _combine(prefix: str) -> pd.DataFrame:
        deltas = [ordered[f"{prefix}_delta_{w}"] for w in ROLLING_WINDOWS]
        weights = [ordered[f"{prefix}_weight_{w}"] for w in ROLLING_WINDOWS]
        weight_sum = sum(weights)
        weighted_delta = sum(delta * weight for delta, weight in zip(deltas, weights, strict=False))
        comp = np.where(weight_sum > 0, weighted_delta / weight_sum, 0.0)
        comp = np.clip(comp, -MAX_LOGIT, MAX_LOGIT)
        reliability = np.clip(weight_sum / len(ROLLING_WINDOWS), 0.0, 1.0)
        return pd.DataFrame(
            {
                f"{prefix}_rolling_component": comp,
                f"{prefix}_rolling_reliability": reliability,
            }
        )

    att_combined = _combine("att")
    def_combined = _combine("def")
    ordered["att_rolling_component"] = att_combined["att_rolling_component"].values
    ordered["att_rolling_reliability"] = att_combined["att_rolling_reliability"].values
    ordered["def_rolling_component"] = def_combined["def_rolling_component"].values
    ordered["def_rolling_reliability"] = def_combined["def_rolling_reliability"].values
    return ordered


def _build_style_features(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    match_ids = summary["match_id"].unique().tolist()
    possession = _fetch_possessions(match_ids)
    events = _fetch_defensive_events(match_ids)

    style = summary[
        [
            "match_id",
            "team_id",
            "competition_season_key",
            "match_number",
            "avg_shot_distance",
            "is_pl1516",
            "goals_for",
            "shots_for",
            "xg_for",
            "goals_against",
            "shots_against",
            "xg_against",
        ]
    ].copy()

    if not possession.empty:
        possession["duration_seconds"] = possession["duration_seconds"].fillna(0.0)
        team_pos = possession.groupby(["match_id", "team_id"], as_index=False)["duration_seconds"].sum()
        match_totals = team_pos.groupby("match_id")["duration_seconds"].transform(
            lambda s: s.sum() if s.sum() > 0 else np.nan
        )
        team_pos["possession_share"] = team_pos["duration_seconds"] / match_totals
        style = style.merge(team_pos[["match_id", "team_id", "possession_share"]], on=["match_id", "team_id"], how="left")
    else:
        style["possession_share"] = np.nan

    if not events.empty:
        pressure = events.loc[events["type"] == "Pressure"].copy()
        pressure_counts = (
            pressure.groupby(["match_id", "team_id"], as_index=False)
            .size()
            .rename(columns={"size": "pressure_events"})
        )
        pressure_loc = (
            pressure.dropna(subset=["location_x"])
            .groupby(["match_id", "team_id"], as_index=False)["location_x"]
            .mean()
            .rename(columns={"location_x": "pressure_location"})
        )
        def_line = (
            events.dropna(subset=["location_x"])
            .groupby(["match_id", "team_id"], as_index=False)["location_x"]
            .mean()
            .rename(columns={"location_x": "def_line_location"})
        )
        style = style.merge(pressure_counts, on=["match_id", "team_id"], how="left")
        style = style.merge(pressure_loc, on=["match_id", "team_id"], how="left")
        style = style.merge(def_line, on=["match_id", "team_id"], how="left")
    else:
        style["pressure_events"] = np.nan
        style["pressure_location"] = np.nan
        style["def_line_location"] = np.nan

    style["press_intensity"] = style["pressure_events"].fillna(0.0) / MATCH_MINUTES
    style["press_height"] = style["pressure_location"].fillna(PITCH_LENGTH / 2.0) / PITCH_LENGTH
    style["def_line_height"] = style["def_line_location"].fillna(PITCH_LENGTH / 2.0) / PITCH_LENGTH

    style["possession_share"] = style["possession_share"].fillna(style["possession_share"].median())
    style["possession_share"] = style["possession_share"].fillna(0.5)
    style["avg_shot_distance"] = style["avg_shot_distance"].fillna(style["avg_shot_distance"].median())
    style["avg_shot_distance"] = style["avg_shot_distance"].fillna(style["avg_shot_distance"].mean()).fillna(18.0)

    feature_cols = ["possession_share", "press_intensity", "avg_shot_distance", "def_line_height", "press_height"]
    style_features = style[feature_cols].fillna(style[feature_cols].median())
    style_features = style_features.fillna(0.0)

    cluster_payload: dict[str, Any] = {"cluster_count": 0, "inertia": None}
    if INCLUDE_PL1516_IN_PRIORS:
        train_mask = pd.Series(True, index=style.index)
    else:
        train_mask = ~style["is_pl1516"].astype(bool)

    if train_mask.sum() < 2:
        style["style_cluster"] = 0
        style["style_att_bias"] = 0.0
        style["style_def_bias"] = 0.0
        style["style_reliability"] = 0.0
        return style, cluster_payload

    cluster_count = min(STYLE_CLUSTER_COUNT, int(train_mask.sum()))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(style_features)
    model = KMeans(n_clusters=cluster_count, n_init="auto", random_state=42)
    model.fit(scaled[train_mask.to_numpy().astype(bool)])
    labels = model.predict(scaled)
    style["style_cluster"] = labels
    cluster_payload = {"cluster_count": cluster_count, "inertia": float(model.inertia_)}

    cluster_stats = (
        style.loc[train_mask]
        .groupby("style_cluster", observed=False)
        .agg(
            att_goals=("goals_for", "sum"),
            att_shots=("shots_for", "sum"),
            att_xg=("xg_for", "sum"),
            def_goals=("goals_against", "sum"),
            def_shots=("shots_against", "sum"),
            def_xg=("xg_against", "sum"),
            match_count=("match_id", "count"),
        )
        .reset_index()
    )
    cluster_stats["style_att_bias"] = _logit(
        _safe_ratio(cluster_stats["att_goals"], cluster_stats["att_shots"]).fillna(0.0)
    ) - _logit(_safe_ratio(cluster_stats["att_xg"], cluster_stats["att_shots"]).fillna(0.0))
    cluster_stats["style_def_bias"] = _logit(
        _safe_ratio(cluster_stats["def_goals"], cluster_stats["def_shots"]).fillna(0.0)
    ) - _logit(_safe_ratio(cluster_stats["def_xg"], cluster_stats["def_shots"]).fillna(0.0))

    style = style.merge(
        cluster_stats[["style_cluster", "style_att_bias", "style_def_bias", "match_count"]],
        on="style_cluster",
        how="left",
    )
    style["style_att_bias"] = style["style_att_bias"].fillna(0.0)
    style["style_def_bias"] = style["style_def_bias"].fillna(0.0)
    style["style_reliability"] = np.clip(style["match_number"] / STYLE_MIN_MATCHES, 0.0, 1.0)
    return style, cluster_payload


def _mean_top_k(series: pd.Series, k: int) -> float:
    valid = series.dropna().sort_values(ascending=False)
    if valid.empty:
        return 0.0
    return float(valid.head(k).mean())


def _build_player_team_features(df: pd.DataFrame) -> pd.DataFrame:
    if "player_id" not in df.columns:
        return pd.DataFrame(columns=["match_id", "team_id", "player_component", "player_reliability"])

    base = df[["shot_id", "match_id", "team_id", "player_id", "statsbomb_xg", "is_goal", "match_date"]].copy()
    base = base.dropna(subset=["player_id"])
    if base.empty:
        return pd.DataFrame(columns=["match_id", "team_id", "player_component", "player_reliability"])

    base["match_date"] = pd.to_datetime(base["match_date"], errors="coerce")
    base = base.sort_values(["player_id", "match_date", "match_id", "shot_id"])

    global_goal = float(df["is_goal"].mean())
    global_xg = float(df["statsbomb_xg"].mean())

    def _rolling(series: pd.Series) -> pd.Series:
        return series.shift(1).rolling(PLAYER_WINDOW, min_periods=1)

    grouped = base.groupby("player_id", group_keys=False, observed=False)
    base["player_goal_rate"] = grouped["is_goal"].apply(lambda s: _rolling(s).mean()).fillna(global_goal)
    base["player_xg_rate"] = grouped["statsbomb_xg"].apply(lambda s: _rolling(s).mean()).fillna(global_xg)
    base["player_shot_volume"] = grouped["shot_id"].apply(lambda s: _rolling(s).count()).fillna(0.0)
    base["player_lift"] = base["player_goal_rate"] - base["player_xg_rate"]

    snapshot = base.sort_values(["match_id", "team_id", "player_id", "shot_id"]) \
        .drop_duplicates(["match_id", "team_id", "player_id"], keep="first")

    team_level = (
        snapshot.groupby(["match_id", "team_id"], observed=False)
        .agg(
            player_bias_top3=("player_lift", lambda s: _mean_top_k(s, PLAYER_TOP_N)),
            player_bias_mean=("player_lift", "mean"),
            player_volume=("player_shot_volume", "sum"),
        )
        .reset_index()
    )
    team_level["player_bias_top3"] = team_level["player_bias_top3"].fillna(team_level["player_bias_mean"].fillna(0.0))
    team_level["player_reliability"] = np.clip(
        team_level["player_volume"] / (PLAYER_WINDOW * PLAYER_TOP_N), 0.0, 1.0
    )
    team_level["player_component"] = team_level["player_bias_top3"] * team_level["player_reliability"]
    return team_level


def _compose_bias_table(summary: pd.DataFrame, style: pd.DataFrame, player_team: pd.DataFrame) -> pd.DataFrame:
    context = summary.merge(
        style[[
            "match_id",
            "team_id",
            "style_cluster",
            "style_att_bias",
            "style_def_bias",
            "style_reliability",
        ]],
        on=["match_id", "team_id"],
        how="left",
    )
    context = context.merge(
        player_team[["match_id", "team_id", "player_component", "player_reliability"]],
        on=["match_id", "team_id"],
        how="left",
    )

    context["player_component"] = context["player_component"].fillna(0.0)
    context["player_reliability"] = context["player_reliability"].fillna(0.0)
    context["style_att_component"] = (
        context["style_att_bias"].fillna(0.0) * context["style_reliability"].fillna(0.0)
    )
    context["style_def_component"] = (
        context["style_def_bias"].fillna(0.0) * context["style_reliability"].fillna(0.0)
    )

    rolling_weight = 0.55
    style_weight = 0.25
    player_weight = 0.20
    def_rolling_weight = 0.65
    def_style_weight = 0.35

    context["finishing_bias_logit"] = (
        rolling_weight * context["att_rolling_component"].fillna(0.0)
        + style_weight * context["style_att_component"].fillna(0.0)
        + player_weight * context["player_component"].fillna(0.0)
    )
    context["concession_bias_logit"] = (
        def_rolling_weight * context["def_rolling_component"].fillna(0.0)
        + def_style_weight * context["style_def_component"].fillna(0.0)
    )

    context["finishing_bias_logit"] = np.clip(context["finishing_bias_logit"], -MAX_LOGIT, MAX_LOGIT)
    context["concession_bias_logit"] = np.clip(context["concession_bias_logit"], -MAX_LOGIT, MAX_LOGIT)
    context["finishing_bias_multiplier"] = np.exp(context["finishing_bias_logit"])
    context["concession_bias_multiplier"] = np.exp(context["concession_bias_logit"])
    return context


def _prepare_outputs(context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_cols = [
        "match_id",
        "match_date",
        "is_home",
        "competition_season_key",
        "competition_statsbomb_id",
        "competition_season",
    ]

    finishing_cols = base_cols + [
        "att_rolling_component",
        "style_att_component",
        "player_component",
        "finishing_bias_logit",
        "finishing_bias_multiplier",
    ]
    concession_cols = base_cols + [
        "def_rolling_component",
        "style_def_component",
        "concession_bias_logit",
        "concession_bias_multiplier",
    ]

    finishing = context[finishing_cols].copy()
    concession = context[concession_cols].copy()
    finishing["match_date"] = finishing["match_date"].dt.strftime("%Y-%m-%d")
    concession["match_date"] = concession["match_date"].dt.strftime("%Y-%m-%d")

    metrics = {
        "total_rows": int(len(context)),
        "finishing_bias_logit_mean": float(context["finishing_bias_logit"].mean()),
        "concession_bias_logit_mean": float(context["concession_bias_logit"].mean()),
        "rolling_windows": list(ROLLING_WINDOWS),
        "player_window": PLAYER_WINDOW,
        "player_coverage": float((context["player_component"].abs() > 1e-6).mean()),
        "excluded_pl1516_rows": int(context["is_pl1516"].sum()),
        "include_pl1516_in_priors": bool(INCLUDE_PL1516_IN_PRIORS),
    }
    return finishing, concession, metrics


def main() -> None:
    df = load_cxg_modeling_dataset()
    df["is_goal"] = df["is_goal"].astype(int)
    summary = _build_match_team_summary(df)
    summary = _attach_rolling_components(summary)
    style, cluster_payload = _build_style_features(summary)
    player_team = _build_player_team_features(df)
    context = _compose_bias_table(summary, style, player_team)
    finishing, concession, metrics = _prepare_outputs(context)
    metrics.update(cluster_payload)

    out_dir = get_modeling_output_dir("cxg", subdir="submodels")
    out_dir.mkdir(parents=True, exist_ok=True)

    finishing_path = out_dir / "finishing_bias_modeled.csv"
    concession_path = out_dir / "concession_bias_modeled.csv"
    finishing.to_csv(finishing_path, index=False)
    concession.to_csv(concession_path, index=False)

    metrics_path = out_dir / "finishing_bias_model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"Saved neutral finishing priors to {finishing_path}")
    print(f"Saved neutral concession priors to {concession_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
