"""CxG pipeline: build shots, features, and opponent profiles.

This module contains the core data pipeline logic for the CxG metric.
It is invoked by ``scripts/run_cxg_pipeline.py`` but can also be used
programmatically.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from opponent_adjusted.config import settings, ensure_directories
from opponent_adjusted.db.session import get_session
from opponent_adjusted.db.models import Shot, Event, Match, Player, Team, RawEvent, ShotFeature
from opponent_adjusted.features.cxg.geometry import (
    calculate_distance,
    calculate_shot_angle,
    calculate_centrality,
)
from opponent_adjusted.features.cxg.context import (
    calculate_minute_bucket_label,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_feature_store_path() -> Path:
    """Get the cxG feature store path."""
    return settings.feature_store_path / "cxg"


def save_parquet(df: pd.DataFrame, name: str, output_dir: Path) -> Path:
    """Save DataFrame as parquet with metadata."""
    filepath = output_dir / f"{name}.parquet"
    df.to_parquet(filepath, index=False, engine="pyarrow")
    logger.info(f"Saved {name}: {len(df):,} rows -> {filepath}")
    return filepath


def _extract_name(value: object) -> str | None:
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name else None
    return None


def _score_state(score_diff: object) -> str:
    if pd.isna(score_diff):
        return "unknown"
    diff = int(score_diff)
    if diff > 0:
        return "leading"
    if diff < 0:
        return "trailing"
    return "drawing"


def _set_piece_category(raw_json: dict, shot_type: object, play_pattern: object) -> str:
    shot_type_name = str(shot_type or "").strip().lower()
    pattern = str(play_pattern or "").strip().lower()
    if shot_type_name == "penalty":
        return "penalty"
    if shot_type_name == "free kick":
        return "free_kick"
    if "corner" in pattern:
        return "corner"
    if "free kick" in pattern:
        return "free_kick"
    if "throw in" in pattern:
        return "throw_in"
    if pattern.startswith("from ") or pattern in {"from goal kick", "from kick off"}:
        return "other_set_piece"
    if raw_json:
        return "open_play"
    return "unknown"


def _set_piece_phase(category: str, raw_json: dict) -> str:
    if category in {"open_play", "unknown"}:
        return "none"
    if category in {"penalty", "free_kick"} and not (raw_json.get("shot") or {}).get("key_pass_id"):
        return "direct"
    if (raw_json.get("shot") or {}).get("key_pass_id"):
        return "first_phase"
    return "second_phase"


def _pressure_state(under_pressure: object, recent_def_actions_count: object) -> str:
    recent_count = 0 if pd.isna(recent_def_actions_count) else int(recent_def_actions_count)
    if bool(under_pressure) or recent_count > 0:
        return "pressured"
    return "unpressured"


def _pressure_proxy_score(under_pressure: object, recent_def_actions_count: object) -> float:
    recent_count = 0 if pd.isna(recent_def_actions_count) else int(recent_def_actions_count)
    return 1.0 if bool(under_pressure) or recent_count > 0 else 0.0


def _def_label(under_pressure: object, recent_def_actions_count: object) -> str:
    count = 0 if pd.isna(recent_def_actions_count) else int(recent_def_actions_count)
    if bool(under_pressure) and count >= 2:
        return "high"
    if bool(under_pressure) or count > 0:
        return "medium"
    return "low"


def _derive_score_context(df: pd.DataFrame) -> pd.DataFrame:
    """Compute score before each shot, excluding the current shot outcome."""

    if df.empty:
        return df
    df = df.sort_values(["match_id", "period", "minute", "second", "event_id"]).copy()
    score_by_match: dict[int, dict[int, int]] = {}
    derived = {
        "score_diff_at_shot": [],
        "is_leading": [],
        "is_trailing": [],
        "is_drawing": [],
    }
    for row in df.itertuples(index=False):
        match_scores = score_by_match.setdefault(int(row.match_id), {})
        team_id = int(row.team_id) if pd.notna(row.team_id) else 0
        opponent_id = int(row.opponent_team_id) if pd.notna(row.opponent_team_id) else 0
        team_score = match_scores.get(team_id, 0)
        opponent_score = match_scores.get(opponent_id, 0)
        score_diff = team_score - opponent_score
        derived["score_diff_at_shot"].append(score_diff)
        derived["is_leading"].append(score_diff > 0)
        derived["is_trailing"].append(score_diff < 0)
        derived["is_drawing"].append(score_diff == 0)

        outcome = str(row.outcome or "").strip().lower()
        if outcome == "goal":
            match_scores[team_id] = team_score + 1
        elif outcome == "own goal":
            # Own goals credit the opponent. If the opponent mapping is missing,
            # keep scores unchanged rather than assigning the goal to a dummy team.
            if opponent_id:
                match_scores[opponent_id] = opponent_score + 1
        else:
            match_scores.setdefault(team_id, team_score)
        match_scores.setdefault(team_id, team_score)
        match_scores.setdefault(opponent_id, opponent_score)

    for column, values in derived.items():
        df[column] = values
    return df


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def build_shots_dataset(session, competition_id: int = None) -> pd.DataFrame:
    """Build base shots dataset from database (all competitions if *competition_id* is ``None``)."""
    logger.info(f"Building shots dataset (competition_id={competition_id or 'ALL'})...")

    stmt = (
        select(
            Shot.id.label("shot_id"),
            Event.id.label("event_id"),
            Shot.match_id,
            Shot.team_id,
            Shot.player_id,
            Shot.opponent_team_id,
            Shot.statsbomb_xg,
            Shot.outcome,
            Shot.body_part,
            Shot.technique,
            Shot.shot_type,
            Shot.first_time,
            Shot.is_blocked,
            Event.period,
            Event.minute,
            Event.second,
            Event.timestamp,
            Event.possession,
            Event.location_x.label("shot_x"),
            Event.location_y.label("shot_y"),
            Event.under_pressure,
            RawEvent.raw_json.label("source_raw_json"),
            ShotFeature.score_diff_at_shot,
            ShotFeature.is_leading,
            ShotFeature.is_trailing,
            ShotFeature.is_drawing,
            ShotFeature.minute_bucket.label("feature_minute_bucket"),
            ShotFeature.possession_sequence_length,
            ShotFeature.possession_duration,
            ShotFeature.previous_action_gap,
            ShotFeature.recent_def_actions_count,
            ShotFeature.pressure_proxy_score,
        )
        .select_from(Shot)
        .join(Event, Event.id == Shot.event_id)
        .join(RawEvent, RawEvent.id == Event.raw_event_id)
        .outerjoin(ShotFeature, ShotFeature.shot_id == Shot.id)
    )

    if competition_id:
        stmt = stmt.join(Match, Match.id == Shot.match_id).where(
            Match.competition_id == competition_id
        )

    stmt = stmt.order_by(Shot.match_id, Event.period, Event.minute, Event.second)

    rows = session.execute(stmt).mappings().all()
    df = pd.DataFrame([dict(r) for r in rows])

    if df.empty:
        logger.warning("No shots found")
        return df

    df = _derive_score_context(df)

    # Add entity names
    player_ids = df["player_id"].dropna().unique().tolist()
    team_ids = df["team_id"].dropna().unique().tolist()

    if player_ids:
        player_stmt = select(Player.id, Player.name).where(Player.id.in_(player_ids))
        player_map = {r.id: r.name for r in session.execute(player_stmt).all()}
        df["player_name"] = df["player_id"].map(player_map)

    if team_ids:
        team_stmt = select(Team.id, Team.name).where(Team.id.in_(team_ids))
        team_map = {r.id: r.name for r in session.execute(team_stmt).all()}
        df["team_name"] = df["team_id"].map(team_map)

    # Derive is_goal
    df["is_goal"] = (df["outcome"] == "Goal").astype(int)

    logger.info(f"Built shots dataset: {len(df):,} rows")
    return df


def add_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add geometric features to shots."""
    logger.info("Adding geometric features...")

    df = df.copy()

    df["shot_distance"] = df.apply(lambda r: calculate_distance(r["shot_x"], r["shot_y"]), axis=1)
    df["shot_angle"] = df.apply(lambda r: calculate_shot_angle(r["shot_x"], r["shot_y"]), axis=1)
    df["centrality"] = df["shot_y"].apply(calculate_centrality)
    df["distance_to_goal_line"] = settings.goal_center_x - df["shot_x"]

    logger.info("Added geometric features")
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add context features to shots."""
    logger.info("Adding context features...")

    df = df.copy()

    raw_json = (
        df["source_raw_json"]
        if "source_raw_json" in df.columns
        else pd.Series([{} for _ in range(len(df))], index=df.index)
    )

    df["play_pattern"] = raw_json.map(
        lambda value: _extract_name((value or {}).get("play_pattern"))
    )
    df["play_pattern"] = df["play_pattern"].fillna("unknown")
    df["set_piece_category"] = [
        _set_piece_category(raw or {}, shot_type, play_pattern)
        for raw, shot_type, play_pattern in zip(
            raw_json,
            df.get("shot_type", pd.Series([None] * len(df), index=df.index)),
            df["play_pattern"],
        )
    ]
    df["set_piece_phase"] = [
        _set_piece_phase(category, raw or {})
        for category, raw in zip(df["set_piece_category"], raw_json)
    ]

    if "score_diff_at_shot" not in df.columns:
        df["score_diff_at_shot"] = 0
    df["score_state"] = df["score_diff_at_shot"].apply(_score_state)
    df["simple_state"] = df["score_state"]

    df["minute_bucket"] = df["minute"].apply(calculate_minute_bucket_label)
    df["minute_bucket_label"] = df["minute_bucket"]
    df["under_pressure_binary"] = df["under_pressure"].fillna(False).astype(int)
    if "time_gap_seconds" not in df.columns:
        df["time_gap_seconds"] = (
            df["previous_action_gap"] if "previous_action_gap" in df.columns else pd.NA
        )
    if "possession_match" not in df.columns:
        df["possession_match"] = (
            df["previous_action_gap"].notna() & df["possession"].notna()
            if "previous_action_gap" in df.columns
            else False
        )
    if "pressure_state" not in df.columns:
        df["pressure_state"] = [
            _pressure_state(pressure, recent)
            for pressure, recent in zip(df["under_pressure"], df["recent_def_actions_count"])
        ]
    df["pressure_proxy_score"] = [
        _pressure_proxy_score(pressure, recent)
        for pressure, recent in zip(df["under_pressure"], df["recent_def_actions_count"])
    ]
    if "def_label" not in df.columns:
        df["def_label"] = [
            _def_label(pressure, recent)
            for pressure, recent in zip(df["under_pressure"], df["recent_def_actions_count"])
        ]

    df["is_first_half"] = (df["period"] == 1).astype(int)
    df["is_second_half"] = (df["period"] == 2).astype(int)
    df["is_extra_time"] = (df["period"] > 2).astype(int)

    df["is_penalty"] = (df["shot_type"] == "Penalty").astype(int)
    df["is_free_kick"] = (df["shot_type"] == "Free Kick").astype(int)
    df["is_open_play"] = (df["shot_type"] == "Open Play").astype(int)

    df["is_header"] = (df["body_part"] == "Head").astype(int)
    df["is_right_foot"] = (df["body_part"] == "Right Foot").astype(int)
    df["is_left_foot"] = (df["body_part"] == "Left Foot").astype(int)

    if "source_raw_json" in df.columns:
        df = df.drop(columns=["source_raw_json"])

    logger.info("Added context features")
    return df


def assign_zone(distance: float, centrality: float) -> str:
    """Assign shot to defensive zone based on distance and centrality.

    Zones (A-F) based on config zone_definitions:
    - A: Close central
    - B: Close wide
    - C: Mid central
    - D: Mid wide
    - E: Far central
    - F: Far wide
    """
    zones = settings.zone_definitions

    for zone_id, bounds in zones.items():
        dist_ok = True
        cent_ok = True

        if "min_distance" in bounds and distance < bounds["min_distance"]:
            dist_ok = False
        if "max_distance" in bounds and distance >= bounds["max_distance"]:
            dist_ok = False
        if "min_centrality" in bounds and centrality < bounds["min_centrality"]:
            cent_ok = False
        if "max_centrality" in bounds and centrality >= bounds["max_centrality"]:
            cent_ok = False

        if dist_ok and cent_ok:
            return zone_id

    return "F"  # Default to far wide


def _shrink(mean_zone: float, n_zone: int, mean_global: float, prior: float = 50.0) -> float:
    """Apply Bayesian shrinkage to zone rating."""
    if n_zone <= 0:
        return mean_global
    return (n_zone * mean_zone + prior * mean_global) / (n_zone + prior)


def build_opponent_profiles(shot_features_df: pd.DataFrame, session) -> pd.DataFrame:
    """Build opponent defensive profiles from shot features.

    For each defending team, compute:
    - Global rating: -mean(conceded_xG) (lower = better defense)
    - Block rate: fraction of shots blocked
    - Zone ratings: per-zone defensive ratings with shrinkage
    """
    logger.info("Building opponent defensive profiles...")

    df = shot_features_df.copy()

    if "opponent_team_id" not in df.columns or df["opponent_team_id"].isna().all():
        logger.warning("No opponent_team_id in data, skipping profiles")
        return pd.DataFrame()

    df["zone"] = df.apply(lambda r: assign_zone(r["shot_distance"], r["centrality"]), axis=1)

    team_ids = df["opponent_team_id"].dropna().unique().tolist()
    if team_ids:
        team_stmt = select(Team.id, Team.name).where(Team.id.in_([int(t) for t in team_ids]))
        team_map = {r.id: r.name for r in session.execute(team_stmt).all()}
    else:
        team_map = {}

    profiles = []

    for team_id, group in df.groupby("opponent_team_id"):
        if pd.isna(team_id):
            continue

        team_id = int(team_id)
        team_name = team_map.get(team_id, f"Team {team_id}")

        n_total = len(group)
        mean_xg = group["statsbomb_xg"].mean() if "statsbomb_xg" in group.columns else 0
        n_blocked = group["is_blocked"].sum() if "is_blocked" in group.columns else 0
        block_rate = n_blocked / n_total if n_total > 0 else 0
        global_rating = -mean_xg

        profiles.append(
            {
                "team_id": team_id,
                "team_name": team_name,
                "zone_id": None,
                "global_rating": global_rating,
                "zone_rating": None,
                "block_rate": block_rate,
                "shots_conceded": n_total,
                "goals_conceded": int(group["is_goal"].sum()) if "is_goal" in group.columns else 0,
                "mean_xg_conceded": mean_xg,
            }
        )

        for zone in ["A", "B", "C", "D", "E", "F"]:
            zone_group = group[group["zone"] == zone]
            n_zone = len(zone_group)

            if n_zone > 0:
                mean_zone_xg = zone_group["statsbomb_xg"].mean()
                zone_blocked = (
                    zone_group["is_blocked"].sum() if "is_blocked" in zone_group.columns else 0
                )
                zone_block_rate = zone_blocked / n_zone
            else:
                mean_zone_xg = mean_xg
                zone_block_rate = block_rate

            shrunk_xg = _shrink(mean_zone_xg, n_zone, mean_xg, prior=50.0)
            zone_rating = -shrunk_xg

            profiles.append(
                {
                    "team_id": team_id,
                    "team_name": team_name,
                    "zone_id": zone,
                    "global_rating": None,
                    "zone_rating": zone_rating,
                    "block_rate": zone_block_rate,
                    "shots_conceded": n_zone,
                    "goals_conceded": (
                        int(zone_group["is_goal"].sum())
                        if n_zone > 0 and "is_goal" in zone_group.columns
                        else 0
                    ),
                    "mean_xg_conceded": mean_zone_xg if n_zone > 0 else None,
                }
            )

    profiles_df = pd.DataFrame(profiles)

    logger.info(f"Built profiles for {profiles_df['team_id'].nunique()} teams")
    logger.info(f"  Global profiles: {len(profiles_df[profiles_df['zone_id'].isna()])}")
    logger.info(f"  Zone profiles: {len(profiles_df[profiles_df['zone_id'].notna()])}")

    return profiles_df


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(competition_id: int = None) -> dict:
    """Run the full cxG pipeline.

    Args:
        competition_id: Filter to specific competition (default: ``None`` = all).

    Returns:
        Dictionary of output file paths.
    """
    ensure_directories()
    output_dir = get_feature_store_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("CxG PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Competition ID: {competition_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Database: {settings.database_url}")
    logger.info("=" * 70)

    outputs = {}

    with get_session() as session:
        # 1. Base shots
        logger.info("\n[1/4] Building base shots dataset...")
        shots_df = build_shots_dataset(session, competition_id=competition_id)
        outputs["shots"] = save_parquet(shots_df, "shots", output_dir)

        # 2. Add geometric features
        logger.info("\n[2/4] Adding geometric features...")
        shots_with_geom = add_geometric_features(shots_df)

        # 3. Add context features
        logger.info("\n[3/4] Adding context features...")
        shot_features_df = add_context_features(shots_with_geom)
        outputs["shot_features"] = save_parquet(shot_features_df, "shot_features", output_dir)

        # 4. Build opponent profiles
        logger.info("\n[4/4] Building opponent defensive profiles...")
        opponent_profiles_df = build_opponent_profiles(shot_features_df, session)
        if not opponent_profiles_df.empty:
            outputs["opponent_profiles"] = save_parquet(
                opponent_profiles_df, "opponent_profiles", output_dir
            )

    # Save metadata
    metadata = {
        "pipeline": "cxg",
        "competition_id": competition_id,
        "created_at": datetime.now().isoformat(),
        "files": {k: str(v) for k, v in outputs.items()},
        "row_counts": {
            "shots": len(shots_df),
            "shot_features": len(shot_features_df),
            "opponent_profiles": len(opponent_profiles_df) if not opponent_profiles_df.empty else 0,
        },
        "summary": {
            "total_shots": len(shots_df),
            "goals": int(shots_df["is_goal"].sum()) if "is_goal" in shots_df.columns else 0,
            "mean_xg": (
                float(shots_df["statsbomb_xg"].mean()) if "statsbomb_xg" in shots_df.columns else 0
            ),
            "teams_profiled": (
                int(opponent_profiles_df["team_id"].nunique())
                if not opponent_profiles_df.empty
                else 0
            ),
        },
    }

    metadata_path = output_dir / "pipeline_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"\nSaved metadata: {metadata_path}")

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Output files:")
    for name, path in outputs.items():
        logger.info(f"  {name}: {path}")

    return outputs
