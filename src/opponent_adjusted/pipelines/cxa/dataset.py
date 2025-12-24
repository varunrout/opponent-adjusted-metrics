"""cxA training dataset construction.

Builds a modeling-ready pass-level dataset with:
- derived geometric features from pass start/end locations
- regression targets for sequence-based xA+ and key-pass xA
- negative examples (passes not attributed to any shot in the window)

CLI wrappers live in `scripts/`.
"""

from __future__ import annotations

import random
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from opponent_adjusted.db.models import Event, PassEvent, Shot, ShotPrediction, ModelRegistry
from opponent_adjusted.features.geometry import calculate_centrality, calculate_distance, calculate_shot_angle


@dataclass(frozen=True)
class ShotCtx:
    shot_id: int
    match_id: int
    team_id: int
    possession: Optional[int]
    shot_raw_event_id: int
    shot_value: float


def _resolve_model_id(session: Session, model_name: str, version: Optional[str]) -> Optional[int]:
    stmt = select(ModelRegistry).where(ModelRegistry.model_name == model_name)
    if version is not None:
        stmt = stmt.where(ModelRegistry.version == version)
    stmt = stmt.order_by(ModelRegistry.version.desc())
    row = session.execute(stmt).scalars().first()
    return int(row.id) if row is not None else None


def _iter_shots(session: Session, *, model_id: Optional[int]) -> Iterable[ShotCtx]:
    if model_id is None:
        stmt = (
            select(
                Shot.id,
                Shot.match_id,
                Shot.team_id,
                Event.possession,
                Event.raw_event_id,
                Shot.statsbomb_xg,
            )
            .join(Event, Event.id == Shot.event_id)
        )
        for r in session.execute(stmt).all():
            if r.statsbomb_xg is None:
                continue
            yield ShotCtx(
                shot_id=int(r.id),
                match_id=int(r.match_id),
                team_id=int(r.team_id),
                possession=r.possession,
                shot_raw_event_id=int(r.raw_event_id),
                shot_value=float(r.statsbomb_xg),
            )
        return

    stmt = (
        select(
            Shot.id,
            Shot.match_id,
            Shot.team_id,
            Event.possession,
            Event.raw_event_id,
            Shot.statsbomb_xg,
            ShotPrediction.is_neutralized,
            ShotPrediction.raw_probability,
            ShotPrediction.neutral_probability,
        )
        .join(Event, Event.id == Shot.event_id)
        .outerjoin(
            ShotPrediction,
            (ShotPrediction.shot_id == Shot.id) & (ShotPrediction.model_id == model_id),
        )
    )

    per_shot: Dict[int, Tuple[ShotCtx, int]] = {}
    for r in session.execute(stmt).all():
        shot_id = int(r.id)

        shot_value: Optional[float] = None
        preference = -1

        if r.neutral_probability is not None:
            shot_value = float(r.neutral_probability)
            preference = 3
        elif bool(r.is_neutralized) and r.raw_probability is not None:
            shot_value = float(r.raw_probability)
            preference = 2
        elif r.statsbomb_xg is not None:
            shot_value = float(r.statsbomb_xg)
            preference = 1

        if shot_value is None:
            continue

        ctx = ShotCtx(
            shot_id=shot_id,
            match_id=int(r.match_id),
            team_id=int(r.team_id),
            possession=r.possession,
            shot_raw_event_id=int(r.raw_event_id),
            shot_value=float(shot_value),
        )

        existing = per_shot.get(shot_id)
        if existing is None or preference > existing[1]:
            per_shot[shot_id] = (ctx, preference)

    for ctx, _pref in per_shot.values():
        yield ctx


def _load_match_events(session: Session, match_id: int) -> Tuple[List[dict], List[int]]:
    stmt = (
        select(
            Event.id,
            Event.raw_event_id,
            Event.match_id,
            Event.team_id,
            Event.player_id,
            Event.type,
            Event.possession,
            Event.period,
            Event.minute,
            Event.second,
            Event.under_pressure,
            Event.location_x,
            Event.location_y,
            PassEvent.outcome.label("pass_outcome"),
            PassEvent.pass_type,
            PassEvent.pass_height,
            PassEvent.body_part,
            PassEvent.is_cross,
            PassEvent.is_through_ball,
            PassEvent.length.label("pass_length"),
            PassEvent.angle.label("pass_angle"),
            PassEvent.end_x,
            PassEvent.end_y,
        )
        .where(Event.match_id == match_id)
        .outerjoin(PassEvent, PassEvent.event_id == Event.id)
        .order_by(Event.raw_event_id)
    )

    rows = session.execute(stmt).mappings().all()
    events = [dict(r) for r in rows]
    raw_ids = [int(e["raw_event_id"]) for e in events]
    return events, raw_ids


def _compute_pass_derived_features(pass_row: dict) -> dict:
    sx = pass_row.get("location_x")
    sy = pass_row.get("location_y")
    ex = pass_row.get("end_x")
    ey = pass_row.get("end_y")

    out: dict = {}

    if sx is not None and sy is not None:
        start_dist_to_goal = float(calculate_distance(float(sx), float(sy)))
        out["start_dist_to_goal"] = start_dist_to_goal
        out["start_angle_to_goal"] = float(calculate_shot_angle(float(sx), float(sy)))
        out["start_centrality"] = float(calculate_centrality(float(sy)))
    else:
        start_dist_to_goal = None
        out["start_dist_to_goal"] = None
        out["start_angle_to_goal"] = None
        out["start_centrality"] = None

    if ex is not None and ey is not None:
        end_dist_to_goal = float(calculate_distance(float(ex), float(ey)))
        out["end_dist_to_goal"] = end_dist_to_goal
        out["end_angle_to_goal"] = float(calculate_shot_angle(float(ex), float(ey)))
        out["end_centrality"] = float(calculate_centrality(float(ey)))
    else:
        end_dist_to_goal = None
        out["end_dist_to_goal"] = None
        out["end_angle_to_goal"] = None
        out["end_centrality"] = None

    if sx is not None and ex is not None:
        out["dx"] = float(ex) - float(sx)
    else:
        out["dx"] = None

    if sy is not None and ey is not None:
        out["dy"] = float(ey) - float(sy)
        out["abs_dy"] = abs(float(ey) - float(sy))
    else:
        out["dy"] = None
        out["abs_dy"] = None

    if start_dist_to_goal is not None and end_dist_to_goal is not None:
        out["progress_to_goal"] = float(start_dist_to_goal) - float(end_dist_to_goal)
    else:
        out["progress_to_goal"] = None

    return out


def _last_k_actions_before_index(
    events: List[dict],
    raw_ids: List[int],
    *,
    raw_event_id: int,
    team_id: int,
    possession: Optional[int],
    k_actions: int,
) -> List[dict]:
    start_idx = bisect_left(raw_ids, int(raw_event_id)) - 1
    if start_idx < 0:
        return []

    out: List[dict] = []
    idx = start_idx
    while idx >= 0:
        ev = events[idx]
        if int(ev["team_id"]) != int(team_id):
            idx -= 1
            continue
        if possession is not None and ev.get("possession") != possession:
            idx -= 1
            continue

        out.append(ev)
        if len(out) >= k_actions:
            break
        idx -= 1

    out.reverse()
    return out


def build_cxa_training_dataset(
    session: Session,
    *,
    model_name: str = "cxg",
    version: Optional[str] = None,
    k_actions: int = 3,
    decay: float = 0.6,
    negatives_ratio: float = 2.0,
    seed: int = 7,
    limit_matches: Optional[int] = None,
    progress_every_matches: int = 10,
) -> pd.DataFrame:
    """Build modeling-ready pass-level dataset for cxA.

    Includes positives (pass attribution rows) and sampled negatives.
    """

    model_id = _resolve_model_id(session, model_name, version)

    shot_map: Dict[Tuple[int, int], ShotCtx] = {}
    for s in _iter_shots(session, model_id=model_id):
        shot_map[(s.match_id, s.shot_raw_event_id)] = s

    match_ids = session.execute(select(Event.match_id).distinct().order_by(Event.match_id)).scalars().all()
    match_ids = list(match_ids)
    if limit_matches is not None:
        match_ids = match_ids[:limit_matches]

    rng = random.Random(seed)

    positive_rows: List[dict] = []
    positive_pass_event_ids: set[int] = set()
    # Negatives are sampled in a second pass using reservoir sampling to avoid
    # holding all negative candidates in memory.

    for mi, match_id in enumerate(match_ids, start=1):
        events, raw_ids = _load_match_events(session, int(match_id))

        for ev in events:
            if ev.get("type") != "Shot":
                continue

            key = (int(ev["match_id"]), int(ev["raw_event_id"]))
            shot_ctx = shot_map.get(key)
            if shot_ctx is None:
                continue

            recent = _last_k_actions_before_index(
                events,
                raw_ids,
                raw_event_id=int(ev["raw_event_id"]),
                team_id=int(ev["team_id"]),
                possession=ev.get("possession"),
                k_actions=k_actions,
            )

            passes = [r for r in recent if r.get("type") == "Pass"]
            if not passes:
                continue

            weights: List[float] = []
            for p_idx, _p in enumerate(passes):
                delta = (len(passes) - 1) - p_idx
                weights.append(float(decay) ** float(delta))
            z = sum(weights)
            if z <= 0:
                continue
            weights = [w / z for w in weights]

            key_pass_event_id = int(passes[-1]["id"])

            for p_idx, p in enumerate(passes):
                pass_event_id = int(p["id"])
                positive_pass_event_ids.add(pass_event_id)

                base = {
                    "match_id": int(p["match_id"]),
                    "possession": p.get("possession"),
                    "period": p.get("period"),
                    "minute": p.get("minute"),
                    "second": p.get("second"),
                    "pass_event_id": pass_event_id,
                    "pass_raw_event_id": int(p["raw_event_id"]),
                    "team_id": int(p["team_id"]),
                    "player_id": int(p["player_id"]) if p.get("player_id") is not None else None,
                    "under_pressure": bool(p.get("under_pressure", False)),
                    "pass_type": p.get("pass_type"),
                    "pass_height": p.get("pass_height"),
                    "pass_body_part": p.get("body_part"),
                    "is_cross": bool(p.get("is_cross") or False),
                    "is_through_ball": bool(p.get("is_through_ball") or False),
                    "pass_length": p.get("pass_length"),
                    "pass_angle": p.get("pass_angle"),
                    "start_x": p.get("location_x"),
                    "start_y": p.get("location_y"),
                    "end_x": p.get("end_x"),
                    "end_y": p.get("end_y"),
                    "pass_outcome": p.get("pass_outcome"),
                    "shot_id": int(shot_ctx.shot_id),
                    "shot_event_raw_event_id": int(shot_ctx.shot_raw_event_id),
                    "shot_value": float(shot_ctx.shot_value),
                    "y_keypass": float(shot_ctx.shot_value) if pass_event_id == key_pass_event_id else 0.0,
                    "y_xa_plus": float(weights[p_idx] * shot_ctx.shot_value),
                    "k_actions": int(k_actions),
                    "decay": float(decay),
                    "model_name": model_name,
                    "model_version": version,
                    "is_negative": False,
                }
                base.update(_compute_pass_derived_features(p))
                positive_rows.append(base)

        if progress_every_matches > 0 and mi % progress_every_matches == 0:
            print(f"Scanned {mi:,}/{len(match_ids):,} matches...")

    n_pos = len(positive_rows)
    n_neg_target = int(max(0.0, negatives_ratio) * n_pos)

    negative_rows: List[dict] = []

    if n_neg_target > 0:
        # Second pass: reservoir sample negative passes (passes not in positive ids).
        seen = 0
        for mi, match_id in enumerate(match_ids, start=1):
            events, _raw_ids = _load_match_events(session, int(match_id))
            for ev in events:
                if ev.get("type") != "Pass":
                    continue
                pass_event_id = int(ev["id"])
                if pass_event_id in positive_pass_event_ids:
                    continue

                seen += 1
                if len(negative_rows) < n_neg_target:
                    negative_rows.append(ev)
                    continue

                j = rng.randrange(seen)
                if j < n_neg_target:
                    negative_rows[j] = ev

            if progress_every_matches > 0 and mi % progress_every_matches == 0:
                print(f"Sampled negatives from {mi:,}/{len(match_ids):,} matches...")

    # Materialize negative rows (with derived features) into final schema.
    rows: List[dict] = []
    rows.extend(positive_rows)

    if n_neg_target > 0 and negative_rows:
        for p in negative_rows:
            base = {
                "match_id": int(p["match_id"]),
                "possession": p.get("possession"),
                "period": p.get("period"),
                "minute": p.get("minute"),
                "second": p.get("second"),
                "pass_event_id": int(p["id"]),
                "pass_raw_event_id": int(p["raw_event_id"]),
                "team_id": int(p["team_id"]),
                "player_id": int(p["player_id"]) if p.get("player_id") is not None else None,
                "under_pressure": bool(p.get("under_pressure", False)),
                "pass_type": p.get("pass_type"),
                "pass_height": p.get("pass_height"),
                "pass_body_part": p.get("body_part"),
                "is_cross": bool(p.get("is_cross") or False),
                "is_through_ball": bool(p.get("is_through_ball") or False),
                "pass_length": p.get("pass_length"),
                "pass_angle": p.get("pass_angle"),
                "start_x": p.get("location_x"),
                "start_y": p.get("location_y"),
                "end_x": p.get("end_x"),
                "end_y": p.get("end_y"),
                "pass_outcome": p.get("pass_outcome"),
                "shot_id": None,
                "shot_event_raw_event_id": None,
                "shot_value": 0.0,
                "y_keypass": 0.0,
                "y_xa_plus": 0.0,
                "k_actions": int(k_actions),
                "decay": float(decay),
                "model_name": model_name,
                "model_version": version,
                "is_negative": True,
            }
            base.update(_compute_pass_derived_features(p))
            rows.append(base)

    print(
        f"Built dataset rows: {len(rows):,} (positives={n_pos:,}, negatives={len(rows) - n_pos:,}, negative_ratio={negatives_ratio})"
    )

    return pd.DataFrame(rows)
