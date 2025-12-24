"""cxA baseline construction.

Provides pass-level attribution tables:
- key-pass xA: shot_value assigned only to last pass before shot
- sequence xA+: decay-weighted allocation of shot_value across last k passes

This module contains reusable core logic; CLI wrappers live in `scripts/`.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from opponent_adjusted.db.models import Event, PassEvent, Shot, ShotPrediction, ModelRegistry


@dataclass(frozen=True)
class ShotContext:
    shot_id: int
    shot_event_id: int
    match_id: int
    team_id: int
    possession: Optional[int]
    shot_raw_event_id: int
    shot_value: float


def _resolve_model_id(session: Session, model_name: str, version: Optional[str]) -> int:
    stmt = select(ModelRegistry).where(ModelRegistry.model_name == model_name)
    if version is not None:
        stmt = stmt.where(ModelRegistry.version == version)
    stmt = stmt.order_by(ModelRegistry.version.desc())
    row = session.execute(stmt).scalars().first()
    if row is None:
        raise ValueError(f"No model in registry for model_name={model_name!r}, version={version!r}")
    return int(row.id)


def _iter_shots(session: Session, *, model_id: Optional[int]) -> Iterable[ShotContext]:
    if model_id is None:
        stmt = (
            select(
                Shot.id,
                Shot.event_id,
                Shot.match_id,
                Shot.team_id,
                Event.possession,
                Event.raw_event_id,
                Shot.statsbomb_xg,
            )
            .join(Event, Event.id == Shot.event_id)
        )
        rows = session.execute(stmt).all()
        for r in rows:
            if r.statsbomb_xg is None:
                continue
            yield ShotContext(
                shot_id=int(r.id),
                shot_event_id=int(r.event_id),
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
            Shot.event_id,
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

    per_shot: Dict[int, Tuple[ShotContext, int]] = {}

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

        ctx = ShotContext(
            shot_id=shot_id,
            shot_event_id=int(r.event_id),
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


def _find_last_k_passes_before_shot(
    events: List[dict],
    raw_ids: List[int],
    *,
    shot_raw_event_id: int,
    team_id: int,
    possession: Optional[int],
    k_actions: int,
) -> List[dict]:
    start_idx = bisect_left(raw_ids, int(shot_raw_event_id)) - 1
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
    return [e for e in out if e.get("type") == "Pass"]


def build_cxa_baselines(
    session: Session,
    *,
    model_name: str = "cxg",
    version: Optional[str] = None,
    k_actions: int = 3,
    decay: float = 0.6,
    limit_shots: Optional[int] = None,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Compute pass-level cxA baselines.

    Returns a DataFrame with one row per attributed pass.
    """

    try:
        model_id: Optional[int] = _resolve_model_id(session, model_name, version)
    except ValueError:
        model_id = None

    shots = list(_iter_shots(session, model_id=model_id))
    shots.sort(key=lambda s: (s.match_id, s.shot_raw_event_id))
    if limit_shots is not None:
        shots = shots[:limit_shots]

    match_cache: Dict[int, Tuple[List[dict], List[int]]] = {}
    pass_rows: List[dict] = []

    total_shots = len(shots)

    for idx_shot, shot in enumerate(shots, start=1):
        cached = match_cache.get(shot.match_id)
        if cached is None:
            events, raw_ids = _load_match_events(session, shot.match_id)
            match_cache[shot.match_id] = (events, raw_ids)
        else:
            events, raw_ids = cached

        if progress_every > 0 and idx_shot % progress_every == 0:
            print(f"Processed {idx_shot:,}/{total_shots:,} shots...")

        passes = _find_last_k_passes_before_shot(
            events,
            raw_ids,
            shot_raw_event_id=shot.shot_raw_event_id,
            team_id=shot.team_id,
            possession=shot.possession,
            k_actions=k_actions,
        )

        if not passes:
            continue

        key_pass = passes[-1]

        weights: List[float] = []
        for p_idx, _p in enumerate(passes):
            delta = (len(passes) - 1) - p_idx
            weights.append(float(decay) ** float(delta))
        z = sum(weights)
        if z <= 0:
            continue
        weights = [w / z for w in weights]

        for p_idx, p in enumerate(passes):
            xa_key = shot.shot_value if int(p["id"]) == int(key_pass["id"]) else 0.0
            xa_plus = weights[p_idx] * shot.shot_value

            pass_rows.append(
                {
                    "match_id": shot.match_id,
                    "possession": shot.possession,
                    "shot_id": shot.shot_id,
                    "shot_event_raw_event_id": shot.shot_raw_event_id,
                    "shot_value": shot.shot_value,
                    "pass_event_id": int(p["id"]),
                    "pass_raw_event_id": int(p["raw_event_id"]),
                    "team_id": int(p["team_id"]),
                    "player_id": p.get("player_id"),
                    "under_pressure": bool(p.get("under_pressure", False)),
                    "pass_type": p.get("pass_type"),
                    "pass_height": p.get("pass_height"),
                    "pass_body_part": p.get("body_part"),
                    "is_cross": bool(p.get("is_cross") or False),
                    "is_through_ball": bool(p.get("is_through_ball") or False),
                    "pass_length": p.get("pass_length"),
                    "pass_angle": p.get("pass_angle"),
                    "pass_end_x": p.get("end_x"),
                    "pass_end_y": p.get("end_y"),
                    "xa_keypass": float(xa_key),
                    "xa_plus": float(xa_plus),
                    "k_actions": int(k_actions),
                    "decay": float(decay),
                    "model_name": model_name,
                    "model_version": version,
                }
            )

    return pd.DataFrame(pass_rows)
