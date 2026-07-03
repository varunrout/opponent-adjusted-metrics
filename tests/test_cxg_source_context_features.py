from opponent_adjusted.db.models import (
    Competition,
    Event,
    Match,
    RawEvent,
    Shot,
    ShotFeature,
    Team,
)
from opponent_adjusted.db.session import session_scope
from opponent_adjusted.pipelines.cxg.pipeline import (
    add_context_features,
    add_geometric_features,
    build_shots_dataset,
)


def _add_event(
    session,
    match_id: int,
    statsbomb_id: str,
    event_type: str,
    team: Team,
    period: int,
    minute: int,
    second: int,
    possession: int,
    raw_json: dict | None = None,
) -> Event:
    payload = {
        "id": statsbomb_id,
        "type": {"name": event_type},
        "team": {"id": team.statsbomb_team_id, "name": team.name},
        "period": period,
        "minute": minute,
        "second": second,
        "possession": possession,
        "location": [105.0, 40.0],
    }
    payload.update(raw_json or {})
    raw = RawEvent(
        match_id=match_id,
        statsbomb_event_id=statsbomb_id,
        raw_json=payload,
        type=event_type,
        period=period,
        minute=minute,
        second=second,
    )
    session.add(raw)
    session.flush()
    event = Event(
        raw_event_id=raw.id,
        match_id=match_id,
        team_id=team.id,
        player_id=None,
        type=event_type,
        period=period,
        minute=minute,
        second=second,
        timestamp=f"00:{minute:02d}:{second:02d}.000",
        possession=possession,
        location_x=payload["location"][0],
        location_y=payload["location"][1],
        under_pressure=bool(payload.get("under_pressure", False)),
        outcome=((payload.get("shot") or {}).get("outcome") or {}).get("name"),
    )
    session.add(event)
    session.flush()
    return event


def _add_shot(
    session,
    event: Event,
    team: Team,
    opponent: Team,
    outcome: str,
    shot_type: str = "Open Play",
) -> Shot:
    shot = Shot(
        event_id=event.id,
        match_id=event.match_id,
        team_id=team.id,
        player_id=None,
        opponent_team_id=opponent.id,
        statsbomb_xg=0.1,
        body_part="Right Foot",
        technique="Normal",
        shot_type=shot_type,
        outcome=outcome,
        first_time=False,
        is_blocked=False,
    )
    session.add(shot)
    session.flush()
    return shot


def test_cxg_feature_store_derives_pre_shot_context_from_source_events(e2e_test_env):
    with session_scope() as session:
        competition = Competition(
            statsbomb_competition_id=1,
            name="Context Cup",
            season="2026",
        )
        home = Team(statsbomb_team_id=10, name="Home")
        away = Team(statsbomb_team_id=20, name="Away")
        session.add_all([competition, home, away])
        session.flush()
        match = Match(
            statsbomb_match_id=100,
            competition_id=competition.id,
            home_team_id=home.id,
            away_team_id=away.id,
            season="2026",
        )
        session.add(match)
        session.flush()

        first_goal_event = _add_event(
            session,
            match.id,
            "shot-1",
            "Shot",
            home,
            1,
            0,
            10,
            1,
            {
                "play_pattern": {"name": "Regular Play"},
                "shot": {
                    "outcome": {"name": "Goal"},
                    "type": {"name": "Open Play"},
                },
            },
        )
        _add_shot(session, first_goal_event, home, away, "Goal")

        _add_event(session, match.id, "pass-1", "Pass", home, 1, 0, 20, 2)
        _add_event(session, match.id, "pressure-1", "Pressure", away, 1, 0, 21, 2)
        pressured_shot_event = _add_event(
            session,
            match.id,
            "shot-2",
            "Shot",
            home,
            1,
            0,
            24,
            2,
            {
                "play_pattern": {"name": "Regular Play"},
                "under_pressure": True,
                "shot": {
                    "outcome": {"name": "Saved"},
                    "type": {"name": "Open Play"},
                },
            },
        )
        second_shot = _add_shot(session, pressured_shot_event, home, away, "Saved")
        session.add(
            ShotFeature(
                shot_id=second_shot.id,
                version_tag="v1",
                possession_sequence_length=3,
                possession_duration=4.0,
                previous_action_gap=4.0,
                recent_def_actions_count=1,
                pressure_proxy_score=1.0,
            )
        )
        _add_event(session, match.id, "carry-after-shot", "Carry", home, 1, 0, 40, 2)

        corner_event = _add_event(
            session,
            match.id,
            "shot-3",
            "Shot",
            away,
            1,
            1,
            5,
            3,
            {
                "play_pattern": {"name": "From Corner"},
                "shot": {
                    "outcome": {"name": "Goal"},
                    "type": {"name": "Open Play"},
                    "key_pass_id": "corner-pass",
                },
            },
        )
        away_shot = _add_shot(session, corner_event, away, home, "Goal")

        shots = build_shots_dataset(session)
        features = add_context_features(add_geometric_features(shots))

        first = features.loc[features["event_id"] == first_goal_event.id].iloc[0]
        assert first["score_diff_at_shot"] == 0
        assert bool(first["is_drawing"])

        second = features.loc[features["shot_id"] == second_shot.id].iloc[0]
        assert second["score_diff_at_shot"] == 1
        assert bool(second["is_leading"])
        assert second["possession_sequence_length"] == 3
        assert second["possession_duration"] == 4.0
        assert second["previous_action_gap"] == 4.0
        assert second["time_gap_seconds"] == 4.0
        assert bool(second["possession_match"])
        assert second["recent_def_actions_count"] == 1
        assert second["pressure_state"] == "pressured"
        assert second["pressure_proxy_score"] == 1.0
        assert second["def_label"] == "medium"

        away_row = features.loc[features["shot_id"] == away_shot.id].iloc[0]
        assert away_row["score_diff_at_shot"] == -1
        assert bool(away_row["is_trailing"])
        assert away_row["play_pattern"] == "From Corner"
        assert away_row["set_piece_category"] == "corner"
        assert away_row["set_piece_phase"] == "first_phase"


def test_cxg_score_context_handles_own_goals_before_later_shot(e2e_test_env):
    with session_scope() as session:
        competition = Competition(
            statsbomb_competition_id=1,
            name="Own Goal Cup",
            season="2026",
        )
        home = Team(statsbomb_team_id=10, name="Home")
        away = Team(statsbomb_team_id=20, name="Away")
        session.add_all([competition, home, away])
        session.flush()
        match = Match(
            statsbomb_match_id=101,
            competition_id=competition.id,
            home_team_id=home.id,
            away_team_id=away.id,
            season="2026",
        )
        session.add(match)
        session.flush()

        own_goal_event = _add_event(
            session,
            match.id,
            "own-goal-shot",
            "Shot",
            home,
            1,
            0,
            10,
            1,
            {
                "play_pattern": {"name": "Regular Play"},
                "shot": {
                    "outcome": {"name": "Own Goal"},
                    "type": {"name": "Open Play"},
                },
            },
        )
        own_goal_shot = _add_shot(session, own_goal_event, home, away, "Own Goal")
        later_shot_event = _add_event(
            session,
            match.id,
            "later-shot",
            "Shot",
            home,
            1,
            0,
            20,
            2,
            {
                "play_pattern": {"name": "Regular Play"},
                "shot": {
                    "outcome": {"name": "Saved"},
                    "type": {"name": "Open Play"},
                },
            },
        )
        later_shot = _add_shot(session, later_shot_event, home, away, "Saved")

        shots = build_shots_dataset(session)

        own_goal_row = shots.loc[shots["shot_id"] == own_goal_shot.id].iloc[0]
        assert own_goal_row["score_diff_at_shot"] == 0
        assert bool(own_goal_row["is_drawing"])

        later_row = shots.loc[shots["shot_id"] == later_shot.id].iloc[0]
        assert later_row["score_diff_at_shot"] == -1
        assert bool(later_row["is_trailing"])
        assert not bool(later_row["is_leading"])
        assert not bool(later_row["is_drawing"])
