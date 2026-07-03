from opponent_adjusted.db.models import (
    Competition,
    Event,
    Match,
    Possession,
    RawEvent,
    Shot,
    ShotFeature,
    Team,
)
from opponent_adjusted.db.session import session_scope
from scripts.build_shot_features import populate_possession_features


def test_shot_features_are_enriched_with_possession_context(e2e_test_env):
    with session_scope() as session:
        competition = Competition(
            statsbomb_competition_id=1,
            name="Fixture Cup",
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

        events = []
        event_plan = (
            ("Pass", home.id, 2),
            ("Carry", home.id, 6),
            ("Pressure", away.id, 7),
            ("Shot", home.id, 8),
            ("Carry", home.id, 14),
        )
        for index, (event_type, team_id, second) in enumerate(event_plan, start=1):
            raw = RawEvent(
                match_id=match.id,
                statsbomb_event_id=f"event-{index}",
                raw_json={"id": f"event-{index}", "type": {"name": event_type}},
                type=event_type,
                period=1,
                minute=0,
                second=second,
            )
            session.add(raw)
            session.flush()
            event = Event(
                raw_event_id=raw.id,
                match_id=match.id,
                team_id=team_id,
                player_id=None,
                type=event_type,
                period=1,
                minute=0,
                second=second,
                possession=7,
                location_x=50.0 + index,
                location_y=40.0,
            )
            session.add(event)
            events.append(event)
        session.flush()

        possession = Possession(
            match_id=match.id,
            possession_number=7,
            team_id=home.id,
            start_event_id=events[0].id,
            end_event_id=events[-1].id,
            start_minute=0,
            end_minute=0,
            duration_seconds=12.0,
            event_count=5,
        )
        shot = Shot(
            event_id=events[3].id,
            match_id=match.id,
            team_id=home.id,
            player_id=None,
            opponent_team_id=away.id,
            outcome="Saved",
        )
        session.add_all([possession, shot])
        session.flush()
        feature = ShotFeature(
            shot_id=shot.id,
            version_tag="v1",
            shot_distance=10.0,
            shot_angle=0.5,
        )
        session.add(feature)
        session.flush()

        updated = populate_possession_features(session, "v1")

        assert updated["updated"] == 1
        assert updated["missing_possession_number"] == 0
        assert updated["missing_possession_row"] == 0
        assert feature.possession_sequence_length == 4
        assert feature.possession_duration == 6.0
        assert feature.previous_action_gap == 1.0
        assert feature.recent_def_actions_count == 1
        assert feature.pressure_proxy_score == 1.0

        second_run = populate_possession_features(session, "v1")

        assert second_run["evaluated"] == 1
        assert second_run["updated"] == 0


def _seed_match(session):
    competition = Competition(
        statsbomb_competition_id=1,
        name="Fixture Cup",
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
    return match, home, away


def _add_event(session, match, statsbomb_id, event_type, team, second, possession):
    raw = RawEvent(
        match_id=match.id,
        statsbomb_event_id=statsbomb_id,
        raw_json={"id": statsbomb_id, "type": {"name": event_type}},
        type=event_type,
        period=1,
        minute=0,
        second=second,
    )
    session.add(raw)
    session.flush()
    event = Event(
        raw_event_id=raw.id,
        match_id=match.id,
        team_id=team.id,
        player_id=None,
        type=event_type,
        period=1,
        minute=0,
        second=second,
        possession=possession,
        location_x=50.0,
        location_y=40.0,
    )
    session.add(event)
    session.flush()
    return event


def _add_shot_feature(session, match, shot_event, team, opponent):
    shot = Shot(
        event_id=shot_event.id,
        match_id=match.id,
        team_id=team.id,
        player_id=None,
        opponent_team_id=opponent.id,
        outcome="Saved",
    )
    session.add(shot)
    session.flush()
    feature = ShotFeature(
        shot_id=shot.id,
        version_tag="v1",
        shot_distance=10.0,
        shot_angle=0.5,
    )
    session.add(feature)
    session.flush()
    return feature


def test_previous_action_gap_uses_actual_previous_event(e2e_test_env):
    with session_scope() as session:
        match, home, away = _seed_match(session)
        _add_event(session, match, "pass-1", "Pass", home, second=5, possession=7)
        _add_event(session, match, "pressure-1", "Pressure", away, second=8, possession=7)
        shot_event = _add_event(session, match, "shot-1", "Shot", home, second=10, possession=7)
        feature = _add_shot_feature(session, match, shot_event, home, away)

        updated = populate_possession_features(session, "v1")

        assert updated["updated"] == 1
        assert feature.previous_action_gap == 2.0
        assert feature.recent_def_actions_count == 1
        assert feature.pressure_proxy_score == 1.0


def test_recent_def_actions_count_uses_bounded_lookback(e2e_test_env):
    with session_scope() as session:
        match, home, away = _seed_match(session)

        _add_event(session, match, "old-pressure", "Pressure", away, second=2, possession=7)
        old_shot_event = _add_event(
            session, match, "shot-old", "Shot", home, second=20, possession=7
        )
        old_feature = _add_shot_feature(session, match, old_shot_event, home, away)

        _add_event(session, match, "recent-pressure", "Pressure", away, second=18, possession=8)
        recent_shot_event = _add_event(
            session,
            match,
            "shot-recent",
            "Shot",
            home,
            second=20,
            possession=8,
        )
        recent_feature = _add_shot_feature(session, match, recent_shot_event, home, away)

        updated = populate_possession_features(session, "v1")

        assert updated["updated"] == 2
        assert old_feature.recent_def_actions_count == 0
        assert old_feature.pressure_proxy_score == 0.0
        assert recent_feature.recent_def_actions_count == 1
        assert recent_feature.pressure_proxy_score == 1.0
