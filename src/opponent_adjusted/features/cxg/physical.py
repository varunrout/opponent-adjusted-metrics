"""Event-only CxG physical shot-state sub-contract."""

from __future__ import annotations

from dataclasses import dataclass

from opponent_adjusted.features.cxg.geometry import ShotGeometry, shot_geometry

CXG_PHYSICAL_STATE_CONTRACT_ID = "cxg_physical_state_v1"

S_FEATURES = (
    "shot_distance_sb",
    "shot_angle_rad",
    "body_part_name",
    "technique_name",
    "shot_type_name",
    "first_time",
    "open_goal",
    "one_on_one",
    "follows_dribble",
    "under_pressure",
)
TARGET_FIELDS = ("is_goal",)
BENCHMARK_FIELDS = ("statsbomb_xg",)
POST_SHOT_FIELDS = (
    "end_x",
    "end_y",
    "end_z",
    "outcome_name",
    "saved_off_target",
    "saved_to_post",
)
LINKAGE_FIELDS = ("key_pass_id",)
EXCLUDED_PHYSICAL_FIELDS = ("deflected", "aerial_won")
RAW_LINEAGE_FIELDS = (
    "event_id",
    "match_id",
    "competition_id",
    "season_id",
    "player_id",
    "team_id",
    "raw_shot_x",
    "raw_shot_y",
    "period",
    "data_version",
    "silver_schema_version",
    "physical_state_contract_id",
)


@dataclass(frozen=True)
class PhysicalShotState:
    """Governed physical shot-state representation without model encoding."""

    event_id: str
    match_id: int
    competition_id: int
    season_id: int
    player_id: int | None
    team_id: int | None
    raw_shot_x: float | None
    raw_shot_y: float | None
    period: int | None
    data_version: str
    silver_schema_version: str
    physical_state_contract_id: str
    geometry: ShotGeometry
    body_part_name: str | None
    technique_name: str | None
    shot_type_name: str | None
    first_time: bool | None
    open_goal: bool | None
    one_on_one: bool | None
    follows_dribble: bool | None
    under_pressure: bool | None
    is_goal: bool | None
    statsbomb_xg: float | None
    end_x: float | None
    end_y: float | None
    end_z: float | None
    outcome_name: str | None
    saved_off_target: bool | None
    saved_to_post: bool | None
    key_pass_id: str | None
    deflected: bool | None
    aerial_won: bool | None

    @property
    def physical_model_eligible(self) -> bool:
        return self.period in {1, 2, 3, 4} and self.geometry.geometry_valid


def build_physical_shot_state(
    *,
    event_id: str,
    match_id: int,
    competition_id: int,
    season_id: int,
    player_id: int | None,
    team_id: int | None,
    raw_shot_x: float | None,
    raw_shot_y: float | None,
    period: int | None,
    data_version: str,
    silver_schema_version: str,
    body_part_name: str | None,
    technique_name: str | None,
    shot_type_name: str | None,
    first_time: bool | None,
    open_goal: bool | None,
    one_on_one: bool | None,
    follows_dribble: bool | None,
    under_pressure: bool | None,
    is_goal: bool | None,
    statsbomb_xg: float | None,
    end_x: float | None,
    end_y: float | None,
    end_z: float | None,
    outcome_name: str | None,
    saved_off_target: bool | None,
    saved_to_post: bool | None,
    key_pass_id: str | None,
    deflected: bool | None,
    aerial_won: bool | None,
) -> PhysicalShotState:
    """Build a physical-state representation while preserving governed source values."""
    return PhysicalShotState(
        event_id=event_id,
        match_id=match_id,
        competition_id=competition_id,
        season_id=season_id,
        player_id=player_id,
        team_id=team_id,
        raw_shot_x=raw_shot_x,
        raw_shot_y=raw_shot_y,
        period=period,
        data_version=data_version,
        silver_schema_version=silver_schema_version,
        physical_state_contract_id=CXG_PHYSICAL_STATE_CONTRACT_ID,
        geometry=shot_geometry(raw_shot_x, raw_shot_y),
        body_part_name=body_part_name,
        technique_name=technique_name,
        shot_type_name=shot_type_name,
        first_time=first_time,
        open_goal=open_goal,
        one_on_one=one_on_one,
        follows_dribble=follows_dribble,
        under_pressure=under_pressure,
        is_goal=is_goal,
        statsbomb_xg=statsbomb_xg,
        end_x=end_x,
        end_y=end_y,
        end_z=end_z,
        outcome_name=outcome_name,
        saved_off_target=saved_off_target,
        saved_to_post=saved_to_post,
        key_pass_id=key_pass_id,
        deflected=deflected,
        aerial_won=aerial_won,
    )
