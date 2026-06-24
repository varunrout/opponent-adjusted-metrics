"""Feature contract helpers for modelling and inference datasets.

A feature contract defines which columns must be present, which columns are
optional, and which leakage-prone columns must never be used as model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


class FeatureContractError(ValueError):
    """Raised when a dataframe violates a feature contract."""


@dataclass(frozen=True)
class FeatureContract:
    """Column-level contract for model datasets."""

    name: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    forbidden_columns: tuple[str, ...] = ()
    target_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def allowed_columns(self) -> set[str]:
        """Return columns allowed by the contract."""

        return set(self.required_columns) | set(self.optional_columns) | set(self.target_columns)

    @property
    def feature_columns(self) -> list[str]:
        """Return ordered model feature columns, excluding targets."""

        return [
            col
            for col in [*self.numeric_columns, *self.categorical_columns]
            if col not in self.target_columns and col not in self.forbidden_columns
        ]

    def validate(self, frame: pd.DataFrame, *, allow_extra: bool = True) -> None:
        """Validate a dataframe against this contract.

        Parameters
        ----------
        frame:
            DataFrame to validate.
        allow_extra:
            Whether extra non-contract columns are allowed. Training datasets
            often carry identifiers and audit columns, so the default is lenient.
        """

        columns = set(frame.columns)
        missing = sorted(set(self.required_columns) - columns)
        forbidden = sorted(set(self.forbidden_columns) & columns)

        if missing:
            raise FeatureContractError(
                f"{self.name} feature contract missing required columns: {missing}"
            )

        if forbidden:
            raise FeatureContractError(
                f"{self.name} feature contract contains forbidden columns: {forbidden}"
            )

        if not allow_extra:
            extra = sorted(columns - self.allowed_columns)
            if extra:
                raise FeatureContractError(
                    f"{self.name} feature contract contains unexpected columns: {extra}"
                )


def validate_contract(
    frame: pd.DataFrame,
    contract: FeatureContract,
    *,
    allow_extra: bool = True,
) -> pd.DataFrame:
    """Validate and return the input dataframe for pipeline chaining."""

    contract.validate(frame, allow_extra=allow_extra)
    return frame


def _as_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


CXG_BASE_CONTRACT = FeatureContract(
    name="cxg_base",
    required_columns=_as_tuple(
        [
            "location_x",
            "location_y",
            "distance_to_goal",
            "centrality",
            "body_part",
            "shot_type",
            "minute",
            "under_pressure",
            "is_goal",
        ]
    ),
    optional_columns=_as_tuple(
        [
            "technique",
            "first_time",
            "score_diff",
            "score_diff_at_shot",
            "possession_duration",
            "possession_length",
            "opponent_team_id",
            "opponent_def_rating_global",
            "opponent_def_zone_rating",
            "statsbomb_xg",
        ]
    ),
    forbidden_columns=_as_tuple(
        [
            "shot_outcome",
            "goal",
            "post_shot_xg",
            "keeper_outcome",
            "resulting_goal",
        ]
    ),
    target_columns=("is_goal",),
    numeric_columns=_as_tuple(
        [
            "location_x",
            "location_y",
            "distance_to_goal",
            "centrality",
            "minute",
            "score_diff",
            "score_diff_at_shot",
            "possession_duration",
            "possession_length",
            "opponent_def_rating_global",
            "opponent_def_zone_rating",
            "statsbomb_xg",
        ]
    ),
    categorical_columns=_as_tuple(["body_part", "shot_type", "technique", "under_pressure"]),
    metadata={
        "purpose": "CxG training and scoring base contract",
        "leakage_rule": "No target-derived post-shot outcome columns may be present.",
    },
)


CXA_ACTION_CONTRACT = FeatureContract(
    name="cxa_action",
    required_columns=_as_tuple(
        [
            "sequence_id",
            "action_type",
            "player_id",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "action_position",
            "shot_created",
        ]
    ),
    optional_columns=_as_tuple(
        [
            "match_id",
            "team_id",
            "under_pressure",
            "is_progressive",
            "enters_box",
            "enters_final_third",
            "resulting_shot_cxg",
            "is_goal",
        ]
    ),
    forbidden_columns=_as_tuple(["shot_outcome", "assist_outcome", "future_goal"]),
    target_columns=("shot_created",),
    numeric_columns=_as_tuple(
        [
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "action_position",
            "resulting_shot_cxg",
        ]
    ),
    categorical_columns=_as_tuple(["action_type", "under_pressure"]),
    metadata={
        "purpose": "CxA action-sequence modelling contract",
        "leakage_rule": "Shot-quality labels may be targets, but must not be used in shot-creation features.",
    },
)


CXT_ACTION_CONTRACT = FeatureContract(
    name="cxt_action",
    required_columns=_as_tuple(
        [
            "action_type",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "completed",
            "start_xt",
        ]
    ),
    optional_columns=_as_tuple(
        [
            "match_id",
            "player_id",
            "team_id",
            "under_pressure",
            "is_progressive",
            "opponent_zone_block_rate",
            "xt_delta",
        ]
    ),
    forbidden_columns=_as_tuple(
        [
            # `xt_delta` is allowed in the value-gain target dataset but must be
            # excluded from the completion feature set because it can imply
            # whether the action completed.
            "xt_delta_as_completion_feature",
            "future_shot",
            "future_goal",
        ]
    ),
    target_columns=("completed", "xt_delta"),
    numeric_columns=_as_tuple(
        [
            "start_x",
            "start_y",
            "end_x",
            "end_y",
            "start_xt",
            "opponent_zone_block_rate",
        ]
    ),
    categorical_columns=_as_tuple(["action_type", "under_pressure"]),
    metadata={
        "purpose": "CxT completion and value-gain modelling contract",
        "leakage_rule": "Do not include realised xT delta as a completion-model feature.",
    },
)


CONTRACTS: dict[str, FeatureContract] = {
    CXG_BASE_CONTRACT.name: CXG_BASE_CONTRACT,
    CXA_ACTION_CONTRACT.name: CXA_ACTION_CONTRACT,
    CXT_ACTION_CONTRACT.name: CXT_ACTION_CONTRACT,
}
