"""Add engineered action feature table.

Revision ID: e5f6a7b8c9d0
Revises: d4f1a2b3c4d5
Create Date: 2026-06-28 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "d4f1a2b3c4d5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "action_features",
        sa.Column("feature_family", sa.String(length=50), nullable=False),
        sa.Column("version_tag", sa.String(length=50), nullable=False),
        sa.Column("action_id", sa.String(length=100), nullable=False),
        sa.Column("event_id", sa.String(length=100), nullable=True),
        sa.Column("match_id", sa.Integer(), nullable=True),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("player_id", sa.Integer(), nullable=True),
        sa.Column("possession_id", sa.String(length=100), nullable=True),
        sa.Column("possession_number", sa.Integer(), nullable=True),
        sa.Column("sequence_id", sa.String(length=100), nullable=True),
        sa.Column("action_type", sa.String(length=50), nullable=True),
        sa.Column("start_x", sa.Float(), nullable=True),
        sa.Column("start_y", sa.Float(), nullable=True),
        sa.Column("end_x", sa.Float(), nullable=True),
        sa.Column("end_y", sa.Float(), nullable=True),
        sa.Column("length", sa.Float(), nullable=True),
        sa.Column("angle", sa.Float(), nullable=True),
        sa.Column("x_progression", sa.Float(), nullable=True),
        sa.Column("y_progression", sa.Float(), nullable=True),
        sa.Column("distance_to_goal_before", sa.Float(), nullable=True),
        sa.Column("distance_to_goal_after", sa.Float(), nullable=True),
        sa.Column("angle_to_goal_before", sa.Float(), nullable=True),
        sa.Column("angle_to_goal_after", sa.Float(), nullable=True),
        sa.Column("start_zone", sa.String(length=50), nullable=True),
        sa.Column("end_zone", sa.String(length=50), nullable=True),
        sa.Column("is_progressive", sa.Boolean(), nullable=True),
        sa.Column("enters_final_third", sa.Boolean(), nullable=True),
        sa.Column("enters_penalty_area", sa.Boolean(), nullable=True),
        sa.Column("target_shot_created", sa.Boolean(), nullable=True),
        sa.Column("target_created_shot_cxg", sa.Float(), nullable=True),
        sa.Column("target_created_shot_id", sa.String(length=100), nullable=True),
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "feature_family",
            "version_tag",
            "action_id",
            name="uq_action_feature_version_action",
        ),
    )
    op.create_index(
        "ix_action_features_family_version",
        "action_features",
        ["feature_family", "version_tag"],
    )
    op.create_index("ix_action_features_match_id", "action_features", ["match_id"])
    op.create_index("ix_action_features_event_id", "action_features", ["event_id"])


def downgrade() -> None:
    op.drop_index("ix_action_features_event_id", table_name="action_features")
    op.drop_index("ix_action_features_match_id", table_name="action_features")
    op.drop_index("ix_action_features_family_version", table_name="action_features")
    op.drop_table("action_features")
