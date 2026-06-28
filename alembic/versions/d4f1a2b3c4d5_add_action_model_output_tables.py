"""add action model output tables

Revision ID: d4f1a2b3c4d5
Revises: c7c3d2a9b1f4
Create Date: 2026-06-28

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4f1a2b3c4d5"
down_revision: Union[str, Sequence[str], None] = "c7c3d2a9b1f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "action_predictions",
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("action_id", sa.String(length=100), nullable=True),
        sa.Column("event_id", sa.String(length=100), nullable=True),
        sa.Column("match_id", sa.Integer(), nullable=True),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("player_id", sa.Integer(), nullable=True),
        sa.Column("possession_id", sa.String(length=100), nullable=True),
        sa.Column("sequence_id", sa.String(length=100), nullable=True),
        sa.Column("action_type", sa.String(length=50), nullable=True),
        sa.Column("predicted_cxa", sa.Float(), nullable=True),
        sa.Column("predicted_value", sa.Float(), nullable=True),
        sa.Column("target_value", sa.Float(), nullable=True),
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.ForeignKeyConstraint(["model_id"], ["model_registry.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id", "action_id", name="uq_action_prediction"),
    )
    op.create_index("ix_action_predictions_match_id", "action_predictions", ["match_id"])
    op.create_index("ix_action_predictions_model_id", "action_predictions", ["model_id"])

    op.create_table(
        "action_threat_predictions",
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("action_id", sa.String(length=100), nullable=True),
        sa.Column("event_id", sa.String(length=100), nullable=True),
        sa.Column("match_id", sa.Integer(), nullable=True),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("player_id", sa.Integer(), nullable=True),
        sa.Column("possession_id", sa.String(length=100), nullable=True),
        sa.Column("sequence_id", sa.String(length=100), nullable=True),
        sa.Column("action_type", sa.String(length=50), nullable=True),
        sa.Column("start_zone", sa.String(length=50), nullable=True),
        sa.Column("end_zone", sa.String(length=50), nullable=True),
        sa.Column("cxt_value", sa.Float(), nullable=True),
        sa.Column("predicted_threat", sa.Float(), nullable=True),
        sa.Column("threat_delta", sa.Float(), nullable=True),
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.ForeignKeyConstraint(["model_id"], ["model_registry.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id", "action_id", name="uq_action_threat_prediction"),
    )
    op.create_index(
        "ix_action_threat_predictions_match_id",
        "action_threat_predictions",
        ["match_id"],
    )
    op.create_index(
        "ix_action_threat_predictions_model_id",
        "action_threat_predictions",
        ["model_id"],
    )

    op.create_table(
        "aggregates_sequence",
        sa.Column("model_id", sa.Integer(), nullable=False),
        sa.Column("model_family", sa.String(length=50), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("match_id", sa.Integer(), nullable=True),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("possession_id", sa.String(length=100), nullable=True),
        sa.Column("sequence_id", sa.String(length=100), nullable=True),
        sa.Column("total_value", sa.Float(), nullable=False),
        sa.Column("action_count", sa.Integer(), nullable=False),
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP"), nullable=False
        ),
        sa.ForeignKeyConstraint(["model_id"], ["model_registry.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_id", "match_id", "sequence_id", name="uq_sequence_aggregate"),
    )
    op.create_index("ix_aggregates_sequence_match_id", "aggregates_sequence", ["match_id"])
    op.create_index("ix_aggregates_sequence_model_id", "aggregates_sequence", ["model_id"])


def downgrade() -> None:
    op.drop_index("ix_aggregates_sequence_model_id", table_name="aggregates_sequence")
    op.drop_index("ix_aggregates_sequence_match_id", table_name="aggregates_sequence")
    op.drop_table("aggregates_sequence")
    op.drop_index("ix_action_threat_predictions_model_id", table_name="action_threat_predictions")
    op.drop_index("ix_action_threat_predictions_match_id", table_name="action_threat_predictions")
    op.drop_table("action_threat_predictions")
    op.drop_index("ix_action_predictions_model_id", table_name="action_predictions")
    op.drop_index("ix_action_predictions_match_id", table_name="action_predictions")
    op.drop_table("action_predictions")
