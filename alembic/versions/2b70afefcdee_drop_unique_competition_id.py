"""drop_unique_competition_id

Revision ID: 2b70afefcdee
Revises: 001_initial
Create Date: 2025-11-12 19:54:27.004852

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "2b70afefcdee"
down_revision: Union[str, Sequence[str], None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop incorrect unique constraint on competitions.statsbomb_competition_id
    #
    # SQLite does not support ALTER TABLE DROP CONSTRAINT. Alembic batch mode
    # recreates the table behind the scenes on SQLite while preserving the
    # direct ALTER behaviour on dialects that support it, such as Postgres.
    with op.batch_alter_table("competitions") as batch_op:
        batch_op.drop_constraint(
            "uq_competitions_statsbomb_competition_id",
            type_="unique",
        )

    # These FKs were detected as missing by autogenerate; ensure they exist
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        op.create_foreign_key(
            op.f("fk_possessions_start_event_id_events"),
            "possessions",
            "events",
            ["start_event_id"],
            ["id"],
        )
        op.create_foreign_key(
            op.f("fk_possessions_end_event_id_events"),
            "possessions",
            "events",
            ["end_event_id"],
            ["id"],
        )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        # Remove added FKs. SQLite cannot drop constraints directly and the
        # upgrade intentionally does not add these constraints on SQLite.
        op.drop_constraint(
            op.f("fk_possessions_end_event_id_events"),
            "possessions",
            type_="foreignkey",
        )
        op.drop_constraint(
            op.f("fk_possessions_start_event_id_events"),
            "possessions",
            type_="foreignkey",
        )
    # Recreate original unique constraint
    with op.batch_alter_table("competitions") as batch_op:
        batch_op.create_unique_constraint(
            "uq_competitions_statsbomb_competition_id",
            ["statsbomb_competition_id"],
        )
