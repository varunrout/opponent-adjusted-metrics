"""drop_unique_competition_id

Revision ID: 2b70afefcdee
Revises: 001_initial
Create Date: 2025-11-12 19:54:27.004852

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2b70afefcdee'
down_revision: Union[str, Sequence[str], None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop incorrect unique constraint on competitions.statsbomb_competition_id
    op.drop_constraint('uq_competitions_statsbomb_competition_id', 'competitions', type_='unique')

    # These FKs were detected as missing by autogenerate; ensure they exist
    op.create_foreign_key(op.f('fk_possessions_start_event_id_events'), 'possessions', 'events', ['start_event_id'], ['id'])
    op.create_foreign_key(op.f('fk_possessions_end_event_id_events'), 'possessions', 'events', ['end_event_id'], ['id'])


def downgrade() -> None:
    """Downgrade schema."""
    # Remove added FKs
    op.drop_constraint(op.f('fk_possessions_end_event_id_events'), 'possessions', type_='foreignkey')
    op.drop_constraint(op.f('fk_possessions_start_event_id_events'), 'possessions', type_='foreignkey')
    # Recreate original unique constraint
    op.create_unique_constraint('uq_competitions_statsbomb_competition_id', 'competitions', ['statsbomb_competition_id'])
