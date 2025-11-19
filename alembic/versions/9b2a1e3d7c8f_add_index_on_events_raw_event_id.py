"""add index on events.raw_event_id

Revision ID: 9b2a1e3d7c8f
Revises: 14f289cc51b0
Create Date: 2025-11-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9b2a1e3d7c8f'
down_revision: Union[str, Sequence[str], None] = '14f289cc51b0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index('ix_events_raw_event_id', 'events', ['raw_event_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_events_raw_event_id', table_name='events')
