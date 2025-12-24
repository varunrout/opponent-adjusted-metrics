"""add pass end location

Revision ID: c7c3d2a9b1f4
Revises: 9b2a1e3d7c8f
Create Date: 2025-12-24

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c7c3d2a9b1f4"
down_revision: Union[str, Sequence[str], None] = "9b2a1e3d7c8f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("passes", sa.Column("end_x", sa.Float(), nullable=True))
    op.add_column("passes", sa.Column("end_y", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("passes", "end_y")
    op.drop_column("passes", "end_x")
