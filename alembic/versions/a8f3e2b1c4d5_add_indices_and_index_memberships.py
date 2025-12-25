"""add indices and index_memberships tables

Revision ID: a8f3e2b1c4d5
Revises: 512fbe0aa3a1
Create Date: 2025-12-25 15:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a8f3e2b1c4d5'
down_revision: Union[str, None] = '512fbe0aa3a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create indices table
    op.create_table(
        'indices',
        sa.Column('index_id', sa.String(50), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('weighting_scheme', sa.String(20), nullable=False),
        sa.Column('base_value', sa.Numeric(), server_default='100.0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Create index_memberships table
    op.create_table(
        'index_memberships',
        sa.Column('index_id', sa.String(50), sa.ForeignKey('indices.index_id'), primary_key=True, nullable=False),
        sa.Column('as_of_time', sa.DateTime(), primary_key=True, nullable=False),
        sa.Column('ticker', sa.String(20), sa.ForeignKey('tickers.ticker'), primary_key=True, nullable=False),
        sa.Column('is_member', sa.Boolean(), server_default='true'),
        sa.Column('source', sa.String(50), nullable=True),
        sa.Column('ingested_at', sa.DateTime(), server_default=sa.func.now()),
    )

    # Create indexes for efficient queries
    op.create_index(
        'idx_membership_index_time',
        'index_memberships',
        ['index_id', 'as_of_time']
    )
    op.create_index(
        'idx_membership_ticker',
        'index_memberships',
        ['ticker']
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index('idx_membership_ticker', table_name='index_memberships')
    op.drop_index('idx_membership_index_time', table_name='index_memberships')

    # Drop tables
    op.drop_table('index_memberships')
    op.drop_table('indices')
