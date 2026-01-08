"""add project soft delete

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-02
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("projects", sa.Column("deleted_at", sa.DateTime, nullable=True))


def downgrade() -> None:
    op.drop_column("projects", "deleted_at")
