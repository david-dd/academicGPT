"""initial

Revision ID: 0001
Revises: 
Create Date: 2024-01-01
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "projects",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("api_key_encrypted", sa.LargeBinary, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "text_blocks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("text_block_id", sa.Integer, sa.ForeignKey("text_blocks.id"), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("openai_previous_response_id", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "turns",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("conversation_id", sa.Integer, sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("response", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.execute(
        """
        CREATE VIRTUAL TABLE turns_fts USING fts5(
            turn_id UNINDEXED,
            prompt,
            response,
            project_id UNINDEXED,
            text_block_id UNINDEXED
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS turns_fts")
    op.drop_table("turns")
    op.drop_table("conversations")
    op.drop_table("text_blocks")
    op.drop_table("projects")
