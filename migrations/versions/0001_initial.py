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
        sa.Column("slug", sa.String(length=200), nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("settings_json", sa.JSON, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "project_secrets",
        sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id"), primary_key=True),
        sa.Column("secret_type", sa.String(length=50), primary_key=True),
        sa.Column("secret_ciphertext", sa.LargeBinary, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "text_blocks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("tb_id", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("working_text", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "conversations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("project_id", sa.Integer, sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("external_id", sa.String(length=200), nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_table(
        "turns",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("conversation_id", sa.Integer, sa.ForeignKey("conversations.id"), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content_text", sa.Text, nullable=False),
        sa.Column("content_json", sa.JSON, nullable=True),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("timestamp", sa.DateTime, nullable=False),
        sa.Column("response_id", sa.String(length=200), nullable=True),
        sa.Column("metadata_json", sa.JSON, nullable=False),
    )
    op.create_table(
        "links",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("text_block_id", sa.Integer, sa.ForeignKey("text_blocks.id"), nullable=False),
        sa.Column(
            "conversation_id", sa.Integer, sa.ForeignKey("conversations.id"), nullable=False
        ),
        sa.Column("relation", sa.String(length=50), nullable=False),
        sa.Column("note", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.execute(
        """
        CREATE VIRTUAL TABLE turns_fts USING fts5(
            turn_id UNINDEXED,
            content_text,
            conversation_id UNINDEXED,
            project_id UNINDEXED
        )
        """
    )
    op.execute(
        """
        CREATE VIRTUAL TABLE text_blocks_fts USING fts5(
            text_block_id UNINDEXED,
            title,
            notes,
            working_text,
            project_id UNINDEXED
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS text_blocks_fts")
    op.execute("DROP TABLE IF EXISTS turns_fts")
    op.drop_table("links")
    op.drop_table("turns")
    op.drop_table("conversations")
    op.drop_table("text_blocks")
    op.drop_table("project_secrets")
    op.drop_table("projects")
