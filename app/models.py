from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utc_now() -> datetime:
    return datetime.utcnow()


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    slug: Mapped[str] = mapped_column(String(200), unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    settings_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    secrets: Mapped[list["ProjectSecret"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    text_blocks: Mapped[list["TextBlock"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class ProjectSecret(Base):
    __tablename__ = "project_secrets"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), primary_key=True)
    secret_type: Mapped[str] = mapped_column(String(50), primary_key=True)
    secret_ciphertext: Mapped[bytes] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)

    project: Mapped[Project] = relationship(back_populates="secrets")


class TextBlock(Base):
    __tablename__ = "text_blocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    tb_id: Mapped[str] = mapped_column(String(50))
    title: Mapped[str] = mapped_column(String(200))
    type: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(50))
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)
    working_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)

    project: Mapped[Project] = relationship(back_populates="text_blocks")
    links: Mapped[list["Link"]] = relationship(
        back_populates="text_block", cascade="all, delete-orphan"
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    title: Mapped[str] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(50))
    external_id: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)

    project: Mapped[Project] = relationship(back_populates="conversations")
    turns: Mapped[list["Turn"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    links: Mapped[list["Link"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Turn(Base):
    __tablename__ = "turns"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    role: Mapped[str] = mapped_column(String(20))
    content_text: Mapped[str] = mapped_column(Text)
    content_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model: Mapped[str] = mapped_column(String(100))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    response_id: Mapped[str | None] = mapped_column(String(200))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)

    conversation: Mapped[Conversation] = relationship(back_populates="turns")


class Link(Base):
    __tablename__ = "links"

    id: Mapped[int] = mapped_column(primary_key=True)
    text_block_id: Mapped[int] = mapped_column(ForeignKey("text_blocks.id"))
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"))
    relation: Mapped[str] = mapped_column(String(50))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    text_block: Mapped[TextBlock] = relationship(back_populates="links")
    conversation: Mapped[Conversation] = relationship(back_populates="links")
