from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app import crud


@pytest.fixture()
def sqlite_db_path() -> Path:
    with NamedTemporaryFile(suffix=".db") as tmp:
        yield Path(tmp.name)


def run_migrations(db_path: Path) -> None:
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite+pysqlite:///{db_path}")
    command.upgrade(alembic_cfg, "head")


def build_session(db_path: Path):
    engine = create_engine(
        f"sqlite+pysqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return session_factory


def test_migrations_apply(sqlite_db_path: Path) -> None:
    run_migrations(sqlite_db_path)
    engine = create_engine(f"sqlite+pysqlite:///{sqlite_db_path}")
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert "projects" in tables
    assert "project_secrets" in tables
    assert "text_blocks" in tables
    assert "conversations" in tables
    assert "turns" in tables
    assert "links" in tables


def test_crud_roundtrip(sqlite_db_path: Path) -> None:
    run_migrations(sqlite_db_path)
    session_factory = build_session(sqlite_db_path)
    with session_factory() as session:
        project = crud.create_project(
            session,
            name="Thesis",
            slug="thesis",
            description="My project",
            settings_json={"language": "de"},
        )
        secret = crud.create_project_secret(
            session,
            project=project,
            secret_type="openai_api_key",
            secret_ciphertext=b"ciphertext",
        )
        block = crud.create_text_block(
            session,
            project=project,
            tb_id="TB-RW-03",
            title="Intro",
            block_type="draft",
            status="active",
            notes="notes",
            working_text="working text",
        )
        conversation = crud.create_conversation(
            session,
            project=project,
            title="Kickoff",
            source="openai_api",
        )
        turn = crud.create_turn(
            session,
            conversation=conversation,
            role="user",
            content_text="Hello world",
            model="gpt-test",
            timestamp=datetime.utcnow(),
            content_json={"foo": "bar"},
            metadata_json={"token_count": 10},
        )
        link = crud.create_link(
            session,
            text_block=block,
            conversation=conversation,
            relation="adopted",
            note="useful",
        )

        fetched_project = crud.get_project(session, project.id)
        assert fetched_project
        assert fetched_project.name == "Thesis"
        assert fetched_project.settings_json == {"language": "de"}
        assert fetched_project.secrets[0].secret_type == "openai_api_key"
        assert fetched_project.text_blocks[0].tb_id == "TB-RW-03"
        assert fetched_project.conversations[0].title == "Kickoff"

        assert secret.project_id == project.id
        assert block.project_id == project.id
        assert conversation.project_id == project.id
        assert turn.conversation_id == conversation.id
        assert link.text_block_id == block.id

        turn_results = crud.search_turns(session, query="Hello")
        assert turn_results
        assert turn_results[0]["turn_id"] == turn.id

        block_results = crud.search_text_blocks(session, query="Intro")
        assert block_results
        assert block_results[0]["text_block_id"] == block.id

        export_rows = crud.list_turns_for_project(session, project_id=project.id)
        assert export_rows
        assert export_rows[0]["project_id"] == project.id
