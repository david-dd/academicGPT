from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import Conversation, Project, TextBlock, Turn


def create_project(
    db: Session, name: str, description: str | None, api_key_encrypted: bytes | None
) -> Project:
    project = Project(name=name, description=description, api_key_encrypted=api_key_encrypted)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def list_projects(db: Session) -> list[Project]:
    return db.scalars(select(Project).order_by(Project.created_at.desc())).all()


def get_project(db: Session, project_id: int) -> Project | None:
    return db.get(Project, project_id)


def create_text_block(
    db: Session, project: Project, title: str, description: str | None
) -> TextBlock:
    block = TextBlock(project=project, title=title, description=description)
    db.add(block)
    db.commit()
    db.refresh(block)
    return block


def create_conversation(db: Session, block: TextBlock, title: str) -> Conversation:
    conversation = Conversation(text_block=block, title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def create_turn(
    db: Session, conversation: Conversation, prompt: str, response: str
) -> Turn:
    turn = Turn(conversation=conversation, prompt=prompt, response=response)
    db.add(turn)
    db.commit()
    db.refresh(turn)

    db.execute(
        text(
            """
            INSERT INTO turns_fts(turn_id, prompt, response, project_id, text_block_id)
            VALUES (:turn_id, :prompt, :response, :project_id, :text_block_id)
            """
        ),
        {
            "turn_id": turn.id,
            "prompt": prompt,
            "response": response,
            "project_id": conversation.text_block.project_id,
            "text_block_id": conversation.text_block_id,
        },
    )
    db.commit()
    return turn


def search_turns(db: Session, query: str, project_id: int | None = None) -> list[dict]:
    sql = (
        "SELECT turn_id, prompt, response, project_id, text_block_id "
        "FROM turns_fts WHERE turns_fts MATCH :query"
    )
    params = {"query": query}
    if project_id is not None:
        sql += " AND project_id = :project_id"
        params["project_id"] = project_id
    rows = db.execute(text(sql), params).mappings().all()
    return list(rows)


def list_turns_for_project(db: Session, project_id: int) -> list[dict]:
    rows = db.execute(
        text(
            """
            SELECT turns.id AS turn_id,
                   turns.prompt AS prompt,
                   turns.response AS response,
                   conversations.id AS conversation_id,
                   text_blocks.id AS text_block_id,
                   projects.id AS project_id
            FROM turns
            JOIN conversations ON conversations.id = turns.conversation_id
            JOIN text_blocks ON text_blocks.id = conversations.text_block_id
            JOIN projects ON projects.id = text_blocks.project_id
            WHERE projects.id = :project_id
            ORDER BY turns.created_at DESC
            """
        ),
        {"project_id": project_id},
    ).mappings().all()
    return list(rows)
