from datetime import datetime

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models import (
    Conversation,
    Link,
    Project,
    ProjectSecret,
    TextBlock,
    Turn,
)

HIGHLIGHT_START = "[[[H]]]"
HIGHLIGHT_END = "[[[/H]]]"


def create_project(
    db: Session,
    name: str,
    slug: str,
    description: str | None = None,
    settings_json: dict | None = None,
) -> Project:
    project = Project(name=name, slug=slug, description=description, settings_json=settings_json)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def get_project(db: Session, project_id: int, include_archived: bool = False) -> Project | None:
    stmt = select(Project).where(Project.id == project_id)
    if not include_archived:
        stmt = stmt.where(Project.deleted_at.is_(None))
    return db.scalar(stmt)


def get_project_by_slug(
    db: Session, slug: str, include_archived: bool = False
) -> Project | None:
    stmt = select(Project).where(Project.slug == slug)
    if not include_archived:
        stmt = stmt.where(Project.deleted_at.is_(None))
    return db.scalar(stmt)


def list_projects(db: Session, include_archived: bool = False) -> list[Project]:
    stmt = select(Project)
    if not include_archived:
        stmt = stmt.where(Project.deleted_at.is_(None))
    return db.scalars(stmt.order_by(Project.created_at.desc())).all()


def project_slug_exists(db: Session, slug: str) -> bool:
    return db.scalar(select(func.count(Project.id)).where(Project.slug == slug)) > 0


def project_name_exists(db: Session, name: str) -> bool:
    return db.scalar(select(func.count(Project.id)).where(Project.name == name)) > 0


def update_project(
    db: Session,
    project: Project,
    name: str | None = None,
    slug: str | None = None,
    description: str | None = None,
    settings_json: dict | None = None,
) -> Project:
    if name is not None:
        project.name = name
    if slug is not None:
        project.slug = slug
    if description is not None:
        project.description = description
    project.settings_json = settings_json
    db.commit()
    db.refresh(project)
    return project


def soft_delete_project(db: Session, project: Project) -> Project:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    project.deleted_at = datetime.utcnow()
    project.name = f"{project.name} (archived {timestamp})"
    project.slug = f"{project.slug}-archived-{timestamp}"
    db.commit()
    db.refresh(project)
    return project


def create_project_secret(
    db: Session, project: Project, secret_type: str, secret_ciphertext: bytes
) -> ProjectSecret:
    secret = ProjectSecret(
        project=project, secret_type=secret_type, secret_ciphertext=secret_ciphertext
    )
    db.add(secret)
    db.commit()
    db.refresh(secret)
    return secret


def upsert_project_secret(
    db: Session, project: Project, secret_type: str, secret_ciphertext: bytes
) -> ProjectSecret:
    secret = db.get(ProjectSecret, (project.id, secret_type))
    if secret:
        secret.secret_ciphertext = secret_ciphertext
        db.commit()
        db.refresh(secret)
        return secret
    return create_project_secret(
        db, project=project, secret_type=secret_type, secret_ciphertext=secret_ciphertext
    )


def create_text_block(
    db: Session,
    project: Project,
    tb_id: str,
    title: str,
    block_type: str,
    status: str,
    notes: str | None = None,
    working_text: str | None = None,
) -> TextBlock:
    block = TextBlock(
        project=project,
        tb_id=tb_id,
        title=title,
        type=block_type,
        status=status,
        notes=notes,
        working_text=working_text,
    )
    db.add(block)
    db.commit()
    db.refresh(block)
    db.execute(
        text(
            """
            INSERT INTO text_blocks_fts(text_block_id, title, notes, working_text, project_id)
            VALUES (:text_block_id, :title, :notes, :working_text, :project_id)
            """
        ),
        {
            "text_block_id": block.id,
            "title": block.title,
            "notes": block.notes or "",
            "working_text": block.working_text or "",
            "project_id": project.id,
        },
    )
    db.commit()
    return block


def update_text_block(
    db: Session,
    block: TextBlock,
    title: str | None = None,
    block_type: str | None = None,
    status: str | None = None,
    notes: str | None = None,
    working_text: str | None = None,
) -> TextBlock:
    if title is not None:
        block.title = title
    if block_type is not None:
        block.type = block_type
    if status is not None:
        block.status = status
    if notes is not None:
        block.notes = notes
    if working_text is not None:
        block.working_text = working_text
    db.commit()
    db.refresh(block)
    db.execute(
        text(
            """
            UPDATE text_blocks_fts
            SET title = :title,
                notes = :notes,
                working_text = :working_text
            WHERE text_block_id = :text_block_id
            """
        ),
        {
            "text_block_id": block.id,
            "title": block.title,
            "notes": block.notes or "",
            "working_text": block.working_text or "",
        },
    )
    db.commit()
    return block


def create_conversation(
    db: Session,
    project: Project,
    title: str,
    source: str,
    external_id: str | None = None,
    notes: str | None = None,
) -> Conversation:
    conversation = Conversation(
        project=project,
        title=title,
        source=source,
        external_id=external_id,
        notes=notes,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def update_conversation(
    db: Session,
    conversation: Conversation,
    title: str | None = None,
    notes: str | None = None,
) -> Conversation:
    if title is not None:
        conversation.title = title
    if notes is not None:
        conversation.notes = notes
    db.commit()
    db.refresh(conversation)
    return conversation


def list_text_blocks_overview(db: Session, project_id: int) -> list[dict]:
    stmt = (
        select(
            TextBlock,
            func.count(func.distinct(Link.conversation_id)).label("conversation_count"),
            func.count(Turn.id).label("turn_count"),
        )
        .outerjoin(Link, Link.text_block_id == TextBlock.id)
        .outerjoin(Turn, Turn.conversation_id == Link.conversation_id)
        .where(TextBlock.project_id == project_id)
        .group_by(TextBlock.id)
        .order_by(TextBlock.created_at.desc(), TextBlock.id.desc())
    )
    rows = db.execute(stmt).all()
    return [
        {
            "block": row[0],
            "conversation_count": row[1] or 0,
            "turn_count": row[2] or 0,
        }
        for row in rows
    ]


def get_turn_counts_for_conversations(
    db: Session, conversation_ids: list[int]
) -> dict[int, int]:
    if not conversation_ids:
        return {}
    stmt = (
        select(Turn.conversation_id, func.count(Turn.id))
        .where(Turn.conversation_id.in_(conversation_ids))
        .group_by(Turn.conversation_id)
    )
    rows = db.execute(stmt).all()
    return {row[0]: row[1] for row in rows}


def create_turn(
    db: Session,
    conversation: Conversation,
    role: str,
    content_text: str,
    model: str,
    timestamp: datetime | None = None,
    content_json: dict | None = None,
    response_id: str | None = None,
    metadata_json: dict | None = None,
) -> Turn:
    turn = Turn(
        conversation=conversation,
        role=role,
        content_text=content_text,
        model=model,
        timestamp=timestamp or datetime.utcnow(),
        content_json=content_json,
        response_id=response_id,
        metadata_json=metadata_json or {},
    )
    db.add(turn)
    db.commit()
    db.refresh(turn)
    db.execute(
        text(
            """
            INSERT INTO turns_fts(turn_id, content_text, conversation_id, project_id)
            VALUES (:turn_id, :content_text, :conversation_id, :project_id)
            """
        ),
        {
            "turn_id": turn.id,
            "content_text": turn.content_text,
            "conversation_id": conversation.id,
            "project_id": conversation.project_id,
        },
    )
    db.commit()
    return turn


def create_link(
    db: Session,
    text_block: TextBlock,
    conversation: Conversation,
    relation: str,
    note: str | None = None,
) -> Link:
    link = Link(
        text_block=text_block, conversation=conversation, relation=relation, note=note
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    return link


def list_turns_for_conversation(db: Session, conversation_id: int) -> list[Turn]:
    stmt = (
        select(Turn)
        .where(Turn.conversation_id == conversation_id)
        .order_by(Turn.timestamp.asc(), Turn.id.asc())
    )
    return db.scalars(stmt).all()


def get_latest_response_id(db: Session, conversation_id: int) -> str | None:
    stmt = (
        select(Turn.response_id)
        .where(
            Turn.conversation_id == conversation_id,
            Turn.role == "assistant",
            Turn.response_id.isnot(None),
        )
        .order_by(Turn.timestamp.desc(), Turn.id.desc())
        .limit(1)
    )
    return db.scalar(stmt)


def get_initial_system_turn(db: Session, conversation_id: int) -> Turn | None:
    stmt = (
        select(Turn)
        .where(Turn.conversation_id == conversation_id, Turn.role == "system")
        .order_by(Turn.timestamp.asc(), Turn.id.asc())
        .limit(1)
    )
    return db.scalar(stmt)


def get_latest_turn_by_role(db: Session, conversation_id: int, role: str) -> Turn | None:
    stmt = (
        select(Turn)
        .where(Turn.conversation_id == conversation_id, Turn.role == role)
        .order_by(Turn.timestamp.desc(), Turn.id.desc())
        .limit(1)
    )
    return db.scalar(stmt)


def upsert_system_turn(
    db: Session, conversation: Conversation, content_text: str
) -> Turn:
    system_turn = get_initial_system_turn(db, conversation_id=conversation.id)
    if system_turn:
        system_turn.content_text = content_text
        system_turn.timestamp = datetime.utcnow()
        system_turn.model = "system"
        db.add(system_turn)
        db.commit()
        db.execute(
            text(
                """
                UPDATE turns_fts
                SET content_text = :content_text
                WHERE turn_id = :turn_id
                """
            ),
            {"content_text": content_text, "turn_id": system_turn.id},
        )
        db.commit()
        db.refresh(system_turn)
        return system_turn
    return create_turn(
        db,
        conversation=conversation,
        role="system",
        content_text=content_text,
        model="system",
    )


def search_turns(db: Session, query: str) -> list[dict]:
    sql = (
        "SELECT turn_id, content_text, conversation_id, project_id "
        "FROM turns_fts WHERE turns_fts MATCH :query"
    )
    rows = db.execute(text(sql), {"query": query}).mappings().all()
    return list(rows)


def search_turns_for_project(db: Session, project_id: int, query: str) -> list[dict]:
    sql = (
        "SELECT turn_id, content_text, conversation_id, project_id "
        "FROM turns_fts WHERE turns_fts MATCH :query AND project_id = :project_id"
    )
    rows = db.execute(text(sql), {"query": query, "project_id": project_id}).mappings().all()
    return list(rows)


def search_turns_for_project_snippets(
    db: Session, project_id: int, query: str, limit: int = 25
) -> list[dict]:
    sql = f"""
        SELECT turns_fts.turn_id,
               turns_fts.conversation_id,
               turns_fts.project_id,
               conversations.title AS conversation_title,
               snippet(turns_fts, 1, '{HIGHLIGHT_START}', '{HIGHLIGHT_END}', '…', 12)
                   AS snippet
        FROM turns_fts
        JOIN turns ON turns.id = turns_fts.turn_id
        JOIN conversations ON conversations.id = turns_fts.conversation_id
        WHERE turns_fts MATCH :query AND turns_fts.project_id = :project_id
        ORDER BY bm25(turns_fts)
        LIMIT :limit
    """
    rows = db.execute(
        text(sql), {"query": query, "project_id": project_id, "limit": limit}
    ).mappings().all()
    return list(rows)


def search_text_blocks(db: Session, query: str) -> list[dict]:
    sql = (
        "SELECT text_block_id, title, notes, working_text, project_id "
        "FROM text_blocks_fts WHERE text_blocks_fts MATCH :query"
    )
    rows = db.execute(text(sql), {"query": query}).mappings().all()
    return list(rows)


def search_text_blocks_for_project_snippets(
    db: Session, project_id: int, query: str, limit: int = 25
) -> list[dict]:
    sql = f"""
        SELECT text_blocks_fts.text_block_id,
               text_blocks.title,
               text_blocks_fts.project_id,
               snippet(text_blocks_fts, 1, '{HIGHLIGHT_START}', '{HIGHLIGHT_END}', '…', 12)
                   AS title_snippet,
               snippet(text_blocks_fts, 2, '{HIGHLIGHT_START}', '{HIGHLIGHT_END}', '…', 12)
                   AS notes_snippet,
               snippet(text_blocks_fts, 3, '{HIGHLIGHT_START}', '{HIGHLIGHT_END}', '…', 12)
                   AS working_snippet
        FROM text_blocks_fts
        JOIN text_blocks ON text_blocks.id = text_blocks_fts.text_block_id
        WHERE text_blocks_fts MATCH :query AND text_blocks_fts.project_id = :project_id
        ORDER BY bm25(text_blocks_fts)
        LIMIT :limit
    """
    rows = db.execute(
        text(sql), {"query": query, "project_id": project_id, "limit": limit}
    ).mappings().all()
    return list(rows)


def list_turns_for_project(db: Session, project_id: int) -> list[dict]:
    rows = db.execute(
        text(
            """
            SELECT turns.id AS turn_id,
                   turns.role AS role,
                   turns.content_text AS content_text,
                   turns.model AS model,
                   turns.timestamp AS timestamp,
                   conversations.id AS conversation_id,
                   projects.id AS project_id,
                   GROUP_CONCAT(links.text_block_id) AS text_block_ids
            FROM turns
            JOIN conversations ON conversations.id = turns.conversation_id
            JOIN projects ON projects.id = conversations.project_id
            LEFT JOIN links ON links.conversation_id = conversations.id
            WHERE projects.id = :project_id
            GROUP BY turns.id
            ORDER BY turns.timestamp DESC
            """
        ),
        {"project_id": project_id},
    ).mappings().all()
    return list(rows)


def get_project_stats(db: Session, project_id: int) -> dict:
    text_blocks = db.scalar(
        select(func.count(TextBlock.id)).where(TextBlock.project_id == project_id)
    )
    conversations = db.scalar(
        select(func.count(Conversation.id)).where(Conversation.project_id == project_id)
    )
    turns = db.scalar(
        select(func.count(Turn.id))
        .join(Conversation, Conversation.id == Turn.conversation_id)
        .where(Conversation.project_id == project_id)
    )
    return {
        "text_blocks": text_blocks or 0,
        "conversations": conversations or 0,
        "turns": turns or 0,
    }
