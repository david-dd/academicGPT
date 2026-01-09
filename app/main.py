import html
import json
import re
from datetime import datetime

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from markupsafe import Markup

from app import crud
from app.db import get_db
from openai import APIStatusError, AuthenticationError, OpenAIError, RateLimitError

from app.models import Conversation, Link, Project, TextBlock
from app.openai_client import get_openai_client
from app.security import EncryptionError, decrypt_secret, encrypt_secret

app = FastAPI(title="AI Logger")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

RELATION_TYPES = ["adopted", "supporting", "ideation", "rejected"]
RELATION_LABELS = {
    "adopted": "adopted",
    "supporting": "supporting",
    "ideation": "ideation (nicht übernommen)",
    "rejected": "rejected",
}


def slugify(value: str) -> str:
    return "-".join(value.lower().strip().split())


def get_project_by_slug_or_404(db: Session, project_slug: str):
    project = crud.get_project_by_slug(db, project_slug)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def get_block_or_404(db: Session, project: Project, block_id: int) -> TextBlock:
    block = db.get(TextBlock, block_id)
    if not block or block.project_id != project.id:
        raise HTTPException(status_code=404, detail="Text block not found")
    return block


def get_conversation_or_404(
    db: Session, project: Project, conversation_id: int
) -> Conversation:
    conversation = db.get(Conversation, conversation_id)
    if not conversation or conversation.project_id != project.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


def get_project_openai_key(project: Project) -> str:
    secret = next((s for s in project.secrets if s.secret_type == "openai_api_key"), None)
    if not secret:
        raise HTTPException(status_code=400, detail="Project API key not configured")
    try:
        return decrypt_secret(secret.secret_ciphertext)
    except EncryptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def get_project_default_model(project: Project) -> str | None:
    if not project.settings_json:
        return None
    model = project.settings_json.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def list_available_models(project: Project) -> tuple[list[str], str | None]:
    try:
        api_key = get_project_openai_key(project)
    except HTTPException as exc:
        return [], str(exc.detail)
    client = get_openai_client(api_key)
    try:
        response = client.models.list()
    except AuthenticationError:
        return [], "Authentication failed. Please check the project API key."
    except RateLimitError:
        return [], "Rate limit reached. Please wait and try again."
    except APIStatusError as exc:
        return [], f"API error: {exc.status_code} {exc.message}"
    except OpenAIError as exc:
        return [], f"API error: {exc}"
    data = getattr(response, "data", response)
    model_ids = []
    for item in data or []:
        model_id = getattr(item, "id", None)
        if not model_id and isinstance(item, dict):
            model_id = item.get("id")
        if model_id:
            model_ids.append(model_id)
    model_ids = sorted(set(model_ids))
    return model_ids, None


def serialize_usage(usage) -> dict | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return dict(usage)


def extract_query_terms(query: str | None) -> list[str]:
    if not query:
        return []
    return re.findall(r"[\wÀ-ÖØ-öø-ÿ]+", query, flags=re.UNICODE)


def build_highlight_pattern(terms: list[str]) -> re.Pattern[str] | None:
    unique_terms = sorted({term for term in terms if term}, key=len, reverse=True)
    if not unique_terms:
        return None
    return re.compile(
        r"(" + "|".join(re.escape(term) for term in unique_terms) + r")",
        flags=re.IGNORECASE,
    )


def highlight_text(text: str | None, pattern: re.Pattern[str] | None) -> Markup:
    if not text:
        return Markup("")
    escaped = html.escape(text)
    if not pattern:
        return Markup(escaped)
    return Markup(pattern.sub(r"<mark>\1</mark>", escaped))


def render_snippet(snippet: str | None) -> Markup:
    if not snippet:
        return Markup("")
    escaped = html.escape(snippet)
    escaped = escaped.replace(crud.HIGHLIGHT_START, "<mark>").replace(
        crud.HIGHLIGHT_END, "</mark>"
    )
    return Markup(escaped)


def render_conversation_detail(
    request: Request,
    project: Project,
    conversation: Conversation,
    db: Session,
    error: str | None = None,
    highlight_query: str | None = None,
    selected_model: str | None = None,
):
    turns = crud.list_turns_for_conversation(db, conversation_id=conversation.id)
    system_turn = crud.get_initial_system_turn(db, conversation_id=conversation.id)
    available_models, model_list_error = list_available_models(project)
    default_model = get_project_default_model(project)
    if not selected_model:
        selected_model = default_model or (available_models[0] if available_models else "gpt-4o-mini")
    highlight_terms = extract_query_terms(highlight_query)
    highlight_pattern = build_highlight_pattern(highlight_terms)
    rendered_turns = []
    for turn in turns:
        rendered_turns.append(
            {
                "id": turn.id,
                "role": turn.role,
                "model": turn.model,
                "timestamp": turn.timestamp,
                "content_html": highlight_text(turn.content_text, highlight_pattern),
                "is_match": bool(
                    highlight_pattern and highlight_pattern.search(turn.content_text or "")
                ),
            }
        )
    return templates.TemplateResponse(
        "conversation_detail.html",
        {
            "request": request,
            "project": project,
            "conversation": conversation,
            "turns": rendered_turns,
            "error": error,
            "relation_types": RELATION_TYPES,
            "relation_labels": RELATION_LABELS,
            "project_blocks": crud.list_text_blocks(db, project_id=project.id),
            "highlight_query": highlight_query or "",
            "system_instruction": system_turn.content_text if system_turn else "",
            "available_models": available_models,
            "model_list_error": model_list_error,
            "selected_model": selected_model,
        },
    )


def render_project_settings(
    request: Request,
    project: Project,
    db: Session,
    settings_payload: str,
    error: str | None = None,
    test_result: str | None = None,
    test_status: str | None = None,
):
    projects = crud.list_projects(db)
    return templates.TemplateResponse(
        "project_settings.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "settings_payload": settings_payload,
            "error": error,
            "test_result": test_result,
            "test_status": test_status,
        },
    )


def filter_links(
    links: list[Link],
    relation: str | None = None,
    query: str | None = None,
    db: Session | None = None,
    project_id: int | None = None,
) -> list[Link]:
    filtered = links
    if relation and relation != "all":
        filtered = [link for link in filtered if link.relation == relation]
    if query:
        lowered = query.lower()
        matched_conversation_ids = {
            link.conversation.id
            for link in filtered
            if lowered in link.conversation.title.lower()
        }
        if db and project_id:
            matches = crud.search_turns_for_project(
                db, project_id=project_id, query=query
            )
            matched_conversation_ids.update(
                row["conversation_id"] for row in matches
            )
        filtered = [
            link
            for link in filtered
            if link.conversation.id in matched_conversation_ids
        ]
    return filtered


def build_unique_slug(db: Session, name: str) -> str:
    base_slug = slugify(name)
    slug = base_slug
    counter = 2
    while crud.project_slug_exists(db, slug):
        slug = f"{base_slug}-{counter}"
        counter += 1
    return slug


def build_search_results(db: Session, project_id: int, query: str | None) -> dict:
    if not query:
        return {"turn_results": [], "text_block_results": []}
    turn_rows = crud.search_turns_for_project_snippets(
        db, project_id=project_id, query=query
    )
    text_block_rows = crud.search_text_blocks_for_project_snippets(
        db, project_id=project_id, query=query
    )

    turn_results = []
    for row in turn_rows:
        turn_results.append(
            {
                "turn_id": row["turn_id"],
                "conversation_id": row["conversation_id"],
                "conversation_title": row["conversation_title"],
                "snippet": render_snippet(row.get("snippet")),
            }
        )

    text_block_results = []
    for row in text_block_rows:
        snippets = [
            row.get("title_snippet"),
            row.get("notes_snippet"),
            row.get("working_snippet"),
        ]
        chosen_snippet = next(
            (snippet for snippet in snippets if snippet and crud.HIGHLIGHT_START in snippet),
            None,
        )
        if not chosen_snippet:
            chosen_snippet = next((snippet for snippet in snippets if snippet), "")
        text_block_results.append(
            {
                "text_block_id": row["text_block_id"],
                "title": row["title"],
                "snippet": render_snippet(chosen_snippet),
            }
        )

    return {"turn_results": turn_results, "text_block_results": text_block_results}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    projects = crud.list_projects(db)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "projects": projects, "error": None},
    )


@app.post("/projects")
def create_project(
    request: Request,
    name: str = Form(...),
    description: str | None = Form(None),
    api_key: str | None = Form(None),
    db: Session = Depends(get_db),
):
    if crud.project_name_exists(db, name):
        projects = crud.list_projects(db)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "projects": projects,
                "error": "Project name already exists.",
            },
            status_code=400,
        )
    try:
        api_key_encrypted = encrypt_secret(api_key) if api_key else None
    except EncryptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    slug = build_unique_slug(db, name)
    project = crud.create_project(db, name=name, slug=slug, description=description, settings_json=None)
    if api_key_encrypted:
        crud.create_project_secret(
            db,
            project=project,
            secret_type="openai_api_key",
            secret_ciphertext=api_key_encrypted,
        )
    return RedirectResponse(url=f"/p/{project.slug}/dashboard", status_code=303)


@app.get("/p/{project_slug}/dashboard", response_class=HTMLResponse)
def project_dashboard(project_slug: str, request: Request, db: Session = Depends(get_db)):
    project = get_project_by_slug_or_404(db, project_slug)
    stats = crud.get_project_stats(db, project_id=project.id)
    text_blocks_overview = crud.list_text_blocks_overview(db, project_id=project.id)
    has_key = any(secret.secret_type == "openai_api_key" for secret in project.secrets)
    projects = crud.list_projects(db)
    return templates.TemplateResponse(
        "project_dashboard.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "stats": stats,
            "has_key": has_key,
            "text_blocks_overview": text_blocks_overview,
            "relation_labels": RELATION_LABELS,
        },
    )


@app.get("/p/{project_slug}/conversations", response_class=HTMLResponse)
def list_conversations(
    project_slug: str, request: Request, db: Session = Depends(get_db)
):
    project = get_project_by_slug_or_404(db, project_slug)
    return templates.TemplateResponse(
        "conversations.html",
        {
            "request": request,
            "project": project,
            "conversations": project.conversations,
        },
    )


@app.post("/p/{project_slug}/conversations")
def create_project_conversation(
    project_slug: str,
    title: str = Form(...),
    system_instruction: str | None = Form(None),
    notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = crud.create_conversation(
        db,
        project=project,
        title=title,
        source="openai",
        external_id=None,
        notes=notes,
    )
    if system_instruction and system_instruction.strip():
        crud.create_turn(
            db,
            conversation=conversation,
            role="system",
            content_text=system_instruction,
            model="system",
            timestamp=datetime.utcnow(),
        )
    return RedirectResponse(
        url=f"/p/{project.slug}/conversations/{conversation.id}", status_code=303
    )


@app.post("/p/{project_slug}/blocks")
def create_block(
    project_slug: str,
    title: str = Form(...),
    notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    block = crud.create_text_block(
        db,
        project=project,
        tb_id=f"TB-{project.id}-{title[:3].upper()}",
        title=title,
        block_type="draft",
        status="active",
        notes=notes,
        working_text=None,
    )
    return RedirectResponse(
        url=f"/p/{project.slug}/blocks?selected={block.id}", status_code=303
    )


@app.get("/p/{project_slug}/blocks/new", response_class=HTMLResponse)
def new_block(
    project_slug: str,
    request: Request,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    projects = crud.list_projects(db)
    return templates.TemplateResponse(
        "text_blocks/new.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
        },
    )


@app.get("/p/{project_slug}/blocks", response_class=HTMLResponse)
def list_blocks(
    project_slug: str,
    request: Request,
    selected: int | None = None,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    blocks = crud.list_text_blocks(db, project_id=project.id)
    selected_block = None
    if selected is not None:
        candidate = db.get(TextBlock, selected)
        if candidate and candidate.project_id == project.id:
            selected_block = candidate
    if not selected_block and blocks:
        selected_block = blocks[0]
    link_relation = "all"
    link_query = ""
    filtered_links = (
        filter_links(
            selected_block.links,
            link_relation,
            link_query,
            db=db,
            project_id=project.id,
        )
        if selected_block
        else []
    )
    conversation_ids = [link.conversation.id for link in filtered_links]
    turn_counts = crud.get_turn_counts_for_conversations(db, conversation_ids)
    projects = crud.list_projects(db)
    return templates.TemplateResponse(
        "text_blocks/index.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "blocks": blocks,
            "selected_block": selected_block,
            "links": filtered_links,
            "turn_counts": turn_counts,
            "link_relation": link_relation,
            "link_query": link_query,
            "relation_types": RELATION_TYPES,
            "relation_labels": RELATION_LABELS,
            "project_conversations": project.conversations,
        },
    )


@app.get("/p/{project_slug}/blocks/archived", response_class=HTMLResponse)
def list_archived_blocks(
    project_slug: str,
    request: Request,
    selected: int | None = None,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    blocks = crud.list_archived_text_blocks(db, project_id=project.id)
    selected_block = None
    if selected is not None:
        candidate = db.get(TextBlock, selected)
        if candidate and candidate.project_id == project.id and candidate.archived:
            selected_block = candidate
    if not selected_block and blocks:
        selected_block = blocks[0]
    link_relation = "all"
    link_query = ""
    filtered_links = (
        filter_links(
            selected_block.links,
            link_relation,
            link_query,
            db=db,
            project_id=project.id,
        )
        if selected_block
        else []
    )
    conversation_ids = [link.conversation.id for link in filtered_links]
    turn_counts = crud.get_turn_counts_for_conversations(db, conversation_ids)
    projects = crud.list_projects(db)
    return templates.TemplateResponse(
        "text_blocks/archived.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "blocks": blocks,
            "selected_block": selected_block,
            "links": filtered_links,
            "turn_counts": turn_counts,
            "link_relation": link_relation,
            "link_query": link_query,
            "search_highlights": [],
            "highlight_query": "",
            "relation_types": RELATION_TYPES,
            "relation_labels": RELATION_LABELS,
            "project_conversations": project.conversations,
        },
    )


@app.get("/p/{project_slug}/blocks/{block_id}", response_class=HTMLResponse)
def block_detail(
    project_slug: str,
    block_id: int,
    request: Request,
    relation: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    block = get_block_or_404(db, project, block_id)
    link_relation = relation or "all"
    link_query = q or ""
    filtered_links = filter_links(
        block.links,
        link_relation,
        link_query,
        db=db,
        project_id=project.id,
    )
    conversation_ids = [link.conversation.id for link in filtered_links]
    turn_counts = crud.get_turn_counts_for_conversations(db, conversation_ids)
    highlight_terms = extract_query_terms(q)
    highlight_pattern = build_highlight_pattern(highlight_terms)
    search_highlights = []
    for label, value in (
        ("Titel", block.title),
        ("Working-Text", block.working_text),
        ("Notes", block.notes),
    ):
        if value and highlight_pattern and highlight_pattern.search(value):
            search_highlights.append(
                {"label": label, "content_html": highlight_text(value, highlight_pattern)}
            )
    context = {
        "request": request,
        "project": project,
        "block": block,
        "links": filtered_links,
        "turn_counts": turn_counts,
        "link_relation": link_relation,
        "link_query": link_query,
        "relation_types": RELATION_TYPES,
        "relation_labels": RELATION_LABELS,
        "project_conversations": project.conversations,
        "search_highlights": search_highlights,
        "highlight_query": q or "",
    }
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("text_blocks/detail.html", context)
    blocks = crud.list_text_blocks(db, project_id=project.id)
    projects = crud.list_projects(db)
    context.update(
        {
            "projects": projects,
            "blocks": blocks,
            "selected_block": block,
        }
    )
    return templates.TemplateResponse("text_blocks/index.html", context)


@app.post("/p/{project_slug}/blocks/{block_id}")
def update_block(
    project_slug: str,
    block_id: int,
    request: Request,
    title: str = Form(...),
    block_type: str = Form(...),
    status: str = Form(...),
    working_text: str | None = Form(None),
    notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    block = get_block_or_404(db, project, block_id)
    crud.update_text_block(
        db,
        block=block,
        title=title,
        block_type=block_type,
        status=status,
        working_text=working_text,
        notes=notes,
    )
    if request.headers.get("HX-Request"):
        return block_detail(
            project_slug,
            block_id,
            request,
            relation="all",
            q="",
            db=db,
        )
    return RedirectResponse(url=f"/p/{project.slug}/blocks?selected={block.id}", status_code=303)


@app.post("/p/{project_slug}/blocks/{block_id}/archive")
def archive_block(
    project_slug: str,
    block_id: int,
    request: Request,
    archived: bool = Form(...),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    block = get_block_or_404(db, project, block_id)
    crud.set_text_block_archived(db, block=block, archived=archived)
    redirect_target = (
        f"/p/{project.slug}/blocks/archived?selected={block.id}"
        if archived
        else f"/p/{project.slug}/blocks?selected={block.id}"
    )
    response = RedirectResponse(url=redirect_target, status_code=303)
    if request.headers.get("HX-Request"):
        response.headers["HX-Redirect"] = redirect_target
    return response


@app.get("/p/{project_slug}/blocks/{block_id}/links", response_class=HTMLResponse)
def block_links(
    project_slug: str,
    block_id: int,
    request: Request,
    relation: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    block = get_block_or_404(db, project, block_id)
    link_relation = relation or "all"
    link_query = q or ""
    filtered_links = filter_links(
        block.links,
        link_relation,
        link_query,
        db=db,
        project_id=project.id,
    )
    conversation_ids = [link.conversation.id for link in filtered_links]
    turn_counts = crud.get_turn_counts_for_conversations(db, conversation_ids)
    context = {
        "request": request,
        "project": project,
        "block": block,
        "links": filtered_links,
        "turn_counts": turn_counts,
        "link_relation": link_relation,
        "link_query": link_query,
        "relation_types": RELATION_TYPES,
        "relation_labels": RELATION_LABELS,
        "project_conversations": project.conversations,
    }
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("text_blocks/links.html", context)
    return RedirectResponse(
        url=f"/p/{project.slug}/blocks/{block.id}?relation={link_relation}&q={link_query}",
        status_code=303,
    )


@app.post("/p/{project_slug}/blocks/{block_id}/links")
def add_block_link(
    project_slug: str,
    block_id: int,
    request: Request,
    relation: str = Form(...),
    conversation_id: int | None = Form(None),
    new_title: str | None = Form(None),
    new_notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    block = get_block_or_404(db, project, block_id)
    if relation not in RELATION_TYPES:
        raise HTTPException(status_code=400, detail="Invalid relation type")
    conversation: Conversation
    if conversation_id:
        conversation = db.get(Conversation, conversation_id)
        if not conversation or conversation.project_id != project.id:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        if not new_title:
            raise HTTPException(status_code=400, detail="Conversation title required")
        conversation = crud.create_conversation(
            db,
            project=project,
            title=new_title,
            source="manual",
            external_id=None,
            notes=new_notes,
        )
    crud.create_link(db, text_block=block, conversation=conversation, relation=relation)
    return block_links(
        project_slug,
        block_id,
        request,
        relation="all",
        q="",
        db=db,
    )


@app.post("/p/{project_slug}/blocks/{block_id}/links/{link_id}")
def update_block_link(
    project_slug: str,
    block_id: int,
    link_id: int,
    request: Request,
    relation: str = Form(...),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    block = get_block_or_404(db, project, block_id)
    link = db.get(Link, link_id)
    if not link or link.text_block_id != block.id:
        raise HTTPException(status_code=404, detail="Link not found")
    if relation not in RELATION_TYPES:
        raise HTTPException(status_code=400, detail="Invalid relation type")
    link.relation = relation
    db.commit()
    return block_links(
        project_slug,
        block_id,
        request,
        relation="all",
        q="",
        db=db,
    )


@app.post("/p/{project_slug}/blocks/{block_id}/conversations")
def create_conversation(
    project_slug: str,
    block_id: int,
    title: str = Form(...),
    notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    block = db.get(TextBlock, block_id)
    if not block or block.project_id != project.id:
        raise HTTPException(status_code=404, detail="Text block not found")
    conversation = crud.create_conversation(
        db,
        project=block.project,
        title=title,
        source="other",
        external_id=None,
        notes=notes,
    )
    crud.create_link(db, text_block=block, conversation=conversation, relation="ideation")
    return RedirectResponse(url=f"/p/{project.slug}/dashboard", status_code=303)


@app.get("/p/{project_slug}/conversations/{conversation_id}", response_class=HTMLResponse)
def conversation_detail(
    project_slug: str,
    conversation_id: int,
    request: Request,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = get_conversation_or_404(db, project, conversation_id)
    return render_conversation_detail(
        request, project, conversation, db, highlight_query=q
    )


@app.post("/p/{project_slug}/conversations/{conversation_id}/links")
def add_conversation_link(
    project_slug: str,
    conversation_id: int,
    block_id: int = Form(...),
    relation: str = Form(...),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = get_conversation_or_404(db, project, conversation_id)
    block = get_block_or_404(db, project, block_id)
    if relation not in RELATION_TYPES:
        raise HTTPException(status_code=400, detail="Invalid relation type")
    crud.create_link(db, text_block=block, conversation=conversation, relation=relation)
    return RedirectResponse(
        url=f"/p/{project.slug}/conversations/{conversation.id}", status_code=303
    )


@app.post("/p/{project_slug}/conversations/{conversation_id}/links/{link_id}")
def update_conversation_link(
    project_slug: str,
    conversation_id: int,
    link_id: int,
    relation: str = Form(...),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = get_conversation_or_404(db, project, conversation_id)
    link = db.get(Link, link_id)
    if not link or link.conversation_id != conversation.id:
        raise HTTPException(status_code=404, detail="Link not found")
    if relation not in RELATION_TYPES:
        raise HTTPException(status_code=400, detail="Invalid relation type")
    link.relation = relation
    db.commit()
    return RedirectResponse(
        url=f"/p/{project.slug}/conversations/{conversation.id}", status_code=303
    )


@app.post("/p/{project_slug}/conversations/{conversation_id}/meta")
def update_conversation_meta(
    project_slug: str,
    conversation_id: int,
    title: str = Form(...),
    notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = get_conversation_or_404(db, project, conversation_id)
    crud.update_conversation(db, conversation=conversation, title=title, notes=notes)
    return RedirectResponse(
        url=f"/p/{project.slug}/conversations/{conversation.id}", status_code=303
    )


@app.post("/p/{project_slug}/conversations/{conversation_id}/messages")
def create_conversation_message(
    project_slug: str,
    conversation_id: int,
    request: Request,
    message: str = Form(...),
    model: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = get_conversation_or_404(db, project, conversation_id)
    default_model = get_project_default_model(project) or "gpt-4o-mini"
    selected_model = (model or default_model).strip()
    if not selected_model:
        selected_model = default_model
    if not message.strip():
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error="Message cannot be empty.",
            selected_model=selected_model,
        )

    try:
        api_key = get_project_openai_key(project)
    except HTTPException as exc:
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error=str(exc.detail),
            selected_model=selected_model,
        )
    client = get_openai_client(api_key)
    previous_response_id = crud.get_latest_response_id(db, conversation_id=conversation.id)
    system_turn = crud.get_initial_system_turn(db, conversation_id=conversation.id)
    latest_assistant_turn = crud.get_latest_turn_by_role(
        db, conversation_id=conversation.id, role="assistant"
    )
    if system_turn and latest_assistant_turn:
        if system_turn.timestamp > latest_assistant_turn.timestamp:
            previous_response_id = None
    input_messages = []
    if not previous_response_id and system_turn:
        input_messages.append({"role": "system", "content": system_turn.content_text})
    input_messages.append({"role": "user", "content": message})

    # We use previous_response_id to keep conversation state on the OpenAI side.
    try:
        api_response = client.responses.create(
            model=selected_model,
            input=input_messages,
            previous_response_id=previous_response_id,
        )
    except RateLimitError:
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error="Rate limit reached. Please wait and try again.",
            selected_model=selected_model,
        )
    except AuthenticationError:
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error="Authentication failed. Please check the project API key.",
            selected_model=selected_model,
        )
    except APIStatusError as exc:
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error=f"OpenAI error: {exc.message}",
            selected_model=selected_model,
        )
    except OpenAIError as exc:
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error=f"OpenAI error: {exc}",
            selected_model=selected_model,
        )

    crud.create_turn(
        db,
        conversation=conversation,
        role="user",
        content_text=message,
        model=selected_model,
        timestamp=datetime.utcnow(),
        metadata_json={"input": input_messages},
    )

    metadata = {
        "usage": serialize_usage(getattr(api_response, "usage", None)),
        "previous_response_id": previous_response_id,
        "status": getattr(api_response, "status", None),
    }
    response_id = getattr(api_response, "id", None)
    model = getattr(api_response, "model", None) or selected_model
    crud.create_turn(
        db,
        conversation=conversation,
        role="assistant",
        content_text=api_response.output_text,
        model=model,
        timestamp=datetime.utcnow(),
        response_id=response_id,
        metadata_json=metadata,
    )
    return RedirectResponse(
        url=f"/p/{project.slug}/conversations/{conversation.id}", status_code=303
    )


@app.post("/p/{project_slug}/conversations/{conversation_id}/system")
def update_conversation_system_instruction(
    project_slug: str,
    conversation_id: int,
    request: Request,
    system_instruction: str = Form(...),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = get_conversation_or_404(db, project, conversation_id)
    if not system_instruction.strip():
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error="System instruction cannot be empty.",
        )
    crud.upsert_system_turn(
        db, conversation=conversation, content_text=system_instruction.strip()
    )
    return RedirectResponse(
        url=f"/p/{project.slug}/conversations/{conversation.id}", status_code=303
    )


@app.get("/p/{project_slug}/search", response_class=HTMLResponse)
def search(
    project_slug: str,
    request: Request,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    results = build_search_results(db, project_id=project.id, query=q)
    projects = crud.list_projects(db)
    return templates.TemplateResponse(
        "project_search.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "turn_results": results["turn_results"],
            "text_block_results": results["text_block_results"],
            "query": q,
        },
    )


@app.get("/p/{project_slug}/search/results", response_class=HTMLResponse)
def search_results(
    project_slug: str,
    request: Request,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    results = build_search_results(db, project_id=project.id, query=q)
    return templates.TemplateResponse(
        "search/results.html",
        {
            "request": request,
            "project": project,
            "turn_results": results["turn_results"],
            "text_block_results": results["text_block_results"],
            "query": q,
        },
    )


@app.get("/p/{project_slug}/export")
def export_project(
    project_slug: str,
    include_archived: bool = False,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)

    rows = crud.list_turns_for_project(
        db, project_id=project.id, include_archived=include_archived
    )

    def generate():
        for row in rows:
            payload = dict(row)
            text_block_ids = payload.get("text_block_tb_ids")
            if text_block_ids:
                payload["text_block_tb_ids"] = [
                    block_id for block_id in text_block_ids.split(",") if block_id
                ]
            else:
                payload["text_block_tb_ids"] = []
            yield json.dumps(payload, ensure_ascii=False) + "\n"

    headers = {"Content-Disposition": f"attachment; filename=project_{project.id}.jsonl"}
    return StreamingResponse(generate(), media_type="application/jsonl", headers=headers)


@app.get("/p/{project_slug}/settings", response_class=HTMLResponse)
def project_settings(project_slug: str, request: Request, db: Session = Depends(get_db)):
    project = get_project_by_slug_or_404(db, project_slug)
    settings_payload = json.dumps(project.settings_json, indent=2) if project.settings_json else ""
    return render_project_settings(
        request=request,
        project=project,
        db=db,
        settings_payload=settings_payload,
        error=None,
        test_result=None,
        test_status=None,
    )


@app.post("/p/{project_slug}/settings")
def update_project_settings(
    project_slug: str,
    request: Request,
    name: str = Form(...),
    description: str | None = Form(None),
    settings_payload: str | None = Form(None),
    api_key: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    parsed_settings: dict | None = None
    if settings_payload:
        try:
            parsed_settings = json.loads(settings_payload)
        except json.JSONDecodeError as exc:
            return render_project_settings(
                request=request,
                project=project,
                db=db,
                settings_payload=settings_payload,
                error=f"Invalid JSON: {exc.msg}",
                test_result=None,
                test_status=None,
            )
    if name != project.name and crud.project_name_exists(db, name):
        return render_project_settings(
            request=request,
            project=project,
            db=db,
            settings_payload=settings_payload or "",
            error="Project name already exists.",
            test_result=None,
            test_status=None,
        )
    slug = project.slug
    if name != project.name:
        slug = build_unique_slug(db, name)
    crud.update_project(
        db,
        project=project,
        name=name,
        slug=slug,
        description=description,
        settings_json=parsed_settings,
    )
    if api_key:
        try:
            api_key_encrypted = encrypt_secret(api_key)
        except EncryptionError as exc:
            return render_project_settings(
                request=request,
                project=project,
                db=db,
                settings_payload=settings_payload or "",
                error=str(exc),
                test_result=None,
                test_status=None,
            )
        crud.upsert_project_secret(
            db,
            project=project,
            secret_type="openai_api_key",
            secret_ciphertext=api_key_encrypted,
        )
    return RedirectResponse(url=f"/p/{slug}/settings", status_code=303)


@app.post("/p/{project_slug}/settings/test-api-key", response_class=HTMLResponse)
def test_project_settings_api_key(
    project_slug: str, request: Request, db: Session = Depends(get_db)
):
    project = get_project_by_slug_or_404(db, project_slug)
    settings_payload = json.dumps(project.settings_json, indent=2) if project.settings_json else ""
    try:
        api_key = get_project_openai_key(project)
        client = get_openai_client(api_key)
        client.models.list()
    except HTTPException as exc:
        return render_project_settings(
            request=request,
            project=project,
            db=db,
            settings_payload=settings_payload,
            error=None,
            test_result=str(exc.detail),
            test_status="error",
        )
    except AuthenticationError:
        return render_project_settings(
            request=request,
            project=project,
            db=db,
            settings_payload=settings_payload,
            error=None,
            test_result="Authentication failed. Please check the project API key.",
            test_status="error",
        )
    except RateLimitError:
        return render_project_settings(
            request=request,
            project=project,
            db=db,
            settings_payload=settings_payload,
            error=None,
            test_result="Rate limit reached. Please wait and try again.",
            test_status="error",
        )
    except APIStatusError as exc:
        return render_project_settings(
            request=request,
            project=project,
            db=db,
            settings_payload=settings_payload,
            error=None,
            test_result=f"API error: {exc.status_code} {exc.message}",
            test_status="error",
        )
    except OpenAIError as exc:
        return render_project_settings(
            request=request,
            project=project,
            db=db,
            settings_payload=settings_payload,
            error=None,
            test_result=f"API error: {exc}",
            test_status="error",
        )
    return render_project_settings(
        request=request,
        project=project,
        db=db,
        settings_payload=settings_payload,
        error=None,
        test_result="API key verified. OpenAI connection looks good.",
        test_status="success",
    )


@app.post("/p/{project_slug}/archive")
def archive_project(project_slug: str, db: Session = Depends(get_db)):
    project = get_project_by_slug_or_404(db, project_slug)
    crud.soft_delete_project(db, project)
    return RedirectResponse(url="/", status_code=303)
