import json
from datetime import datetime

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

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


def serialize_usage(usage) -> dict | None:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return dict(usage)


def render_conversation_detail(
    request: Request,
    project: Project,
    conversation: Conversation,
    db: Session,
    error: str | None = None,
):
    turns = crud.list_turns_for_conversation(db, conversation_id=conversation.id)
    return templates.TemplateResponse(
        "conversation_detail.html",
        {
            "request": request,
            "project": project,
            "conversation": conversation,
            "turns": turns,
            "error": error,
        },
    )


def filter_links(
    links: list[Link], relation: str | None = None, query: str | None = None
) -> list[Link]:
    filtered = links
    if relation and relation != "all":
        filtered = [link for link in filtered if link.relation == relation]
    if query:
        lowered = query.lower()
        filtered = [
            link for link in filtered if lowered in link.conversation.title.lower()
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
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = crud.create_conversation(
        db, project=project, title=title, source="openai", external_id=None
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


@app.get("/p/{project_slug}/blocks", response_class=HTMLResponse)
def list_blocks(
    project_slug: str,
    request: Request,
    selected: int | None = None,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    blocks = list(project.text_blocks)
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
        filter_links(selected_block.links, link_relation, link_query)
        if selected_block
        else []
    )
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
            "link_relation": link_relation,
            "link_query": link_query,
            "relation_types": RELATION_TYPES,
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
    filtered_links = filter_links(block.links, link_relation, link_query)
    context = {
        "request": request,
        "project": project,
        "block": block,
        "links": filtered_links,
        "link_relation": link_relation,
        "link_query": link_query,
        "relation_types": RELATION_TYPES,
        "project_conversations": project.conversations,
    }
    if request.headers.get("HX-Request"):
        return templates.TemplateResponse("text_blocks/detail.html", context)
    blocks = list(project.text_blocks)
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
    filtered_links = filter_links(block.links, link_relation, link_query)
    context = {
        "request": request,
        "project": project,
        "block": block,
        "links": filtered_links,
        "link_relation": link_relation,
        "link_query": link_query,
        "relation_types": RELATION_TYPES,
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
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    block = db.get(TextBlock, block_id)
    if not block or block.project_id != project.id:
        raise HTTPException(status_code=404, detail="Text block not found")
    conversation = crud.create_conversation(
        db, project=block.project, title=title, source="other", external_id=None
    )
    crud.create_link(db, text_block=block, conversation=conversation, relation="ideation")
    return RedirectResponse(url=f"/p/{project.slug}/dashboard", status_code=303)


@app.get("/p/{project_slug}/conversations/{conversation_id}", response_class=HTMLResponse)
def conversation_detail(
    project_slug: str,
    conversation_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = get_conversation_or_404(db, project, conversation_id)
    return render_conversation_detail(request, project, conversation, db)


@app.post("/p/{project_slug}/conversations/{conversation_id}/messages")
def create_conversation_message(
    project_slug: str,
    conversation_id: int,
    request: Request,
    message: str = Form(...),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = get_conversation_or_404(db, project, conversation_id)
    if not message.strip():
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error="Message cannot be empty.",
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
        )
    client = get_openai_client(api_key)
    previous_response_id = crud.get_latest_response_id(db, conversation_id=conversation.id)
    system_turn = crud.get_initial_system_turn(db, conversation_id=conversation.id)
    input_messages = []
    if not previous_response_id and system_turn:
        input_messages.append({"role": "system", "content": system_turn.content_text})
    input_messages.append({"role": "user", "content": message})

    # We use previous_response_id to keep conversation state on the OpenAI side.
    try:
        api_response = client.responses.create(
            model="gpt-4o-mini",
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
        )
    except AuthenticationError:
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error="Authentication failed. Please check the project API key.",
        )
    except APIStatusError as exc:
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error=f"OpenAI error: {exc.message}",
        )
    except OpenAIError as exc:
        return render_conversation_detail(
            request,
            project,
            conversation,
            db,
            error=f"OpenAI error: {exc}",
        )

    crud.create_turn(
        db,
        conversation=conversation,
        role="user",
        content_text=message,
        model="user",
        timestamp=datetime.utcnow(),
        metadata_json={"input": input_messages},
    )

    metadata = {
        "usage": serialize_usage(getattr(api_response, "usage", None)),
        "previous_response_id": previous_response_id,
        "status": getattr(api_response, "status", None),
    }
    response_id = getattr(api_response, "id", None)
    model = getattr(api_response, "model", None) or "gpt-4o-mini"
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


@app.get("/p/{project_slug}/search", response_class=HTMLResponse)
def search(
    project_slug: str,
    request: Request,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    results = crud.search_turns_for_project(db, project_id=project.id, query=q) if q else []
    projects = crud.list_projects(db)
    return templates.TemplateResponse(
        "project_search.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "results": results,
            "query": q,
        },
    )


@app.get("/p/{project_slug}/export")
def export_project(project_slug: str, db: Session = Depends(get_db)):
    project = get_project_by_slug_or_404(db, project_slug)

    rows = crud.list_turns_for_project(db, project_id=project.id)

    def generate():
        for row in rows:
            yield json.dumps(dict(row), ensure_ascii=False) + "\n"

    headers = {"Content-Disposition": f"attachment; filename=project_{project.id}.jsonl"}
    return StreamingResponse(generate(), media_type="application/jsonl", headers=headers)


@app.get("/p/{project_slug}/settings", response_class=HTMLResponse)
def project_settings(project_slug: str, request: Request, db: Session = Depends(get_db)):
    project = get_project_by_slug_or_404(db, project_slug)
    projects = crud.list_projects(db)
    settings_payload = json.dumps(project.settings_json, indent=2) if project.settings_json else ""
    return templates.TemplateResponse(
        "project_settings.html",
        {
            "request": request,
            "project": project,
            "projects": projects,
            "settings_payload": settings_payload,
            "error": None,
        },
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
            projects = crud.list_projects(db)
            return templates.TemplateResponse(
                "project_settings.html",
                {
                    "request": request,
                    "project": project,
                    "projects": projects,
                    "settings_payload": settings_payload,
                    "error": f"Invalid JSON: {exc.msg}",
                },
                status_code=400,
            )
    if name != project.name and crud.project_name_exists(db, name):
        projects = crud.list_projects(db)
        return templates.TemplateResponse(
            "project_settings.html",
            {
                "request": request,
                "project": project,
                "projects": projects,
                "settings_payload": settings_payload or "",
                "error": "Project name already exists.",
            },
            status_code=400,
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
            projects = crud.list_projects(db)
            return templates.TemplateResponse(
                "project_settings.html",
                {
                    "request": request,
                    "project": project,
                    "projects": projects,
                    "settings_payload": settings_payload or "",
                    "error": str(exc),
                },
                status_code=400,
            )
        crud.upsert_project_secret(
            db,
            project=project,
            secret_type="openai_api_key",
            secret_ciphertext=api_key_encrypted,
        )
    return RedirectResponse(url=f"/p/{slug}/settings", status_code=303)


@app.post("/p/{project_slug}/archive")
def archive_project(project_slug: str, db: Session = Depends(get_db)):
    project = get_project_by_slug_or_404(db, project_slug)
    crud.soft_delete_project(db, project)
    return RedirectResponse(url="/", status_code=303)
