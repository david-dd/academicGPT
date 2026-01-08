import json
from datetime import datetime

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app import crud
from app.db import get_db
from app.models import Conversation, TextBlock
from app.openai_client import get_openai_client
from app.security import EncryptionError, decrypt_secret, encrypt_secret

app = FastAPI(title="AI Logger")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def slugify(value: str) -> str:
    return "-".join(value.lower().strip().split())


def get_project_by_slug_or_404(db: Session, project_slug: str):
    project = crud.get_project_by_slug(db, project_slug)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


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


@app.post("/p/{project_slug}/blocks")
def create_block(
    project_slug: str,
    title: str = Form(...),
    notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    crud.create_text_block(
        db,
        project=project,
        tb_id=f"TB-{project.id}-{title[:3].upper()}",
        title=title,
        block_type="draft",
        status="active",
        notes=notes,
        working_text=None,
    )
    return RedirectResponse(url=f"/p/{project.slug}/dashboard", status_code=303)


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


@app.post("/p/{project_slug}/conversations/{conversation_id}/turns")
def create_turn(
    project_slug: str,
    conversation_id: int,
    prompt: str = Form(...),
    response: str | None = Form(None),
    use_openai: bool | None = Form(False),
    db: Session = Depends(get_db),
):
    project = get_project_by_slug_or_404(db, project_slug)
    conversation = db.get(Conversation, conversation_id)
    if not conversation or conversation.project_id != project.id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    resolved_response = response or ""
    response_id: str | None = None

    if use_openai:
        secret = next(
            (s for s in project.secrets if s.secret_type == "openai_api_key"), None
        )
        if not secret:
            raise HTTPException(status_code=400, detail="Project API key not configured")
        try:
            api_key = decrypt_secret(secret.secret_ciphertext)
        except EncryptionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        client = get_openai_client(api_key)
        api_response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        resolved_response = api_response.output_text
        response_id = api_response.id

    crud.create_turn(
        db,
        conversation=conversation,
        role="user",
        content_text=prompt,
        model="manual",
        timestamp=datetime.utcnow(),
    )
    crud.create_turn(
        db,
        conversation=conversation,
        role="assistant",
        content_text=resolved_response,
        model="gpt-4o-mini" if use_openai else "manual",
        timestamp=datetime.utcnow(),
        response_id=response_id,
    )
    return RedirectResponse(url=f"/p/{project.slug}/dashboard", status_code=303)


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
