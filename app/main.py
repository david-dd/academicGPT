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


@app.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)):
    projects = crud.list_projects(db)
    return templates.TemplateResponse("index.html", {"request": request, "projects": projects})


@app.post("/projects")
def create_project(
    name: str = Form(...),
    description: str | None = Form(None),
    api_key: str | None = Form(None),
    db: Session = Depends(get_db),
):
    try:
        api_key_encrypted = encrypt_secret(api_key) if api_key else None
    except EncryptionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    project = crud.create_project(
        db, name=name, slug=slugify(name), description=description, settings_json=None
    )
    if api_key_encrypted:
        crud.create_project_secret(
            db,
            project=project,
            secret_type="openai_api_key",
            secret_ciphertext=api_key_encrypted,
        )
    return RedirectResponse(url=f"/projects/{project.id}", status_code=303)


@app.get("/projects/{project_id}", response_class=HTMLResponse)
def project_detail(project_id: int, request: Request, db: Session = Depends(get_db)):
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return templates.TemplateResponse(
        "project.html", {"request": request, "project": project}
    )


@app.post("/projects/{project_id}/blocks")
def create_block(
    project_id: int,
    title: str = Form(...),
    notes: str | None = Form(None),
    db: Session = Depends(get_db),
):
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    crud.create_text_block(
        db,
        project=project,
        tb_id=f"TB-{project_id}-{title[:3].upper()}",
        title=title,
        block_type="draft",
        status="active",
        notes=notes,
        working_text=None,
    )
    return RedirectResponse(url=f"/projects/{project_id}", status_code=303)


@app.post("/blocks/{block_id}/conversations")
def create_conversation(
    block_id: int,
    title: str = Form(...),
    db: Session = Depends(get_db),
):
    block = db.get(TextBlock, block_id)
    if not block:
        raise HTTPException(status_code=404, detail="Text block not found")
    conversation = crud.create_conversation(
        db, project=block.project, title=title, source="other", external_id=None
    )
    crud.create_link(db, text_block=block, conversation=conversation, relation="ideation")
    return RedirectResponse(url=f"/projects/{block.project_id}", status_code=303)


@app.post("/conversations/{conversation_id}/turns")
def create_turn(
    conversation_id: int,
    prompt: str = Form(...),
    response: str | None = Form(None),
    use_openai: bool | None = Form(False),
    db: Session = Depends(get_db),
):
    conversation = db.get(Conversation, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    project = conversation.project
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
    return RedirectResponse(url=f"/projects/{project.id}", status_code=303)


@app.get("/search", response_class=HTMLResponse)
def search(
    request: Request,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    results = crud.search_turns(db, query=q) if q else []
    return templates.TemplateResponse(
        "search.html",
        {"request": request, "results": results, "query": q},
    )


@app.get("/projects/{project_id}/export")
def export_project(project_id: int, db: Session = Depends(get_db)):
    project = crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    rows = crud.list_turns_for_project(db, project_id=project_id)

    def generate():
        for row in rows:
            yield json.dumps(dict(row), ensure_ascii=False) + "\n"

    headers = {"Content-Disposition": f"attachment; filename=project_{project_id}.jsonl"}
    return StreamingResponse(generate(), media_type="application/jsonl", headers=headers)
