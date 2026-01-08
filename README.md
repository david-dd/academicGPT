# AI Logger

Local web client for structured documentation of AI usage in academic projects. Each project stores its
own OpenAI API key (encrypted with a master passphrase), text blocks, conversations, and logged turns
that are searchable via SQLite FTS5.

## Why Poetry?

Poetry offers deterministic dependency locking and an easy, single-tool workflow for installing, running,
and packaging the app. For a small MVP, it keeps commands consistent across dev machines.

## Requirements

- Python 3.12
- Poetry
- SQLite with FTS5 enabled (default for modern SQLite builds)

## Setup

```bash
py -3.12 -m poetry install
py -3.12 -m poetry run alembic upgrade head
```

Set a master passphrase (required for encrypting project API keys) within the .env file:

```bash
AILOGGER_MASTER_PASSPHRASE="your-strong-passphrase"
```

Run the database migration:

```bash
poetry run alembic upgrade head
```

## Run

```bash
py -3.12 -m poetry run uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000`.


## Docker (optional)

```bash
docker compose up --build
```

The container uses a bind mount so local changes are reflected.
