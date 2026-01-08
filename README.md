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
poetry install
```

Set a master passphrase (required for encrypting project API keys):

```bash
export AILOGGER_MASTER_PASSPHRASE="your-strong-passphrase"
```

Run the database migration:

```bash
poetry run alembic upgrade head
```

## Run

```bash
make dev
```

Visit `http://127.0.0.1:8000`.

## Tests

```bash
make test
```

## Formatting & Linting

```bash
make format
make lint
```

## Docker (optional)

```bash
docker compose up --build
```

The container uses a bind mount so local changes are reflected.
