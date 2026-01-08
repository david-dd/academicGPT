.PHONY: dev test lint format

dev:
	poetry run uvicorn app.main:app --reload

test:
	poetry run pytest

lint:
	poetry run ruff check .

format:
	poetry run ruff format .
