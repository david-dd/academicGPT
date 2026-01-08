from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_db
from app.main import app
from app.settings import settings


def build_test_client():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    Base.metadata.create_all(bind=engine)
    with engine.connect() as connection:
        connection.execute(
            text(
                """
                CREATE VIRTUAL TABLE turns_fts USING fts5(
                    turn_id UNINDEXED,
                    prompt,
                    response,
                    project_id UNINDEXED,
                    text_block_id UNINDEXED
                )
                """
            )
        )

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)


def test_create_project_flow():
    settings.master_passphrase = "test-passphrase"
    client = build_test_client()

    response = client.post(
        "/projects",
        data={"name": "Thesis", "description": "My project", "api_key": "sk-test"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    project_url = response.headers["location"]
    project_page = client.get(project_url)
    assert project_page.status_code == 200
    assert "Thesis" in project_page.text

    block_response = client.post(
        f"{project_url}/blocks",
        data={"title": "Intro", "description": "Opening"},
        follow_redirects=False,
    )
    assert block_response.status_code == 303
