"""Basic startup tests for the MVP FastAPI scaffold."""

from fastapi.testclient import TestClient

from src.api.server import app


def test_imports_and_app_startup() -> None:
    """Verify the app imports and responds on the health endpoint."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_run_task_endpoint_creates_placeholder_run() -> None:
    """Verify the placeholder run route is wired and returns a run identifier."""
    client = TestClient(app)
    response = client.post("/run-task", json={"intent": "Navigate and click a button.", "headless": True})
    body = response.json()
    assert response.status_code == 202
    assert body["intent"] == "Navigate and click a button."
    assert body["status"] == "pending"
    assert body["run_id"]
