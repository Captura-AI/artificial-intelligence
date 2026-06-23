from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.security import require_api_key


def build_client(api_key) -> TestClient:
    """Tiny app with one route guarded by require_api_key, with settings overridden."""
    app = FastAPI()

    @app.get("/guarded", dependencies=[Depends(require_api_key)])
    def guarded() -> dict:
        return {"ok": True}

    app.dependency_overrides[get_settings] = lambda: Settings(api_key=api_key)
    return TestClient(app)


def test_no_key_configured_allows_request():
    # When no api_key is set, auth is disabled (dev convenience).
    client = build_client(api_key=None)
    response = client.get("/guarded")
    assert response.status_code == 200


def test_missing_header_rejected_when_key_configured():
    client = build_client(api_key="secret-123")
    response = client.get("/guarded")
    assert response.status_code == 401


def test_wrong_key_rejected():
    client = build_client(api_key="secret-123")
    response = client.get("/guarded", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_correct_key_accepted():
    client = build_client(api_key="secret-123")
    response = client.get("/guarded", headers={"X-API-Key": "secret-123"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
