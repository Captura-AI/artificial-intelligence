from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routers.embedding import router


def build_client() -> TestClient:
    app = FastAPI()
    app.include_router(router)

    return TestClient(app)


def test_embed_text_returns_vector(monkeypatch):
    def fake_embedding(query: str, model_name: str) -> list[float]:
        assert query == "black motorcycle"
        assert model_name == "ViT-B/32"
        return [0.1] * 512

    monkeypatch.setattr("app.routers.embedding.get_text_embedding", fake_embedding)

    client = build_client()
    response = client.post("/embed/text", json={"query": " black motorcycle "})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "black motorcycle"
    assert body["model"] == "ViT-B/32"
    assert body["vector_dimension"] == 512
    assert len(body["embedding"]) == 512


def test_embed_text_rejects_blank_query():
    client = build_client()
    response = client.post("/embed/text", json={"query": "   "})

    assert response.status_code == 422


def test_embed_text_returns_unavailable_when_model_fails(monkeypatch):
    def fake_embedding(query: str, model_name: str) -> None:
        return None

    monkeypatch.setattr("app.routers.embedding.get_text_embedding", fake_embedding)

    client = build_client()
    response = client.post("/embed/text", json={"query": "red jacket"})

    assert response.status_code == 503
