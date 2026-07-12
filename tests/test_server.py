"""
Tests d'integration pour server.py.
"""

import json

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture
def client(monkeypatch):
    # lifespan() appelle scoring_module.init() au demarrage -- on le
    # neutralise pour ne pas charger de vrai checkpoint en test.
    monkeypatch.setattr(server.scoring_module, "init", lambda: None)
    with TestClient(server.app) as c:
        yield c


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_score_success(client, monkeypatch):
    monkeypatch.setattr(
        server.scoring_module,
        "run",
        lambda raw: json.dumps({"pred_score": 0.2, "pred_label": 0, "is_anomalous": False}),
    )
    response = client.post("/score", json={"image_b64": "peu-importe", "image_size": [256, 256]})
    assert response.status_code == 200
    body = response.json()
    assert body["is_anomalous"] is False
    assert body["pred_score"] == pytest.approx(0.2)


def test_score_propagates_scoring_error_as_400(client, monkeypatch):
    monkeypatch.setattr(
        server.scoring_module,
        "run",
        lambda raw: json.dumps({"error": "Cannot decode image: bad data"}),
    )
    response = client.post("/score", json={"image_b64": "invalide"})
    assert response.status_code == 400
    assert "Cannot decode image" in response.json()["detail"]


def test_score_invalid_image_returns_400_end_to_end(client):
    """Pas de mock ici : on passe un vrai base64 invalide a travers toute
    la chaine FastAPI -> endpoint_score.run() reel -> verifie le 400."""
    response = client.post("/score", json={"image_b64": "!!!pas-une-image!!!"})
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_score_missing_required_field_returns_422(client):
    # image_b64 est un champ requis du modele Pydantic ScoreRequest
    response = client.post("/score", json={})
    assert response.status_code == 422


def test_score_uses_default_image_size_when_omitted(client, monkeypatch):
    captured = {}

    def fake_run(raw: str) -> str:
        captured["payload"] = json.loads(raw)
        return json.dumps({"pred_score": 0.1, "pred_label": 0, "is_anomalous": False})

    monkeypatch.setattr(server.scoring_module, "run", fake_run)
    client.post("/score", json={"image_b64": "peu-importe"})
    assert captured["payload"]["image_size"] == [256, 256]