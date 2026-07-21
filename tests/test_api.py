"""Tests del servicio de scoring (requiere models/model.joblib generado)."""
import pytest
from fastapi.testclient import TestClient

from src import config

pytestmark = pytest.mark.skipif(
    not (config.MODELS_DIR / "model.joblib").exists(),
    reason="modelo no entrenado; corre run_pipeline.py primero",
)

GOOD = {
    "AMT_INCOME_TOTAL": 250_000, "AMT_CREDIT": 400_000, "AMT_ANNUITY": 25_000,
    "DAYS_BIRTH": -40 * 365, "DAYS_EMPLOYED": -12 * 365,
    "EXT_SOURCE_1": 0.85, "EXT_SOURCE_2": 0.80, "EXT_SOURCE_3": 0.82,
}
RISKY = {
    "AMT_INCOME_TOTAL": 90_000, "AMT_CREDIT": 900_000, "AMT_ANNUITY": 60_000,
    "DAYS_BIRTH": -23 * 365, "DAYS_EMPLOYED": -60,
    "EXT_SOURCE_1": 0.10, "EXT_SOURCE_2": 0.12, "EXT_SOURCE_3": 0.08,
    "BUREAU_OVERDUE_ANY": 1, "PREV_REFUSED_COUNT": 3, "PREV_APP_COUNT": 4,
}


@pytest.fixture(scope="module")
def client():
    from app.main import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["trained_on"] in {"kaggle", "synthetic"}


def test_score_contract(client):
    r = client.post("/score", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["decision"] in {"approve", "reject"}


def test_risk_ordering(client):
    """Un perfil claramente riesgoso debe recibir mayor PD que uno sólido."""
    p_good = client.post("/score", json=GOOD).json()["probability_of_default"]
    p_risky = client.post("/score", json=RISKY).json()["probability_of_default"]
    assert p_risky > p_good


def test_validation_rejects_bad_input(client):
    r = client.post("/score", json={"AMT_INCOME_TOTAL": -5, "AMT_CREDIT": 100, "DAYS_BIRTH": -9000})
    assert r.status_code == 422
