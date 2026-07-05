import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import src.api.main as api
from src.data.preprocess import ALL_FEATURES, build_preprocessor

VALID_INPUT = {
    "age": 57,
    "sex": 1,
    "cp": 4,
    "trestbps": 140,
    "chol": 260,
    "fbs": 0,
    "restecg": 2,
    "thalach": 120,
    "exang": 1,
    "oldpeak": 2.1,
    "slope": 2,
    "ca": 1,
    "thal": 7,
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API client backed by a tiny model trained on synthetic data."""
    rng = np.random.default_rng(1)
    n = 80
    X = pd.DataFrame([VALID_INPUT] * n)
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        X[col] = X[col] + rng.normal(0, 10, n).round(1)
    y = rng.integers(0, 2, n)

    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("model", LogisticRegression())]
    )
    pipeline.fit(X[ALL_FEATURES], y)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    monkeypatch.setattr(api, "MODEL_PATH", model_path)
    monkeypatch.setattr(api, "_model", None)
    return TestClient(api.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_prediction_and_confidence(client):
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0


def test_predict_accepts_missing_optional_features(client):
    payload = {k: v for k, v in VALID_INPUT.items() if k not in ("ca", "thal")}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_rejects_invalid_input(client):
    bad = dict(VALID_INPUT, age=-5)
    assert client.post("/predict", json=bad).status_code == 422
    incomplete = {"age": 50}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_metrics_endpoint(client):
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
