"""FastAPI serving app.

Endpoints:
- GET /health   — liveness/readiness probe
- POST /predict — JSON patient features in, prediction + confidence out
- GET /metrics  — Prometheus metrics (request counts, latency, predictions)

The model is the full sklearn pipeline saved by src.models.train (preprocessing
included), so raw feature values are accepted as-is.
"""

import logging
import os
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("heart-disease-api")

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["path"]
)
PREDICTION_COUNT = Counter(
    "predictions_total", "Predictions served by class", ["prediction"]
)

app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts presence of heart disease from patient health data.",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {MODEL_PATH}; run `python -m src.models.train`.",
            )
        _model = joblib.load(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    return _model


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    REQUEST_LATENCY.labels(path).observe(elapsed)
    logger.info(
        "%s %s -> %s in %.1fms", request.method, path, response.status_code, elapsed * 1000
    )
    return response


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., gt=0, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., gt=0, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., gt=0, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: float | None = Field(None, ge=0, le=3, description="Major vessels colored (0-3)")
    thal: float | None = Field(None, description="Thalassemia (3, 6, or 7)")


class Prediction(BaseModel):
    prediction: int = Field(description="1 = heart disease present, 0 = absent")
    confidence: float = Field(description="Probability of the predicted class")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: PatientFeatures) -> Prediction:
    model = get_model()
    row = pd.DataFrame([features.model_dump()])
    proba_disease = float(model.predict_proba(row)[0, 1])
    prediction = int(proba_disease >= 0.5)
    confidence = proba_disease if prediction == 1 else 1 - proba_disease
    PREDICTION_COUNT.labels(str(prediction)).inc()
    return Prediction(prediction=prediction, confidence=round(confidence, 4))


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
