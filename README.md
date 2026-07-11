# Heart Disease Risk Prediction — MLOps Pipeline

End-to-end ML solution for the AIMLCZG523 MLOps Assignment 01: a heart disease
risk classifier (UCI Heart Disease dataset) built, tracked, tested, containerized,
and deployed with modern MLOps practices.

## Architecture

```
UCI dataset → download script → EDA / preprocessing (sklearn ColumnTransformer)
            → GridSearchCV training (LogReg, Random Forest) → MLflow tracking
            → best model artifact → FastAPI /predict → Docker → Kubernetes
            → request logging + Prometheus /metrics
```

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC (test) | ROC-AUC (5-fold CV) |
|---|---|---|---|---|---|---|
| Logistic Regression (best) | 0.869 | 0.813 | 0.929 | 0.867 | **0.958** | 0.900 |
| Random Forest | 0.885 | 0.839 | 0.929 | 0.881 | 0.943 | 0.900 |

Both models are tuned with 5-fold GridSearchCV over the full preprocessing +
estimator pipeline; the best model by test ROC-AUC is exported for serving.

## Project structure

```
├── data/               # raw + processed data (gitignored; use download script)
├── notebooks/01_eda.ipynb  # EDA: distributions, correlations, class balance
├── src/
│   ├── data/           # download.py, preprocess.py (ColumnTransformer pipeline)
│   ├── models/         # train.py (GridSearchCV + MLflow logging)
│   └── api/            # main.py (FastAPI: /health, /predict, /metrics)
├── tests/              # pytest: preprocessing + API contract tests
├── deployment/         # Kubernetes Deployment + LoadBalancer Service
├── .github/workflows/  # CI: lint → test → train → docker build + smoke test
├── models/             # exported best model + metadata
├── screenshots/        # MLflow / CI / deployment evidence for the report
└── report/             # final written report
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Download the dataset (303 rows from the UCI repository)
python -m src.data.download

# 2. Train + tune both models; logs params/metrics/plots/models to MLflow
python -m src.models.train

# 3. Browse experiment runs
mlflow ui

# 4. Lint + tests (same commands CI runs)
flake8 src tests
pytest

# 5. Serve the API locally
uvicorn src.api.main:app --reload
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample_input.json
# → {"prediction":1,"confidence":0.988}
```

Interactive API docs (Swagger UI) at `http://localhost:8000/docs`.

## Monitoring

Every request is logged (method, path, status, latency). Prometheus metrics are
exposed at `/metrics`: request counts by endpoint/status, latency histograms,
and served predictions by class — scrapeable by a Prometheus + Grafana stack.

## Docker

```bash
python -m src.models.train        # produce models/model.joblib first
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
curl -sf localhost:8000/health
```

CI performs this exact build + `/predict` smoke test on every push.

## Kubernetes (Minikube / Docker Desktop)

```bash
# Build the image inside minikube's docker daemon, then:
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
kubectl get svc heart-disease-api   # LoadBalancer; `minikube tunnel` for local LB
```

The Deployment runs 2 replicas with liveness/readiness probes on `/health`.
