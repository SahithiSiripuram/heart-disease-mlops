# Heart Disease Risk Prediction — MLOps Pipeline

End-to-end ML solution for the AIMLCZG523 MLOps Assignment 01: a heart disease
risk classifier (UCI Heart Disease dataset) built, tracked, tested, containerized,
and deployed with modern MLOps practices.

## Architecture

```
UCI dataset → download script → EDA / preprocessing (sklearn Pipeline)
            → model training (LogReg, Random Forest + tuning) → MLflow tracking
            → best model artifact → FastAPI /predict → Docker → Kubernetes
            → logging & monitoring
```

## Project structure

```
├── data/               # raw + processed data (gitignored; use download script)
├── notebooks/          # EDA and experimentation notebooks
├── src/
│   ├── data/           # download + preprocessing
│   ├── models/         # training, tuning, evaluation
│   └── api/            # FastAPI serving app
├── tests/              # pytest unit tests
├── deployment/         # Kubernetes manifests
├── .github/workflows/  # CI pipeline (lint, test, train)
├── screenshots/        # MLflow / CI / deployment evidence for the report
├── report/             # final written report
└── docs/               # assignment PDFs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Download the dataset
python -m src.data.download

# Train models (logs to MLflow)
python -m src.models.train

# View experiment runs
mlflow ui

# Run tests
pytest

# Serve the API locally
uvicorn src.api.main:app --reload
```

## Docker

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample_input.json
```

## Kubernetes

```bash
kubectl apply -f deployment/deployment.yaml
kubectl apply -f deployment/service.yaml
```
