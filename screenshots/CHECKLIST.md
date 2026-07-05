# Evidence checklist (screenshots + video)

Each item maps to a placeholder in `report/REPORT.md`.

## Screenshots

| # | What to capture | How to get it |
|---|---|---|
| 1 | MLflow experiment run list (both runs) | `source .venv/bin/activate && mlflow ui` → http://localhost:5000 |
| 2 | MLflow run detail: metrics + artifacts (ROC curve, confusion matrix) | click the `logistic_regression` run |
| 3 | EDA plots | already embedded in `notebooks/01_eda.ipynb` — export if needed |
| 4 | Green GitHub Actions run (both jobs) | repo → Actions tab, after first push |
| 5 | `kubectl get pods,svc` showing 2 replicas + LoadBalancer | after `kubectl apply -f deployment/` |
| 6 | `/predict` response | `curl -X POST <url>/predict -H 'Content-Type: application/json' -d @sample_input.json` |
| 7 | `/metrics` Prometheus output | `curl <url>/metrics` or browser |
| 8 | Swagger UI | http://localhost:8000/docs |

## Demo video (~3 min, QuickTime: File → New Screen Recording)

1. (20s) Repo tour: structure, README results table.
2. (30s) `pytest` + `flake8 src tests` passing.
3. (40s) `python -m src.models.train` → open MLflow UI, show runs, metrics, ROC artifact.
4. (40s) Start API (`uvicorn src.api.main:app`), Swagger UI, `curl /predict` with sample input, show `/metrics`.
5. (30s) GitHub Actions page: green pipeline (lint → test → train → docker smoke test).
6. (20s) `kubectl get pods,svc` + `/predict` through the LoadBalancer.
