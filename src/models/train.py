"""Train and tune classifiers, logging everything to MLflow.

Usage: python -m src.models.train

For each candidate model: GridSearchCV (5-fold, ROC-AUC) over the full
preprocessing + estimator pipeline, then one MLflow run logging best params,
cross-validated and held-out-test metrics, ROC curve, confusion matrix, and the
fitted pipeline. The best model by test ROC-AUC is saved to models/model.joblib
for the serving API.
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.data.preprocess import RANDOM_STATE, build_preprocessor, train_test_data

EXPERIMENT_NAME = "heart-disease-classification"
MODEL_DIR = Path("models")

CANDIDATES = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "param_grid": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 3, 5],
        },
    },
}


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }


def log_plots(pipeline: Pipeline, X_test, y_test, name: str) -> None:
    for plot_cls, fname in [
        (RocCurveDisplay, f"{name}_roc_curve.png"),
        (ConfusionMatrixDisplay, f"{name}_confusion_matrix.png"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 5))
        plot_cls.from_estimator(pipeline, X_test, y_test, ax=ax)
        ax.set_title(f"{name} — {fname.split('_', 1)[1].removesuffix('.png')}")
        fig.tight_layout()
        mlflow.log_figure(fig, fname)
        plt.close(fig)


def train_one(name: str, spec: dict, X_train, X_test, y_train, y_test) -> dict:
    pipeline = Pipeline(
        [("preprocess", build_preprocessor()), ("model", spec["estimator"])]
    )
    search = GridSearchCV(
        pipeline,
        spec["param_grid"],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        refit=True,
    )

    with mlflow.start_run(run_name=name):
        search.fit(X_train, y_train)
        best = search.best_estimator_

        metrics = evaluate(best, X_test, y_test)
        metrics["cv_roc_auc"] = search.best_score_

        mlflow.log_params(search.best_params_)
        mlflow.log_param("model_type", name)
        mlflow.log_metrics(metrics)
        log_plots(best, X_test, y_test, name)
        mlflow.sklearn.log_model(best, name="model", serialization_format="cloudpickle")

        print(f"{name}: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")
        return {"name": name, "pipeline": best, **metrics}


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = train_test_data()

    results = [
        train_one(name, spec, X_train, X_test, y_train, y_test)
        for name, spec in CANDIDATES.items()
    ]

    best = max(results, key=lambda r: r["test_roc_auc"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_DIR / "model.joblib")
    metadata = {k: v for k, v in best.items() if k != "pipeline"}
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(
        f"Best model: {best['name']} "
        f"(test ROC-AUC {best['test_roc_auc']:.4f}) → models/model.joblib"
    )


if __name__ == "__main__":
    main()
