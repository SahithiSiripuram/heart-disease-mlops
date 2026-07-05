"""Data cleaning and the reusable preprocessing pipeline.

The UCI Heart Disease target is 0-4 (0 = no disease, 1-4 = increasing severity);
we binarize it to presence/absence. Preprocessing is a sklearn ColumnTransformer
so the exact same transforms run at training and inference time:

- numeric features: median imputation + standard scaling
- categorical features: most-frequent imputation + one-hot encoding
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_CSV = Path("data/raw/heart_disease.csv")
PROCESSED_CSV = Path("data/processed/heart_disease_clean.csv")

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target"

RANDOM_STATE = 42


def load_clean_data(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV and binarize the target (0 = no disease, 1 = disease)."""
    df = pd.read_csv(path)
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df[ALL_FEATURES + [TARGET]]


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def build_preprocessor() -> ColumnTransformer:
    """Preprocessing transformer reused by every model and at inference."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def train_test_data(df: pd.DataFrame | None = None, test_size: float = 0.2):
    """Stratified train/test split of features and binary target."""
    if df is None:
        df = load_clean_data()
    X = df[ALL_FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


if __name__ == "__main__":
    clean = load_clean_data()
    out = save_processed(clean)
    print(f"Saved cleaned data ({len(clean)} rows) to {out}")
