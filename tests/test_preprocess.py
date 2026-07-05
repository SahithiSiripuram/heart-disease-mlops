import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    ALL_FEATURES,
    TARGET,
    build_preprocessor,
    load_clean_data,
    train_test_data,
)


@pytest.fixture()
def raw_df(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "age": rng.integers(30, 75, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.integers(100, 180, n),
            "chol": rng.integers(150, 350, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(90, 200, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.uniform(0, 4, n).round(1),
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], n),
            "target": rng.integers(0, 5, n),
        }
    )
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    path = tmp_path / "raw.csv"
    df.to_csv(path, index=False)
    return path


def test_load_clean_data_binarizes_target(raw_df):
    df = load_clean_data(raw_df)
    assert set(df[TARGET].unique()) <= {0, 1}
    assert list(df.columns) == ALL_FEATURES + [TARGET]


def test_preprocessor_handles_missing_values(raw_df):
    df = load_clean_data(raw_df)
    X = df[ALL_FEATURES]
    transformed = build_preprocessor().fit_transform(X)
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()
    assert transformed.shape[0] == len(X)


def test_train_test_split_is_stratified(raw_df):
    df = load_clean_data(raw_df)
    X_train, X_test, y_train, y_test = train_test_data(df, test_size=0.25)
    assert len(X_train) + len(X_test) == len(df)
    # stratification keeps class balance within a few points
    assert abs(y_train.mean() - y_test.mean()) < 0.2
