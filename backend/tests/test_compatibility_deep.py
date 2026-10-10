"""Tests for advanced model-dataset compatibility checks."""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import joblib

from app.services.compatibility_service import check_compatibility


@pytest.fixture
def named_features_model(tmp_path):
    X = np.random.randn(50, 3)
    y = np.random.choice([0, 1], 50)
    df = pd.DataFrame(X, columns=["feature_a", "feature_b", "feature_c"])
    
    model = LogisticRegression()
    model.fit(df, y)
    
    path = tmp_path / "named_model.pkl"
    joblib.dump(model, path)
    return str(path)


def test_feature_names_in_reordering(tmp_path, named_features_model):
    # CSV has columns in reverse order: c, b, a
    df = pd.DataFrame({
        "feature_c": np.random.randn(20),
        "feature_b": np.random.randn(20),
        "feature_a": np.random.randn(20),
        "target": np.random.choice([0, 1], 20),
    })
    data_path = tmp_path / "reversed.csv"
    df.to_csv(data_path, index=False)
    
    res = check_compatibility(
        model_path=named_features_model,
        dataset_path=str(data_path),
        target_column="target",
    )
    assert res.compatible is True
    assert res.dataset_info.get("aligned_with_feature_names_in") is True
    assert res.dataset_info["feature_columns"] == ["feature_a", "feature_b", "feature_c"]


def test_missing_named_features(tmp_path, named_features_model):
    # Missing feature_c
    df = pd.DataFrame({
        "feature_a": np.random.randn(20),
        "feature_b": np.random.randn(20),
        "target": np.random.choice([0, 1], 20),
    })
    data_path = tmp_path / "missing_col.csv"
    df.to_csv(data_path, index=False)
    
    res = check_compatibility(
        model_path=named_features_model,
        dataset_path=str(data_path),
        target_column="target",
    )
    assert res.compatible is False
    assert any("feature_c" in m for m in res.messages)


def test_all_nan_target_column(tmp_path, named_features_model):
    df = pd.DataFrame({
        "feature_a": np.random.randn(20),
        "feature_b": np.random.randn(20),
        "feature_c": np.random.randn(20),
        "target": [np.nan] * 20,
    })
    data_path = tmp_path / "all_nan.csv"
    df.to_csv(data_path, index=False)
    
    res = check_compatibility(
        model_path=named_features_model,
        dataset_path=str(data_path),
        target_column="target",
    )
    assert res.compatible is False
    assert any("NaN" in m for m in res.messages)


def test_non_numeric_target_for_regression(tmp_path):
    X = np.random.randn(30, 2)
    y = np.random.randn(30)
    model = RandomForestRegressor(n_estimators=5)
    model.fit(X, y)
    
    model_path = tmp_path / "reg.pkl"
    joblib.dump(model, model_path)
    
    df = pd.DataFrame({
        "feat_0": np.random.randn(20),
        "feat_1": np.random.randn(20),
        "price": ["cheap", "expensive"] * 10,  # String labels for regression
    })
    data_path = tmp_path / "bad_reg.csv"
    df.to_csv(data_path, index=False)
    
    res = check_compatibility(
        model_path=str(model_path),
        dataset_path=str(data_path),
        target_column="price",
    )
    assert res.compatible is False
    assert any("non-numeric" in m for m in res.messages)
