"""Tests for format-aware model adapters."""
import pytest
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from app.adapters.factory import get_adapter
from app.adapters.sklearn_adapter import SklearnAdapter
from app.adapters.keras_adapter import KerasAdapter
from app.adapters.onnx_adapter import OnnxAdapter


@pytest.fixture
def sample_sklearn_joblib(tmp_path):
    X = np.random.randn(20, 4)
    y = np.random.choice([0, 1], 20)
    clf = LogisticRegression()
    clf.fit(X, y)
    p = tmp_path / "model.joblib"
    joblib.dump(clf, p)
    return str(p)


@pytest.fixture
def sample_sklearn_pickle(tmp_path):
    X = np.random.randn(20, 3)
    y = np.random.randn(20)
    reg = RandomForestRegressor(n_estimators=3)
    reg.fit(X, y)
    p = tmp_path / "model.pickle"
    with open(p, "wb") as f:
        pickle.dump(reg, f)
    return str(p)


@pytest.fixture
def sample_keras_model(tmp_path):
    try:
        import keras
        from keras import layers
    except ImportError:
        import tensorflow.keras as keras
        from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(4,)),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    p = tmp_path / "model.keras"
    model.save(p)
    return str(p)


def test_factory_resolves_sklearn_joblib(sample_sklearn_joblib):
    adapter = get_adapter(sample_sklearn_joblib)
    assert isinstance(adapter, SklearnAdapter)
    assert adapter.framework_name == "scikit-learn"
    assert adapter.model_type == "classification"
    meta = adapter.inspect()
    assert meta["has_predict"] is True
    assert meta["has_predict_proba"] is True
    assert meta["feature_count"] == 4


def test_factory_resolves_sklearn_pickle(sample_sklearn_pickle):
    adapter = get_adapter(sample_sklearn_pickle)
    assert isinstance(adapter, SklearnAdapter)
    assert adapter.model_type == "regression"
    meta = adapter.inspect()
    assert meta["feature_count"] == 3


def test_factory_resolves_keras_model(sample_keras_model):
    adapter = get_adapter(sample_keras_model)
    assert isinstance(adapter, KerasAdapter)
    assert adapter.framework_name == "keras / tensorflow"
    assert adapter.model_type == "classification"
    meta = adapter.inspect()
    assert meta["has_predict"] is True
    assert meta["feature_count"] == 4

    # Test bounded prediction and probability execution
    X_sample = np.random.randn(5, 4)
    preds = adapter.predict(X_sample)
    assert len(preds) == 5
    assert all(p in (0, 1) for p in preds)

    probs = adapter.predict_proba(X_sample)
    assert probs is not None
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_factory_rejects_unsupported_format(tmp_path):
    bad_file = tmp_path / "model.pt"
    bad_file.write_text("dummy")
    with pytest.raises(ValueError) as exc:
        get_adapter(str(bad_file))
    assert "Unsupported model file format" in str(exc.value)


def test_onnx_adapter_graceful_missing_runtime(tmp_path):
    p = tmp_path / "model.onnx"
    p.write_bytes(b"dummy onnx bytes")
    adapter = get_adapter(str(p))
    assert isinstance(adapter, OnnxAdapter)
    assert adapter.framework_name == "onnx"
