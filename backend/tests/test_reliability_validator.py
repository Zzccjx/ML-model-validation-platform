import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from app.validators.reliability import ReliabilityValidator
from app.validators.base import ValidationContext, ValidatorStatus


@pytest.fixture
def reliability_validator():
    return ReliabilityValidator()


@pytest.fixture
def binary_classification_data():
    np.random.seed(42)
    X = np.random.randn(200, 4)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, df, "target"


@pytest.fixture
def multiclass_data():
    np.random.seed(42)
    X = np.random.randn(150, 4)
    y = np.random.choice([0, 1, 2], size=150)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["target"] = y

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, df, "target"


def test_reliability_metadata(reliability_validator):
    assert reliability_validator.category == "reliability"
    assert reliability_validator.name == "reliability"
    assert reliability_validator.version == "1.0.0"


def test_binary_reliability_validation(reliability_validator, binary_classification_data):
    model, df, target = binary_classification_data
    context = ValidationContext(
        model=model,
        model_type="classification",
        framework="sklearn",
        dataset=df,
        target_column=target,
        feature_columns=[f"f{i}" for i in range(4)],
    )

    result = reliability_validator.validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.category == "reliability"
    assert result.score is not None and 0.0 <= result.score <= 100.0
    metrics = result.metrics
    assert metrics["classification_mode"] == "binary"
    assert "brier_score" in metrics
    assert "expected_calibration_error" in metrics
    assert "maximum_calibration_error" in metrics
    assert "overconfident_errors_count" in metrics
    assert "calibration_bins" in metrics
    assert len(metrics["calibration_bins"]) == 10
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0


def test_multiclass_reliability_validation(reliability_validator, multiclass_data):
    model, df, target = multiclass_data
    context = ValidationContext(
        model=model,
        model_type="classification",
        framework="sklearn",
        dataset=df,
        target_column=target,
        feature_columns=[f"f{i}" for i in range(4)],
    )

    result = reliability_validator.validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.category == "reliability"
    assert result.score is not None and 0.0 <= result.score <= 100.0
    metrics = result.metrics
    assert metrics["classification_mode"] == "multiclass"
    assert "brier_score" in metrics
    assert "expected_calibration_error" in metrics
    assert len(metrics["calibration_bins"]) == 10


def test_regression_reliability_validation(reliability_validator):
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2.0 * X[:, 0] + np.random.normal(0, 0.2, 100)
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["target"] = y

    model = LinearRegression()
    model.fit(X, y)

    context = ValidationContext(
        model=model,
        model_type="regression",
        framework="sklearn",
        dataset=df,
        target_column="target",
        feature_columns=["a", "b", "c"],
    )

    result = reliability_validator.validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.category == "reliability"
    assert result.score is not None and 0.0 <= result.score <= 100.0
    assert result.metrics["prediction_type"] == "regression"
    assert "dispersion_ratio" in result.metrics
    assert "empirical_coverage_95" in result.metrics


def test_model_without_predict_proba(reliability_validator):
    class NoProbModel:
        def predict(self, X):
            return np.ones(len(X))

    df = pd.DataFrame({"f1": range(20), "target": [1] * 10 + [0] * 10})
    context = ValidationContext(
        model=NoProbModel(),
        model_type="classification",
        framework="custom",
        dataset=df,
        target_column="target",
        feature_columns=["f1"],
    )

    result = reliability_validator.validate(context)

    assert result.status == ValidatorStatus.NOT_EVALUATED
    assert result.score is None
    assert result.metrics["supports_probabilities"] is False
    assert any("predict_proba" in f.description for f in result.findings)
