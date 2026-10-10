"""Tests for Module 5: Robustness & Adversarial Testing Validator."""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression

from app.validators.robustness.validator import RobustnessValidator
from app.validators.base import ValidationContext, ValidatorStatus


class FragileMockModel:
    """Mock model that changes predictions on the slightest perturbation near zero."""

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Volatile boundary based on parity of sum of rounded small floats
        vals = X.iloc[:, 0].values
        # Changes sign / label rapidly on small noise
        return (np.sin(vals * 100) > 0).astype(int)


@pytest.fixture
def sample_classification_data():
    np.random.seed(42)
    n = 200
    # Two well-separated clusters
    X1 = np.random.normal(5, 1, (100, 3))
    X2 = np.random.normal(-5, 1, (100, 3))
    X = np.vstack([X1, X2])
    y = np.array([1] * 100 + [0] * 100)

    df = pd.DataFrame(X, columns=["feat_1", "feat_2", "feat_3"])
    df["target"] = y
    return df


@pytest.fixture
def sample_regression_data():
    np.random.seed(42)
    n = 200
    X = np.random.normal(0, 1, (n, 3))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + np.random.normal(0, 0.1, n)

    df = pd.DataFrame(X, columns=["feat_1", "feat_2", "feat_3"])
    df["target"] = y
    return df


@pytest.mark.asyncio
async def test_robustness_validator_skipped_when_no_model(sample_classification_data):
    """Robustness requires a trained model; should return SKIPPED when model is None."""
    context = ValidationContext(
        model=None,
        dataset=sample_classification_data,
        target_column="target",
        feature_columns=["feat_1", "feat_2", "feat_3"],
    )
    validator = RobustnessValidator()
    assert validator.is_applicable(context) is False

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SKIPPED


@pytest.mark.asyncio
async def test_robustness_validator_classification_resilient(sample_classification_data):
    """Well-separated classification clusters should be highly resilient to noise."""
    df = sample_classification_data
    X = df[["feat_1", "feat_2", "feat_3"]]
    y = df["target"]
    clf = LogisticRegression()
    clf.fit(X, y)

    context = ValidationContext(
        model=clf,
        model_type="classification",
        dataset=df,
        target_column="target",
        feature_columns=["feat_1", "feat_2", "feat_3"],
    )
    validator = RobustnessValidator()
    assert validator.is_applicable(context) is True

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SUCCESS
    assert result.score is not None
    assert result.score >= 80.0
    assert result.metrics["robustness_rating"] in ("RESILIENT", "MODERATE")
    assert result.metrics["flip_rate_5pct"] < 0.15
    assert len(result.metrics["perturbation_curve"]) == 4


@pytest.mark.asyncio
async def test_robustness_validator_brittle_model(sample_classification_data):
    """Brittle decision boundary should trigger high flip rates and severe findings."""
    model = FragileMockModel()
    context = ValidationContext(
        model=model,
        model_type="classification",
        dataset=sample_classification_data,
        target_column="target",
        feature_columns=["feat_1", "feat_2", "feat_3"],
    )
    validator = RobustnessValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.metrics["flip_rate_5pct"] >= 0.20
    assert any(f.severity in ("critical", "high") for f in result.findings)
    assert result.score < 70.0


@pytest.mark.asyncio
async def test_robustness_validator_regression(sample_regression_data):
    """Regression model evaluated under perturbation curve."""
    df = sample_regression_data
    X = df[["feat_1", "feat_2", "feat_3"]]
    y = df["target"]
    reg = LinearRegression()
    reg.fit(X, y)

    context = ValidationContext(
        model=reg,
        model_type="regression",
        dataset=df,
        target_column="target",
        feature_columns=["feat_1", "feat_2", "feat_3"],
    )
    validator = RobustnessValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.score is not None
    assert "perturbation_curve" in result.metrics
    assert "masking_sensitivity" in result.metrics