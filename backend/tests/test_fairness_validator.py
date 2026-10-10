"""Tests for Module 3: Fairness & Bias Validator."""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.validators.fairness.validator import FairnessValidator
from app.validators.base import ValidationContext, ValidatorStatus


class DummyBiasedModel:
    """Mock model that predicts positive label predominantly for group M."""

    def __init__(self, sensitive_col: str = "gender"):
        self.sensitive_col = sensitive_col
        self.classes_ = np.array([0, 1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.sensitive_col in X.columns:
            return np.where(X[self.sensitive_col] == "M", 1, 0)
        return np.ones(len(X), dtype=int)


class DummyFairModel:
    """Mock model with identical predictions across demographic groups."""

    def __init__(self):
        self.classes_ = np.array([0, 1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)


@pytest.fixture
def demographic_dataset():
    np.random.seed(42)
    n = 200
    genders = ["M"] * 100 + ["F"] * 100
    y = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    f1 = np.random.randn(n)
    f2 = np.random.randn(n)

    df = pd.DataFrame({
        "feature_1": f1,
        "feature_2": f2,
        "gender": genders,
        "target": y,
    })
    return df


@pytest.mark.asyncio
async def test_fairness_validator_skipped_when_no_sensitive_cols(demographic_dataset):
    """Fairness validator must return SKIPPED when sensitive_columns is empty/None."""
    context = ValidationContext(
        model=DummyFairModel(),
        dataset=demographic_dataset,
        target_column="target",
        feature_columns=["feature_1", "feature_2"],
        sensitive_columns=None,
    )
    validator = FairnessValidator()
    assert validator.is_applicable(context) is False

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SKIPPED
    assert result.score is None
    assert "not applicable" in result.error_message.lower()


@pytest.mark.asyncio
async def test_fairness_validator_skipped_when_columns_not_in_dataset(demographic_dataset):
    """Fairness validator must return SKIPPED if requested sensitive columns are absent."""
    context = ValidationContext(
        model=DummyFairModel(),
        dataset=demographic_dataset,
        target_column="target",
        feature_columns=["feature_1", "feature_2"],
        sensitive_columns=["non_existent_column"],
    )
    validator = FairnessValidator()
    assert validator.is_applicable(context) is False

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SKIPPED


@pytest.mark.asyncio
async def test_fairness_compliant_model(demographic_dataset):
    """Fair model with balanced outcomes should pass Four-Fifths rule and score high."""
    X = demographic_dataset[["feature_1", "feature_2"]]
    y = demographic_dataset["target"]
    clf = LogisticRegression()
    clf.fit(X, y)

    context = ValidationContext(
        model=clf,
        dataset=demographic_dataset,
        target_column="target",
        feature_columns=["feature_1", "feature_2"],
        sensitive_columns=["gender"],
    )
    validator = FairnessValidator()
    assert validator.is_applicable(context) is True

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SUCCESS
    assert result.score is not None
    assert "gender" in result.metrics["attributes"]
    attr_metrics = result.metrics["attributes"]["gender"]
    assert "disparate_impact_ratio" in attr_metrics
    assert "demographic_parity_difference" in attr_metrics
    assert "equal_opportunity_difference" in attr_metrics
    assert "subgroups" in attr_metrics
    assert "M" in attr_metrics["subgroups"]
    assert "F" in attr_metrics["subgroups"]


@pytest.mark.asyncio
async def test_fairness_severe_disparate_impact():
    """Biased predictions should trigger critical/high findings and penalize score."""
    df = pd.DataFrame({
        "gender": ["M"] * 100 + ["F"] * 100,
        "feature_1": [1.0] * 100 + [0.0] * 100,
        "target": [1] * 100 + [1] * 100,
    })

    model = DummyBiasedModel(sensitive_col="gender")

    context = ValidationContext(
        model=model,
        dataset=df,
        target_column="target",
        feature_columns=["feature_1", "gender"],
        sensitive_columns=["gender"],
    )
    validator = FairnessValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    attr = result.metrics["attributes"]["gender"]
    assert attr["four_fifths_compliant"] is False
    assert attr["disparate_impact_ratio"] < 0.80

    severities = [f.severity for f in result.findings]
    assert "critical" in severities or "high" in severities
    assert result.score < 70.0


@pytest.mark.asyncio
async def test_small_subgroup_sample_size_warning():
    """Subgroups with fewer than 30 samples should trigger info finding."""
    df = pd.DataFrame({
        "ethnicity": ["GroupA"] * 80 + ["GroupB"] * 15,
        "feature": np.random.randn(95),
        "target": np.random.choice([0, 1], 95),
    })

    model = DummyFairModel()

    context = ValidationContext(
        model=model,
        dataset=df,
        target_column="target",
        feature_columns=["feature"],
        sensitive_columns=["ethnicity"],
    )
    validator = FairnessValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    info_titles = [f.title for f in result.findings if f.severity == "info"]
    assert any("Small Subgroup Sample Size" in t for t in info_titles)