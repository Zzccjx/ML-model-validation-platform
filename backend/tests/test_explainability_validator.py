"""Tests for Module 6: Model Explainability & Feature Importance Validator."""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from app.validators.explainability.validator import ExplainabilityValidator
from app.validators.base import ValidationContext, ValidatorStatus


class DominantFeatureMockModel:
    """Mock model with a single ultra-dominant feature."""

    def __init__(self):
        self.feature_importances_ = np.array([0.92, 0.04, 0.03, 0.01])
        self.classes_ = np.array([0, 1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)


class DeadFeaturesMockModel:
    """Mock model with several zero-importance features."""

    def __init__(self):
        self.feature_importances_ = np.array([0.60, 0.40, 0.0, 0.0, 0.0])
        self.classes_ = np.array([0, 1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.ones(len(X), dtype=int)


@pytest.fixture
def explainability_dataset():
    np.random.seed(42)
    n = 150
    X1 = np.random.normal(0, 1, n)
    X2 = np.random.normal(2, 1, n)
    X3 = np.random.normal(-1, 2, n)
    X4 = np.random.uniform(0, 5, n)
    y = ((X1 + X2) > 1).astype(int)

    df = pd.DataFrame({
        "feat_1": X1,
        "feat_2": X2,
        "feat_3": X3,
        "feat_4": X4,
        "target": y,
    })
    return df


@pytest.mark.asyncio
async def test_explainability_validator_skipped_when_no_model(explainability_dataset):
    """Explainability requires a model; should return SKIPPED when model is None."""
    context = ValidationContext(
        model=None,
        dataset=explainability_dataset,
        target_column="target",
        feature_columns=["feat_1", "feat_2", "feat_3", "feat_4"],
    )
    validator = ExplainabilityValidator()
    assert validator.is_applicable(context) is False

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SKIPPED


@pytest.mark.asyncio
async def test_explainability_validator_tree_model(explainability_dataset):
    """RandomForest model should extract tree_feature_importances and normalize to 100%."""
    df = explainability_dataset
    X = df[["feat_1", "feat_2", "feat_3", "feat_4"]]
    y = df["target"]
    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    rf.fit(X, y)

    context = ValidationContext(
        model=rf,
        model_type="classification",
        dataset=df,
        target_column="target",
        feature_columns=["feat_1", "feat_2", "feat_3", "feat_4"],
    )
    validator = ExplainabilityValidator()
    assert validator.is_applicable(context) is True

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SUCCESS
    assert result.score is not None
    assert result.metrics["attribution_method"] == "tree_feature_importances"
    assert result.metrics["total_features_analyzed"] == 4

    features = result.metrics["features"]
    assert len(features) == 4
    total_rel = sum(f["relative_percentage"] for f in features)
    assert pytest.approx(total_rel, 0.1) == 100.0
    assert features[-1]["cumulative_percentage"] == 100.0


@pytest.mark.asyncio
async def test_explainability_validator_linear_model(explainability_dataset):
    """LogisticRegression model should extract linear_coefficients."""
    df = explainability_dataset
    X = df[["feat_1", "feat_2", "feat_3", "feat_4"]]
    y = df["target"]
    lr = LogisticRegression()
    lr.fit(X, y)

    context = ValidationContext(
        model=lr,
        model_type="classification",
        dataset=df,
        target_column="target",
        feature_columns=["feat_1", "feat_2", "feat_3", "feat_4"],
    )
    validator = ExplainabilityValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.metrics["attribution_method"] == "linear_coefficients"
    assert "top_feature_name" in result.metrics
    assert result.score >= 70.0


@pytest.mark.asyncio
async def test_explainability_validator_single_feature_dominance(explainability_dataset):
    """Extreme single-feature dominance should trigger CRITICAL/HIGH finding and score penalty."""
    model = DominantFeatureMockModel()
    context = ValidationContext(
        model=model,
        model_type="classification",
        dataset=explainability_dataset,
        target_column="target",
        feature_columns=["feat_1", "feat_2", "feat_3", "feat_4"],
    )
    validator = ExplainabilityValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.metrics["distribution_balance"] == "DOMINATED"
    assert result.metrics["top_feature_share_percentage"] >= 80.0
    assert any(f.severity in ("critical", "high") for f in result.findings)
    assert result.score < 70.0


@pytest.mark.asyncio
async def test_explainability_validator_zero_impact_features():
    """Features with zero attribution should trigger low-impact finding."""
    model = DeadFeaturesMockModel()
    df = pd.DataFrame(np.random.randn(50, 5), columns=["f1", "f2", "f3", "f4", "f5"])
    df["target"] = np.random.choice([0, 1], 50)

    context = ValidationContext(
        model=model,
        model_type="classification",
        dataset=df,
        target_column="target",
        feature_columns=["f1", "f2", "f3", "f4", "f5"],
    )
    validator = ExplainabilityValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.metrics["zero_impact_features_count"] == 3
    assert any("Uninformative Features Detected" in f.title for f in result.findings)