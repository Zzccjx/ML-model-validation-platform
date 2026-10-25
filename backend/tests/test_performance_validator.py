"""Tests for Module 1: Model Performance Validator."""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from app.validators.performance.validator import PerformanceValidator
from app.validators.base import ValidationContext, ValidatorStatus


@pytest.fixture
def binary_classification_setup():
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    feature_names = [f"f_{i}" for i in range(5)]
    
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    context = ValidationContext(
        model=model,
        model_type="classification",
        framework="scikit-learn",
        dataset=df,
        target_column="target",
        feature_columns=feature_names,
    )
    return context


@pytest.fixture
def regression_setup():
    np.random.seed(42)
    X = np.random.randn(150, 4)
    y = X[:, 0] * 3.5 + X[:, 1] * 1.5 + np.random.randn(150) * 0.5
    feature_names = [f"feat_{i}" for i in range(4)]
    
    model = RandomForestRegressor(n_estimators=20, random_state=42)
    model.fit(X, y)
    
    df = pd.DataFrame(X, columns=feature_names)
    df["price"] = y
    
    context = ValidationContext(
        model=model,
        model_type="regression",
        framework="scikit-learn",
        dataset=df,
        target_column="price",
        feature_columns=feature_names,
    )
    return context


@pytest.mark.asyncio
async def test_classification_performance_validator_binary(binary_classification_setup):
    validator = PerformanceValidator()
    assert validator.is_applicable(binary_classification_setup) is True
    
    result = await validator.safe_validate(binary_classification_setup)
    
    assert result.status == ValidatorStatus.SUCCESS
    assert result.score is not None
    assert 0 <= result.score <= 100
    
    # Verify metrics
    metrics = result.metrics
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "f1_macro" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert "per_class_metrics" in metrics
    
    assert metrics["confusion_matrix"]["labels"] == ["0", "1"]
    assert len(metrics["confusion_matrix"]["matrix"]) == 2
    assert "0" in metrics["per_class_metrics"]
    assert "1" in metrics["per_class_metrics"]


@pytest.mark.asyncio
async def test_classification_without_probabilities():
    class NoProbaClassifier:
        def __init__(self):
            self.classes_ = np.array([0, 1])
        def predict(self, X):
            return np.ones(len(X), dtype=int)
            
    df = pd.DataFrame({"x1": [1, 2, 3, 4], "target": [0, 1, 0, 1]})
    context = ValidationContext(
        model=NoProbaClassifier(),
        model_type="classification",
        framework="custom",
        dataset=df,
        target_column="target",
        feature_columns=["x1"],
    )
    validator = PerformanceValidator()
    result = await validator.safe_validate(context)
    
    assert result.status == ValidatorStatus.SUCCESS
    assert result.metrics["roc_auc"] is None
    # Emits info finding about missing probabilities
    info_findings = [f for f in result.findings if f.severity == "info"]
    assert len(info_findings) >= 1
    assert "Probability" in info_findings[0].title


@pytest.mark.asyncio
async def test_regression_performance_validator(regression_setup):
    validator = PerformanceValidator()
    assert validator.is_applicable(regression_setup) is True
    
    result = await validator.safe_validate(regression_setup)
    
    assert result.status == ValidatorStatus.SUCCESS
    assert result.score is not None
    assert result.score > 70  # Well-fitted model should achieve strong R^2
    
    metrics = result.metrics
    assert "r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "residuals" in metrics
    assert "quantiles" in metrics["residuals"]
    assert metrics["r2"] > 0.70
    assert metrics["residuals"]["mean_residual"] is not None


@pytest.mark.asyncio
async def test_regression_negative_r2_finding():
    class TerriblyFittedModel:
        def predict(self, X):
            return np.full(len(X), 1000.0)  # Grossly wrong constant prediction
            
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.0, 0.5, 1.0]})
    context = ValidationContext(
        model=TerriblyFittedModel(),
        model_type="regression",
        framework="custom",
        dataset=df,
        target_column="y",
        feature_columns=["x"],
    )
    validator = PerformanceValidator()
    result = await validator.safe_validate(context)
    
    assert result.status == ValidatorStatus.SUCCESS
    assert result.score == 0.0
    crit_findings = [f for f in result.findings if f.severity == "critical"]
    assert len(crit_findings) >= 1
    assert "Negative R²" in crit_findings[0].title


@pytest.mark.asyncio
async def test_accuracy_paradox_finding():
    class MajorityPredictor:
        def __init__(self):
            self.classes_ = np.array([0, 1])
        def predict(self, X):
            return np.zeros(len(X), dtype=int)  # Always predicts majority class 0
            
    # 95 instances of 0, 5 instances of 1 -> 95% accuracy, but 50% balanced accuracy
    y = [0] * 95 + [1] * 5
    df = pd.DataFrame({"x": np.random.randn(100), "target": y})
    context = ValidationContext(
        model=MajorityPredictor(),
        model_type="classification",
        framework="custom",
        dataset=df,
        target_column="target",
        feature_columns=["x"],
    )
    validator = PerformanceValidator()
    result = await validator.safe_validate(context)
    
    assert result.status == ValidatorStatus.SUCCESS
    # Should catch the disparity and emit HIGH finding
    disparity_findings = [f for f in result.findings if "Accuracy Paradox" in f.title]
    assert len(disparity_findings) == 1
    assert disparity_findings[0].severity == "high"


@pytest.mark.asyncio
async def test_classification_row_level_predictions(binary_classification_setup):
    """Deepchecks reference: Verify that row-level predictions match independent calculation."""
    validator = PerformanceValidator()
    result = await validator.safe_validate(binary_classification_setup)
    
    assert "predictions" in result.metrics
    preds = result.metrics["predictions"]
    assert preds["total_rows"] == 200
    assert preds["has_probabilities"] is True
    assert "sample_rows" in preds
    
    # Check first row matches model.predict and predict_proba exactly
    df = binary_classification_setup.dataset
    target = binary_classification_setup.target_column
    X = df[binary_classification_setup.feature_columns]
    y_true = df[target].values
    
    raw_pred = binary_classification_setup.model.predict(X)
    raw_proba = binary_classification_setup.model.predict_proba(X)
    
    first_sample = preds["sample_rows"][0]
    assert first_sample["actual"] == int(y_true[0])
    assert first_sample["predicted"] == int(raw_pred[0])
    assert first_sample["correct"] == bool(y_true[0] == raw_pred[0])
    assert "0" in first_sample["probabilities"]
    assert "1" in first_sample["probabilities"]
    assert pytest.approx(first_sample["probabilities"]["1"], 0.001) == raw_proba[0, 1]


@pytest.mark.asyncio
async def test_regression_row_level_predictions(regression_setup):
    """Deepchecks reference: Verify that regression row-level predictions and errors match independent calculation."""
    validator = PerformanceValidator()
    result = await validator.safe_validate(regression_setup)
    
    assert "predictions" in result.metrics
    preds = result.metrics["predictions"]
    assert preds["total_rows"] == 150
    assert "sample_rows" in preds
    
    df = regression_setup.dataset
    target = regression_setup.target_column
    X = df[regression_setup.feature_columns]
    y_true = df[target].values
    raw_pred = regression_setup.model.predict(X)
    
    first_sample = preds["sample_rows"][0]
    assert pytest.approx(first_sample["actual"], 0.001) == float(y_true[0])
    assert pytest.approx(first_sample["predicted"], 0.001) == float(raw_pred[0])
    assert pytest.approx(first_sample["error"], 0.001) == float(y_true[0] - raw_pred[0])
    assert pytest.approx(first_sample["abs_error"], 0.001) == abs(float(y_true[0] - raw_pred[0]))

