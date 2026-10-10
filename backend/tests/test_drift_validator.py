"""Tests for Module 4: Data & Concept Drift Validator."""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.validators.drift.validator import DriftValidator, calculate_psi
from app.validators.base import ValidationContext, ValidatorStatus


@pytest.fixture
def baseline_dataset():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "num_feature_1": np.random.normal(0, 1, n),
        "num_feature_2": np.random.uniform(10, 20, n),
        "cat_feature": np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "target": np.random.choice([0, 1], n),
    })
    return df


def test_calculate_psi():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 500)
    curr_same = np.random.normal(0, 1, 500)
    curr_drifted = np.random.normal(3, 1, 500)

    psi_same = calculate_psi(ref, curr_same)
    psi_drifted = calculate_psi(ref, curr_drifted)

    assert psi_same < 0.10
    assert psi_drifted > 0.25


@pytest.mark.asyncio
async def test_drift_validator_sequential_partition(baseline_dataset):
    """When reference_dataset is None, DriftValidator applies sequential partition."""
    context = ValidationContext(
        model=None,
        dataset=baseline_dataset,
        target_column="target",
        feature_columns=["num_feature_1", "num_feature_2", "cat_feature"],
    )
    validator = DriftValidator()
    assert validator.is_applicable(context) is True

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SUCCESS
    assert result.metrics["is_sequential_partition"] is True
    assert result.score is not None
    assert any("Sequential Partition Applied" in f.title for f in result.findings)


@pytest.mark.asyncio
async def test_drift_validator_stable_reference_dataset(baseline_dataset):
    """When reference and current datasets share the same distribution, drift is minimal."""
    np.random.seed(42)
    n = 200
    current_df = pd.DataFrame({
        "num_feature_1": np.random.normal(0, 1, n),
        "num_feature_2": np.random.uniform(10, 20, n),
        "cat_feature": np.random.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2]),
        "target": np.random.choice([0, 1], n),
    })

    context = ValidationContext(
        model=None,
        dataset=current_df,
        reference_dataset=baseline_dataset,
        target_column="target",
        feature_columns=["num_feature_1", "num_feature_2", "cat_feature"],
    )
    validator = DriftValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.score >= 85.0
    assert result.metrics["overall_drift_detected"] is False
    assert result.metrics["overall_drift_status"] == "stable"


@pytest.mark.asyncio
async def test_drift_validator_severe_drift(baseline_dataset):
    """Severe distribution shift should trigger HIGH/CRITICAL findings and score penalty."""
    np.random.seed(99)
    n = 200
    severely_drifted = pd.DataFrame({
        "num_feature_1": np.random.normal(10, 5, n),  # Huge shift from N(0, 1)
        "num_feature_2": np.random.uniform(50, 100, n),  # Huge shift from U(10, 20)
        "cat_feature": np.random.choice(["A", "B", "C"], n, p=[0.05, 0.05, 0.90]),  # Shifted categories
        "target": np.random.choice([0, 1], n),
    })

    context = ValidationContext(
        model=None,
        dataset=severely_drifted,
        reference_dataset=baseline_dataset,
        target_column="target",
        feature_columns=["num_feature_1", "num_feature_2", "cat_feature"],
    )
    validator = DriftValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.score < 60.0
    assert result.metrics["overall_drift_detected"] is True
    assert result.metrics["drifted_features_count"] >= 2
    assert any(f.severity in ("critical", "high") for f in result.findings)


@pytest.mark.asyncio
async def test_drift_validator_concept_drift(baseline_dataset):
    """When model prediction distributions diverge between reference and current data."""
    X_ref = baseline_dataset[["num_feature_1", "num_feature_2"]]
    y_ref = (X_ref["num_feature_1"] > 0).astype(int)
    clf = LogisticRegression()
    clf.fit(X_ref, y_ref)

    # Current dataset where feature_1 is all positive, causing prediction shift
    n = 200
    shifted_df = pd.DataFrame({
        "num_feature_1": np.random.normal(5, 1, n),
        "num_feature_2": np.random.uniform(10, 20, n),
        "cat_feature": ["A"] * n,
        "target": [1] * n,
    })

    context = ValidationContext(
        model=clf,
        model_type="classification",
        dataset=shifted_df,
        reference_dataset=baseline_dataset,
        target_column="target",
        feature_columns=["num_feature_1", "num_feature_2"],
    )
    validator = DriftValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert "concept_drift" in result.metrics
    assert "prediction_psi" in result.metrics["concept_drift"]