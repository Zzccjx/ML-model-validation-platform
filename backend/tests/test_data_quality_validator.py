"""Tests for Module 2: Data Quality Validator."""
import pytest
import numpy as np
import pandas as pd

from app.validators.data_quality.validator import DataQualityValidator
from app.validators.base import ValidationContext, ValidatorStatus


@pytest.fixture
def clean_dataset():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n) * 2 + 1,
        "feature_3": np.random.uniform(0, 10, n),
        "target": np.random.choice([0, 1], n),
    })
    return df


@pytest.mark.asyncio
async def test_clean_dataset_quality(clean_dataset):
    context = ValidationContext(
        model=None,
        dataset=clean_dataset,
        target_column="target",
        feature_columns=["feature_1", "feature_2", "feature_3"],
    )
    validator = DataQualityValidator()
    assert validator.is_applicable(context) is True

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SUCCESS
    assert result.score is not None
    assert result.score >= 90.0
    assert result.metrics["overall_missing_percentage"] == 0.0
    assert result.metrics["duplicate_rows_count"] == 0
    assert len(result.metrics["constant_columns"]) == 0


@pytest.mark.asyncio
async def test_missing_values_detection(clean_dataset):
    df = clean_dataset.copy()
    # Inject 30% missing into feature_1
    df.loc[:29, "feature_1"] = np.nan

    context = ValidationContext(
        model=None,
        dataset=df,
        target_column="target",
        feature_columns=["feature_1", "feature_2", "feature_3"],
    )
    validator = DataQualityValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.metrics["overall_missing_percentage"] > 0
    assert "feature_1" in result.metrics["missing_values"]

    # Should emit HIGH finding for > 20% missing
    high_findings = [f for f in result.findings if f.severity == "high" and "feature_1" in f.title]
    assert len(high_findings) >= 1
    assert "High Missing Rate" in high_findings[0].title


@pytest.mark.asyncio
async def test_duplicate_rows_and_constant_columns(clean_dataset):
    df = clean_dataset.copy()
    # Make feature_3 constant
    df["feature_3"] = 42.0
    # Duplicate first 15 rows
    df = pd.concat([df, df.iloc[:15]], ignore_index=True)

    context = ValidationContext(
        model=None,
        dataset=df,
        target_column="target",
        feature_columns=["feature_1", "feature_2", "feature_3"],
    )
    validator = DataQualityValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert "feature_3" in result.metrics["constant_columns"]
    assert result.metrics["duplicate_rows_count"] >= 15

    titles = [f.title for f in result.findings]
    assert any("Constant Feature" in t for t in titles)
    assert any("Duplicate" in t for t in titles)


@pytest.mark.asyncio
async def test_extreme_class_imbalance():
    # 98 instances of 0, 2 instances of 1
    y = [0] * 98 + [1] * 2
    df = pd.DataFrame({
        "feat_a": np.random.randn(100),
        "target": y,
    })
    context = ValidationContext(
        model=None,
        model_type="classification",
        dataset=df,
        target_column="target",
        feature_columns=["feat_a"],
    )
    validator = DataQualityValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.metrics["imbalance_level"] == "EXTREME"
    imb_findings = [f for f in result.findings if "Class Imbalance" in f.title]
    assert len(imb_findings) >= 1
    assert imb_findings[0].severity == "high"


@pytest.mark.asyncio
async def test_potential_target_leakage():
    np.random.seed(42)
    y = np.random.choice([0, 1], 100)
    # feature almost perfectly mirrors target
    leaky_feature = y * 1.0 + np.random.normal(0, 0.001, 100)
    df = pd.DataFrame({
        "clean_feat": np.random.randn(100),
        "leak_feat": leaky_feature,
        "target": y,
    })
    context = ValidationContext(
        model=None,
        dataset=df,
        target_column="target",
        feature_columns=["clean_feat", "leak_feat"],
    )
    validator = DataQualityValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert len(result.metrics["potential_leakage_columns"]) >= 1
    leak_findings = [f for f in result.findings if "Leakage" in f.title]
    assert len(leak_findings) >= 1
    assert leak_findings[0].severity in ("critical", "high")
    assert "leak_feat" in leak_findings[0].title


@pytest.mark.asyncio
async def test_infinite_values_detection(clean_dataset):
    df = clean_dataset.copy()
    df.loc[5, "feature_1"] = np.inf

    context = ValidationContext(
        model=None,
        dataset=df,
        target_column="target",
        feature_columns=["feature_1", "feature_2", "feature_3"],
    )
    validator = DataQualityValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert len(result.metrics["infinite_columns"]) >= 1
    inf_findings = [f for f in result.findings if "Infinite" in f.title]
    assert len(inf_findings) >= 1
    assert inf_findings[0].severity == "high"
