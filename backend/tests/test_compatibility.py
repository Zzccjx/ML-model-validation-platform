"""Tests for compatibility service."""
import pytest
from app.services.compatibility_service import check_compatibility

@pytest.mark.asyncio
async def test_compatible_model_dataset(sample_model_path, sample_dataset_path):
    features = [f"feature_{i}" for i in range(5)]
    result = await check_compatibility(
        model_path=sample_model_path,
        dataset_path=sample_dataset_path,
        target_column="target",
        feature_columns=features
    )
    assert result.is_compatible is True

@pytest.mark.asyncio
async def test_wrong_feature_count(sample_model_path, sample_dataset_path):
    features = ["feature_0", "feature_1", "feature_2"]
    result = await check_compatibility(
        model_path=sample_model_path,
        dataset_path=sample_dataset_path,
        target_column="target",
        feature_columns=features
    )
    assert result.is_compatible is False
    assert result.errors is not None

@pytest.mark.asyncio
async def test_missing_target_column(sample_model_path, sample_dataset_path):
    features = [f"feature_{i}" for i in range(5)]
    result = await check_compatibility(
        model_path=sample_model_path,
        dataset_path=sample_dataset_path,
        target_column="wrong_target",
        feature_columns=features
    )
    assert result.is_compatible is False
    assert result.errors is not None

@pytest.mark.asyncio
async def test_model_file_not_found(sample_dataset_path):
    features = [f"feature_{i}" for i in range(5)]
    result = await check_compatibility(
        model_path="nonexistent_model.pkl",
        dataset_path=sample_dataset_path,
        target_column="target",
        feature_columns=features
    )
    assert result.is_compatible is False

@pytest.mark.asyncio
async def test_dataset_file_not_found(sample_model_path):
    features = [f"feature_{i}" for i in range(5)]
    result = await check_compatibility(
        model_path=sample_model_path,
        dataset_path="nonexistent_data.csv",
        target_column="target",
        feature_columns=features
    )
    assert result.is_compatible is False
