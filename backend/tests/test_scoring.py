"""Tests for scoring service."""
import pytest
from app.services.scoring_service import calculate_scores
from app.validators.base import ValidatorResult, ValidatorStatus

@pytest.mark.asyncio
async def test_calculate_scores_basic():
    results = [
        ValidatorResult(validator_name="val1", category="fairness", status=ValidatorStatus.PASSED, score=0.8),
        ValidatorResult(validator_name="val2", category="robustness", status=ValidatorStatus.PASSED, score=0.9),
    ]
    scores = calculate_scores(results, weights=None)
    
    assert "overall" in scores
    assert "categories" in scores
    assert abs(scores["overall"] - 0.85) < 1e-6
    assert scores["categories"]["fairness"] == 0.8
    assert scores["categories"]["robustness"] == 0.9

@pytest.mark.asyncio
async def test_calculate_scores_empty():
    results = []
    scores = calculate_scores(results, weights=None)
    assert scores["overall"] is None
    assert len(scores["categories"]) == 0

@pytest.mark.asyncio
async def test_calculate_scores_with_skipped():
    results = [
        ValidatorResult(validator_name="val1", category="fairness", status=ValidatorStatus.PASSED, score=0.8),
        ValidatorResult(validator_name="val2", category="robustness", status=ValidatorStatus.SKIPPED, score=0.0),
        ValidatorResult(validator_name="val3", category="robustness", status=ValidatorStatus.ERROR, score=0.0),
    ]
    scores = calculate_scores(results, weights=None)
    
    assert abs(scores["overall"] - 0.8) < 1e-6
    assert scores["categories"]["fairness"] == 0.8
    assert scores["categories"].get("robustness") is None

@pytest.mark.asyncio
async def test_calculate_scores_custom_weights():
    results = [
        ValidatorResult(validator_name="val1", category="fairness", status=ValidatorStatus.PASSED, score=1.0),
        ValidatorResult(validator_name="val2", category="robustness", status=ValidatorStatus.PASSED, score=0.5),
    ]
    weights = {"fairness": 0.8, "robustness": 0.2}
    scores = calculate_scores(results, weights=weights)
    
    assert abs(scores["overall"] - 0.9) < 1e-6
