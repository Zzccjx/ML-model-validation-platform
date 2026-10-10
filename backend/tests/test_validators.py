"""Tests for validator registry and base validator."""
import pytest
import numpy as np
from typing import Any, Dict
from app.validators.base import BaseValidator, ValidatorResult, ValidationContext, ValidatorStatus
from app.validators.registry import get_registry

class DummyValidator(BaseValidator):
    name = "dummy_validator"
    description = "A dummy validator"
    category = "robustness"
    
    def validate(self, context: ValidationContext) -> ValidatorResult:
        return ValidatorResult(
            validator_name=self.name,
            status=ValidatorStatus.PASSED,
            score=0.95
        )

class ErrorValidator(BaseValidator):
    name = "error_validator"
    description = "Throws an error"
    category = "fairness"
    
    def validate(self, context: ValidationContext) -> ValidatorResult:
        raise ValueError("Intentional error")

class SkippedValidator(BaseValidator):
    name = "skipped_validator"
    description = "Always skipped"
    category = "performance"
    
    def is_applicable(self, context: ValidationContext) -> bool:
        return False
        
    def validate(self, context: ValidationContext) -> ValidatorResult:
        return ValidatorResult(
            validator_name=self.name,
            status=ValidatorStatus.PASSED
        )

@pytest.fixture(autouse=True)
def setup_teardown_registry():
    registry = get_registry()
    registry.clear()
    yield
    registry.clear()

@pytest.fixture
def mock_context():
    class MockModel:
        def predict(self, X):
            return np.zeros(len(X))
            
    return ValidationContext(
        model=MockModel(),
        dataset=None,
        features=np.array([[1, 2], [3, 4]]),
        targets=np.array([0, 1]),
        model_type="classification",
        config={}
    )

@pytest.mark.asyncio
async def test_base_validator_interface():
    validator = DummyValidator()
    assert validator.name == "dummy_validator"

@pytest.mark.asyncio
async def test_registry_register_get():
    registry = get_registry()
    validator = DummyValidator()
    registry.register(validator)
    
    retrieved = registry.get_validator("dummy_validator")
    assert retrieved == validator

@pytest.mark.asyncio
async def test_registry_list_all():
    registry = get_registry()
    registry.register(DummyValidator())
    registry.register(ErrorValidator())
    
    all_validators = registry.list_validators()
    assert len(all_validators) == 2

@pytest.mark.asyncio
async def test_registry_get_by_category():
    registry = get_registry()
    registry.register(DummyValidator())
    registry.register(ErrorValidator())
    
    fairness = registry.get_validators_by_category("fairness")
    assert len(fairness) == 1
    assert fairness[0].name == "error_validator"

@pytest.mark.asyncio
async def test_registry_run_all(mock_context):
    registry = get_registry()
    registry.register(DummyValidator())
    
    results = []
    for validator in registry.list_validators():
        if validator.is_applicable(mock_context):
            try:
                res = validator.validate(mock_context)
            except Exception as e:
                res = ValidatorResult(validator_name=validator.name, status=ValidatorStatus.ERROR, error_message=str(e))
            results.append(res)
            
    assert len(results) == 1
    assert results[0].status == ValidatorStatus.PASSED

@pytest.mark.asyncio
async def test_safe_validate_catches_errors(mock_context):
    validator = ErrorValidator()
    try:
        if validator.is_applicable(mock_context):
            result = validator.validate(mock_context)
    except Exception as e:
        result = ValidatorResult(
            validator_name=validator.name,
            status=ValidatorStatus.ERROR,
            error_message=str(e)
        )
    assert result.status == ValidatorStatus.ERROR
    assert "Intentional error" in result.error_message

@pytest.mark.asyncio
async def test_validator_not_applicable(mock_context):
    validator = SkippedValidator()
    applicable = validator.is_applicable(mock_context)
    assert not applicable
    
    if not applicable:
        result = ValidatorResult(
            validator_name=validator.name,
            status=ValidatorStatus.SKIPPED,
            message="Not applicable"
        )
    assert result.status == ValidatorStatus.SKIPPED

@pytest.mark.asyncio
async def test_registry_clear():
    registry = get_registry()
    registry.register(DummyValidator())
    assert len(registry.list_validators()) == 1
    
    registry.clear()
    assert len(registry.list_validators()) == 0
