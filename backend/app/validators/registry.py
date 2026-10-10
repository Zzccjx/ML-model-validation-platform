"""Validator Registry for dynamic validator management.

The registry allows validators to be registered, discovered, and executed
dynamically. This enables a plugin architecture where new validators
(including future AI code validators and autonomous agent validators)
can be added without modifying the core system.
"""
import logging
from typing import Any
from app.validators.base import BaseValidator, ValidatorResult, ValidationContext

logger = logging.getLogger('ai_validation')


class ValidatorRegistry:
    """Singleton registry for all validators."""
    
    _instance: 'ValidatorRegistry | None' = None
    _validators: dict[str, BaseValidator]
    
    def __new__(cls) -> 'ValidatorRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._validators = {}
        return cls._instance
    
    def register(self, validator: BaseValidator) -> None:
        """Register a validator instance."""
        if validator.name in self._validators:
            logger.warning('Validator %s is already registered. Overwriting.', validator.name)
        self._validators[validator.name] = validator
        logger.info('Registered validator: %s v%s [%s]', validator.name, validator.version, validator.category)
    
    def unregister(self, name: str) -> None:
        """Remove a validator from the registry."""
        if name in self._validators:
            del self._validators[name]
            logger.info('Unregistered validator: %s', name)
    
    def get(self, name: str) -> BaseValidator | None:
        """Get a validator by name."""
        return self._validators.get(name)
    
    get_validator = get

    def get_all(self) -> list[BaseValidator]:
        """Get all registered validators."""
        return list(self._validators.values())
    
    list_validators = get_all

    def get_by_category(self, category: str) -> list[BaseValidator]:
        """Get all validators in a given category."""
        return [v for v in self._validators.values() if v.category == category]
    
    get_validators_by_category = get_by_category
    
    def list_names(self) -> list[str]:
        """List all registered validator names."""
        return list(self._validators.keys())
    
    async def run_all(self, context: ValidationContext) -> list[ValidatorResult]:
        """Run all registered validators on the given context.
        
        Each validator runs via safe_validate() so individual failures
        don't crash the entire run.
        """
        results = []
        for validator in self._validators.values():
            logger.info('Running validator: %s v%s', validator.name, validator.version)
            result = await validator.safe_validate(context)
            results.append(result)
            logger.info(
                'Validator %s completed: status=%s score=%s time=%.2fs',
                validator.name, result.status.value, result.score, result.execution_time_seconds
            )
        return results
    
    async def run_by_category(self, category: str, context: ValidationContext) -> list[ValidatorResult]:
        """Run all validators in a given category."""
        validators = self.get_by_category(category)
        results = []
        for validator in validators:
            result = await validator.safe_validate(context)
            results.append(result)
        return results
    
    def clear(self) -> None:
        """Clear all registered validators. Mainly for testing."""
        self._validators.clear()


# Module-level convenience function
def get_registry() -> ValidatorRegistry:
    """Get the global validator registry instance."""
    return ValidatorRegistry()
