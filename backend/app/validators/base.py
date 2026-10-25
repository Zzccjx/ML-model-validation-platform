"""Base validator interface for the AI/ML Validation Platform.

All validation modules must inherit from BaseValidator and implement
the validate() method. This enables a plugin-style architecture where
new validators can be added without modifying the core system.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import time
import enum


class ValidatorStatus(str, enum.Enum):
    """Status of a validator execution."""
    SUCCESS = 'success'
    PASSED = 'success'
    WARNING = 'warning'
    ERROR = 'error'
    SKIPPED = 'skipped'
    NOT_EVALUATED = 'not_evaluated'
    UNSUPPORTED = 'unsupported'


@dataclass
class ValidationFinding:
    """A structured finding from a validator."""
    severity: str  # critical, high, medium, low, info
    category: str
    title: str
    description: str
    metric_name: str | None = None
    metric_value: float | None = None
    threshold: str | None = None
    recommendation: str | None = None


@dataclass
class ValidatorResult:
    """Structured result from a validator execution."""
    validator_name: str
    validator_version: str = '0.1.0'
    status: ValidatorStatus = ValidatorStatus.SUCCESS
    score: float | None = None  # 0-100
    findings: list[ValidationFinding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    execution_time_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    category: str | None = None
    message: str | None = None
    explanation: str | None = None
    status_label: str = 'SUCCESS'


@dataclass
class ValidationContext:
    """Context passed to each validator containing model, data, and config."""
    model: Any  # The loaded ML model
    model_type: str = 'classification'  # 'classification' or 'regression'
    framework: str = 'scikit-learn'
    dataset: Any = None  # pandas DataFrame
    target_column: str = ''
    feature_columns: list[str] = field(default_factory=list)
    sensitive_columns: list[str] = field(default_factory=list)
    reference_dataset: Any | None = None  # For drift detection
    config: dict[str, Any] = field(default_factory=dict)
    features: Any = None
    targets: Any = None
    adapter: Any = None  # BaseModelAdapter instance


class BaseValidator(ABC):
    """Abstract base class for all validators.
    
    Every validation module must inherit from this class and implement
    the validate() method. Validators are registered with the
    ValidatorRegistry and can be discovered and executed dynamically.
    """
    
    name: str = 'base'
    version: str = '0.1.0'
    category: str = 'general'  # performance, data_quality, fairness, drift, robustness, explainability, reliability
    
    def is_applicable(self, context: ValidationContext) -> bool:
        """Check if this validator can run on the given context.
        
        Override this method to add preconditions. For example, a fairness
        validator might require sensitive_columns to be non-empty.
        """
        return True

    def get_inapplicability_details(self, context: ValidationContext) -> tuple[ValidatorStatus, str]:
        """Return (status, reason) when is_applicable() returns False."""
        return (
            ValidatorStatus.NOT_EVALUATED,
            f"Prerequisites missing: Validator '{self.name}' could not be evaluated on current context."
        )
    
    @abstractmethod
    def validate(self, context: ValidationContext) -> Any:
        """Execute the validation and return structured results."""
        ...
    
    async def safe_validate(self, context: ValidationContext) -> ValidatorResult:
        """Wrapper that catches unexpected exceptions gracefully and maintains module independence."""
        import inspect
        start_time = time.time()
        try:
            if not self.is_applicable(context):
                inapp_status, inapp_reason = self.get_inapplicability_details(context)
                err_msg = f"Validator {self.name} is not applicable to this context: {inapp_reason}"
                return ValidatorResult(
                    validator_name=self.name,
                    validator_version=self.version,
                    status=ValidatorStatus.SKIPPED,
                    score=None,
                    error_message=err_msg,
                    explanation=inapp_reason,
                    status_label=inapp_status.value.upper(),
                    category=self.category,
                )
            
            res = self.validate(context)
            if inspect.isawaitable(res):
                result = await res
            else:
                result = res

            result.execution_time_seconds = time.time() - start_time
            if not getattr(result, 'status_label', None):
                result.status_label = result.status.value.upper()
            return result
        except Exception as e:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.ERROR,
                status_label='ERROR',
                execution_time_seconds=time.time() - start_time,
                error_message=f'Validator {self.name} failed unexpectedly: {str(e)}',
                explanation=f'Unexpected runtime error: {str(e)}',
                category=self.category,
            )
