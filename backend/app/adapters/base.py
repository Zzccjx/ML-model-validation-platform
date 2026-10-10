"""Base model adapter defining interface for format-aware model inspection and safe execution."""
from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('ai_validation')


class BaseModelAdapter(ABC):
    """Abstract base adapter for loaded ML models."""

    def __init__(self, model_path: str, raw_model: Any = None):
        self.model_path = model_path
        self.raw_model = raw_model
        self.format_ext = Path(model_path).suffix.lower()

    @classmethod
    @abstractmethod
    def can_handle(cls, file_path: str) -> bool:
        """Return True if this adapter supports the given file format."""
        ...

    @abstractmethod
    def load(self) -> Any:
        """Load model from disk safely."""
        ...

    @abstractmethod
    def inspect(self) -> dict[str, Any]:
        """Return complete architectural metadata (shapes, parameters, framework, task type)."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Execute bounded sample prediction returning standard 1D/2D numpy array."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        """Return prediction probabilities if supported, else None."""
        ...

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Identify ML framework."""
        ...

    @property
    @abstractmethod
    def model_type(self) -> str:
        """classification, regression, or neural_network."""
        ...

    def __getattr__(self, name: str) -> Any:
        if self.raw_model is not None and hasattr(self.raw_model, name):
            return getattr(self.raw_model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
