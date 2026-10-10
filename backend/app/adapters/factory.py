"""Model adapter factory."""
from pathlib import Path
from typing import Any

from app.adapters.base import BaseModelAdapter
from app.adapters.sklearn_adapter import SklearnAdapter
from app.adapters.keras_adapter import KerasAdapter
from app.adapters.onnx_adapter import OnnxAdapter

ADAPTER_CLASSES: list[type[BaseModelAdapter]] = [
    SklearnAdapter,
    KerasAdapter,
    OnnxAdapter,
]


def get_adapter(file_path: str, raw_model: Any = None) -> BaseModelAdapter:
    """Return appropriate adapter instance for the model file format."""
    path = Path(file_path)
    for adapter_cls in ADAPTER_CLASSES:
        if adapter_cls.can_handle(file_path):
            return adapter_cls(file_path, raw_model=raw_model)
    
    raise ValueError(
        f"Unsupported model file format: '{path.suffix}'. "
        f"Supported formats: .pkl, .joblib, .pickle, .keras, .h5, .onnx"
    )
