"""ONNX adapter for .onnx models with graceful capability detection."""
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd

from app.adapters.base import BaseModelAdapter


class OnnxAdapter(BaseModelAdapter):
    """Adapter for ONNX models using onnxruntime when available."""

    SUPPORTED_EXTENSIONS = {'.onnx'}

    @classmethod
    def can_handle(cls, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in cls.SUPPORTED_EXTENSIONS

    def load(self) -> Any:
        if self.raw_model is not None:
            return self.raw_model
        try:
            import onnxruntime as ort
            self.raw_model = ort.InferenceSession(self.model_path)
            return self.raw_model
        except ImportError:
            raise RuntimeError(
                "ONNX Runtime ('onnxruntime') is not installed in the Python environment. "
                "Please install onnxruntime to execute or inspect .onnx models."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

    @property
    def framework_name(self) -> str:
        return 'onnx'

    @property
    def model_type(self) -> str:
        return 'neural_network'

    def inspect(self) -> dict[str, Any]:
        sess = self.load()
        inputs = sess.get_inputs()
        outputs = sess.get_outputs()
        
        feature_count = None
        input_shape = None
        if inputs:
            shape = inputs[0].shape
            input_shape = str(shape)
            if len(shape) > 1 and isinstance(shape[-1], int):
                feature_count = shape[-1]

        return {
            'framework': self.framework_name,
            'model_type': self.model_type,
            'model_class': 'InferenceSession',
            'module': 'onnxruntime',
            'input_names': [i.name for i in inputs],
            'output_names': [o.name for o in outputs],
            'input_shape': input_shape,
            'output_shape': str(outputs[0].shape) if outputs else None,
            'feature_count': feature_count,
            'has_predict': True,
            'has_predict_proba': False,
        }

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        sess = self.load()
        inputs = sess.get_inputs()
        if not inputs:
            raise RuntimeError("ONNX model has no input nodes.")
        input_name = inputs[0].name
        
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)

        outputs = sess.run(None, {input_name: arr})
        raw = np.asarray(outputs[0])
        return raw.squeeze() if raw.ndim == 2 and raw.shape[1] == 1 else raw

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        try:
            sess = self.load()
            outputs = sess.get_outputs()
            # If multiple outputs, check if second is probabilities
            if len(outputs) > 1 and 'proba' in outputs[1].name.lower():
                inputs = sess.get_inputs()
                if isinstance(X, pd.DataFrame):
                    arr = X.to_numpy(dtype=np.float32)
                else:
                    arr = np.asarray(X, dtype=np.float32)
                outs = sess.run([outputs[1].name], {inputs[0].name: arr})
                return np.asarray(outs[0])
        except Exception:
            pass
        return None
