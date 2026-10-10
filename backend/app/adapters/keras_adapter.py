"""Keras / TensorFlow adapter for .keras and .h5 models."""
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd

from app.adapters.base import BaseModelAdapter


class KerasAdapter(BaseModelAdapter):
    """Adapter for Keras and TensorFlow models."""

    SUPPORTED_EXTENSIONS = {'.keras', '.h5'}

    @classmethod
    def can_handle(cls, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in cls.SUPPORTED_EXTENSIONS

    def load(self) -> Any:
        if self.raw_model is not None:
            return self.raw_model
        try:
            import keras
            self.raw_model = keras.models.load_model(self.model_path)
            return self.raw_model
        except Exception:
            try:
                import tensorflow as tf
                self.raw_model = tf.keras.models.load_model(self.model_path)
                return self.raw_model
            except Exception as e:
                raise RuntimeError(f"Failed to load Keras/TensorFlow model: {str(e)}")

    @property
    def framework_name(self) -> str:
        return 'keras / tensorflow'

    @property
    def model_type(self) -> str:
        m = self.load()
        if hasattr(m, 'output_shape') or hasattr(m, 'layers'):
            try:
                last_layer = getattr(m, 'layers', [])[-1] if getattr(m, 'layers', None) else None
                act = getattr(getattr(last_layer, 'activation', None), '__name__', '')
                if act in ('sigmoid', 'softmax'):
                    return 'classification'
                if act in ('linear', 'relu'):
                    return 'regression'
            except Exception:
                pass
        return 'neural_network'

    def inspect(self) -> dict[str, Any]:
        m = self.load()
        info: dict[str, Any] = {
            'framework': self.framework_name,
            'model_type': self.model_type,
            'model_class': type(m).__name__,
            'module': type(m).__module__,
        }
        
        # input shape and feature count
        feature_count = None
        if hasattr(m, 'input_shape') and m.input_shape:
            shape = m.input_shape
            info['input_shape'] = str(shape)
            if isinstance(shape, tuple) and len(shape) > 1 and shape[-1] is not None:
                feature_count = int(shape[-1])
        elif hasattr(m, 'inputs') and m.inputs:
            shape = tuple(m.inputs[0].shape)
            info['input_shape'] = str(shape)
            if len(shape) > 1 and shape[-1] is not None:
                feature_count = int(shape[-1])
        info['feature_count'] = feature_count

        if hasattr(m, 'output_shape'):
            info['output_shape'] = str(m.output_shape)
        if hasattr(m, 'count_params'):
            try:
                info['total_parameters'] = int(m.count_params())
            except Exception:
                pass
        if hasattr(m, 'layers'):
            info['layer_count'] = len(m.layers)

        # Detect classification output classes
        if hasattr(m, 'output_shape') and m.output_shape:
            out_shape = m.output_shape
            if isinstance(out_shape, tuple) and len(out_shape) > 1 and out_shape[-1] is not None:
                num_outs = int(out_shape[-1])
                if num_outs == 1:
                    info['classes'] = ['0', '1']
                    info['class_count'] = 2
                elif num_outs > 1:
                    info['classes'] = [str(i) for i in range(num_outs)]
                    info['class_count'] = num_outs

        info['has_predict'] = True
        info['has_predict_proba'] = (self.model_type == 'classification')
        return info

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        m = self.load()
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)

        raw_preds = m.predict(arr, verbose=0)
        raw_arr = np.asarray(raw_preds)

        # Check if classification
        m_type = self.model_type
        if m_type == 'classification':
            if raw_arr.ndim == 2:
                if raw_arr.shape[1] == 1:
                    # Binary sigmoid: threshold at 0.5
                    return (raw_arr.squeeze(-1) >= 0.5).astype(int)
                else:
                    # Multi-class softmax: argmax
                    return np.argmax(raw_arr, axis=1)
            return raw_arr.squeeze()
        elif m_type == 'regression':
            return raw_arr.squeeze()
        else:
            # Neural network generic
            if raw_arr.ndim == 2 and raw_arr.shape[1] == 1:
                return raw_arr.squeeze(-1)
            return raw_arr

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        m = self.load()
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(X, dtype=np.float32)

        raw_preds = m.predict(arr, verbose=0)
        raw_arr = np.asarray(raw_preds)

        if raw_arr.ndim == 2:
            if raw_arr.shape[1] == 1:
                # Binary probabilities: [1 - p, p]
                p = raw_arr.squeeze(-1)
                return np.column_stack([1.0 - p, p])
            return raw_arr
        elif raw_arr.ndim == 1:
            p = raw_arr
            return np.column_stack([1.0 - p, p])
        return raw_arr

    def __getattr__(self, name: str) -> Any:
        # Delegate attributes (e.g. classes_, input_shape, etc.) to raw underlying model if present
        if self.raw_model is not None and hasattr(self.raw_model, name):
            return getattr(self.raw_model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
