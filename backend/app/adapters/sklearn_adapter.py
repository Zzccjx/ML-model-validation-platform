"""Scikit-learn adapter for .pkl, .joblib, .pickle models."""
from typing import Any
from pathlib import Path
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor

from app.adapters.base import BaseModelAdapter


class SklearnAdapter(BaseModelAdapter):
    """Adapter for Scikit-learn estimators and pipelines."""

    SUPPORTED_EXTENSIONS = {'.pkl', '.joblib', '.pickle'}

    @classmethod
    def can_handle(cls, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in cls.SUPPORTED_EXTENSIONS

    def load(self) -> Any:
        if self.raw_model is not None:
            return self.raw_model
        ext = self.format_ext
        if ext in ('.pkl', '.joblib'):
            self.raw_model = joblib.load(self.model_path)
        elif ext == '.pickle':
            with open(self.model_path, 'rb') as f:
                self.raw_model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported sklearn extension: {ext}")
        return self.raw_model

    @property
    def framework_name(self) -> str:
        m = self.load()
        mod = type(m).__module__
        if mod.startswith('xgboost'):
            return 'xgboost'
        if mod.startswith('lightgbm'):
            return 'lightgbm'
        return 'scikit-learn'

    @property
    def model_type(self) -> str:
        m = self.load()
        if is_classifier(m):
            return 'classification'
        if is_regressor(m):
            return 'regression'
        return 'unknown'

    def inspect(self) -> dict[str, Any]:
        m = self.load()
        info: dict[str, Any] = {
            'framework': self.framework_name,
            'model_type': self.model_type,
            'model_class': type(m).__name__,
            'module': type(m).__module__,
        }
        if hasattr(m, 'n_features_in_'):
            info['feature_count'] = int(m.n_features_in_)
        if hasattr(m, 'feature_names_in_'):
            info['feature_names'] = [str(f) for f in m.feature_names_in_]
        if hasattr(m, 'classes_'):
            info['classes'] = [str(c) for c in m.classes_]
            info['class_count'] = len(info['classes'])
        info['has_predict'] = hasattr(m, 'predict') and callable(getattr(m, 'predict'))
        info['has_predict_proba'] = hasattr(m, 'predict_proba') and callable(getattr(m, 'predict_proba'))
        info['has_decision_function'] = hasattr(m, 'decision_function') and callable(getattr(m, 'decision_function'))
        return info

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        m = self.load()
        preds = m.predict(X)
        return np.asarray(preds)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray | None:
        m = self.load()
        if hasattr(m, 'predict_proba') and callable(getattr(m, 'predict_proba')):
            return np.asarray(m.predict_proba(X))
        if hasattr(m, 'decision_function') and callable(getattr(m, 'decision_function')):
            # fallback decision function converted to 2D
            df = m.decision_function(X)
            return np.asarray(df)
        return None
