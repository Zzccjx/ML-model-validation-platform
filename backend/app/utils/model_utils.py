import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np

logger = logging.getLogger('ai_validation')

# SECURITY WARNING: pickle/joblib deserialization can execute arbitrary code.
# In production, model loading MUST happen in an isolated sandbox/container.
# This implementation is for local development only.

def load_model_safely(file_path: str) -> Any:
    """Load a model from disk.
    
    WARNING: Deserialization can execute arbitrary code depending on format.
    In production, this should run in an isolated execution sandbox/worker.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'Model file not found: {file_path}')
    
    ext = path.suffix.lower()
    logger.info('Loading model file %s (extension %s)', path.name, ext)
    
    if ext in ('.pkl', '.joblib'):
        return joblib.load(file_path)
    elif ext == '.pickle':
        import pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    elif ext in ('.keras', '.h5'):
        try:
            import keras
            return keras.models.load_model(file_path)
        except ImportError:
            try:
                import tensorflow as tf
                return tf.keras.models.load_model(file_path)
            except ImportError:
                raise RuntimeError(
                    'TensorFlow / Keras runtime is not installed in the environment to load .keras/.h5 models.'
                )
    elif ext == '.onnx':
        try:
            import onnxruntime as ort
            return ort.InferenceSession(file_path)
        except ImportError:
            raise RuntimeError(
                'ONNX Runtime (onnxruntime) is not installed in the environment to inspect/execute .onnx models.'
            )
    else:
        raise ValueError(f'Unsupported model file format: {ext}')

def detect_model_type(model: Any) -> str:
    """Detect whether model is a classifier or regressor."""
    from sklearn.base import is_classifier, is_regressor
    if is_classifier(model):
        return 'classification'
    elif is_regressor(model):
        return 'regression'
    
    # Check Keras / Neural Network output layer
    if hasattr(model, 'output_shape') or hasattr(model, 'layers'):
        try:
            # Check last layer activation
            last_layer = getattr(model, 'layers', [])[-1] if hasattr(model, 'layers') and model.layers else None
            activation = getattr(getattr(last_layer, 'activation', None), '__name__', '')
            if activation in ('sigmoid', 'softmax'):
                return 'classification'
            elif activation in ('linear', 'relu'):
                return 'regression'
        except Exception:
            pass
        return 'neural_network'
    
    return 'unknown'

def detect_framework(model: Any) -> str:
    """Detect the ML framework of the model."""
    module = type(model).__module__
    if module.startswith('sklearn'):
        return 'scikit-learn'
    elif module.startswith('xgboost'):
        return 'xgboost'
    elif module.startswith('lightgbm'):
        return 'lightgbm'
    elif module.startswith(('keras', 'tensorflow')):
        return 'keras / tensorflow'
    elif 'onnx' in module.lower():
        return 'onnx'
    return 'Python Estimator'

def get_model_metadata(model: Any) -> dict:
    """Extract metadata from a loaded model."""
    metadata = {
        'model_class': type(model).__name__,
        'module': type(model).__module__,
    }
    
    # Feature count & Keras shapes
    feature_count = None
    if hasattr(model, 'n_features_in_'):
        feature_count = int(model.n_features_in_)
    elif hasattr(model, 'input_shape') and model.input_shape:
        # e.g. (None, 10)
        shape = model.input_shape
        if isinstance(shape, tuple) and len(shape) > 1 and shape[-1] is not None:
            feature_count = int(shape[-1])
        metadata['input_shape'] = str(shape)
    elif hasattr(model, 'input_names'):
        metadata['input_names'] = list(model.input_names)
    metadata['feature_count'] = feature_count

    # Output shape & parameters (Keras / PyTorch / ONNX)
    if hasattr(model, 'output_shape'):
        metadata['output_shape'] = str(model.output_shape)
    if hasattr(model, 'count_params'):
        try:
            metadata['total_parameters'] = int(model.count_params())
        except Exception:
            pass
    if hasattr(model, 'layers'):
        metadata['layer_count'] = len(model.layers)
    
    # Classes for classifiers
    classes = None
    class_count = None
    if hasattr(model, 'classes_'):
        classes = [str(c) for c in model.classes_]
        class_count = len(classes)
    metadata['classes'] = classes
    metadata['class_count'] = class_count
    
    # Feature names if available
    if hasattr(model, 'feature_names_in_'):
        metadata['feature_names'] = list(model.feature_names_in_)
    
    # Probability support & prediction interface
    has_predict_proba = hasattr(model, 'predict_proba') and callable(getattr(model, 'predict_proba'))
    # In Keras classification models with sigmoid/softmax, predict() directly yields probabilities
    if not has_predict_proba and hasattr(model, 'output_shape') and hasattr(model, 'layers'):
        try:
            last_layer = getattr(model, 'layers', [])[-1]
            act = getattr(getattr(last_layer, 'activation', None), '__name__', '')
            if act in ('sigmoid', 'softmax'):
                has_predict_proba = True
        except Exception:
            pass

    metadata['has_predict_proba'] = has_predict_proba
    metadata['has_decision_function'] = hasattr(model, 'decision_function') and callable(getattr(model, 'decision_function'))
    metadata['has_predict'] = hasattr(model, 'predict') and callable(getattr(model, 'predict'))
    
    # Model params
    if hasattr(model, 'get_params'):
        try:
            params = model.get_params()
            clean_params = {}
            for k, v in params.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    clean_params[k] = v
                else:
                    clean_params[k] = str(v)
            metadata['params'] = clean_params
        except Exception:
            pass
    
    return metadata

