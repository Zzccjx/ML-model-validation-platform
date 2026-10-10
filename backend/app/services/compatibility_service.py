"""Model-dataset compatibility checking service."""
import logging
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from app.adapters.factory import get_adapter
from app.utils.dataset_utils import load_dataset

logger = logging.getLogger('ai_validation')


@dataclass
class CompatibilityResult:
    """Result of a model-dataset compatibility check."""
    compatible: bool = True
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_info: dict[str, Any] = field(default_factory=dict)
    dataset_info: dict[str, Any] = field(default_factory=dict)

    @property
    def is_compatible(self) -> bool:
        return self.compatible

    @is_compatible.setter
    def is_compatible(self, val: bool) -> None:
        self.compatible = val

    @property
    def errors(self) -> list[str]:
        return self.messages

    def __await__(self):
        async def _identity():
            return self
        return _identity().__await__()


def check_compatibility(
    model_path: str,
    dataset_path: str,
    target_column: str,
    feature_columns: list[str] | None = None,
) -> CompatibilityResult:
    """Check if a model and dataset are compatible for validation.
    
    Verifies:
    1. Model file can be loaded via format-aware adapter
    2. Dataset file can be parsed
    3. Target column exists and is valid (if provided)
    4. Feature count and column alignment match
    5. Prediction interface works on a sample
    """
    result = CompatibilityResult()
    
    # 1. Load model via format adapter
    try:
        adapter = get_adapter(model_path)
        model = adapter.load()
        insp = adapter.inspect()
        result.model_info['class'] = insp.get('model_class', type(model).__name__)
        result.model_info['framework'] = adapter.framework_name
        result.model_info['type'] = adapter.model_type
        if 'input_shape' in insp:
            result.model_info['input_shape'] = insp['input_shape']
        if 'output_shape' in insp:
            result.model_info['output_shape'] = insp['output_shape']
        if 'total_parameters' in insp:
            result.model_info['total_parameters'] = insp['total_parameters']
    except FileNotFoundError:
        result.compatible = False
        result.messages.append(f'Model file not found: {model_path}')
        return result
    except Exception as e:
        result.compatible = False
        result.messages.append(f'Failed to load model: {str(e)}')
        return result
    
    # 2. Load dataset
    try:
        df = load_dataset(dataset_path)
        result.dataset_info['rows'] = len(df)
        result.dataset_info['columns'] = list(df.columns)
    except FileNotFoundError:
        result.compatible = False
        result.messages.append(f'Dataset file not found: {dataset_path}')
        return result
    except Exception as e:
        result.compatible = False
        result.messages.append(f'Failed to load dataset: {str(e)}')
        return result
    
    # 3. Target detection / check
    target_series = None
    if not target_column:
        expected_names = getattr(model, 'feature_names_in_', None)
        if expected_names is not None:
            candidates = [c for c in df.columns if c not in set(expected_names)]
            if len(candidates) == 1:
                target_column = candidates[0]
                result.dataset_info['auto_detected_target'] = target_column
        elif 'target' in df.columns:
            target_column = 'target'
        elif 'label' in df.columns:
            target_column = 'label'
        elif 'y' in df.columns:
            target_column = 'y'

    if target_column:
        if target_column in df.columns:
            target_series = df[target_column]
            if target_series.isna().all():
                result.compatible = False
                result.messages.append(f'Target column "{target_column}" contains only NaN values.')
                return result
            result.dataset_info['target_column'] = target_column
        else:
            result.compatible = False
            result.messages.append(f'Target column "{target_column}" not found in dataset.')
            return result
    else:
        result.warnings.append(
            'No ground-truth target column specified or detected. Performance validator will report "Not Evaluated", but full unsupervised, quality, robustness, and explainability validation will run.'
        )

    # 4. Determine feature columns & alignment
    expected_names = getattr(model, 'feature_names_in_', None)
    if expected_names is not None:
        expected_list = [str(f) for f in expected_names]
        missing_features = [f for f in expected_list if f not in df.columns]
        if missing_features:
            result.compatible = False
            result.messages.append(
                f'Model requires specific feature names that are missing from dataset: {missing_features}'
            )
            return result
        # Re-order features exactly as expected by the model
        X = df[expected_list]
        feature_columns = expected_list
        result.dataset_info['aligned_with_feature_names_in'] = True
    elif feature_columns:
        # Exclude target column from features if present
        actual_features = [c for c in feature_columns if c != target_column] if target_column else feature_columns
        missing = [c for c in actual_features if c not in df.columns]
        if missing:
            result.compatible = False
            result.messages.append(f'Missing feature columns: {missing}')
            return result
        X = df[actual_features]
        feature_columns = actual_features
    else:
        # Use all columns except target
        cols_to_drop = [target_column] if target_column and target_column in df.columns else []
        X = df.drop(columns=cols_to_drop)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric = [c for c in X.columns if c not in numeric_cols]
        if non_numeric:
            result.warnings.append(
                f'Non-numeric columns excluded from features: {non_numeric}'
            )
            X = X[numeric_cols]
        feature_columns = list(X.columns)
    
    result.dataset_info['feature_columns'] = feature_columns
    result.dataset_info['feature_count'] = len(feature_columns)
    
    # 5. Check feature count
    expected_feat = None
    if hasattr(model, 'n_features_in_'):
        expected_feat = int(model.n_features_in_)
    elif insp.get('feature_count') is not None:
        expected_feat = int(insp['feature_count'])
    
    if expected_feat is not None:
        actual = len(feature_columns)
        result.model_info['expected_features'] = expected_feat
        if expected_feat != actual:
            result.compatible = False
            result.messages.append(
                f'Feature count mismatch: model expects {expected_feat} features, '
                f'dataset has {actual} feature columns'
            )
            return result
    
    # 6. Target problem-type compatibility
    if target_series is not None:
        m_type = adapter.model_type
        if m_type == 'classification':
            if hasattr(model, 'classes_'):
                model_classes = set(model.classes_)
                target_unique = set(target_series.dropna().unique())
                unseen = target_unique - model_classes
                if unseen:
                    result.warnings.append(
                        f'Target contains labels not present in model classes_: {list(unseen)}'
                    )
        elif m_type == 'regression':
            if not pd.api.types.is_numeric_dtype(target_series.dtype):
                result.compatible = False
                result.messages.append(
                    f'Target column "{target_column}" has non-numeric dtype ({target_series.dtype}), '
                    f'incompatible with regression model {result.model_info.get("class")}.'
                )
                return result

    # 7. Check NaN tolerances
    nan_count = int(X.isna().sum().sum())
    if nan_count > 0:
        result.dataset_info['nan_feature_count'] = nan_count
        result.warnings.append(
            f'Dataset contains {nan_count} missing (NaN) values across feature columns.'
        )

    # 8. Test prediction on a sample
    try:
        sample = X.head(min(5, len(X)))
        try:
            preds = adapter.predict(sample)
        except Exception:
            # If native prediction fails due to NaNs, test with zero-imputation
            preds = adapter.predict(sample.fillna(0))
            result.warnings.append('Model required missing value imputation (zeros) to produce predictions on sample.')

        result.model_info['prediction_works'] = True
        flat_preds = np.asarray(preds).squeeze()
        if flat_preds.ndim == 0:
            flat_preds = np.array([flat_preds.item()])
        result.model_info['sample_predictions'] = [
            float(p) if isinstance(p, (np.floating, float)) else (int(p) if isinstance(p, (np.integer, int)) else str(p))
            for p in flat_preds[:3]
        ]
        
        # Check probability output capability
        proba_res = None
        try:
            proba_res = adapter.predict_proba(sample.fillna(0))
        except Exception:
            pass
        result.model_info['supports_probabilities'] = (proba_res is not None)
        
    except Exception as e:
        result.compatible = False
        result.messages.append(f'Model prediction failed on sample data: {str(e)}')
        return result
    
    if result.compatible:
        result.messages.append('Model and dataset are compatible for validation.')
    
    return result
