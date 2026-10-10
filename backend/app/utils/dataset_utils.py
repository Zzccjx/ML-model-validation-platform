import logging
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger('ai_validation')

def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f'Dataset file not found: {file_path}')
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f'Failed to parse CSV file: {e}')
    
    if df.empty:
        raise ValueError('Dataset is empty')
    
    return df

def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Get a comprehensive statistical and quality summary of the complete dataset."""
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_cols = numeric_df.columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # Infinite values in numeric columns
    infinite_counts = {}
    for col in numeric_cols:
        inf_count = int(np.isinf(df[col]).sum())
        if inf_count > 0:
            infinite_counts[col] = inf_count

    # Constant / zero-variance columns
    constant_columns = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

    # Outlier counts via IQR method on numerical columns
    outlier_counts = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 10:
            q25, q75 = np.percentile(series, 25), np.percentile(series, 75)
            iqr = q75 - q25
            if iqr > 0:
                lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
                count = int(((series < lower) | (series > upper)).sum())
                if count > 0:
                    outlier_counts[col] = count

    # Class / label distribution for low-cardinality candidate columns
    class_distributions = {}
    for col in df.columns:
        n_unique = df[col].nunique(dropna=True)
        if 2 <= n_unique <= 10:
            top_vals = df[col].value_counts(dropna=False).head(5).to_dict()
            class_distributions[col] = {str(k): int(v) for k, v in top_vals.items()}

    return {
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': list(df.columns),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': {col: int(count) for col, count in df.isnull().sum().items() if count > 0},
        'missing_percentage': float(df.isnull().mean().mean() * 100),
        'duplicate_rows': int(df.duplicated().sum()),
        'infinite_counts': infinite_counts,
        'constant_columns': constant_columns,
        'outlier_counts': outlier_counts,
        'class_distributions': class_distributions,
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
    }


def validate_target_column(df: pd.DataFrame, target_column: str) -> dict:
    """Validate that a target column exists and has valid values."""
    result = {'valid': True, 'messages': []}
    
    if target_column not in df.columns:
        result['valid'] = False
        result['messages'].append(f'Target column "{target_column}" not found in dataset. Available columns: {list(df.columns)}')
        return result
    
    target = df[target_column]
    null_count = int(target.isnull().sum())
    if null_count > 0:
        result['messages'].append(f'Target column has {null_count} missing values ({null_count/len(df)*100:.1f}%)')
    
    unique_count = int(target.nunique())
    result['unique_values'] = unique_count
    result['dtype'] = str(target.dtype)
    
    if unique_count == 1:
        result['valid'] = False
        result['messages'].append('Target column has only one unique value')
    elif unique_count == len(df):
        result['messages'].append('Target column has all unique values - might be an ID column')
    
    return result


def get_dataset_preview(df: pd.DataFrame, max_rows: int = 15) -> dict:
    """Return top rows, dtypes, nulls, duplicates, and column categorization."""
    preview_df = df.head(max_rows).copy()
    # Replace NaN/inf with None for JSON serialization
    preview_df = preview_df.replace({np.nan: None, np.inf: None, -np.inf: None})
    records = preview_df.to_dict(orient='records')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    
    return {
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'columns': list(df.columns),
        'preview_rows': records,
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_counts': {col: int(count) for col, count in df.isnull().sum().items()},
        'duplicate_rows': int(df.duplicated().sum()),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
    }

