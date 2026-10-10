"""
Sample Data Generator for AI/ML Validation Platform

Generates synthetic datasets and trained models for development and testing.

Usage:
    python scripts/create_sample_data.py

Outputs saved to: scripts/sample_data/
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import joblib

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'sample_data')

def ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Output directory: {OUTPUT_DIR}')

def create_classification_data():
    """Create a binary classification dataset and trained models."""
    print('\n--- Classification Dataset ---')
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=7,
        n_redundant=2, n_classes=2, random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add group attributes for fairness testing
    np.random.seed(42)
    df['gender'] = np.random.choice(['male', 'female'], size=len(df))
    df['region'] = np.random.choice(['north', 'south', 'east', 'west'], size=len(df))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_names], df['target'], test_size=0.3, random_state=42
    )
    
    # Save test dataset (with group attributes)
    test_df = X_test.copy()
    test_df['target'] = y_test.values
    test_df['gender'] = df.loc[X_test.index, 'gender'].values
    test_df['region'] = df.loc[X_test.index, 'region'].values
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'classification_test.csv'), index=False)
    print(f'  Test dataset: {len(test_df)} rows, {len(test_df.columns)} columns')
    
    # Train LogisticRegression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(OUTPUT_DIR, 'logistic_regression.pkl'))
    print(f'  LogisticRegression accuracy: {lr.score(X_test, y_test):.4f}')
    
    # Train RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, os.path.join(OUTPUT_DIR, 'random_forest_classifier.pkl'))
    print(f'  RandomForestClassifier accuracy: {rf.score(X_test, y_test):.4f}')

def create_regression_data():
    """Create a regression dataset and trained model."""
    print('\n--- Regression Dataset ---')
    X, y = make_regression(
        n_samples=800, n_features=8, n_informative=5,
        noise=10.0, random_state=42
    )
    feature_names = [f'feat_{i}' for i in range(8)]
    df = pd.DataFrame(X, columns=feature_names)
    df['price'] = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_names], df['price'], test_size=0.3, random_state=42
    )
    
    test_df = X_test.copy()
    test_df['price'] = y_test.values
    test_df.to_csv(os.path.join(OUTPUT_DIR, 'regression_test.csv'), index=False)
    print(f'  Test dataset: {len(test_df)} rows')
    
    rfr = RandomForestRegressor(n_estimators=50, random_state=42)
    rfr.fit(X_train, y_train)
    joblib.dump(rfr, os.path.join(OUTPUT_DIR, 'random_forest_regressor.pkl'))
    print(f'  RandomForestRegressor R²: {rfr.score(X_test, y_test):.4f}')

def create_missing_values_data():
    """Create a dataset with deliberate missing values."""
    print('\n--- Missing Values Dataset ---')
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n).astype(float),
        'income': np.random.normal(50000, 15000, n),
        'score': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'target': np.random.choice([0, 1], n),
    })
    # Insert missing values
    mask = np.random.random(df.shape) < 0.08
    # Only apply to numeric columns
    for i, col in enumerate(['age', 'income', 'score']):
        df.loc[mask[:, i], col] = np.nan
    
    df.to_csv(os.path.join(OUTPUT_DIR, 'missing_values_test.csv'), index=False)
    print(f'  Missing values: {df.isnull().sum().sum()} total')

def create_imbalanced_data():
    """Create a dataset with severe class imbalance."""
    print('\n--- Imbalanced Dataset ---')
    X, y = make_classification(
        n_samples=1000, n_features=6, n_informative=4,
        n_classes=2, weights=[0.95, 0.05], random_state=42
    )
    feature_names = [f'f_{i}' for i in range(6)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df.to_csv(os.path.join(OUTPUT_DIR, 'imbalanced_test.csv'), index=False)
    print(f'  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}')

def create_drift_data():
    """Create reference and current datasets with deliberate distribution drift."""
    print('\n--- Drift Datasets ---')
    np.random.seed(42)
    n = 500
    
    # Reference (training) distribution
    ref = pd.DataFrame({
        'feature_a': np.random.normal(0, 1, n),
        'feature_b': np.random.normal(5, 2, n),
        'feature_c': np.random.choice(['cat', 'dog', 'bird'], n, p=[0.5, 0.3, 0.2]),
        'target': np.random.choice([0, 1], n),
    })
    ref.to_csv(os.path.join(OUTPUT_DIR, 'drift_reference.csv'), index=False)
    
    # Current (production) distribution — shifted
    cur = pd.DataFrame({
        'feature_a': np.random.normal(0.8, 1.5, n),  # Mean and std shifted
        'feature_b': np.random.normal(5, 2, n),      # No drift
        'feature_c': np.random.choice(['cat', 'dog', 'bird'], n, p=[0.2, 0.5, 0.3]),  # Category shift
        'target': np.random.choice([0, 1], n),
    })
    cur.to_csv(os.path.join(OUTPUT_DIR, 'drift_current.csv'), index=False)
    print(f'  Reference: {len(ref)} rows, Current: {len(cur)} rows')

def main():
    print('='*60)
    print('AI/ML Validation Platform — Sample Data Generator')
    print('='*60)
    
    ensure_dir()
    create_classification_data()
    create_regression_data()
    create_missing_values_data()
    create_imbalanced_data()
    create_drift_data()
    
    print('\n' + '='*60)
    print(f'All sample data saved to: {OUTPUT_DIR}')
    files = os.listdir(OUTPUT_DIR)
    print(f'Files created: {len(files)}')
    for f in sorted(files):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f'  {f}: {size/1024:.1f} KB')
    print('='*60)

if __name__ == '__main__':
    main()
