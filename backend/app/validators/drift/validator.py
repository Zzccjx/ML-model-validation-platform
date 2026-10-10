"""Data & Concept Drift Validator module.

Evaluates distribution shifts between baseline reference data and current evaluation data
using two-sample Kolmogorov-Smirnov (KS) tests, Population Stability Index (PSI), and
prediction / concept drift analysis.
"""
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

from app.validators.base import (
    BaseValidator,
    ValidatorResult,
    ValidationContext,
    ValidationFinding,
    ValidatorStatus,
)


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for a continuous feature."""
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0
    
    # Generate quantile bin edges from reference distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    try:
        bin_edges = np.percentile(ref_clean, percentiles)
    except Exception:
        return 0.0
    
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)
    
    ref_pct = (ref_counts + 1e-4) / (len(ref_clean) + 1e-4 * len(ref_counts))
    curr_pct = (curr_counts + 1e-4) / (len(curr_clean) + 1e-4 * len(curr_counts))
    
    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_val))


def calculate_categorical_psi(ref_series: pd.Series, curr_series: pd.Series) -> float:
    """Calculate Population Stability Index for a categorical feature."""
    all_categories = list(set(ref_series.dropna().unique()) | set(curr_series.dropna().unique()))
    if not all_categories:
        return 0.0
    
    ref_counts = ref_series.value_counts()
    curr_counts = curr_series.value_counts()
    
    n_ref = len(ref_series.dropna())
    n_curr = len(curr_series.dropna())
    if n_ref == 0 or n_curr == 0:
        return 0.0
    
    psi_total = 0.0
    k = len(all_categories)
    for cat in all_categories:
        r_pct = (ref_counts.get(cat, 0) + 1e-4) / (n_ref + 1e-4 * k)
        c_pct = (curr_counts.get(cat, 0) + 1e-4) / (n_curr + 1e-4 * k)
        psi_total += (c_pct - r_pct) * np.log(c_pct / r_pct)
        
    return float(max(0.0, psi_total))


class DriftValidator(BaseValidator):
    """Validator detecting feature distribution drift and concept drift."""

    name = "drift"
    version = "1.0.0"
    category = "drift"

    def is_applicable(self, context: ValidationContext) -> bool:
        """Applicable when dataset is provided with at least 10 records."""
        if context.dataset is None or len(context.dataset) < 10:
            return False
        return True

    def validate(self, context: ValidationContext) -> ValidatorResult:
        """Execute distribution and concept drift evaluation."""
        df: pd.DataFrame = context.dataset
        target_col = context.target_column
        findings: list[ValidationFinding] = []
        metrics: dict[str, Any] = {}
        recommendations: list[str] = []

        # Determine reference and current partitions
        if context.reference_dataset is not None and len(context.reference_dataset) > 0:
            ref_df: pd.DataFrame = context.reference_dataset
            curr_df: pd.DataFrame = df
            is_partitioned = False
        else:
            # Sequential / temporal 50-50 split
            half_idx = len(df) // 2
            ref_df = df.iloc[:half_idx].copy()
            curr_df = df.iloc[half_idx:].copy()
            is_partitioned = True
            findings.append(
                ValidationFinding(
                    severity="info",
                    category="drift",
                    title="Sequential Partition Applied for Drift Baseline",
                    description=(
                        f"No external reference dataset provided. Evaluated drift by partitioning dataset "
                        f"chronologically into baseline reference ({len(ref_df)} rows) and evaluation sample ({len(curr_df)} rows)."
                    ),
                    recommendation="Upload a dedicated reference dataset (e.g. historical training distribution) for production drift monitoring.",
                )
            )

        # Identify candidate feature columns
        if context.feature_columns:
            feat_cols = [c for c in context.feature_columns if c in df.columns and c in ref_df.columns]
        else:
            feat_cols = [c for c in df.columns if c != target_col and c in ref_df.columns]

        if not feat_cols:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SKIPPED,
                score=None,
                error_message="No matching feature columns found between reference and evaluation datasets.",
                category=self.category,
            )

        feature_drift_results: dict[str, Any] = {}
        drifted_count = 0
        max_psi = 0.0

        for col in feat_cols:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            if is_numeric:
                ref_vals = ref_df[col].dropna().values.astype(float)
                curr_vals = curr_df[col].dropna().values.astype(float)
                
                if len(ref_vals) < 5 or len(curr_vals) < 5:
                    continue
                
                # Two-sample KS test
                ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, curr_vals)
                psi_val = calculate_psi(ref_vals, curr_vals)
                
                # Status based on PSI and KS p-value
                if psi_val >= 0.25:
                    status = "significant"
                    drifted = True
                elif psi_val >= 0.10 or ks_pvalue < 0.01:
                    status = "moderate"
                    drifted = True
                else:
                    status = "stable"
                    drifted = False
                
                feature_drift_results[col] = {
                    "type": "numeric",
                    "ks_statistic": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(ks_pvalue), 5),
                    "psi": round(float(psi_val), 4),
                    "drift_detected": drifted,
                    "drift_status": status,
                    "reference_mean": round(float(np.mean(ref_vals)), 4),
                    "current_mean": round(float(np.mean(curr_vals)), 4),
                    "reference_std": round(float(np.std(ref_vals)), 4),
                    "current_std": round(float(np.std(curr_vals)), 4),
                }
            else:
                # Categorical column drift
                psi_val = calculate_categorical_psi(ref_df[col], curr_df[col])
                if psi_val >= 0.25:
                    status = "significant"
                    drifted = True
                elif psi_val >= 0.10:
                    status = "moderate"
                    drifted = True
                else:
                    status = "stable"
                    drifted = False
                
                feature_drift_results[col] = {
                    "type": "categorical",
                    "ks_statistic": None,
                    "ks_pvalue": None,
                    "psi": round(float(psi_val), 4),
                    "drift_detected": drifted,
                    "drift_status": status,
                }
            
            if psi_val > max_psi:
                max_psi = psi_val
                
            if drifted:
                drifted_count += 1
                sev = "critical" if psi_val >= 0.35 else ("high" if psi_val >= 0.25 else "medium")
                findings.append(
                    ValidationFinding(
                        severity=sev,
                        category="drift",
                        title=f"{status.capitalize()} Distribution Drift Detected in '{col}' (PSI={psi_val:.3f})",
                        description=(
                            f"Feature '{col}' exhibits {status} distribution shift. "
                            f"Population Stability Index is {psi_val:.3f} "
                            + (f"with KS p-value {feature_drift_results[col]['ks_pvalue']}." if is_numeric else ".")
                        ),
                        metric_name=f"PSI ({col})",
                        metric_value=round(psi_val, 4),
                        threshold="< 0.10 (Stable)",
                        recommendation="Evaluate upstream data pipeline transformations or recalibrate/retrain model on recent data distributions.",
                    )
                )

        drift_share = float(drifted_count / len(feat_cols)) if feat_cols else 0.0
        overall_drift_detected = drift_share >= 0.30 or max_psi >= 0.25

        # --- Concept / Prediction Drift ---
        concept_drift_info: dict[str, Any] = {}
        if context.model is not None:
            try:
                expected_names = getattr(context.model, "feature_names_in_", None)
                if expected_names is not None:
                    model_cols = [str(f) for f in expected_names if f in df.columns]
                else:
                    model_cols = [c for c in feat_cols if c in df.columns]
                
                X_ref = ref_df[model_cols].copy().fillna(0)
                X_curr = curr_df[model_cols].copy().fillna(0)
                
                y_pred_ref = context.model.predict(X_ref)
                y_pred_curr = context.model.predict(X_curr)
                
                # Check prediction distribution shift
                if context.model_type == "classification" or hasattr(context.model, "classes_"):
                    # Class frequency distribution shift
                    ref_pred_series = pd.Series(y_pred_ref)
                    curr_pred_series = pd.Series(y_pred_curr)
                    pred_psi = calculate_categorical_psi(ref_pred_series, curr_pred_series)
                    
                    concept_drift_detected = pred_psi >= 0.15
                    concept_drift_info = {
                        "prediction_psi": round(pred_psi, 4),
                        "concept_drift_detected": concept_drift_detected,
                        "reference_class_distribution": {str(k): round(float(v), 4) for k, v in ref_pred_series.value_counts(normalize=True).items()},
                        "current_class_distribution": {str(k): round(float(v), 4) for k, v in curr_pred_series.value_counts(normalize=True).items()},
                    }
                    if concept_drift_detected:
                        findings.append(
                            ValidationFinding(
                                severity="high",
                                category="drift",
                                title=f"Prediction Concept Drift Detected (Pred PSI={pred_psi:.3f})",
                                description=(
                                    f"Model prediction output distribution has shifted significantly between reference and current samples (PSI: {pred_psi:.3f})."
                                ),
                                metric_name="Prediction Output PSI",
                                metric_value=round(pred_psi, 4),
                                threshold="< 0.15",
                                recommendation="Investigate concept drift; model decision boundaries may no longer match real-world class distributions.",
                            )
                        )
                else:
                    # Continuous regression prediction shift
                    pred_psi = calculate_psi(np.array(y_pred_ref, dtype=float), np.array(y_pred_curr, dtype=float))
                    concept_drift_detected = pred_psi >= 0.15
                    concept_drift_info = {
                        "prediction_psi": round(pred_psi, 4),
                        "concept_drift_detected": concept_drift_detected,
                        "reference_pred_mean": round(float(np.mean(y_pred_ref)), 4),
                        "current_pred_mean": round(float(np.mean(y_pred_curr)), 4),
                    }
            except Exception as e:
                concept_drift_info = {"error": f"Could not compute prediction drift: {str(e)}"}

        # --- Aggregate Metrics ---
        metrics["reference_row_count"] = len(ref_df)
        metrics["current_row_count"] = len(curr_df)
        metrics["is_sequential_partition"] = is_partitioned
        metrics["total_features_evaluated"] = len(feat_cols)
        metrics["drifted_features_count"] = drifted_count
        metrics["drifted_features_percentage"] = round(drift_share * 100, 2)
        metrics["max_psi"] = round(float(max_psi), 4)
        metrics["overall_drift_detected"] = overall_drift_detected
        metrics["overall_drift_status"] = "significant" if max_psi >= 0.25 or drift_share >= 0.50 else ("moderate" if overall_drift_detected else "stable")
        metrics["feature_drift"] = feature_drift_results
        metrics["concept_drift"] = concept_drift_info

        # --- Overall Category Score (0–100) ---
        score = 100.0
        # Penalize for drifted features proportion (up to 40 pts)
        score -= min(40.0, drift_share * 60.0)
        # Penalize for max PSI severity (up to 30 pts)
        if max_psi >= 0.35:
            score -= 30.0
        elif max_psi >= 0.25:
            score -= 20.0
        elif max_psi >= 0.10:
            score -= 10.0
        # Penalize for concept drift if detected (up to 20 pts)
        if concept_drift_info.get("concept_drift_detected", False):
            score -= 15.0

        final_score = max(0.0, min(100.0, round(score, 1)))

        return ValidatorResult(
            validator_name=self.name,
            validator_version=self.version,
            status=ValidatorStatus.SUCCESS,
            score=final_score,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            category=self.category,
        )