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


def classify_drift_status(
    psi: float,
    threshold_stable: float = 0.10,
    threshold_moderate: float = 0.25,
) -> tuple[str, bool, str]:
    """Classify drift status consistently based on configurable standard thresholds.
    
    Standard convention:
    - PSI < 0.10: LOW / STABLE
    - 0.10 <= PSI < 0.25: MODERATE
    - PSI >= 0.25: HIGH / SIGNIFICANT DRIFT
    
    Returns:
        (status_name, is_drifted, severity)
        status_name: 'stable' | 'moderate' | 'significant'
        is_drifted: True if status is 'moderate' or 'significant'
        severity: 'info' | 'medium' | 'high'
    """
    if psi is None or np.isnan(psi):
        return "stable", False, "info"
    
    # Strictly follow standard screening conventions
    if psi >= threshold_moderate:
        return "significant", True, "high"
    elif psi >= threshold_stable:
        return "moderate", True, "medium"
    else:
        return "stable", False, "info"


def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for a continuous feature with robust quantile binning.
    
    - Handles NaN and infinite values cleanly.
    - Handles constant features without crashing.
    - Uses identical bin edges derived from reference distribution.
    - Uses a documented Laplace smoothing constant (1e-4) to avoid log(0) or division by zero.
    """
    # Clean non-finite and missing values
    ref_clean = reference[np.isfinite(reference)]
    curr_clean = current[np.isfinite(current)]
    
    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0
    
    # Constant or identical feature check
    if np.all(ref_clean == ref_clean[0]) and np.all(curr_clean == curr_clean[0]):
        return 0.0 if ref_clean[0] == curr_clean[0] else 1.0
    
    # Quantile binning on reference
    # Use between 2 and num_bins depending on available sample size
    effective_bins = min(num_bins, max(2, min(len(ref_clean), len(curr_clean)) // 5))
    percentiles = np.linspace(0, 100, effective_bins + 1)
    try:
        bin_edges = np.percentile(ref_clean, percentiles)
    except Exception:
        return 0.0
    
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        # Reference is constant or lacks variance; create two bins around the value
        val = ref_clean[0]
        bin_edges = np.array([-np.inf, val, np.inf])
    else:
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)
    
    k = len(ref_counts)
    smooth = 1e-4
    ref_pct = (ref_counts + smooth) / (len(ref_clean) + smooth * k)
    curr_pct = (curr_counts + smooth) / (len(curr_clean) + smooth * k)
    
    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_val))


def calculate_categorical_psi(ref_series: pd.Series, curr_series: pd.Series) -> tuple[float, float, float | None]:
    """Calculate Population Stability Index, Total Variation Distance (TVD), and Chi-squared p-value for categorical features.
    
    - Aligns category distributions across both datasets.
    - Handles categories present in only one dataset without NaN/inf.
    - Applies consistent smoothing (1e-4).
    """
    r_clean = ref_series.dropna().astype(str)
    c_clean = curr_series.dropna().astype(str)
    all_categories = sorted(list(set(r_clean.unique()) | set(c_clean.unique())))
    if not all_categories or len(r_clean) == 0 or len(c_clean) == 0:
        return 0.0, 0.0, None
    
    ref_counts = r_clean.value_counts()
    curr_counts = c_clean.value_counts()
    
    n_ref = len(r_clean)
    n_curr = len(c_clean)
    k = len(all_categories)
    smooth = 1e-4
    
    psi_total = 0.0
    tvd_total = 0.0
    
    ref_obs = []
    curr_obs = []
    for cat in all_categories:
        rc = ref_counts.get(cat, 0)
        cc = curr_counts.get(cat, 0)
        ref_obs.append(rc)
        curr_obs.append(cc)
        
        r_pct = (rc + smooth) / (n_ref + smooth * k)
        c_pct = (cc + smooth) / (n_curr + smooth * k)
        psi_total += (c_pct - r_pct) * np.log(c_pct / r_pct)
        tvd_total += 0.5 * abs(rc / n_ref - cc / n_curr)
        
    chi2_p = None
    if n_ref >= 10 and n_curr >= 10 and k >= 2:
        try:
            contingency = np.array([ref_obs, curr_obs])
            if contingency.sum() > 0:
                res = stats.chi2_contingency(contingency)
                chi2_p = float(res.pvalue)
        except Exception:
            chi2_p = None
            
    return float(max(0.0, psi_total)), float(tvd_total), chi2_p


class DriftValidator(BaseValidator):
    """Validator detecting feature distribution drift and concept drift."""

    name = "drift"
    version = "1.0.0"
    category = "drift"

    def is_applicable(self, context: ValidationContext) -> bool:
        """Applicable when evaluation dataset has at least 10 samples."""
        if context.dataset is None or len(context.dataset) < 10:
            return False
        return True

    def get_inapplicability_details(self, context: ValidationContext) -> tuple[ValidatorStatus, str]:
        """Return clear status and explanation when drift cannot be evaluated."""
        return (
            ValidatorStatus.NOT_EVALUATED,
            "Insufficient sample size: Drift evaluation requires at least 10 evaluation samples."
        )

    def validate(self, context: ValidationContext) -> ValidatorResult:
        """Execute distribution and concept drift evaluation."""
        if context.dataset is None or len(context.dataset) < 10:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.NOT_EVALUATED,
                status_label='NOT_EVALUATED',
                score=None,
                error_message="Insufficient sample size: Drift evaluation requires at least 10 evaluation samples.",
                explanation="Insufficient sample size: Drift evaluation requires at least 10 evaluation samples.",
                category=self.category,
            )

        df: pd.DataFrame = context.dataset
        target_col = context.target_column
        findings: list[ValidationFinding] = []
        metrics: dict[str, Any] = {}
        recommendations: list[str] = []

        is_partitioned = False
        is_small_sample = False

        if context.reference_dataset is not None and len(context.reference_dataset) > 0:
            ref_df: pd.DataFrame = context.reference_dataset
            curr_df: pd.DataFrame = df
            status_label = "EVALUATED"
            if len(ref_df) < 30 or len(curr_df) < 30:
                is_small_sample = True
                findings.append(
                    ValidationFinding(
                        severity="info",
                        category="drift",
                        title="Small Sample Size Warning (N < 30 per partition)",
                        description=(
                            f"Sample size is limited (Reference: {len(ref_df)}, Evaluation: {len(curr_df)}). "
                            "Statistical power is low; two-sample KS and PSI estimates have high variance and must be interpreted cautiously."
                        ),
                        metric_name="Sample Size",
                        metric_value=float(min(len(ref_df), len(curr_df))),
                        threshold=">= 30 per partition",
                        recommendation="Collect at least 50-100 baseline and inference observations for production-grade drift significance.",
                    )
                )
        else:
            # Exploratory internal split when no separate reference/baseline dataset is provided
            is_partitioned = True
            half = len(df) // 2
            ref_df = df.iloc[:half].copy().reset_index(drop=True)
            curr_df = df.iloc[half:].copy().reset_index(drop=True)
            status_label = "INTERNAL_SPLIT_DIAGNOSTIC"
            if half < 30:
                is_small_sample = True
            findings.append(
                ValidationFinding(
                    severity="info",
                    category="drift",
                    title="Sequential Partition Applied: Internal Split Diagnostic (No External Baseline)",
                    description=(
                        "INTERNAL_SPLIT_DIAGNOSTIC: No external reference dataset was provided. "
                        f"Evaluation dataset was partitioned sequentially (First {half} vs Last {len(df) - half} rows) "
                        "for intra-dataset stability diagnosis only. Sequential or arbitrary partitions cannot establish real-world production drift."
                    ),
                    metric_name="Partition Split",
                    metric_value=50.0,
                    threshold="N/A",
                    recommendation="Provide an explicit external baseline dataset representing production or training distributions.",
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
                status_label="SKIPPED",
                score=None,
                error_message="No matching feature columns found between reference and evaluation datasets.",
                explanation="No matching feature columns found between reference and evaluation datasets.",
                category=self.category,
            )

        feature_drift_results: dict[str, Any] = {}
        drifted_count = 0
        valid_feature_psis: list[float] = []

        for col in feat_cols:
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            if is_numeric:
                ref_vals = ref_df[col].dropna().values.astype(float)
                curr_vals = curr_df[col].dropna().values.astype(float)
                
                if len(ref_vals) < 2 or len(curr_vals) < 2:
                    continue
                
                # Two-sample KS test (effect size and statistical significance)
                try:
                    ks_stat, ks_pvalue = stats.ks_2samp(ref_vals, curr_vals)
                except Exception:
                    ks_stat, ks_pvalue = 0.0, 1.0
                    
                psi_val = calculate_psi(ref_vals, curr_vals)
                valid_feature_psis.append(psi_val)
                
                # Shared uniform classification rule
                status, drifted, sev = classify_drift_status(psi_val)
                
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
                psi_val, tvd_val, chi2_p = calculate_categorical_psi(ref_df[col], curr_df[col])
                valid_feature_psis.append(psi_val)
                
                # Shared uniform classification rule
                status, drifted, sev = classify_drift_status(psi_val)
                
                feature_drift_results[col] = {
                    "type": "categorical",
                    "ks_statistic": None,
                    "ks_pvalue": None,
                    "psi": round(float(psi_val), 4),
                    "total_variation_distance": round(float(tvd_val), 4),
                    "chi2_pvalue": round(float(chi2_p), 5) if chi2_p is not None else None,
                    "drift_detected": drifted,
                    "drift_status": status,
                }
            
            if drifted:
                drifted_count += 1
                if is_partitioned or is_small_sample:
                    finding_sev = "medium" if status == "significant" else "low"
                    title_prefix = f"Intra-Split {status.capitalize()} Divergence"
                else:
                    finding_sev = sev
                    title_prefix = f"{status.capitalize()} Distribution Drift Detected"

                findings.append(
                    ValidationFinding(
                        severity=finding_sev,
                        category="drift",
                        title=f"{title_prefix} in '{col}' (PSI={psi_val:.3f})",
                        description=(
                            f"Feature '{col}' exhibits {status} distribution divergence "
                            f"(PSI={psi_val:.3f}"
                            + (f", KS p={feature_drift_results[col]['ks_pvalue']}" if is_numeric else f", TVD={feature_drift_results[col]['total_variation_distance']}")
                            + f"). "
                            + ("Note: Intra-dataset diagnostic split; does not prove production drift." if is_partitioned else "")
                        ),
                        metric_name=f"PSI ({col})",
                        metric_value=round(psi_val, 4),
                        threshold="< 0.10 (Stable)",
                        recommendation="Evaluate data pipeline transformations and monitor distribution across production batches.",
                    )
                )

        # Max PSI from valid feature PSIs only
        max_psi = max(valid_feature_psis) if valid_feature_psis else 0.0
        drift_share = float(drifted_count / len(feat_cols)) if feat_cols else 0.0
        
        # Overall status derived from shared threshold rules
        if max_psi >= 0.25:
            overall_drift_status = "significant"
            overall_drift_detected = True
        elif max_psi >= 0.10:
            overall_drift_status = "moderate"
            overall_drift_detected = True
        else:
            overall_drift_status = "stable"
            overall_drift_detected = False

        # --- Concept / Prediction Drift ---
        concept_drift_info: dict[str, Any] = {}
        if context.model is not None and not is_small_sample:
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
                    pred_psi, pred_tvd, _ = calculate_categorical_psi(ref_pred_series, curr_pred_series)
                    
                    concept_drift_detected = pred_psi >= 0.10 and not is_partitioned
                    concept_drift_info = {
                        "prediction_psi": round(pred_psi, 4),
                        "prediction_tvd": round(pred_tvd, 4),
                        "concept_drift_detected": concept_drift_detected,
                        "reference_class_distribution": {str(k): round(float(v), 4) for k, v in ref_pred_series.value_counts(normalize=True).items()},
                        "current_class_distribution": {str(k): round(float(v), 4) for k, v in curr_pred_series.value_counts(normalize=True).items()},
                    }
                    if concept_drift_detected:
                        findings.append(
                            ValidationFinding(
                                severity="medium",
                                category="drift",
                                title=f"Prediction Distribution Shift Detected (Pred PSI={pred_psi:.3f})",
                                description=(
                                    f"Model prediction output distribution has shifted between reference and current samples (PSI: {pred_psi:.3f})."
                                ),
                                metric_name="Prediction Output PSI",
                                metric_value=round(pred_psi, 4),
                                threshold="< 0.10",
                                recommendation="Investigate concept drift; compare against ground-truth labels if available.",
                            )
                        )
                else:
                    # Continuous regression prediction shift
                    pred_psi = calculate_psi(np.array(y_pred_ref, dtype=float), np.array(y_pred_curr, dtype=float))
                    concept_drift_detected = pred_psi >= 0.10 and not is_partitioned
                    concept_drift_info = {
                        "prediction_psi": round(pred_psi, 4),
                        "concept_drift_detected": concept_drift_detected,
                        "reference_pred_mean": round(float(np.mean(y_pred_ref)), 4),
                        "current_pred_mean": round(float(np.mean(y_pred_curr)), 4),
                    }
            except Exception as e:
                concept_drift_info = {"error": f"Could not compute prediction drift: {str(e)}"}
        elif is_small_sample or is_partitioned:
            concept_drift_info = {
                "concept_drift_detected": False,
                "note": "Concept drift bypassed: Requires external baseline dataset and sample size N >= 30."
            }

        # --- Aggregate Metrics ---
        metrics["reference_row_count"] = len(ref_df)
        metrics["current_row_count"] = len(curr_df)
        metrics["is_sequential_partition"] = is_partitioned
        metrics["is_small_sample"] = is_small_sample
        metrics["total_features_evaluated"] = len(feat_cols)
        metrics["drifted_features_count"] = drifted_count
        metrics["drifted_features_percentage"] = round(drift_share * 100, 2)
        metrics["max_psi"] = round(float(max_psi), 4)
        metrics["overall_drift_detected"] = overall_drift_detected
        metrics["overall_drift_status"] = overall_drift_status
        metrics["feature_drift"] = feature_drift_results
        metrics["concept_drift"] = concept_drift_info
        metrics["status_label"] = status_label

        # --- Overall Category Score (0–100) ---
        score = 100.0
        if is_partitioned:
            # Internal split diagnostic: cap maximum penalty to at most 15 points
            split_penalty = min(15.0, (drift_share * 10.0) + (10.0 if max_psi >= 0.25 else (5.0 if max_psi >= 0.10 else 0.0)))
            score -= split_penalty
        else:
            # External baseline: standard evidence-based penalty
            score -= min(40.0, drift_share * 60.0)
            if max_psi >= 0.25:
                score -= 30.0
            elif max_psi >= 0.10:
                score -= 15.0
            if concept_drift_info.get("concept_drift_detected", False):
                score -= 15.0

        final_score = max(0.0, min(100.0, round(score, 1)))

        return ValidatorResult(
            validator_name=self.name,
            validator_version=self.version,
            status=ValidatorStatus.SUCCESS,
            status_label=status_label,
            score=final_score,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            category=self.category,
        )