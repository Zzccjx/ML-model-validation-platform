"""Model Robustness & Adversarial Testing Validator module.

Evaluates decision boundary stability, prediction flip rate, and metric degradation
under calibrated Gaussian noise injection and feature missingness perturbations.
"""
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from app.validators.base import (
    BaseValidator,
    ValidatorResult,
    ValidationContext,
    ValidationFinding,
    ValidatorStatus,
)


class RobustnessValidator(BaseValidator):
    """Validator evaluating model resilience against noise and feature perturbations."""

    name = "robustness"
    version = "1.0.0"
    category = "robustness"

    def is_applicable(self, context: ValidationContext) -> bool:
        """Applicable when model and dataset are both provided."""
        if context.model is None or context.dataset is None or len(context.dataset) < 10:
            return False
        return True

    def validate(self, context: ValidationContext) -> ValidatorResult:
        """Execute robustness and adversarial perturbation evaluation."""
        if context.model is None or context.dataset is None:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SKIPPED,
                score=None,
                error_message="Model or dataset unavailable for robustness testing.",
                category=self.category,
            )

        df: pd.DataFrame = context.dataset
        target_col = context.target_column
        has_ground_truth = target_col in df.columns

        # Determine feature columns
        expected_names = getattr(context.model, "feature_names_in_", None)
        if expected_names is not None:
            feat_cols = [str(f) for f in expected_names if f in df.columns]
        elif context.feature_columns:
            feat_cols = [c for c in context.feature_columns if c in df.columns]
        else:
            feat_cols = [c for c in df.columns if c != target_col]

        numeric_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]

        if not numeric_cols:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SKIPPED,
                score=None,
                error_message="No numeric features found for perturbation testing.",
                category=self.category,
            )

        X_clean = df[feat_cols].copy().fillna(0)
        y_clean_pred = context.model.predict(X_clean)
        n_samples = len(df)

        is_classification = context.model_type == "classification" or hasattr(context.model, "classes_")

        # Baseline performance on clean data
        clean_metric_val = None
        if has_ground_truth:
            y_true = df[target_col].values
            if is_classification:
                clean_metric_val = float(accuracy_score(y_true, y_clean_pred))
            else:
                try:
                    clean_metric_val = float(r2_score(y_true, y_clean_pred))
                except Exception:
                    clean_metric_val = 0.0

        findings: list[ValidationFinding] = []
        metrics: dict[str, Any] = {}
        recommendations: list[str] = []

        # --- 1. Gaussian Noise Injection at Calibrated Intensities ---
        noise_levels = [0.01, 0.05, 0.10, 0.20]  # 1%, 5%, 10%, 20% of feature standard deviations
        col_stds = {c: float(df[c].std() if df[c].std() > 1e-6 else 1.0) for c in numeric_cols}

        np.random.seed(42)
        perturbation_curve = []
        flip_rates = {}

        for sigma in noise_levels:
            X_perturbed = X_clean.copy()
            for col in numeric_cols:
                noise = np.random.normal(0, sigma * col_stds[col], size=n_samples)
                X_perturbed[col] += noise

            y_pert_pred = context.model.predict(X_perturbed)

            if is_classification:
                flips = int(np.sum(y_pert_pred != y_clean_pred))
                flip_rate = float(flips / n_samples)
                pert_metric = float(accuracy_score(y_true, y_pert_pred)) if has_ground_truth else None
                deg_delta = round(clean_metric_val - pert_metric, 4) if (clean_metric_val is not None and pert_metric is not None) else None
            else:
                # Continuous prediction deviation threshold (> 10% std of target)
                target_std = float(np.std(y_clean_pred)) if np.std(y_clean_pred) > 1e-6 else 1.0
                deviations = np.abs(y_pert_pred - y_clean_pred) > (0.10 * target_std)
                flips = int(np.sum(deviations))
                flip_rate = float(flips / n_samples)
                try:
                    pert_metric = float(r2_score(y_true, y_pert_pred)) if has_ground_truth else None
                except Exception:
                    pert_metric = 0.0
                deg_delta = round(clean_metric_val - pert_metric, 4) if (clean_metric_val is not None and pert_metric is not None) else None

            flip_rates[f"noise_{int(sigma*100)}pct"] = round(flip_rate, 4)
            perturbation_curve.append({
                "noise_sigma": sigma,
                "noise_percentage": f"{int(sigma*100)}%",
                "prediction_flip_rate": round(flip_rate, 4),
                "flipped_count": flips,
                "performance_score": round(pert_metric, 4) if pert_metric is not None else None,
                "metric_degradation": deg_delta,
            })

        # --- 2. Incomplete Input / Feature Masking Resilience ---
        masking_results = {}
        for mask_pct in [0.10, 0.20]:
            X_masked = X_clean.copy()
            mask = np.random.rand(*X_masked[numeric_cols].shape) < mask_pct
            for col_idx, col in enumerate(numeric_cols):
                median_val = df[col].median()
                col_mask = mask[:, col_idx]
                X_masked.loc[col_mask, col] = median_val

            y_masked_pred = context.model.predict(X_masked)
            if is_classification:
                m_flips = int(np.sum(y_masked_pred != y_clean_pred))
            else:
                target_std = float(np.std(y_clean_pred)) if np.std(y_clean_pred) > 1e-6 else 1.0
                m_flips = int(np.sum(np.abs(y_masked_pred - y_clean_pred) > (0.10 * target_std)))
            m_flip_rate = float(m_flips / n_samples)
            masking_results[f"mask_{int(mask_pct*100)}pct"] = {
                "mask_percentage": f"{int(mask_pct*100)}%",
                "flip_rate": round(m_flip_rate, 4),
                "flipped_count": m_flips,
            }

        # --- 3. Rule-Based Findings Generation ---
        flip_5pct = flip_rates.get("noise_5pct", 0.0)
        flip_10pct = flip_rates.get("noise_10pct", 0.0)

        if flip_5pct >= 0.30:
            findings.append(
                ValidationFinding(
                    severity="critical",
                    category="robustness",
                    title=f"Severe Decision Boundary Fragility at 5% Noise ({flip_5pct*100:.1f}% Flip Rate)",
                    description=(
                        f"Injecting mild 5% standard deviation noise causes {flip_5pct*100:.1f}% of model predictions "
                        "to change. The decision boundaries are extremely brittle and sensitive to small input variations."
                    ),
                    metric_name="5% Noise Flip Rate",
                    metric_value=flip_5pct,
                    threshold="< 0.15",
                    recommendation="Apply adversarial training, data augmentation, regularization, or ensemble averaging to stabilize decision boundaries.",
                )
            )
        elif flip_5pct >= 0.15:
            findings.append(
                ValidationFinding(
                    severity="high",
                    category="robustness",
                    title=f"Elevated Vulnerability to Perturbation ({flip_5pct*100:.1f}% Flip Rate at 5% Noise)",
                    description=(
                        f"5% input noise induces a {flip_5pct*100:.1f}% prediction flip rate, "
                        "indicating heightened susceptibility to sensor noise or measurement variations."
                    ),
                    metric_name="5% Noise Flip Rate",
                    metric_value=flip_5pct,
                    threshold="< 0.15",
                    recommendation="Evaluate feature importance and apply smoothing techniques or input normalization constraints.",
                )
            )
        elif flip_10pct >= 0.25:
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="robustness",
                    title=f"Moderate Perturbation Sensitivity at 10% Noise ({flip_10pct*100:.1f}% Flip Rate)",
                    description=f"Model predictions flip on {flip_10pct*100:.1f}% of instances when subjected to 10% noise.",
                    metric_name="10% Noise Flip Rate",
                    metric_value=flip_10pct,
                    threshold="< 0.20",
                    recommendation="Consider noise injection during training to improve out-of-distribution tolerance.",
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    severity="info",
                    category="robustness",
                    title="Decision Boundaries Resilient to Mild Input Jitter",
                    description=(
                        f"Under 5% Gaussian noise, flip rate remained low at {flip_5pct*100:.1f}%. "
                        "The model demonstrates satisfactory local stability around tested data points."
                    ),
                    metric_name="5% Noise Flip Rate",
                    metric_value=flip_5pct,
                    threshold="< 0.15",
                    recommendation="Maintain current regularization hyperparameters during future model updates.",
                )
            )

        # Rating assignment
        if flip_5pct < 0.08 and flip_10pct < 0.18:
            robustness_rating = "RESILIENT"
        elif flip_5pct < 0.20:
            robustness_rating = "MODERATE"
        else:
            robustness_rating = "VULNERABLE"

        # --- 4. Robustness Category Scoring (0–100) ---
        score = 100.0
        # Penalize for 5% noise flip rate (up to 45 pts)
        score -= min(45.0, flip_5pct * 150.0)
        # Penalize for 10% noise flip rate (up to 25 pts)
        score -= min(25.0, max(0.0, flip_10pct - 0.10) * 80.0)
        # Penalize for 20% masking degradation (up to 15 pts)
        mask_20_flips = masking_results.get("mask_20pct", {}).get("flip_rate", 0.0)
        if mask_20_flips > 0.20:
            score -= min(15.0, (mask_20_flips - 0.20) * 50.0)

        final_score = max(0.0, min(100.0, round(score, 1)))

        metrics["clean_metric_value"] = clean_metric_val
        metrics["robustness_rating"] = robustness_rating
        metrics["flip_rate_5pct"] = flip_5pct
        metrics["flip_rate_10pct"] = flip_10pct
        metrics["perturbation_curve"] = perturbation_curve
        metrics["masking_sensitivity"] = masking_results
        metrics["numeric_features_tested"] = len(numeric_cols)

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