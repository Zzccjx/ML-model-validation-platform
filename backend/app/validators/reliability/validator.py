"""Model Reliability & Probability Calibration Validator module.

Calculates Brier Score, Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
prediction confidence statistics, overconfident error identification, and calibration curve bins.
"""
from typing import Any
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from app.validators.base import (
    BaseValidator,
    ValidatorResult,
    ValidationContext,
    ValidationFinding,
    ValidatorStatus,
)
from app.core.logging import logger


class ReliabilityValidator(BaseValidator):
    """Validator evaluating model reliability, probability calibration, and predictive confidence."""

    name = "reliability"
    version = "1.0.0"
    category = "reliability"

    def is_applicable(self, context: ValidationContext) -> bool:
        """Applicable when model and dataset are both provided."""
        if context.model is None or context.dataset is None or len(context.dataset) < 10:
            return False
        return True

    def validate(self, context: ValidationContext) -> ValidatorResult:
        """Execute probability calibration and predictive reliability evaluation."""
        if context.model is None or context.dataset is None:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SKIPPED,
                score=None,
                error_message="Model or dataset not provided for reliability analysis.",
                category=self.category,
            )

        df: pd.DataFrame = context.dataset
        target_col = context.target_column

        # Determine feature columns
        expected_names = getattr(context.model, "feature_names_in_", None)
        if expected_names is not None:
            feat_cols = [str(f) for f in expected_names if f in df.columns]
        elif context.feature_columns:
            feat_cols = [c for c in context.feature_columns if c in df.columns]
        else:
            feat_cols = [c for c in df.columns if c != target_col]

        if not feat_cols or target_col not in df.columns:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SKIPPED,
                score=None,
                error_message="Required feature columns or target column missing for reliability analysis.",
                category=self.category,
            )

        X = df[feat_cols].copy().fillna(0)
        y_true = df[target_col].values

        pred_type = context.model_type or "classification"
        if pred_type == "regression":
            return self._validate_regression_reliability(context.model, X, y_true)

        # Classification evaluation
        findings: list[ValidationFinding] = []
        metrics: dict[str, Any] = {}
        recommendations: list[str] = []

        has_predict_proba = hasattr(context.model, "predict_proba")
        if not has_predict_proba:
            findings.append(
                ValidationFinding(
                    severity="LOW",
                    category=self.category,
                    title="Model Does Not Output Probability Estimates",
                    description=(
                        "The model does not provide a predict_proba method. Discrete calibration metrics "
                        "(Brier score, Expected Calibration Error) cannot be computed directly."
                    ),
                    metric_name="has_predict_proba",
                    metric_value=0.0,
                    threshold="Available",
                    recommendation=(
                        "If probability calibration is required for confidence scoring or thresholding, "
                        "calibrate predictions using sklearn.calibration.CalibratedClassifierCV."
                    ),
                )
            )
            recommendations.append(
                "Wrap classifier in CalibratedClassifierCV to enable probabilistic calibration monitoring."
            )
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SUCCESS,
                score=70.0,
                findings=findings,
                metrics={
                    "supports_probabilities": False,
                    "reliability_status": "PROBABILITIES_UNAVAILABLE",
                },
                recommendations=recommendations,
                category=self.category,
            )

        try:
            probabilities = context.model.predict_proba(X)
            y_pred = context.model.predict(X)
        except Exception as e:
            logger.error(f"Error during probability generation: {e}")
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.FAILED,
                score=0.0,
                error_message=f"Model failed to generate probability estimates: {str(e)}",
                category=self.category,
            )

        n_samples = len(y_true)
        n_classes = probabilities.shape[1]
        classes = getattr(context.model, "classes_", np.unique(y_true))

        if n_classes == 2:
            # Binary classification
            pos_label = classes[1] if len(classes) > 1 else 1
            y_binary = (y_true == pos_label).astype(int)
            p_pos = probabilities[:, 1]

            # Brier Score: Mean squared error of probabilities
            brier_score = float(np.mean((p_pos - y_binary) ** 2))

            # Expected Calibration Error (ECE) & Maximum Calibration Error (MCE)
            bin_edges = np.linspace(0.0, 1.0, 11)
            bin_assignments = np.digitize(p_pos, bin_edges[1:-1])
            ece = 0.0
            mce = 0.0
            calibration_bins = []

            for b in range(10):
                bin_mask = bin_assignments == b
                bin_size = int(np.sum(bin_mask))
                if bin_size > 0:
                    bin_acc = float(np.mean(y_binary[bin_mask]))
                    bin_conf = float(np.mean(p_pos[bin_mask]))
                    bin_diff = abs(bin_acc - bin_conf)
                    ece += (bin_size / n_samples) * bin_diff
                    mce = max(mce, bin_diff)
                    calibration_bins.append({
                        "bin_index": b,
                        "bin_range": f"{bin_edges[b]:.1f} - {bin_edges[b+1]:.1f}",
                        "sample_count": bin_size,
                        "mean_confidence": round(bin_conf, 4),
                        "observed_accuracy": round(bin_acc, 4),
                        "calibration_gap": round(bin_diff, 4),
                    })
                else:
                    calibration_bins.append({
                        "bin_index": b,
                        "bin_range": f"{bin_edges[b]:.1f} - {bin_edges[b+1]:.1f}",
                        "sample_count": 0,
                        "mean_confidence": None,
                        "observed_accuracy": None,
                        "calibration_gap": 0.0,
                    })

            # Calibration curve points
            prob_true, prob_pred = calibration_curve(y_binary, p_pos, n_bins=10, strategy="uniform")
            curve_points = [
                {"true_prob": round(float(t), 4), "pred_prob": round(float(p), 4)}
                for t, p in zip(prob_true, prob_pred)
            ]

            max_confs = np.max(probabilities, axis=1)
            correct_mask = (y_pred == y_true)
            overconfident_errors_mask = (~correct_mask) & (max_confs >= 0.90)
            overconfident_errors_count = int(np.sum(overconfident_errors_mask))
            total_errors = int(np.sum(~correct_mask))
            overconfident_error_rate = float(overconfident_errors_count / total_errors) if total_errors > 0 else 0.0
            overall_overconfident_rate = float(overconfident_errors_count / n_samples)

            metrics = {
                "classification_mode": "binary",
                "brier_score": round(brier_score, 4),
                "expected_calibration_error": round(ece, 4),
                "maximum_calibration_error": round(mce, 4),
                "mean_prediction_confidence": round(float(np.mean(max_confs)), 4),
                "median_prediction_confidence": round(float(np.median(max_confs)), 4),
                "correct_mean_confidence": round(float(np.mean(max_confs[correct_mask])), 4) if np.any(correct_mask) else 0.0,
                "error_mean_confidence": round(float(np.mean(max_confs[~correct_mask])), 4) if np.any(~correct_mask) else 0.0,
                "overconfident_errors_count": overconfident_errors_count,
                "overconfident_error_rate": round(overconfident_error_rate, 4),
                "overall_overconfident_rate": round(overall_overconfident_rate, 4),
                "calibration_curve_points": curve_points,
                "calibration_bins": calibration_bins,
            }

        else:
            # Multiclass formulation
            one_hot_y = pd.get_dummies(y_true).reindex(columns=classes, fill_value=0).values
            brier_score = float(np.mean(np.sum((probabilities - one_hot_y) ** 2, axis=1)))

            winning_confs = np.max(probabilities, axis=1)
            correct_mask = (y_pred == y_true)

            bin_edges = np.linspace(0.0, 1.0, 11)
            bin_assignments = np.digitize(winning_confs, bin_edges[1:-1])
            ece = 0.0
            mce = 0.0
            calibration_bins = []

            for b in range(10):
                bin_mask = bin_assignments == b
                bin_size = int(np.sum(bin_mask))
                if bin_size > 0:
                    bin_acc = float(np.mean(correct_mask[bin_mask]))
                    bin_conf = float(np.mean(winning_confs[bin_mask]))
                    bin_diff = abs(bin_acc - bin_conf)
                    ece += (bin_size / n_samples) * bin_diff
                    mce = max(mce, bin_diff)
                    calibration_bins.append({
                        "bin_index": b,
                        "bin_range": f"{bin_edges[b]:.1f} - {bin_edges[b+1]:.1f}",
                        "sample_count": bin_size,
                        "mean_confidence": round(bin_conf, 4),
                        "observed_accuracy": round(bin_acc, 4),
                        "calibration_gap": round(bin_diff, 4),
                    })
                else:
                    calibration_bins.append({
                        "bin_index": b,
                        "bin_range": f"{bin_edges[b]:.1f} - {bin_edges[b+1]:.1f}",
                        "sample_count": 0,
                        "mean_confidence": None,
                        "observed_accuracy": None,
                        "calibration_gap": 0.0,
                    })

            overconfident_errors_mask = (~correct_mask) & (winning_confs >= 0.90)
            overconfident_errors_count = int(np.sum(overconfident_errors_mask))
            total_errors = int(np.sum(~correct_mask))
            overconfident_error_rate = float(overconfident_errors_count / total_errors) if total_errors > 0 else 0.0
            overall_overconfident_rate = float(overconfident_errors_count / n_samples)

            metrics = {
                "classification_mode": "multiclass",
                "brier_score": round(brier_score, 4),
                "expected_calibration_error": round(ece, 4),
                "maximum_calibration_error": round(mce, 4),
                "mean_prediction_confidence": round(float(np.mean(winning_confs)), 4),
                "median_prediction_confidence": round(float(np.median(winning_confs)), 4),
                "correct_mean_confidence": round(float(np.mean(winning_confs[correct_mask])), 4) if np.any(correct_mask) else 0.0,
                "error_mean_confidence": round(float(np.mean(winning_confs[~correct_mask])), 4) if np.any(~correct_mask) else 0.0,
                "overconfident_errors_count": overconfident_errors_count,
                "overconfident_error_rate": round(overconfident_error_rate, 4),
                "overall_overconfident_rate": round(overall_overconfident_rate, 4),
                "calibration_bins": calibration_bins,
            }

        # Calculate score (0 to 100)
        score = 100.0
        score -= min(40.0, (ece / 0.20) * 40.0)
        score -= min(30.0, (brier_score / 0.35) * 30.0)
        score -= min(30.0, overconfident_error_rate * 50.0)
        final_score = max(0.0, min(100.0, round(score, 1)))

        # Generate findings
        if overconfident_error_rate > 0.15:
            findings.append(
                ValidationFinding(
                    severity="CRITICAL",
                    category=self.category,
                    title="Severe Overconfident Errors Detected",
                    description=(
                        f"{overconfident_errors_count} prediction errors ({overconfident_error_rate:.1%} of mistakes) "
                        "occurred with >= 90% confidence. The model exhibits high certainty on erroneous decisions."
                    ),
                    metric_name="Overconfident Error Rate",
                    metric_value=round(overconfident_error_rate, 4),
                    threshold="< 15%",
                    recommendation="Apply temperature scaling or Platt scaling via CalibratedClassifierCV before autonomous actuation.",
                )
            )
            recommendations.append("Apply post-hoc probability scaling to mitigate overconfident misclassifications.")
        elif overconfident_errors_count > 0:
            findings.append(
                ValidationFinding(
                    severity="MEDIUM",
                    category=self.category,
                    title="Moderate Overconfident Errors Observed",
                    description=(
                        f"{overconfident_errors_count} instances were classified erroneously with high confidence (>= 90%)."
                    ),
                    metric_name="Overconfident Errors Count",
                    metric_value=float(overconfident_errors_count),
                    threshold="0",
                    recommendation="Review high-confidence failure cases to inspect label ambiguity or feature outliers.",
                )
            )

        if ece > 0.15:
            findings.append(
                ValidationFinding(
                    severity="HIGH",
                    category=self.category,
                    title="Poor Probability Calibration (High ECE)",
                    description=(
                        f"Expected Calibration Error is {ece:.3f} (> 0.15). Probabilities diverge notably from empirical outcomes."
                    ),
                    metric_name="Expected Calibration Error",
                    metric_value=round(ece, 4),
                    threshold="<= 0.15",
                    recommendation="Calibrate probabilities using isotonic regression or Platt scaling.",
                )
            )
            recommendations.append("Calibrate probability predictions using CalibratedClassifierCV.")
        elif ece > 0.08:
            findings.append(
                ValidationFinding(
                    severity="MEDIUM",
                    category=self.category,
                    title="Moderate Calibration Discrepancy",
                    description=f"Expected Calibration Error is {ece:.3f} (moderate divergence).",
                    metric_name="Expected Calibration Error",
                    metric_value=round(ece, 4),
                    threshold="<= 0.08",
                    recommendation="Verify whether downstream decision thresholds require calibrated probabilities.",
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    severity="INFO",
                    category=self.category,
                    title="Well-Calibrated Probability Distribution",
                    description=f"Expected Calibration Error is low ({ece:.3f} <= 0.08). Confidence closely tracks empirical frequency.",
                    metric_name="Expected Calibration Error",
                    metric_value=round(ece, 4),
                    threshold="<= 0.08",
                    recommendation="Maintain probability calibration standards in continuous monitoring.",
                )
            )

        if brier_score > 0.25:
            findings.append(
                ValidationFinding(
                    severity="HIGH",
                    category=self.category,
                    title="Elevated Brier Score",
                    description=f"Brier score is {brier_score:.3f} (> 0.25), indicating substantial mean squared probability discrepancy.",
                    metric_name="Brier Score",
                    metric_value=round(brier_score, 4),
                    threshold="<= 0.25",
                    recommendation="Evaluate feature representation or optimize with log-loss or probabilistic scoring rules.",
                )
            )
            recommendations.append("Refit model with cross-entropy/log-loss objective to improve probability scoring.")

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

    def _validate_regression_reliability(
        self,
        model: Any,
        X: pd.DataFrame,
        y_true: np.ndarray,
    ) -> ValidatorResult:
        findings: list[ValidationFinding] = []
        recommendations: list[str] = []
        try:
            y_pred = model.predict(X)
            residuals = y_true - y_pred
            residual_std = float(np.std(residuals))
            y_std = float(np.std(y_true))
            dispersion_ratio = residual_std / y_std if y_std > 0 else 0.0

            lower_bound = y_pred - 1.96 * residual_std
            upper_bound = y_pred + 1.96 * residual_std
            coverage_95 = float(np.mean((y_true >= lower_bound) & (y_true <= upper_bound)))

            score = 100.0 - min(60.0, dispersion_ratio * 40.0) - abs(coverage_95 - 0.95) * 100.0
            final_score = max(0.0, min(100.0, round(score, 1)))

            if dispersion_ratio > 0.70:
                findings.append(
                    ValidationFinding(
                        severity="HIGH",
                        category=self.category,
                        title="High Residual Dispersion",
                        description=f"Residual standard deviation is {residual_std:.3f} ({dispersion_ratio:.1%} of target std).",
                        metric_name="Dispersion Ratio",
                        metric_value=round(dispersion_ratio, 4),
                        threshold="<= 0.70",
                        recommendation="Consider quantile regression or conformal prediction to model uncertainty intervals.",
                    )
                )
                recommendations.append("Adopt conformal prediction or quantile regression for prediction intervals.")
            else:
                findings.append(
                    ValidationFinding(
                        severity="INFO",
                        category=self.category,
                        title="Acceptable Residual Uncertainty",
                        description=f"Empirical 95% interval coverage is {coverage_95:.1%}.",
                        metric_name="Empirical Coverage 95%",
                        metric_value=round(coverage_95, 4),
                        threshold="~95%",
                        recommendation="Residual dispersion falls within normal boundaries.",
                    )
                )

            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SUCCESS,
                score=final_score,
                findings=findings,
                metrics={
                    "prediction_type": "regression",
                    "residual_mean": round(float(np.mean(residuals)), 4),
                    "residual_std": round(residual_std, 4),
                    "dispersion_ratio": round(dispersion_ratio, 4),
                    "empirical_coverage_95": round(coverage_95, 4),
                },
                recommendations=recommendations,
                category=self.category,
            )
        except Exception as e:
            logger.error(f"Error evaluating regression reliability: {e}")
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.FAILED,
                score=0.0,
                error_message=f"Regression uncertainty evaluation failed: {str(e)}",
                category=self.category,
            )
