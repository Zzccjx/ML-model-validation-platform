"""Model Performance Validator module.

Evaluates ML models across standard performance metrics for both
classification and regression tasks.
"""
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy import stats

from app.validators.base import (
    BaseValidator,
    ValidatorResult,
    ValidationContext,
    ValidationFinding,
    ValidatorStatus,
)


class PerformanceValidator(BaseValidator):
    """Validator that assesses model performance for classification and regression."""

    name = "performance"
    version = "1.0.0"
    category = "performance"

    def is_applicable(self, context: ValidationContext) -> bool:
        """Applicable when dataset and target column exist."""
        if context.dataset is None or not context.target_column:
            return False
        return context.target_column in context.dataset.columns

    def validate(self, context: ValidationContext) -> ValidatorResult:
        """Execute performance validation."""
        df = context.dataset
        target_col = context.target_column

        # Align features
        expected_names = getattr(context.model, "feature_names_in_", None)
        if expected_names is not None:
            feature_cols = [str(f) for f in expected_names if f in df.columns]
        elif context.feature_columns:
            feature_cols = [c for c in context.feature_columns if c in df.columns]
        else:
            feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle NaNs in features if model requires it
        has_nans = X.isna().sum().sum() > 0
        if has_nans:
            X = X.fillna(0)

        # Generate predictions
        y_pred = context.model.predict(X)

        model_type = context.model_type
        if model_type == "classification":
            return self._validate_classification(context, X, y, y_pred, has_nans)
        elif model_type == "regression":
            return self._validate_regression(context, X, y, y_pred, has_nans)
        else:
            # Infer from prediction output
            if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 30:
                return self._validate_regression(context, X, y, y_pred, has_nans)
            else:
                return self._validate_classification(context, X, y, y_pred, has_nans)

    def _validate_classification(
        self,
        context: ValidationContext,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: np.ndarray,
        has_nans: bool,
    ) -> ValidatorResult:
        findings: list[ValidationFinding] = []
        metrics: dict[str, Any] = {}
        recommendations: list[str] = []

        labels = (
            list(context.model.classes_)
            if hasattr(context.model, "classes_")
            else sorted(list(np.unique(y)))
        )
        labels_str = [str(l) for l in labels]

        # Basic Classification Metrics
        acc = float(accuracy_score(y, y_pred))
        bacc = float(balanced_accuracy_score(y, y_pred))
        prec_macro = float(precision_score(y, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y, y_pred, average="weighted", zero_division=0))

        metrics["accuracy"] = round(acc, 4)
        metrics["balanced_accuracy"] = round(bacc, 4)
        metrics["precision_macro"] = round(prec_macro, 4)
        metrics["recall_macro"] = round(rec_macro, 4)
        metrics["f1_macro"] = round(f1_macro, 4)
        metrics["f1_weighted"] = round(f1_weighted, 4)
        metrics["num_classes"] = len(labels)

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred, labels=labels)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.nan_to_num(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis])

        metrics["confusion_matrix"] = {
            "labels": labels_str,
            "matrix": cm.tolist(),
            "normalized": np.round(cm_norm, 4).tolist(),
        }

        # Per-class report
        report = classification_report(y, y_pred, labels=labels, output_dict=True, zero_division=0)
        per_class = {}
        for l in labels:
            str_l = str(l)
            if str_l in report:
                per_class[str_l] = {
                    "precision": round(report[str_l]["precision"], 4),
                    "recall": round(report[str_l]["recall"], 4),
                    "f1_score": round(report[str_l]["f1-score"], 4),
                    "support": int(report[str_l]["support"]),
                }
        metrics["per_class_metrics"] = per_class

        # Probabilities & ROC/PR curves
        y_proba = None
        if hasattr(context.model, "predict_proba") and callable(context.model.predict_proba):
            try:
                y_proba = context.model.predict_proba(X)
            except Exception:
                pass
        elif hasattr(context.model, "decision_function") and callable(context.model.decision_function):
            try:
                y_proba = context.model.decision_function(X)
            except Exception:
                pass

        roc_auc = None
        pr_auc = None
        if y_proba is not None:
            try:
                if len(labels) == 2:
                    score_col = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    # Convert binary labels to 0/1 indicator if needed
                    pos_label = labels[1]
                    y_binary = (y == pos_label).astype(int)
                    roc_auc = float(roc_auc_score(y_binary, score_col))
                    pr_auc = float(average_precision_score(y_binary, score_col))
                else:
                    if y_proba.ndim > 1 and y_proba.shape[1] == len(labels):
                        roc_auc = float(roc_auc_score(y, y_proba, multi_class="ovr", average="weighted"))
            except Exception:
                pass

        metrics["roc_auc"] = round(roc_auc, 4) if roc_auc is not None else None
        metrics["pr_auc"] = round(pr_auc, 4) if pr_auc is not None else None

        # --- Rule-Based Finding Emitting ---
        if acc - bacc > 0.12:
            findings.append(
                ValidationFinding(
                    severity="high",
                    category="performance",
                    title="Accuracy Paradox: Class Imbalance Performance Disparity",
                    description=(
                        f"Overall accuracy is {acc*100:.1f}%, but balanced accuracy is only {bacc*100:.1f}%. "
                        "The model over-indexes on majority classes while severely failing on minority classes."
                    ),
                    metric_name="Accuracy - Balanced Accuracy",
                    metric_value=round(acc - bacc, 4),
                    threshold="<= 0.12",
                    recommendation="Review minority class representation and consider class re-weighting or cost-sensitive thresholding.",
                )
            )

        for l_name, p_metrics in per_class.items():
            c_rec = p_metrics["recall"]
            if c_rec < 0.40 and p_metrics["support"] > 5:
                findings.append(
                    ValidationFinding(
                        severity="high",
                        category="performance",
                        title=f"Severe Under-Recall on Class '{l_name}'",
                        description=f"Class '{l_name}' achieved a recall of only {c_rec*100:.1f}% ({p_metrics['support']} test samples).",
                        metric_name=f"Recall ({l_name})",
                        metric_value=round(c_rec, 4),
                        threshold=">= 0.50",
                        recommendation=f"Add more training instances or adjust the decision boundary for class '{l_name}'.",
                    )
                )

        if f1_macro < 0.50:
            findings.append(
                ValidationFinding(
                    severity="critical",
                    category="performance",
                    title="Critical Underperformance: Very Low Macro F1",
                    description=f"Macro-averaged F1 score is {f1_macro:.3f}, indicating poor overall discriminative capability.",
                    metric_name="Macro F1",
                    metric_value=round(f1_macro, 4),
                    threshold=">= 0.60",
                    recommendation="Re-evaluate feature representation and test alternative model architectures.",
                )
            )
        elif f1_macro < 0.70:
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="performance",
                    title="Moderate Model Performance (Macro F1 < 0.70)",
                    description=f"Macro-averaged F1 score is {f1_macro:.3f}. Room for performance optimization exists.",
                    metric_name="Macro F1",
                    metric_value=round(f1_macro, 4),
                    threshold=">= 0.75",
                    recommendation="Perform systematic hyperparameter tuning or incorporate domain features.",
                )
            )

        if y_proba is None:
            findings.append(
                ValidationFinding(
                    severity="info",
                    category="performance",
                    title="Probability Calibration / ROC Analysis Unavailable",
                    description="The evaluated model does not expose predict_proba or decision_function; ROC-AUC and PR-AUC could not be evaluated.",
                    recommendation="Enable probability outputs (e.g. probability=True in SVC) if risk confidence calibration is required.",
                )
            )

        # --- Row-Level Prediction Comparisons (Deepchecks Reference) ---
        n_rows = len(y)
        y_true_list = list(y.values) if hasattr(y, "values") else list(y)
        y_pred_list = list(y_pred)
        
        row_comparisons: list[dict[str, Any]] = []
        incorrect_indices: list[int] = []
        
        for idx in range(n_rows):
            actual_val = y_true_list[idx]
            pred_val = y_pred_list[idx]
            is_correct = bool(str(actual_val) == str(pred_val))
            if not is_correct:
                incorrect_indices.append(idx)
            
            # Format row probabilities if available
            row_probs: dict[str, float] | None = None
            if y_proba is not None and idx < len(y_proba):
                p_row = y_proba[idx]
                if hasattr(p_row, "__len__") and len(p_row) == len(labels):
                    row_probs = {labels_str[c_idx]: round(float(p_row[c_idx]), 4) for c_idx in range(len(labels))}
                elif hasattr(p_row, "__len__") and len(p_row) == 1 and len(labels) == 2:
                    pos_prob = float(p_row[0])
                    row_probs = {labels_str[0]: round(1.0 - pos_prob, 4), labels_str[1]: round(pos_prob, 4)}
                elif not hasattr(p_row, "__len__") and len(labels) == 2:
                    pos_prob = float(p_row)
                    row_probs = {labels_str[0]: round(1.0 - pos_prob, 4), labels_str[1]: round(pos_prob, 4)}
            
            # Native python type conversion
            clean_actual = actual_val.item() if hasattr(actual_val, "item") else actual_val
            clean_pred = pred_val.item() if hasattr(pred_val, "item") else pred_val

            row_comparisons.append({
                "row_index": idx,
                "actual": clean_actual,
                "predicted": clean_pred,
                "correct": is_correct,
                "probabilities": row_probs,
            })

        # Bounded sample for storage (first 50 + top 50 misclassified)
        sample_rows: list[dict[str, Any]] = []
        seen_indices = set()
        for r in row_comparisons[:50]:
            sample_rows.append(r)
            seen_indices.add(r["row_index"])
        
        for idx in incorrect_indices[:50]:
            if idx not in seen_indices:
                sample_rows.append(row_comparisons[idx])
                seen_indices.add(idx)

        metrics["predictions"] = {
            "total_rows": n_rows,
            "correct_rows": n_rows - len(incorrect_indices),
            "incorrect_rows": len(incorrect_indices),
            "misclassification_rate": round(len(incorrect_indices) / max(1, n_rows), 4),
            "sample_rows": sample_rows,
            "has_probabilities": y_proba is not None,
        }

        # Category Score Calculation (0-100)
        if roc_auc is not None:
            score = 0.4 * bacc + 0.3 * f1_macro + 0.3 * roc_auc
        else:
            score = 0.5 * bacc + 0.5 * f1_macro

        final_score = float(np.clip(score * 100, 0, 100))

        recommendations.append(f"Primary classification benchmark: Balanced Accuracy={bacc*100:.1f}%, Macro F1={f1_macro:.3f}.")
        if roc_auc is not None:
            recommendations.append(f"Discriminative capability: ROC-AUC={roc_auc:.3f}.")

        return ValidatorResult(
            validator_name=self.name,
            validator_version=self.version,
            status=ValidatorStatus.SUCCESS,
            score=round(final_score, 1),
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            category=self.category,
        )

    def _validate_regression(
        self,
        context: ValidationContext,
        X: pd.DataFrame,
        y: pd.Series,
        y_pred: np.ndarray,
        has_nans: bool,
    ) -> ValidatorResult:
        findings: list[ValidationFinding] = []
        metrics: dict[str, Any] = {}
        recommendations: list[str] = []

        mae = float(mean_absolute_error(y, y_pred))
        mse = float(mean_squared_error(y, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y, y_pred))

        # Safe MAPE
        nz_mask = np.abs(y.values) > 1e-6
        if nz_mask.sum() > 0:
            mape = float(np.mean(np.abs((y.values[nz_mask] - y_pred[nz_mask]) / y.values[nz_mask])) * 100)
        else:
            mape = None

        # Residuals
        residuals = y.values - y_pred
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))
        max_err = float(np.max(np.abs(residuals)))
        skew_res = float(stats.skew(residuals))

        quantiles = {
            "p25": float(np.percentile(np.abs(residuals), 25)),
            "p50": float(np.percentile(np.abs(residuals), 50)),
            "p75": float(np.percentile(np.abs(residuals), 75)),
            "p90": float(np.percentile(np.abs(residuals), 90)),
            "p95": float(np.percentile(np.abs(residuals), 95)),
            "p99": float(np.percentile(np.abs(residuals), 99)),
        }

        metrics["mae"] = round(mae, 4)
        metrics["mse"] = round(mse, 4)
        metrics["rmse"] = round(rmse, 4)
        metrics["r2"] = round(r2, 4)
        metrics["mape"] = round(mape, 2) if mape is not None else None
        metrics["residuals"] = {
            "mean_residual": round(mean_res, 4),
            "std_residual": round(std_res, 4),
            "max_error": round(max_err, 4),
            "skewness": round(skew_res, 4),
            "quantiles": {k: round(v, 4) for k, v in quantiles.items()},
        }

        # Rule-Based Findings for Regression
        if r2 < 0.0:
            findings.append(
                ValidationFinding(
                    severity="critical",
                    category="performance",
                    title="Severe Model Failure: Negative R² Score",
                    description=f"Model R² score is {r2:.3f}. Predictions perform worse than a constant mean baseline.",
                    metric_name="R² Score",
                    metric_value=round(r2, 4),
                    threshold=">= 0.0",
                    recommendation="Investigate target scaling, severe data distribution mismatch, or corrupted training features.",
                )
            )
        elif r2 < 0.50:
            findings.append(
                ValidationFinding(
                    severity="high",
                    category="performance",
                    title="Low Explanatory Power (R² < 0.50)",
                    description=f"Model R² is {r2:.3f}, meaning over {(1-r2)*100:.1f}% of variance is unexplained.",
                    metric_name="R² Score",
                    metric_value=round(r2, 4),
                    threshold=">= 0.70",
                    recommendation="Test non-linear regression ensembles (Gradient Boosting, Random Forest) or add predictive covariates.",
                )
            )

        if std_res > 1e-6 and abs(mean_res) > 0.15 * std_res:
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="performance",
                    title="Systematic Residual Bias Detected",
                    description=f"Mean residual is {mean_res:.3f} (std={std_res:.3f}), indicating the model consistently under- or over-predicts.",
                    metric_name="Mean Residual",
                    metric_value=round(mean_res, 4),
                    threshold="<= 0.15 * std",
                    recommendation="Calibrate the model intercept or inspect omitted baseline variables.",
                )
            )

        if rmse > 1e-6 and max_err > 3.0 * rmse:
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="performance",
                    title="Extreme Error Outliers in Tail Predictions",
                    description=f"Maximum absolute error ({max_err:.2f}) exceeds 3x the RMSE ({rmse:.2f}).",
                    metric_name="Max Error / RMSE Ratio",
                    metric_value=round(max_err / rmse, 2),
                    threshold="<= 3.0",
                    recommendation="Examine worst-case samples for target annotation errors or train with Huber/robust loss.",
                )
            )

        # --- Row-Level Prediction Comparisons (Deepchecks Regression Reference) ---
        n_rows = len(y)
        y_true_vals = np.asarray(y.values if hasattr(y, "values") else y, dtype=float)
        y_pred_vals = np.asarray(y_pred, dtype=float)

        row_comparisons: list[dict[str, Any]] = []
        for idx in range(n_rows):
            act = float(y_true_vals[idx])
            prd = float(y_pred_vals[idx])
            err = float(act - prd)
            abs_err = float(abs(err))
            pct_err = float((abs_err / abs(act)) * 100) if abs(act) > 1e-6 else None
            
            row_comparisons.append({
                "row_index": idx,
                "actual": round(act, 4),
                "predicted": round(prd, 4),
                "error": round(err, 4),
                "abs_error": round(abs_err, 4),
                "pct_error": round(pct_err, 2) if pct_err is not None else None,
            })

        # Identify rows with largest absolute errors
        sorted_by_error = sorted(row_comparisons, key=lambda r: r["abs_error"], reverse=True)
        top_error_indices = {r["row_index"] for r in sorted_by_error[:50]}

        sample_rows: list[dict[str, Any]] = []
        seen_indices = set()
        for r in row_comparisons[:50]:
            sample_rows.append(r)
            seen_indices.add(r["row_index"])

        for r in sorted_by_error[:50]:
            if r["row_index"] not in seen_indices:
                sample_rows.append(r)
                seen_indices.add(r["row_index"])

        metrics["predictions"] = {
            "total_rows": n_rows,
            "mean_error": round(float(np.mean(y_true_vals - y_pred_vals)), 4),
            "median_abs_error": round(float(np.median(np.abs(y_true_vals - y_pred_vals))), 4),
            "max_abs_error": round(float(np.max(np.abs(y_true_vals - y_pred_vals))), 4),
            "sample_rows": sample_rows,
            "has_probabilities": False,
        }

        # Score calculation (0-100)
        if r2 <= 0.0:
            score = 0.0
        else:
            score = r2 * 100
            if mape is not None and mape > 40.0:
                score -= 10.0

        final_score = float(np.clip(score, 0, 100))

        recommendations.append(f"Regression summary: R²={r2:.3f}, MAE={mae:.3f}, RMSE={rmse:.3f}.")
        if mape is not None:
            recommendations.append(f"Average percentage error: MAPE={mape:.1f}%.")

        return ValidatorResult(
            validator_name=self.name,
            validator_version=self.version,
            status=ValidatorStatus.SUCCESS,
            score=round(final_score, 1),
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            category=self.category,
        )
