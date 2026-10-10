"""Model Fairness & Bias Validator module.

Evaluates algorithmic bias, demographic parity, disparate impact (four-fifths rule),
equal opportunity, and equalized odds across protected / sensitive demographic groups.
"""
from typing import Any
import numpy as np
import pandas as pd

from app.validators.base import (
    BaseValidator,
    ValidatorResult,
    ValidationContext,
    ValidationFinding,
    ValidatorStatus,
)


class FairnessValidator(BaseValidator):
    """Validator assessing group fairness and bias across sensitive attributes."""

    name = "fairness"
    version = "1.0.0"
    category = "fairness"

    def is_applicable(self, context: ValidationContext) -> bool:
        """Applicable when sensitive columns are provided and exist in dataset."""
        if not context.sensitive_columns or context.dataset is None:
            return False
        return any(c in context.dataset.columns for c in context.sensitive_columns)

    def validate(self, context: ValidationContext) -> ValidatorResult:
        """Execute group fairness evaluation."""
        df: pd.DataFrame = context.dataset
        target_col = context.target_column
        sensitive_cols = [c for c in context.sensitive_columns if c in df.columns]

        if not sensitive_cols:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SKIPPED,
                score=None,
                error_message="No sensitive attributes specified. Fairness evaluation skipped.",
                category=self.category,
            )

        # Generate model predictions
        expected_names = getattr(context.model, "feature_names_in_", None)
        if expected_names is not None:
            feature_cols = [str(f) for f in expected_names if f in df.columns]
        elif context.feature_columns:
            feature_cols = [c for c in context.feature_columns if c in df.columns]
        else:
            feature_cols = [c for c in df.columns if c != target_col]

        X = df[feature_cols].copy().fillna(0)
        y_pred = context.model.predict(X)

        # Determine binary outcome orientation
        target_series = df[target_col]
        labels = (
            list(context.model.classes_)
            if hasattr(context.model, "classes_")
            else sorted(list(np.unique(target_series)))
        )
        pos_label = labels[-1]  # Typically 1 in {0, 1}

        y_true_bin = (target_series == pos_label).astype(int).values
        y_pred_bin = (pd.Series(y_pred) == pos_label).astype(int).values

        findings: list[ValidationFinding] = []
        metrics: dict[str, Any] = {}
        recommendations: list[str] = []
        attribute_scores: list[float] = []

        metrics["sensitive_attributes"] = sensitive_cols
        metrics["positive_label"] = str(pos_label)
        metrics["attributes"] = {}

        for s_col in sensitive_cols:
            groups = [g for g in df[s_col].dropna().unique()]
            if len(groups) < 2:
                continue

            group_metrics = {}
            selection_rates = []
            tprs = []
            fprs = []

            for g in groups:
                mask = (df[s_col] == g).values
                n_samples = int(mask.sum())
                if n_samples == 0:
                    continue

                # Small sample size check
                if n_samples < 30:
                    findings.append(
                        ValidationFinding(
                            severity="info",
                            category="fairness",
                            title=f"Small Subgroup Sample Size for '{s_col}={g}' (N={n_samples})",
                            description=(
                                f"Subgroup '{g}' has only {n_samples} records. "
                                "Fairness disparity metrics for this subgroup have wider confidence intervals."
                            ),
                            metric_name=f"Subgroup Size ({g})",
                            metric_value=float(n_samples),
                            threshold=">= 30 samples",
                            recommendation="Collect additional validation instances for low-sample subgroups to improve statistical power.",
                        )
                    )

                # Selection Rate P(y_pred = 1 | group)
                sel_rate = float(y_pred_bin[mask].sum() / n_samples)
                selection_rates.append(sel_rate)

                # True Positive Rate (Recall) P(y_pred = 1 | y_true = 1, group)
                actual_pos = (y_true_bin[mask] == 1).sum()
                if actual_pos > 0:
                    tpr = float(((y_pred_bin[mask] == 1) & (y_true_bin[mask] == 1)).sum() / actual_pos)
                else:
                    tpr = 0.0
                tprs.append(tpr)

                # False Positive Rate P(y_pred = 1 | y_true = 0, group)
                actual_neg = (y_true_bin[mask] == 0).sum()
                if actual_neg > 0:
                    fpr = float(((y_pred_bin[mask] == 1) & (y_true_bin[mask] == 0)).sum() / actual_neg)
                else:
                    fpr = 0.0
                fprs.append(fpr)

                # Accuracy in group
                acc = float((y_pred_bin[mask] == y_true_bin[mask]).sum() / n_samples)

                group_metrics[str(g)] = {
                    "sample_count": n_samples,
                    "selection_rate": round(sel_rate, 4),
                    "true_positive_rate": round(tpr, 4),
                    "false_positive_rate": round(fpr, 4),
                    "accuracy": round(acc, 4),
                }

            # Calculate Disparities across groups
            max_sr = max(selection_rates) if selection_rates else 0.0
            min_sr = min(selection_rates) if selection_rates else 0.0
            dp_diff = round(float(max_sr - min_sr), 4)

            # Disparate Impact Ratio (80% rule: min_sr / max_sr)
            if max_sr > 1e-6:
                di_ratio = round(float(min_sr / max_sr), 4)
            else:
                di_ratio = 1.0  # Equal zero rates

            four_fifths_compliant = di_ratio >= 0.80

            # Equal Opportunity Difference (Delta TPR)
            max_tpr = max(tprs) if tprs else 0.0
            min_tpr = min(tprs) if tprs else 0.0
            eo_diff = round(float(max_tpr - min_tpr), 4)

            # Equalized Odds Difference (max(Delta TPR, Delta FPR))
            max_fpr = max(fprs) if fprs else 0.0
            min_fpr = min(fprs) if fprs else 0.0
            fpr_diff = round(float(max_fpr - min_fpr), 4)
            eq_odds_diff = max(eo_diff, fpr_diff)

            metrics["attributes"][s_col] = {
                "demographic_parity_difference": dp_diff,
                "disparate_impact_ratio": di_ratio,
                "four_fifths_compliant": four_fifths_compliant,
                "equal_opportunity_difference": eo_diff,
                "equalized_odds_difference": eq_odds_diff,
                "subgroups": group_metrics,
            }

            # --- Rule-Based Finding Emitting ---
            if di_ratio < 0.60:
                findings.append(
                    ValidationFinding(
                        severity="critical",
                        category="fairness",
                        title=f"Severe Disparate Impact on '{s_col}' (Ratio: {di_ratio:.2f})",
                        description=(
                            f"Disparate impact ratio is {di_ratio:.2f}, heavily violating the Four-Fifths (80%) Rule. "
                            f"Positive selection rates vary widely from {min_sr*100:.1f}% to {max_sr*100:.1f}%."
                        ),
                        metric_name=f"Disparate Impact Ratio ({s_col})",
                        metric_value=di_ratio,
                        threshold=">= 0.80 (Four-Fifths Rule)",
                        recommendation="Audit training distribution for demographic bias and apply threshold mitigation or fair representations.",
                    )
                )
            elif not four_fifths_compliant:
                findings.append(
                    ValidationFinding(
                        severity="high",
                        category="fairness",
                        title=f"Four-Fifths Rule Violation on '{s_col}' (Ratio: {di_ratio:.2f})",
                        description=(
                            f"Disparate impact ratio ({di_ratio:.2f}) falls below the 0.80 threshold. "
                            f"Demographic parity difference is {dp_diff:.3f}."
                        ),
                        metric_name=f"Disparate Impact Ratio ({s_col})",
                        metric_value=di_ratio,
                        threshold=">= 0.80",
                        recommendation="Inspect feature correlations with sensitive attributes and evaluate post-processing group calibration.",
                    )
                )

            if eo_diff > 0.15:
                findings.append(
                    ValidationFinding(
                        severity="high",
                        category="fairness",
                        title=f"Significant Equal Opportunity Disparity on '{s_col}' (ΔTPR: {eo_diff:.3f})",
                        description=(
                            f"True positive rates vary by {eo_diff*100:.1f}% across '{s_col}' subgroups "
                            f"(ranging from {min_tpr*100:.1f}% to {max_tpr*100:.1f}%)."
                        ),
                        metric_name=f"Equal Opportunity Difference ({s_col})",
                        metric_value=eo_diff,
                        threshold="<= 0.10",
                        recommendation="Calibrate group-level decision thresholds to equalize true positive rates across protected categories.",
                    )
                )
            elif eo_diff > 0.10:
                findings.append(
                    ValidationFinding(
                        severity="medium",
                        category="fairness",
                        title=f"Moderate Equal Opportunity Disparity on '{s_col}' (ΔTPR: {eo_diff:.3f})",
                        description=f"True positive rate gap of {eo_diff*100:.1f}% between groups.",
                        metric_name=f"Equal Opportunity Difference ({s_col})",
                        metric_value=eo_diff,
                        threshold="<= 0.10",
                        recommendation="Consider group threshold tuning to equalize opportunity across demographic segments.",
                    )
                )

            if dp_diff > 0.15 and four_fifths_compliant:
                findings.append(
                    ValidationFinding(
                        severity="medium",
                        category="fairness",
                        title=f"Notable Demographic Parity Difference on '{s_col}' ({dp_diff:.3f})",
                        description=f"Selection rates differ by {dp_diff*100:.1f}% across groups.",
                        metric_name=f"Demographic Parity Difference ({s_col})",
                        metric_value=dp_diff,
                        threshold="<= 0.10",
                        recommendation="Examine underlying group base rate differences versus model algorithmic bias.",
                    )
                )

            # Score calculation for this attribute
            attr_score = 100.0
            if di_ratio < 0.80:
                attr_score -= ((0.80 - di_ratio) / 0.80) * 35.0
            if eo_diff > 0.10:
                attr_score -= min(25.0, (eo_diff - 0.10) * 100.0)
            if dp_diff > 0.10:
                attr_score -= min(25.0, (dp_diff - 0.10) * 100.0)
            if eq_odds_diff > 0.15:
                attr_score -= min(15.0, (eq_odds_diff - 0.15) * 50.0)

            attribute_scores.append(float(np.clip(attr_score, 0.0, 100.0)))

        # Overall fairness score
        final_score = float(np.mean(attribute_scores)) if attribute_scores else 100.0
        final_score = float(np.clip(final_score, 0.0, 100.0))

        # Scientific communication language
        recommendations.append(
            f"Within the tested sensitive attributes ({', '.join(sensitive_cols)}) and sample dataset, "
            f"fairness score evaluated to {final_score:.1f}/100. "
            "Note: Algorithmic fairness evaluations are conditional on the provided dataset and attributes."
        )

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
