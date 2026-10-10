"""Model Explainability & Feature Importance Validator module.

Calculates global feature attributions, normalized relative importance percentages,
cumulative contribution profiles, and detects single-feature dominance and dead features.
"""
from typing import Any
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from app.validators.base import (
    BaseValidator,
    ValidatorResult,
    ValidationContext,
    ValidationFinding,
    ValidatorStatus,
)


class ExplainabilityValidator(BaseValidator):
    """Validator evaluating model explainability, feature contributions, and attribution balance."""

    name = "explainability"
    version = "1.0.0"
    category = "explainability"

    def is_applicable(self, context: ValidationContext) -> bool:
        """Applicable when model and dataset are both provided."""
        if context.model is None or context.dataset is None or len(context.dataset) < 10:
            return False
        return True

    def validate(self, context: ValidationContext) -> ValidatorResult:
        """Execute feature importance and explainability evaluation."""
        if context.model is None or context.dataset is None:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SKIPPED,
                score=None,
                error_message="Model not provided for explainability analysis.",
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

        if not feat_cols:
            return ValidatorResult(
                validator_name=self.name,
                validator_version=self.version,
                status=ValidatorStatus.SKIPPED,
                score=None,
                error_message="No matching feature columns found for explainability evaluation.",
                category=self.category,
            )

        X = df[feat_cols].copy().fillna(0)
        n_features = len(feat_cols)
        findings: list[ValidationFinding] = []
        metrics: dict[str, Any] = {}
        recommendations: list[str] = []

        method = "unknown"
        raw_importances = None

        # 1. Native tree feature importance (RandomForest, ExtraTrees, GBDT, etc.)
        if hasattr(context.model, "feature_importances_"):
            try:
                imp = np.array(context.model.feature_importances_, dtype=float)
                if len(imp) == n_features:
                    raw_importances = np.maximum(0.0, imp)
                    method = "tree_feature_importances"
            except Exception:
                pass

        # 2. Linear model coefficients (LogisticRegression, LinearRegression, etc.)
        if raw_importances is None and hasattr(context.model, "coef_"):
            try:
                coef = np.array(context.model.coef_, dtype=float)
                if coef.ndim == 2:
                    imp = np.mean(np.abs(coef), axis=0)
                else:
                    imp = np.abs(coef)
                if len(imp) == n_features:
                    raw_importances = np.maximum(0.0, imp)
                    method = "linear_coefficients"
            except Exception:
                pass

        # 3. Permutation feature importance fallback
        if raw_importances is None and target_col in df.columns:
            try:
                sample_n = min(200, len(df))
                df_sample = df.sample(n=sample_n, random_state=42)
                X_sample = df_sample[feat_cols].fillna(0)
                y_sample = df_sample[target_col]
                perm = permutation_importance(context.model, X_sample, y_sample, n_repeats=3, random_state=42)
                raw_importances = np.maximum(0.0, perm.importances_mean)
                method = "permutation_importance"
            except Exception:
                pass

        # 4. Fallback if no method succeeded
        if raw_importances is None:
            raw_importances = np.ones(n_features, dtype=float) / n_features
            method = "uniform_fallback"

        # Normalize relative importances (sum to 100%)
        total_imp = float(np.sum(raw_importances))
        if total_imp > 1e-9:
            rel_percentages = (raw_importances / total_imp) * 100.0
        else:
            rel_percentages = np.full(n_features, 100.0 / n_features)

        # Build sorted feature breakdown
        feature_entries = []
        for i, col in enumerate(feat_cols):
            feature_entries.append({
                "feature": col,
                "importance_score": round(float(raw_importances[i]), 5),
                "relative_percentage": round(float(rel_percentages[i]), 2),
            })

        feature_entries.sort(key=lambda x: x["relative_percentage"], reverse=True)

        # Compute cumulative contribution
        cumulative = 0.0
        for rank, entry in enumerate(feature_entries, 1):
            cumulative += entry["relative_percentage"]
            entry["cumulative_percentage"] = round(min(100.0, cumulative), 2)
            entry["rank"] = rank

        top_feature = feature_entries[0]
        top_feature_name = top_feature["feature"]
        top_feature_share = top_feature["relative_percentage"]
        top_3_share = sum(e["relative_percentage"] for e in feature_entries[:3]) if len(feature_entries) >= 3 else 100.0

        low_impact_features = [e["feature"] for e in feature_entries if e["relative_percentage"] < 1.0]
        zero_impact_features = [e["feature"] for e in feature_entries if e["relative_percentage"] <= 0.05]

        # --- Rule-Based Finding Emitting ---
        if top_feature_share >= 80.0:
            distribution_balance = "DOMINATED"
            findings.append(
                ValidationFinding(
                    severity="critical",
                    category="explainability",
                    title=f"Critical Single-Feature Dominance: '{top_feature_name}' ({top_feature_share:.1f}%)",
                    description=(
                        f"Feature '{top_feature_name}' accounts for {top_feature_share:.1f}% of total model attribution. "
                        "The model is heavily dependent on this single feature, creating a potential single point of failure "
                        "if this signal drifts, degrades, or is compromised."
                    ),
                    metric_name=f"Attribution Share ({top_feature_name})",
                    metric_value=top_feature_share,
                    threshold="< 60.0%",
                    recommendation="Investigate if this feature is causing data leakage or collinearity. Train with feature subsampling (max_features) or regularization to encourage diverse signal utilization.",
                )
            )
        elif top_feature_share >= 60.0:
            distribution_balance = "DOMINATED"
            findings.append(
                ValidationFinding(
                    severity="high",
                    category="explainability",
                    title=f"High Feature Dominance: '{top_feature_name}' ({top_feature_share:.1f}%)",
                    description=(
                        f"Feature '{top_feature_name}' drives {top_feature_share:.1f}% of all model decisions. "
                        "All other features collectively contribute less than 40%."
                    ),
                    metric_name=f"Attribution Share ({top_feature_name})",
                    metric_value=top_feature_share,
                    threshold="< 60.0%",
                    recommendation="Audit correlation with target and evaluate model performance when this feature is masked.",
                )
            )
        elif top_feature_share >= 45.0:
            distribution_balance = "MODERATE"
            findings.append(
                ValidationFinding(
                    severity="medium",
                    category="explainability",
                    title=f"Moderate Feature Concentration: '{top_feature_name}' ({top_feature_share:.1f}%)",
                    description=(
                        f"Feature '{top_feature_name}' contributes {top_feature_share:.1f}% of overall model weight."
                    ),
                    metric_name=f"Attribution Share ({top_feature_name})",
                    metric_value=top_feature_share,
                    threshold="< 45.0%",
                    recommendation="Monitor this feature closely in production drift monitoring pipelines.",
                )
            )
        else:
            distribution_balance = "BALANCED"
            findings.append(
                ValidationFinding(
                    severity="info",
                    category="explainability",
                    title=f"Balanced Feature Attribution Across {n_features} Features",
                    description=(
                        f"Leading feature '{top_feature_name}' represents {top_feature_share:.1f}% of model attribution. "
                        f"Top 3 features represent {top_3_share:.1f}%, indicating balanced signal utilization."
                    ),
                    metric_name="Top 3 Attribution Share",
                    metric_value=round(top_3_share, 2),
                    threshold="< 85.0%",
                    recommendation="Model exhibits healthy multi-feature reliance without single-signal dependency.",
                )
            )

        if len(zero_impact_features) > 0:
            findings.append(
                ValidationFinding(
                    severity="low",
                    category="explainability",
                    title=f"Uninformative Features Detected ({len(zero_impact_features)} Features)",
                    description=(
                        f"The following features provide effectively zero attribution (<= 0.05%): "
                        f"{', '.join(zero_impact_features[:5])}{' and others' if len(zero_impact_features) > 5 else ''}."
                    ),
                    metric_name="Zero-Impact Features Count",
                    metric_value=float(len(zero_impact_features)),
                    threshold="0 features",
                    recommendation="Consider pruning these uninformative features during retraining to improve latency and reduce pipeline complexity.",
                )
            )

        # --- Category Score (0–100) ---
        score = 100.0
        # Deductions for dominance
        if top_feature_share >= 80.0:
            score -= 35.0
        elif top_feature_share >= 60.0:
            score -= 20.0
        elif top_feature_share >= 45.0:
            score -= 10.0

        # Deductions if excessive dead features
        if len(zero_impact_features) > (n_features * 0.40):
            score -= 15.0

        final_score = max(0.0, min(100.0, round(score, 1)))

        metrics["attribution_method"] = method
        metrics["distribution_balance"] = distribution_balance
        metrics["total_features_analyzed"] = n_features
        metrics["top_feature_name"] = top_feature_name
        metrics["top_feature_share_percentage"] = top_feature_share
        metrics["top_3_share_percentage"] = round(top_3_share, 2)
        metrics["low_impact_features_count"] = len(low_impact_features)
        metrics["zero_impact_features_count"] = len(zero_impact_features)
        metrics["features"] = feature_entries

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