"""Interactive Visualization and Analytics Engine for AI/ML Model Validation Platform.

Generates structured, chart-ready datasets from persisted validation run results
across all 7 assurance pillars:
1. Performance (Classification & Regression)
2. Data Quality & Schema Health
3. Data & Prediction Drift
4. Fairness & Bias
5. Feature Explainability & Attribution
6. Adversarial Robustness & Perturbation
7. Probability Calibration & Reliability
"""
import io
import csv
from typing import Any
from app.models.validation_run import ValidationRun
from app.schemas.validation import (
    ChartDefinition,
    ChartAxis,
    ChartSeries,
    ChartThreshold,
    ChartProvenance,
    ChartsResponse,
)


def generate_validation_charts(run: ValidationRun) -> list[ChartDefinition]:
    """Generate all applicable chart definitions from a validation run."""
    cfg = run.config_json or {}
    metrics = cfg.get("metrics", {})
    statuses = cfg.get("validator_statuses", {})
    charts: list[ChartDefinition] = []

    # -------------------------------------------------------------
    # 1. Performance Charts
    # -------------------------------------------------------------
    perf = metrics.get("performance", {})
    if perf:
        charts.extend(_build_performance_charts(perf, statuses.get("performance", {})))

    # -------------------------------------------------------------
    # 2. Data Quality Charts
    # -------------------------------------------------------------
    dq = metrics.get("data_quality", {})
    if dq:
        charts.extend(_build_data_quality_charts(dq, statuses.get("data_quality", {})))

    # -------------------------------------------------------------
    # 3. Drift Charts
    # -------------------------------------------------------------
    drift = metrics.get("drift", {})
    if drift:
        charts.extend(_build_drift_charts(drift, statuses.get("drift", {})))

    # -------------------------------------------------------------
    # 4. Fairness Charts
    # -------------------------------------------------------------
    fairness = metrics.get("fairness")
    fairness_status = statuses.get("fairness", {})
    charts.extend(_build_fairness_charts(fairness, fairness_status))

    # -------------------------------------------------------------
    # 5. Explainability Charts
    # -------------------------------------------------------------
    expl = metrics.get("explainability", {})
    if expl:
        charts.extend(_build_explainability_charts(expl, statuses.get("explainability", {})))

    # -------------------------------------------------------------
    # 6. Robustness Charts
    # -------------------------------------------------------------
    rob = metrics.get("robustness", {})
    if rob:
        charts.extend(_build_robustness_charts(rob, statuses.get("robustness", {})))

    # -------------------------------------------------------------
    # 7. Reliability Charts
    # -------------------------------------------------------------
    rel = metrics.get("reliability", {})
    rel_status = statuses.get("reliability", {})
    charts.extend(_build_reliability_charts(rel, rel_status))

    return charts


def _build_performance_charts(perf: dict[str, Any], status_info: dict[str, Any]) -> list[ChartDefinition]:
    charts = []
    is_classification = "confusion_matrix" in perf or "accuracy" in perf
    is_regression = "r2" in perf or "mae" in perf

    if is_classification:
        cm_data = perf.get("confusion_matrix")
        if cm_data and "labels" in cm_data and "matrix" in cm_data:
            labels = cm_data["labels"]
            matrix = cm_data["matrix"]
            norm_matrix = cm_data.get("normalized", matrix)

            heatmap_data = []
            for i, actual in enumerate(labels):
                for j, predicted in enumerate(labels):
                    count = matrix[i][j] if i < len(matrix) and j < len(matrix[i]) else 0
                    norm = norm_matrix[i][j] if i < len(norm_matrix) and j < len(norm_matrix[i]) else 0.0
                    heatmap_data.append({
                        "actual": f"Class {actual}",
                        "predicted": f"Class {predicted}",
                        "count": count,
                        "percentage": round(float(norm) * 100, 2),
                        "is_correct": i == j,
                    })

            charts.append(ChartDefinition(
                chart_id="perf_confusion_matrix",
                title="Confusion Matrix Heatmap",
                description="Cross-tabulation of ground-truth actual classes versus model-predicted classes.",
                category="performance",
                chart_type="heatmap",
                status="EVALUATED",
                x_axis=ChartAxis(key="predicted", label="Predicted Class"),
                y_axis=ChartAxis(key="actual", label="Actual Class"),
                series=[
                    ChartSeries(key="percentage", label="Normalized Rate (%)", color="#3b82f6", type="bar"),
                    ChartSeries(key="count", label="Sample Count", color="#1d4ed8", type="bar"),
                ],
                data=heatmap_data,
                matrix={
                    "labels": [f"Class {l}" for l in labels],
                    "raw_matrix": matrix,
                    "normalized_matrix": norm_matrix,
                },
                interpretation=(
                    f"Overall Accuracy: {perf.get('accuracy', 0)*100:.1f}%, Balanced Accuracy: {perf.get('balanced_accuracy', 0)*100:.1f}%. "
                    "Diagonal cells represent correct classifications; off-diagonal cells highlight specific misclassification patterns."
                ),
                provenance=ChartProvenance(
                    source_validator="performance",
                    status="EVALUATED",
                    sample_count=sum(sum(row) for row in matrix) if matrix else None,
                ),
            ))

        # Per-class performance breakdown (Precision, Recall, F1)
        per_class = perf.get("per_class_metrics", {})
        if per_class:
            class_chart_data = []
            for c_name, c_met in per_class.items():
                class_chart_data.append({
                    "class": f"Class {c_name}",
                    "precision": round(c_met.get("precision", 0) * 100, 1),
                    "recall": round(c_met.get("recall", 0) * 100, 1),
                    "f1_score": round(c_met.get("f1_score", 0) * 100, 1),
                    "support": c_met.get("support", 0),
                })

            charts.append(ChartDefinition(
                chart_id="perf_per_class_metrics",
                title="Class-Level Performance Breakdown",
                description="Comparison of Precision, Recall, and F1-Score across all target classes.",
                category="performance",
                chart_type="bar",
                status="EVALUATED",
                x_axis=ChartAxis(key="class", label="Target Class"),
                y_axis=ChartAxis(key="percentage", label="Metric Score (%)", unit="%"),
                series=[
                    ChartSeries(key="precision", label="Precision", color="#3b82f6", type="bar"),
                    ChartSeries(key="recall", label="Recall", color="#10b981", type="bar"),
                    ChartSeries(key="f1_score", label="F1-Score", color="#8b5cf6", type="bar"),
                ],
                data=class_chart_data,
                thresholds=[
                    ChartThreshold(value=50.0, label="Minimum Adequate Recall (50%)", color="#ef4444", style="dashed"),
                ],
                interpretation="Highlights class-specific under-recall or precision penalties caused by dataset imbalance.",
                provenance=ChartProvenance(source_validator="performance", status="EVALUATED"),
            ))

    elif is_regression:
        pred_data = perf.get("predictions", {})
        sample_rows = pred_data.get("sample_rows", [])

        if sample_rows:
            scatter_data = []
            for r in sample_rows[:100]:
                actual_val = r.get("actual")
                pred_val = r.get("predicted")
                if actual_val is not None and pred_val is not None:
                    scatter_data.append({
                        "index": r.get("row_index", len(scatter_data)),
                        "actual": round(float(actual_val), 3),
                        "predicted": round(float(pred_val), 3),
                        "abs_error": round(float(r.get("abs_error", abs(actual_val - pred_val))), 3),
                        "ideal": round(float(actual_val), 3),
                    })

            charts.append(ChartDefinition(
                chart_id="perf_actual_vs_predicted",
                title="Actual vs Predicted Regression Parity",
                description="Evaluated continuous predictions compared directly against ground truth targets.",
                category="performance",
                chart_type="scatter",
                status="EVALUATED",
                x_axis=ChartAxis(key="actual", label="Actual Ground Truth", type="number"),
                y_axis=ChartAxis(key="predicted", label="Model Predicted Value", type="number"),
                series=[
                    ChartSeries(key="predicted", label="Predicted Points", color="#3b82f6", type="scatter"),
                    ChartSeries(key="ideal", label="Ideal Parity Line (y = x)", color="#94a3b8", type="line"),
                ],
                data=scatter_data,
                interpretation=f"R² Fit: {perf.get('r2', 'N/A')}, MAE: {perf.get('mae', 'N/A')}, RMSE: {perf.get('rmse', 'N/A')}.",
                provenance=ChartProvenance(
                    source_validator="performance",
                    status="EVALUATED",
                    sample_count=len(scatter_data),
                ),
            ))

            # Residual Distribution
            residual_bins = _bin_continuous_values([r["predicted"] - r["actual"] for r in scatter_data], n_bins=8)
            charts.append(ChartDefinition(
                chart_id="perf_residual_distribution",
                title="Residual Error Distribution",
                description="Frequency distribution of prediction residuals (Predicted - Actual).",
                category="performance",
                chart_type="bar",
                status="EVALUATED",
                x_axis=ChartAxis(key="bin", label="Residual Error Range"),
                y_axis=ChartAxis(key="count", label="Observation Frequency", type="number"),
                series=[ChartSeries(key="count", label="Residual Frequency", color="#6366f1", type="bar")],
                data=residual_bins,
                interpretation="Symmetric distribution centered near zero signifies unbiased regression predictions.",
                provenance=ChartProvenance(source_validator="performance", status="EVALUATED"),
            ))

    return charts


def _build_data_quality_charts(dq: dict[str, Any], status_info: dict[str, Any]) -> list[ChartDefinition]:
    charts = []
    missing_dict = dq.get("missing_values", {})
    feature_stats = dq.get("feature_stats", {})
    total_rows = dq.get("row_count", 0)

    # Missing Values Chart
    missing_data = []
    if missing_dict:
        for col, info in missing_dict.items():
            missing_data.append({
                "column": col,
                "missing_count": info.get("count", 0),
                "missing_percentage": info.get("percentage", 0.0),
            })
    elif feature_stats:
        # If no missing values reported, show top columns with 0% missing
        for col in list(feature_stats.keys())[:10]:
            missing_data.append({
                "column": col,
                "missing_count": 0,
                "missing_percentage": 0.0,
            })

    if missing_data:
        charts.append(ChartDefinition(
            chart_id="dq_missing_values",
            title="Missing Value Rates by Feature",
            description="Percentage of null/missing records across evaluated feature columns.",
            category="data_quality",
            chart_type="bar",
            status="EVALUATED",
            x_axis=ChartAxis(key="column", label="Feature Column"),
            y_axis=ChartAxis(key="missing_percentage", label="Missing Rate (%)", unit="%"),
            series=[
                ChartSeries(key="missing_percentage", label="Missing Percentage", color="#f97316", type="bar"),
            ],
            data=missing_data,
            thresholds=[
                ChartThreshold(value=5.0, label="Imputation Warning (5%)", color="#f59e0b", style="dashed"),
                ChartThreshold(value=20.0, label="Critical Missingness (20%)", color="#ef4444", style="dashed"),
            ],
            interpretation=(
                "Features exceeding 5% missingness require imputation; features exceeding 20% risk degrading generalization."
                if any(d["missing_percentage"] > 0 for d in missing_data)
                else "No missing values detected across feature columns. Data integrity verified."
            ),
            provenance=ChartProvenance(
                source_validator="data_quality",
                status="EVALUATED",
                sample_count=total_rows,
            ),
        ))

    # Feature Cardinality / Unique Counts
    if feature_stats:
        cardinality_data = []
        for col, stat in feature_stats.items():
            cardinality_data.append({
                "column": col,
                "unique_count": stat.get("unique_count", 0),
                "variance": stat.get("variance") or 0.0,
                "is_numeric": stat.get("is_numeric", False),
            })

        charts.append(ChartDefinition(
            chart_id="dq_feature_cardinality",
            title="Feature Cardinality & Variance Health",
            description="Count of distinct unique values per feature to screen for single-value or zero-variance columns.",
            category="data_quality",
            chart_type="bar",
            status="EVALUATED",
            x_axis=ChartAxis(key="column", label="Feature"),
            y_axis=ChartAxis(key="unique_count", label="Distinct Values Count", type="number"),
            series=[
                ChartSeries(key="unique_count", label="Unique Value Count", color="#0ea5e9", type="bar"),
            ],
            data=cardinality_data,
            interpretation="Single unique value indicates constant feature with zero explanatory power.",
            provenance=ChartProvenance(source_validator="data_quality", status="EVALUATED"),
        ))

    return charts


def _build_drift_charts(drift: dict[str, Any], status_info: dict[str, Any]) -> list[ChartDefinition]:
    charts = []
    feat_drift = drift.get("feature_drift", {})
    status_label = drift.get("status_label", "EVALUATED")
    is_partitioned = drift.get("is_sequential_partition", False)

    if feat_drift:
        psi_data = []
        for col, res in feat_drift.items():
            psi_val = res.get("psi", 0.0)
            psi_data.append({
                "feature": col,
                "psi": round(float(psi_val), 4),
                "ks_statistic": res.get("ks_statistic"),
                "status": res.get("drift_status", "stable").upper(),
                "drift_detected": res.get("drift_detected", False),
            })

        # Sort by descending PSI
        psi_data.sort(key=lambda x: x["psi"], reverse=True)

        mode_note = (
            "Evaluation Mode: INTERNAL_SPLIT_DIAGNOSTIC (single dataset split). "
            "Internal partitions highlight split variance rather than confirmed production shift."
            if is_partitioned
            else "Evaluation Mode: EXTERNAL_BASELINE. Verified against designated reference distribution."
        )

        charts.append(ChartDefinition(
            chart_id="drift_feature_psi",
            title="Feature Population Stability Index (PSI)",
            description="Empirical distribution divergence per feature between baseline reference and current dataset.",
            category="drift",
            chart_type="bar",
            status="EVALUATED",
            x_axis=ChartAxis(key="feature", label="Feature"),
            y_axis=ChartAxis(key="psi", label="PSI Value", type="number"),
            series=[
                ChartSeries(key="psi", label="Population Stability Index (PSI)", color="#8b5cf6", type="bar"),
            ],
            data=psi_data,
            thresholds=[
                ChartThreshold(value=0.10, label="Moderate Divergence (0.10)", color="#f59e0b", style="dashed"),
                ChartThreshold(value=0.25, label="Significant Drift (0.25)", color="#ef4444", style="dashed"),
            ],
            interpretation=(
                f"{mode_note} Features with PSI < 0.10 are Stable; 0.10-0.25 exhibit Moderate shift; "
                f">= 0.25 represent Significant Drift requiring review."
            ),
            provenance=ChartProvenance(
                source_validator="drift",
                status="EVALUATED",
                sample_count=drift.get("current_row_count"),
                note=mode_note,
            ),
        ))

        # Two-Sample Kolmogorov-Smirnov Test Chart for numeric features
        ks_data = [
            {
                "feature": d["feature"],
                "ks_statistic": round(float(d["ks_statistic"]), 4) if d["ks_statistic"] is not None else 0.0,
            }
            for d in psi_data
            if d.get("ks_statistic") is not None
        ]
        if ks_data:
            charts.append(ChartDefinition(
                chart_id="drift_ks_statistics",
                title="Kolmogorov-Smirnov Statistic (KS Test)",
                description="Maximum vertical distance between reference and current empirical cumulative distributions.",
                category="drift",
                chart_type="bar",
                status="EVALUATED",
                x_axis=ChartAxis(key="feature", label="Feature"),
                y_axis=ChartAxis(key="ks_statistic", label="KS Distance (D-stat)", type="number"),
                series=[
                    ChartSeries(key="ks_statistic", label="KS Statistic (D)", color="#06b6d4", type="bar"),
                ],
                data=ks_data,
                interpretation="Higher KS statistics indicate substantial shape or location distribution changes.",
                provenance=ChartProvenance(source_validator="drift", status="EVALUATED"),
            ))

    return charts


def _build_fairness_charts(fairness: dict[str, Any] | None, status_info: dict[str, Any]) -> list[ChartDefinition]:
    charts = []

    # Check if fairness was evaluated or skipped/not_evaluated
    if not fairness or not fairness.get("attributes"):
        explanation = (
            status_info.get("explanation")
            or "No sensitive demographic attributes (e.g., gender, ethnicity, age_group) were designated or auto-detected in the dataset."
        )
        charts.append(ChartDefinition(
            chart_id="fairness_disparate_impact",
            title="Demographic Parity & Four-Fifths Rule",
            description="Ratio of positive selection rates across sensitive demographic cohorts.",
            category="fairness",
            chart_type="bar",
            status="NOT_EVALUATED",
            x_axis=ChartAxis(key="attribute", label="Sensitive Attribute"),
            y_axis=ChartAxis(key="ratio", label="Disparate Impact Ratio", type="number"),
            series=[],
            data=[],
            interpretation=f"NOT EVALUATED: {explanation}",
            provenance=ChartProvenance(
                source_validator="fairness",
                status="NOT_EVALUATED",
                note=explanation,
            ),
        ))
        return charts

    attrs = fairness.get("attributes", {})
    di_data = []
    subgroup_rates = []

    for attr_name, attr_info in attrs.items():
        di_ratio = attr_info.get("disparate_impact_ratio", 1.0)
        di_data.append({
            "attribute": attr_name,
            "disparate_impact_ratio": round(float(di_ratio), 3),
            "compliant": attr_info.get("four_fifths_compliant", True),
            "demographic_parity_diff": round(float(attr_info.get("demographic_parity_difference", 0.0)), 3),
            "equal_opportunity_diff": round(float(attr_info.get("equal_opportunity_difference", 0.0)), 3),
        })

        subgroups = attr_info.get("subgroups", {})
        for grp_val, grp_metrics in subgroups.items():
            subgroup_rates.append({
                "attribute_group": f"{attr_name}: {grp_val}",
                "selection_rate": round(grp_metrics.get("selection_rate", 0.0) * 100, 1),
                "tpr": round(grp_metrics.get("true_positive_rate", 0.0) * 100, 1),
                "accuracy": round(grp_metrics.get("accuracy", 0.0) * 100, 1),
                "sample_count": grp_metrics.get("sample_count", 0),
            })

    if di_data:
        charts.append(ChartDefinition(
            chart_id="fairness_disparate_impact",
            title="Disparate Impact Ratio (80% Rule Benchmark)",
            description="Ratio of selection rates (min / max) per demographic attribute vs EEOC 0.80 benchmark.",
            category="fairness",
            chart_type="bar",
            status="EVALUATED",
            x_axis=ChartAxis(key="attribute", label="Demographic Attribute"),
            y_axis=ChartAxis(key="disparate_impact_ratio", label="Disparate Impact Ratio", type="number"),
            series=[
                ChartSeries(key="disparate_impact_ratio", label="Disparate Impact Ratio", color="#10b981", type="bar"),
            ],
            data=di_data,
            thresholds=[
                ChartThreshold(value=0.80, label="EEOC 80% Rule Threshold (0.80)", color="#ef4444", style="dashed"),
            ],
            interpretation=(
                "Ratios below 0.80 indicate potential adverse impact under the Uniform Guidelines on Employee Selection Procedures."
            ),
            provenance=ChartProvenance(source_validator="fairness", status="EVALUATED"),
        ))

    if subgroup_rates:
        charts.append(ChartDefinition(
            chart_id="fairness_subgroup_rates",
            title="Subgroup Selection Rate & True Positive Rate",
            description="Positive prediction frequency and True Positive Rate (Equal Opportunity) per demographic group.",
            category="fairness",
            chart_type="bar",
            status="EVALUATED",
            x_axis=ChartAxis(key="attribute_group", label="Cohort"),
            y_axis=ChartAxis(key="rate", label="Rate (%)", unit="%"),
            series=[
                ChartSeries(key="selection_rate", label="Selection Rate (%)", color="#3b82f6", type="bar"),
                ChartSeries(key="tpr", label="True Positive Rate (%)", color="#8b5cf6", type="bar"),
            ],
            data=subgroup_rates,
            interpretation="Significant divergence in TPR between cohorts signifies violation of Equal Opportunity criterion.",
            provenance=ChartProvenance(source_validator="fairness", status="EVALUATED"),
        ))

    return charts


def _build_explainability_charts(expl: dict[str, Any], status_info: dict[str, Any]) -> list[ChartDefinition]:
    charts = []
    features = expl.get("features", [])
    method = expl.get("method", "permutation_importance")

    if features:
        top_features = features[:10]  # Top 10 features
        chart_data = []
        for f in top_features:
            chart_data.append({
                "feature": f.get("feature", "unknown"),
                "relative_share": round(float(f.get("relative_percentage", 0.0)), 1),
                "importance_score": round(float(f.get("importance_score", 0.0)), 4),
                "cumulative_share": round(float(f.get("cumulative_percentage", 0.0)), 1),
                "rank": f.get("rank", 1),
            })

        method_names = {
            "tree_feature_importances": "Ensemble Tree Gini / Gain Importances",
            "linear_coefficients": "Normalized Linear Model Coefficients",
            "shap_approx": "TreeSHAP Kernel Attributions",
            "permutation_importance": "Permutation Feature Importance",
            "uniform_fallback": "Uniform Feature Baseline",
        }
        method_label = method_names.get(method, method)

        charts.append(ChartDefinition(
            chart_id="explainability_feature_importance",
            title="Global Feature Importance & Attribution Share",
            description=f"Relative attribution weight per feature computed via {method_label}.",
            category="explainability",
            chart_type="bar",
            status="EVALUATED",
            x_axis=ChartAxis(key="feature", label="Feature Name"),
            y_axis=ChartAxis(key="relative_share", label="Relative Contribution (%)", unit="%"),
            series=[
                ChartSeries(key="relative_share", label="Importance Share (%)", color="#6366f1", type="bar"),
                ChartSeries(key="cumulative_share", label="Cumulative Contribution (%)", color="#ec4899", type="line"),
            ],
            data=chart_data,
            thresholds=[
                ChartThreshold(value=50.0, label="Single-Feature Dominance (50%)", color="#ef4444", style="dashed"),
            ],
            interpretation=(
                f"Computed using {method_label}. Top feature '{expl.get('top_feature', '')}' "
                f"accounts for {expl.get('top_feature_share', 0):.1f}% of predictive weight. "
                f"80% of model predictive variance is captured by top {expl.get('features_for_80_percent', len(chart_data))} features."
            ),
            provenance=ChartProvenance(
                source_validator="explainability",
                status="EVALUATED",
                note=f"Attribution Method: {method_label}",
            ),
        ))

    return charts


def _build_robustness_charts(rob: dict[str, Any], status_info: dict[str, Any]) -> list[ChartDefinition]:
    charts = []
    curve = rob.get("perturbation_curve", [])

    if curve:
        chart_data = []
        for pt in curve:
            chart_data.append({
                "noise_intensity": f"{int(pt.get('noise_pct', 0))}% σ",
                "sigma": pt.get("noise_pct", 0),
                "flip_rate": round(float(pt.get("flip_rate", 0.0)) * 100, 2),
                "metric_degradation": round(float(pt.get("metric_degradation") or 0.0) * 100, 2),
                "flip_count": pt.get("flip_count", 0),
            })

        charts.append(ChartDefinition(
            chart_id="robustness_perturbation_curve",
            title="Adversarial Noise Perturbation Sensitivity",
            description="Prediction flip rate and performance degradation under calibrated Gaussian noise injection (1% to 20% σ).",
            category="robustness",
            chart_type="line",
            status="EVALUATED",
            x_axis=ChartAxis(key="noise_intensity", label="Gaussian Noise Level (σ)"),
            y_axis=ChartAxis(key="flip_rate", label="Prediction Flip Rate (%)", unit="%"),
            series=[
                ChartSeries(key="flip_rate", label="Prediction Flip Rate (%)", color="#ef4444", type="line"),
                ChartSeries(key="metric_degradation", label="Accuracy Degradation Δ (%)", color="#f97316", type="line"),
            ],
            data=chart_data,
            thresholds=[
                ChartThreshold(value=15.0, label="Acceptable Stability Limit (15%)", color="#f59e0b", style="dashed"),
            ],
            interpretation=(
                f"Overall Flip Rate: {rob.get('overall_flip_rate', 0)*100:.1f}%. "
                "Steep curves indicate high sensitivity to measurement noise and precarious decision boundaries."
            ),
            provenance=ChartProvenance(
                source_validator="robustness",
                status="EVALUATED",
                sample_count=rob.get("samples_evaluated"),
            ),
        ))

    return charts


def _build_reliability_charts(rel: dict[str, Any], status_info: dict[str, Any]) -> list[ChartDefinition]:
    charts = []

    # Check if probabilities were supported
    supports_probabilities = rel.get("supports_probabilities", True)
    if not supports_probabilities or rel.get("reliability_status") == "NOT_EVALUATED" or status_info.get("status") == "not_evaluated":
        explanation = (
            rel.get("explanation")
            or status_info.get("explanation")
            or "Model does not output continuous posterior probability estimates (predict_proba unavailable). Discrete outputs cannot evaluate probabilistic calibration."
        )
        charts.append(ChartDefinition(
            chart_id="reliability_calibration_curve",
            title="Probability Calibration Reliability",
            description="Mean predicted confidence versus empirical observation accuracy across probability deciles.",
            category="reliability",
            chart_type="line",
            status="NOT_EVALUATED",
            x_axis=ChartAxis(key="pred_prob", label="Mean Predicted Probability", type="number"),
            y_axis=ChartAxis(key="true_prob", label="Empirical Fraction of Positives", type="number"),
            series=[],
            data=[],
            interpretation=f"NOT EVALUATED: {explanation}",
            provenance=ChartProvenance(
                source_validator="reliability",
                status="NOT_EVALUATED",
                note=explanation,
            ),
        ))
        return charts

    # Reliability Calibration Curve
    curve_points = rel.get("calibration_curve_points", [])
    if not curve_points and rel.get("calibration_bins"):
        curve_points = [
            {"pred_prob": b["mean_confidence"], "true_prob": b["observed_accuracy"]}
            for b in rel["calibration_bins"]
            if b.get("sample_count", 0) > 0 and b.get("mean_confidence") is not None and b.get("observed_accuracy") is not None
        ]

    if curve_points:
        chart_data = []
        for pt in curve_points:
            chart_data.append({
                "pred_prob": pt.get("pred_prob"),
                "true_prob": pt.get("true_prob"),
                "perfect_calibration": pt.get("pred_prob"),
            })

        charts.append(ChartDefinition(
            chart_id="reliability_calibration_curve",
            title="Probability Calibration Reliability Curve",
            description="Comparison of model predicted probabilities against actual empirical accuracy (Expected Calibration Error).",
            category="reliability",
            chart_type="line",
            status="EVALUATED",
            x_axis=ChartAxis(key="pred_prob", label="Mean Predicted Probability", type="number"),
            y_axis=ChartAxis(key="true_prob", label="Observed Empirical Accuracy", type="number"),
            series=[
                ChartSeries(key="true_prob", label="Model Empirical Calibration", color="#2563eb", type="line"),
                ChartSeries(key="perfect_calibration", label="Perfect Calibration (y = x)", color="#94a3b8", type="line"),
            ],
            data=chart_data,
            interpretation=(
                f"Expected Calibration Error (ECE): {rel.get('expected_calibration_error', 'N/A')}, "
                f"Brier Score: {rel.get('brier_score', 'N/A')}. "
                "Points lying below the diagonal indicate model overconfidence; points above indicate underconfidence."
            ),
            provenance=ChartProvenance(source_validator="reliability", status="EVALUATED"),
        ))

    # Calibration Bins Distribution
    bins = rel.get("calibration_bins", [])
    if bins:
        bin_data = []
        for b in bins:
            bin_data.append({
                "bin_range": b.get("bin_range", ""),
                "sample_count": b.get("sample_count", 0),
                "calibration_gap": round(b.get("calibration_gap", 0.0) * 100, 1),
                "mean_confidence": round((b.get("mean_confidence") or 0.0) * 100, 1),
                "observed_accuracy": round((b.get("observed_accuracy") or 0.0) * 100, 1),
            })

        charts.append(ChartDefinition(
            chart_id="reliability_confidence_histogram",
            title="Prediction Confidence Decile Distribution",
            description="Sample count distribution and calibration discrepancy across confidence deciles.",
            category="reliability",
            chart_type="bar",
            status="EVALUATED",
            x_axis=ChartAxis(key="bin_range", label="Confidence Range"),
            y_axis=ChartAxis(key="sample_count", label="Sample Count", type="number"),
            series=[
                ChartSeries(key="sample_count", label="Sample Count", color="#3b82f6", type="bar"),
                ChartSeries(key="calibration_gap", label="Calibration Gap (%)", color="#ef4444", type="bar"),
            ],
            data=bin_data,
            interpretation="Highlights over-concentration of sample predictions in extreme confidence bands.",
            provenance=ChartProvenance(source_validator="reliability", status="EVALUATED"),
        ))

    return charts


def chart_to_csv(chart: ChartDefinition) -> str:
    """Export chart tabular data series to CSV formatted string."""
    output = io.StringIO()
    if not chart.data:
        output.write("No tabular data available for this chart\n")
        return output.getvalue()

    headers = list(chart.data[0].keys())
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for row in chart.data:
        writer.writerow(row)
    return output.getvalue()


def _bin_continuous_values(values: list[float], n_bins: int = 8) -> list[dict[str, Any]]:
    """Helper to bin a list of continuous numbers into histogram counts."""
    if not values:
        return []
    min_v, max_v = min(values), max(values)
    if abs(max_v - min_v) < 1e-6:
        return [{"bin": f"{min_v:.2f}", "count": len(values)}]

    step = (max_v - min_v) / n_bins
    bins = [min_v + i * step for i in range(n_bins + 1)]
    counts = [0] * n_bins

    for v in values:
        assigned = False
        for i in range(n_bins):
            if i == n_bins - 1:
                if bins[i] <= v <= bins[i + 1]:
                    counts[i] += 1
                    assigned = True
                    break
            else:
                if bins[i] <= v < bins[i + 1]:
                    counts[i] += 1
                    assigned = True
                    break
        if not assigned:
            counts[-1] += 1

    return [
        {"bin": f"{bins[i]:.2f} to {bins[i+1]:.2f}", "count": counts[i]}
        for i in range(n_bins)
    ]
