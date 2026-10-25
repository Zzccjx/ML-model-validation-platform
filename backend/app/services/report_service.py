"""Report generation service for AI/ML Model Validation & Assurance Platform.

Implements the standard 15-section audit framework:
1. Executive Summary
2. Model Information & Lineage
3. Dataset Metadata & Quality
4. Overall Validation Health Score & Category Ratings
5. Performance Analysis (Module 1)
6. Data Quality & Integrity (Module 2)
7. Fairness & Bias Audit (Module 3)
8. Data & Prediction Drift (Module 4)
9. Adversarial Robustness & Noise Perturbation (Module 5)
10. Feature Explainability & Attribution Balance (Module 6)
11. Probability Calibration & Reliability (Module 7)
12. Comprehensive Findings Inventory (Critical -> Info)
13. Prioritized Actionable Recommendations
14. Validation Methodology & Standards
15. Assumptions, Operational Boundaries & Limitations
"""
import io
import uuid
from datetime import datetime, timezone
from typing import Any
from fpdf import FPDF

from app.models.validation_run import ValidationRun
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.models.finding import Finding


class ValidationReportPDF(FPDF):
    """Custom FPDF2 class with header, footer, and professional styling."""

    def __init__(self, run_number: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.run_number = run_number
        self.set_auto_page_break(auto=True, margin=15)

    @staticmethod
    def clean_text(text: Any) -> str:
        """Convert Greek/special unicode symbols to Latin-1 / ASCII safe equivalents."""
        if text is None:
            return ""
        s = str(text)
        replacements = {
            "\u0394": "Delta ",
            "\u03c3": "sigma",
            "\u2265": ">=",
            "\u2264": "<=",
            "\u2260": "!=",
            "\u2248": "~",
            "\u2014": "-",
            "\u2013": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        # Fallback encode to latin-1 ignoring unrepresentable characters
        return s.encode("latin-1", "replace").decode("latin-1")

    def header(self):
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(100, 116, 139)  # Slate-500
        self.cell(0, 6, f"AI/ML MODEL VALIDATION AUDIT REPORT | {self.clean_text(self.run_number)}", border=0, align="L")
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, "CONFIDENTIAL & PROPRIETARY", border=0, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(226, 232, 240)
        self.line(10, 16, 200, 16)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 6, f"Page {self.page_no()}/{{nb}} | Automated AI/ML Validation Platform", border=0, align="C")

    def section_title(self, num: int, title: str):
        self.ln(4)
        self.set_fill_color(241, 245, 249)  # Slate-100
        self.set_text_color(15, 23, 42)      # Slate-900
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, f" {num}. {self.clean_text(title.upper())}", border=0, align="L", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def table_header(self, cols: list[tuple[str, int]]):
        self.set_fill_color(30, 41, 59)     # Slate-800
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 8)
        for name, width in cols:
            self.cell(width, 6, self.clean_text(name), border=1, align="C", fill=True)
        self.ln()

    def table_row(self, row: list[tuple[str, int, str]], fill: bool = False):
        if fill:
            self.set_fill_color(248, 250, 252)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_text_color(30, 41, 59)
        self.set_font("Helvetica", "", 8)
        for val, width, align in row:
            self.cell(width, 6, self.clean_text(val), border=1, align=align, fill=True)
        self.ln()


def build_report_data(
    run: ValidationRun,
    model: MLModel,
    dataset: Dataset,
    findings: list[Finding],
) -> dict[str, Any]:
    """Assemble standard 15-section audit report data dictionary."""
    metrics_map = (run.config_json or {}).get("metrics", {})
    cat_scores = run.category_scores_json or {}

    # Sort findings: critical -> high -> medium -> low -> info
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_findings = sorted(
        findings,
        key=lambda f: sev_order.get(f.severity.value if hasattr(f.severity, "value") else str(f.severity).lower(), 99),
    )

    sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for f in findings:
        sev = f.severity.value if hasattr(f.severity, "value") else str(f.severity).lower()
        if sev in sev_counts:
            sev_counts[sev] += 1

    overall = run.overall_score or 0.0
    if overall >= 85:
        health_status = "HEALTHY"
        recommendation_summary = "Model exhibits strong multi-dimensional readiness. Suitable for production with standard observability."
    elif overall >= 70:
        health_status = "WARNING"
        recommendation_summary = "Model requires mitigation on moderate findings before full automated deployment."
    else:
        health_status = "CRITICAL"
        recommendation_summary = "High risk detected across critical validation pillars. Do not deploy without addressing flagged items."

    report = {
        "report_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_version": "1.0.0",
            "standard": "AI/ML Model Assurance Standard 2026",
            "run_id": str(run.id),
            "run_number": run.run_number,
        },
        "section_1_executive_summary": {
            "run_number": run.run_number,
            "overall_health_score": round(overall, 2),
            "health_status": health_status,
            "total_findings": len(findings),
            "findings_by_severity": sev_counts,
            "validation_duration_seconds": round(run.duration_seconds or 0.0, 2),
            "summary_verdict": recommendation_summary,
        },
        "section_2_model_information": {
            "model_id": str(model.id),
            "model_name": model.name,
            "framework": model.framework or "scikit-learn",
            "model_type": model.model_type or "classification",
            "feature_count": model.feature_count,
            "class_count": model.class_count,
            "classes": model.classes_json,
            "file_size_bytes": model.file_size,
            "uploaded_at": model.uploaded_at.isoformat() if model.uploaded_at else None,
        },
        "section_3_dataset_information": {
            "dataset_id": str(dataset.id),
            "dataset_name": dataset.name,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "target_column": dataset.target_column,
            "feature_columns": dataset.feature_columns_json,
            "sensitive_columns": dataset.sensitive_columns_json,
            "file_size_bytes": dataset.file_size,
        },
        "section_4_overall_health_score": {
            "overall_score": round(overall, 2),
            "category_scores": {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in cat_scores.items()},
        },
        "section_5_performance": {
            "score": cat_scores.get("performance"),
            "metrics": metrics_map.get("performance", {}),
        },
        "section_6_data_quality": {
            "score": cat_scores.get("data_quality"),
            "metrics": metrics_map.get("data_quality", {}),
        },
        "section_7_fairness": {
            "score": cat_scores.get("fairness"),
            "metrics": metrics_map.get("fairness", {}),
        },
        "section_8_drift": {
            "score": cat_scores.get("drift"),
            "metrics": metrics_map.get("drift", {}),
        },
        "section_9_robustness": {
            "score": cat_scores.get("robustness"),
            "metrics": metrics_map.get("robustness", {}),
        },
        "section_10_explainability": {
            "score": cat_scores.get("explainability"),
            "metrics": metrics_map.get("explainability", {}),
        },
        "section_11_reliability": {
            "score": cat_scores.get("reliability"),
            "metrics": metrics_map.get("reliability", {}),
        },
        "section_12_findings_inventory": [
            {
                "id": str(f.id),
                "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
                "category": f.category,
                "title": f.title,
                "description": f.description,
                "metric_name": f.metric_name,
                "metric_value": f.metric_value,
                "threshold": f.threshold,
                "recommendation": f.recommendation,
                "validator_name": f.validator_name,
            }
            for f in sorted_findings
        ],
        "section_13_actionable_recommendations": [
            {
                "priority": i + 1,
                "category": f.category,
                "action": f.recommendation or f.title,
                "associated_finding": f.title,
                "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
            }
            for i, f in enumerate(sorted_findings[:10])
            if f.recommendation
        ],
        "section_14_methodology": {
            "evaluation_pillars": 7,
            "engine": "Deterministic empirical verification",
            "standards_referenced": [
                "EEOC 80% Four-Fifths Uniform Guidelines",
                "Two-Sample Kolmogorov-Smirnov Distribution Divergence",
                "Population Stability Index (PSI Quantile Binning)",
                "Expected Calibration Error (ECE 10-bin discretization)",
                "Feature Attribution & Single-Factor Dominance Analysis",
            ],
        },
        "section_15_limitations": {
            "bounds": [
                "Evaluations are constrained to the statistical characteristics of the supplied test dataset.",
                "Adversarial perturbations test Gaussian additive noise and missingness; structured adversarial attacks (FGSM/PGD) are out of scope.",
                "Calibration assessments assume the target distribution in production matches the verification dataset.",
            ],
            "disclaimer": "This report provides empirical assurance metrics and does not constitute a legal certification of fitness for automated decisions.",
        },
    }

    try:
        from app.services.visualization_service import generate_validation_charts
        charts = generate_validation_charts(run)
        report["visual_analytics"] = {
            "total_charts": len(charts),
            "charts": [c.model_dump() for c in charts],
        }
    except Exception as e:
        report["visual_analytics"] = {"total_charts": 0, "charts": [], "error": str(e)}

    return report


def generate_json_report(
    run: ValidationRun,
    model: MLModel,
    dataset: Dataset,
    findings: list[Finding],
) -> dict[str, Any]:
    """Generate structured JSON report."""
    return build_report_data(run, model, dataset, findings)


def generate_pdf_report(
    run: ValidationRun,
    model: MLModel,
    dataset: Dataset,
    findings: list[Finding],
) -> bytes:
    """Generate professional PDF validation audit report using fpdf2."""
    data = build_report_data(run, model, dataset, findings)
    pdf = ValidationReportPDF(run_number=run.run_number)
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title Banner
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "Model Assurance & Validation Audit Report", border=0, align="L", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(
        0,
        5,
        f"Run Number: {run.run_number}   |   Execution Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}   |   Status: {run.status.value.upper()}",
        border=0,
        align="L",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)

    # 1. Executive Summary
    pdf.section_title(1, "Executive Summary")
    exec_summary = data["section_1_executive_summary"]
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(
        0,
        5,
        f"This validation audit performed an automated multi-dimensional evaluation of {model.name} against test dataset {dataset.name}. "
        f"Overall Health Score is {exec_summary['overall_health_score']}/100 ({exec_summary['health_status']}). "
        f"A total of {exec_summary['total_findings']} findings were logged across 7 validation pillars in {exec_summary['validation_duration_seconds']}s.\n"
        f"Verdict: {exec_summary['summary_verdict']}",
    )
    pdf.ln(2)

    # Severity Summary Table
    pdf.table_header([("Severity Level", 50), ("Finding Count", 40), ("Status Description", 100)])
    sev_map = exec_summary["findings_by_severity"]
    sev_rows = [
        ("CRITICAL", str(sev_map["critical"]), "Immediate blocker; model non-conformity"),
        ("HIGH", str(sev_map["high"]), "High risk; requires remediation prior to deployment"),
        ("MEDIUM", str(sev_map["medium"]), "Moderate risk; review and optimize"),
        ("LOW / INFO", str(sev_map["low"] + sev_map["info"]), "Informational findings & optimization suggestions"),
    ]
    for i, (lvl, cnt, desc) in enumerate(sev_rows):
        pdf.table_row([(lvl, 50, "L"), (cnt, 40, "C"), (desc, 100, "L")], fill=(i % 2 == 1))
    pdf.ln(2)

    # 2 & 3. Model & Dataset Information
    pdf.section_title(2, "Model & Dataset Lineage")
    pdf.table_header([("Asset Dimension", 60), ("Model Details", 65), ("Dataset Details", 65)])
    m_info = data["section_2_model_information"]
    d_info = data["section_3_dataset_information"]
    lineage_rows = [
        ("Name", m_info["model_name"][:30], d_info["dataset_name"][:30]),
        ("Type / Columns", f"{m_info['model_type']} ({m_info['framework']})", f"{d_info['row_count']} rows, {d_info['column_count']} cols"),
        ("Target Column", "N/A (Learned Outcome)", str(d_info["target_column"] or "None")),
        ("Feature Count", f"{m_info['feature_count'] or 'N/A'} features", f"{len(d_info['feature_columns'] or [])} recognized features"),
        ("Sensitive Attributes", "None configured", ", ".join(d_info["sensitive_columns"] or []) or "None"),
    ]
    for i, (dim, m_val, d_val) in enumerate(lineage_rows):
        pdf.table_row([(dim, 60, "L"), (m_val, 65, "L"), (d_val, 65, "L")], fill=(i % 2 == 1))
    pdf.ln(2)

    # 4. Overall Health Score & Category Ratings
    pdf.section_title(4, "Health Score & Pillar Ratings")
    cat_scores = data["section_4_overall_health_score"]["category_scores"]
    pdf.table_header([("Validation Pillar", 70), ("Score (0-100)", 40), ("Evaluation Status", 80)])
    pillar_names = [
        ("performance", "Performance & Predictive Accuracy"),
        ("data_quality", "Data Quality & Missingness Integrity"),
        ("fairness", "Fairness & Disparate Impact Audit"),
        ("drift", "Feature & Prediction Drift"),
        ("robustness", "Robustness & Noise Perturbation"),
        ("explainability", "Feature Explainability & Attribution"),
        ("reliability", "Reliability & Calibration"),
    ]
    for i, (k, label) in enumerate(pillar_names):
        sc = cat_scores.get(k)
        sc_str = f"{sc:.1f}" if isinstance(sc, (int, float)) else "N/A"
        if isinstance(sc, (int, float)):
            st = "OPTIMAL" if sc >= 85 else ("WARNING" if sc >= 70 else "CRITICAL")
        else:
            st = "NOT EVALUATED"
        pdf.table_row([(label, 70, "L"), (sc_str, 40, "C"), (st, 80, "L")], fill=(i % 2 == 1))
    pdf.ln(2)

    # New Page for Modules 5 - 11
    pdf.add_page()

    # 5. Performance
    pdf.section_title(5, "Performance & Accuracy Metrics (Module 1)")
    p_metrics = data["section_5_performance"]["metrics"]
    if "r2" in p_metrics:
        # Regression metrics
        pdf.table_header([("Regression Metric Name", 60), ("Calculated Value", 40), ("Guideline / Benchmark", 90)])
        p_rows = [
            ("R² Explanatory Power", str(p_metrics.get("r2", "N/A")), "R² >= 0.70 desired (flagged if < 0.50)"),
            ("Mean Absolute Error (MAE)", str(p_metrics.get("mae", "N/A")), "Lower is better; average error magnitude"),
            ("Root Mean Squared Error (RMSE)", str(p_metrics.get("rmse", "N/A")), "Penalizes large outlier errors"),
            ("Mean Absolute Percentage Error (MAPE)", f"{p_metrics.get('mape', 'N/A')}%" if p_metrics.get("mape") is not None else "N/A", "MAPE <= 20% well-fitted"),
        ]
    else:
        # Classification metrics
        pdf.table_header([("Classification Metric Name", 60), ("Calculated Value", 40), ("Guideline / Benchmark", 90)])
        p_rows = [
            ("Accuracy", str(p_metrics.get("accuracy", "N/A")), "Baseline accuracy threshold"),
            ("F1 Macro", str(p_metrics.get("f1_macro", "N/A")), "Macro balanced F1 score across classes"),
            ("ROC-AUC", str(p_metrics.get("roc_auc", "N/A")), "Discriminative area under ROC curve"),
            ("PR-AUC", str(p_metrics.get("pr_auc", "N/A")), "Precision-recall curve area"),
        ]
    for i, (m, v, b) in enumerate(p_rows):
        pdf.table_row([(m, 60, "L"), (v, 40, "C"), (b, 90, "L")], fill=(i % 2 == 1))
    pdf.ln(2)

    # Prediction Diagnostics Sample Table (Deepchecks Reference)
    pred_info = p_metrics.get("predictions", {})
    sample_rows = pred_info.get("sample_rows", [])
    if sample_rows:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 5, "Sample Row-Level Prediction Comparisons:", new_x="LMARGIN", new_y="NEXT")
        if "r2" in p_metrics:
            pdf.table_header([("Row #", 25), ("Actual Value", 45), ("Predicted Value", 45), ("Absolute Error", 40), ("Percent Error", 35)])
            for j, s_row in enumerate(sample_rows[:6]):
                pdf.table_row([
                    (str(s_row.get("row_index", j)), 25, "C"),
                    (str(s_row.get("actual", "")), 45, "C"),
                    (str(s_row.get("predicted", "")), 45, "C"),
                    (str(s_row.get("abs_error", "")), 40, "C"),
                    (f"{s_row.get('pct_error', '')}%" if s_row.get("pct_error") is not None else "N/A", 35, "C"),
                ], fill=(j % 2 == 1))
        else:
            pdf.table_header([("Row #", 25), ("Actual Label", 50), ("Predicted Label", 50), ("Status", 35), ("Top Probability", 30)])
            for j, s_row in enumerate(sample_rows[:6]):
                st_label = "CORRECT" if s_row.get("correct") else "MISCLASSIFIED"
                probs = s_row.get("probabilities")
                top_p = f"{max(probs.values()):.2f}" if probs else "N/A"
                pdf.table_row([
                    (str(s_row.get("row_index", j)), 25, "C"),
                    (str(s_row.get("actual", "")), 50, "C"),
                    (str(s_row.get("predicted", "")), 50, "C"),
                    (st_label, 35, "C"),
                    (top_p, 30, "C"),
                ], fill=(j % 2 == 1))
        pdf.ln(2)

    # 6. Data Quality
    pdf.section_title(6, "Data Quality & Schema Health (Module 2)")
    dq_metrics = data["section_6_data_quality"]["metrics"]
    pdf.table_header([("Quality Check", 60), ("Observed Value", 40), ("Status", 90)])
    dq_rows = [
        ("Missing Values Total", str(dq_metrics.get("missing_values_count", "0")), "Zero missingness preferred"),
        ("Duplicate Rows", str(dq_metrics.get("duplicate_rows_count", "0")), "Zero duplicates required"),
        ("Target Leakage Columns", str(len(dq_metrics.get("leakage_features", []))), "High correlation leakage checks"),
        ("Constant Features", str(len(dq_metrics.get("constant_features", []))), "Zero-variance features flag"),
    ]
    for i, (chk, obs, st) in enumerate(dq_rows):
        pdf.table_row([(chk, 60, "L"), (obs, 40, "C"), (st, 90, "L")], fill=(i % 2 == 1))
    pdf.ln(2)

    # 7. Fairness
    pdf.section_title(7, "Fairness & Disparate Impact (Module 3)")
    f_metrics = data["section_7_fairness"]["metrics"]
    f_attrs = f_metrics.get("attributes", {})
    if f_metrics.get("_status") in ("not_evaluated", "unsupported", "skipped") or not f_attrs:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        expl = f_metrics.get("_explanation") or "Prerequisites missing: Sensitive demographic attributes were not configured or detected for this validation run."
        status_lbl = f_metrics.get("_status_label", "NOT EVALUATED")
        pdf.multi_cell(0, 5, pdf.clean_text(f"[{status_lbl}] {expl}"))
    else:
        pdf.table_header([("Sensitive Attribute", 50), ("Disparate Impact", 40), ("Parity Diff", 40), ("Equal Opportunity", 60)])
        for i, (attr_name, attr_data) in enumerate(f_attrs.items()):
            di_str = str(attr_data.get("disparate_impact_ratio", "N/A"))
            dp_str = str(attr_data.get("demographic_parity_difference", "N/A"))
            eo_str = str(attr_data.get("equal_opportunity_difference", "N/A"))
            pdf.table_row([
                (attr_name[:22], 50, "L"),
                (di_str, 40, "C"),
                (dp_str, 40, "C"),
                (eo_str, 60, "C"),
            ], fill=(i % 2 == 1))
    pdf.ln(2)

    # 8. Drift
    pdf.section_title(8, "Data & Prediction Drift (Module 4)")
    d_metrics = data["section_8_drift"]["metrics"]
    if d_metrics.get("_status") in ("not_evaluated", "unsupported", "skipped") or not d_metrics:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        expl = d_metrics.get("_explanation") or "Prerequisites missing: Baseline reference dataset was not provided for drift evaluation."
        status_lbl = d_metrics.get("_status_label", "NOT EVALUATED")
        pdf.multi_cell(0, 5, pdf.clean_text(f"[{status_lbl}] {expl}"))
    else:
        is_split = d_metrics.get("is_sequential_partition", False)
        mode_str = "Internal Split Diagnostic (Exploratory)" if is_split else "Reference Baseline Comparison"
        pdf.table_header([("Drift Indicator", 60), ("Score", 40), ("Interpretation / Benchmark", 90)])
        dr_rows = [
            ("Evaluation Mode", mode_str, "Separate baseline required for verified production drift"),
            ("Overall Drift Status", str(d_metrics.get("overall_drift_status", "stable")).upper(), "PSI < 0.10 Stable | 0.10-0.25 Moderate | >= 0.25 High"),
            ("Drifted Features Count", f"{d_metrics.get('drifted_features_count', 0)} / {d_metrics.get('total_features_evaluated', 0)}", f"{d_metrics.get('drifted_features_percentage', 0)}% of features flagged"),
            ("Max Population Stability Index", str(d_metrics.get("max_psi", 0.0)), "PSI < 0.10 (Stable), PSI >= 0.25 (High Drift)"),
            ("Prediction Output Shift", str(d_metrics.get("concept_drift", {}).get("concept_drift_detected", False)), f"Pred PSI: {d_metrics.get('concept_drift', {}).get('prediction_psi', 'N/A')}"),
        ]
        for i, (m, v, b) in enumerate(dr_rows):
            pdf.table_row([(m, 60, "L"), (v, 40, "C"), (b, 90, "L")], fill=(i % 2 == 1))
    pdf.ln(2)

    # 9. Robustness
    pdf.section_title(9, "Adversarial Robustness & Noise Sensitivity (Module 5)")
    r_metrics = data["section_9_robustness"]["metrics"]
    pdf.table_header([("Perturbation Test", 60), ("Flip Rate / Delta", 40), ("Stability Assessment", 90)])
    rb_rows = [
        ("5% Gaussian Noise Flip Rate", f"{r_metrics.get('flip_rate_5pct', 0.0):.2%}", "Perturbation stability (< 10% expected)"),
        ("10% Gaussian Noise Flip Rate", f"{r_metrics.get('flip_rate_10pct', 0.0):.2%}", "Moderate perturbation sensitivity"),
        ("Missingness Resilience (10%)", f"{r_metrics.get('accuracy_drop_10pct_missing', 0.0):.2%}", "Drop in performance when features masked"),
        ("Robustness Classification", str(r_metrics.get("robustness_tier", "RESILIENT")), "Overall perturbation classification"),
    ]
    for i, (m, v, b) in enumerate(rb_rows):
        pdf.table_row([(m, 60, "L"), (v, 40, "C"), (b, 90, "L")], fill=(i % 2 == 1))
    pdf.ln(2)

    # 10 & 11. Explainability & Reliability
    pdf.add_page()
    pdf.section_title(10, "Explainability & Feature Attribution (Module 6)")
    exp_metrics = data["section_10_explainability"]["metrics"]
    pdf.table_header([("Attribution Dimension", 60), ("Observed Value", 40), ("Design Interpretation", 90)])
    exp_rows = [
        ("Attribution Method", str(exp_metrics.get("attribution_method", "SHAP")).replace("_", " ").upper(), "Explainability framework (SHAP Tree / Linear / Permutation)"),
        ("Primary Feature Share", f"{exp_metrics.get('top_feature_share_percentage', 0.0):.1f}%", "< 60% balanced (flagged if > 60%)"),
        ("Top Feature Name", str(exp_metrics.get("top_feature_name", "None")), "Dominant single driver of decisions"),
        ("Top 3 Features Share", f"{exp_metrics.get('top_3_share_percentage', 0.0):.1f}%", "Concentration of attribution"),
        ("Dead / Zero-Impact Features", str(exp_metrics.get("zero_impact_features_count", 0)), "Features with zero attribution"),
    ]
    for i, (m, v, b) in enumerate(exp_rows):
        pdf.table_row([(m, 60, "L"), (v, 40, "C"), (b, 90, "L")], fill=(i % 2 == 1))
    pdf.ln(2)

    pdf.section_title(11, "Probability Calibration & Reliability (Module 7)")
    rel_metrics = data["section_11_reliability"]["metrics"]
    pdf.table_header([("Reliability Metric", 60), ("Calculated Value", 40), ("Calibration Benchmark", 90)])
    rel_rows = [
        ("Expected Calibration Error (ECE)", str(rel_metrics.get("expected_calibration_error", "N/A")), "<= 0.08 well-calibrated (flag if > 0.15)"),
        ("Brier Score", str(rel_metrics.get("brier_score", "N/A")), "<= 0.25 acceptable probabilistic divergence"),
        ("Mean Prediction Confidence", str(rel_metrics.get("mean_prediction_confidence", "N/A")), "Average certainty across test set"),
        ("Overconfident Misclassifications", str(rel_metrics.get("overconfident_errors_count", 0)), "Mistakes with >= 90% confidence"),
    ]
    for i, (m, v, b) in enumerate(rel_rows):
        pdf.table_row([(m, 60, "L"), (v, 40, "C"), (b, 90, "L")], fill=(i % 2 == 1))
    pdf.ln(2)

    # 12. Findings Inventory
    pdf.section_title(12, "Findings Inventory")
    inv = data["section_12_findings_inventory"]
    if inv:
        pdf.table_header([("Severity", 25), ("Category", 35), ("Title", 60), ("Metric", 35), ("Threshold", 35)])
        for i, f in enumerate(inv[:15]):  # Up to top 15 findings
            sev = f["severity"].upper()
            pdf.table_row([
                (sev, 25, "C"),
                (f["category"][:18], 35, "L"),
                (f["title"][:30], 60, "L"),
                (str(f["metric_value"])[:18] if f["metric_value"] is not None else "-", 35, "C"),
                (str(f["threshold"] or "-")[:18], 35, "C"),
            ], fill=(i % 2 == 1))
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "No adverse findings logged during validation run.", border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # 13, 14, 15. Recommendations, Methodology & Limitations
    pdf.add_page()
    pdf.section_title(13, "Prioritized Actionable Recommendations")
    recs = data["section_13_actionable_recommendations"]
    if recs:
        for r in recs[:6]:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(15, 23, 42)
            header_txt = f"[{r['severity'].upper()}] {r['category'].upper()}: {r['associated_finding']}"
            pdf.cell(0, 5, pdf.clean_text(header_txt), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(51, 65, 85)
            action_txt = f"Action: {r['action']}"
            pdf.multi_cell(0, 4, pdf.clean_text(action_txt))
            pdf.ln(1)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "No corrective actions required.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.section_title(14, "Validation Methodology & Standards")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(
        0,
        4,
        "Validation evaluations are performed strictly using deterministic algorithms across 7 standard pillars: "
        "Classification/Regression Performance, Data Hygiene & Leakage, Demographic Subgroup Parity, "
        "Two-Sample KS & PSI Drift Detection, Adversarial Gaussian Noise Flip Rates, Feature Attribution Concentration, "
        "and 10-bin Expected Calibration Error. Thresholds comply with EEOC Uniform Guidelines and industry ML assurance norms.",
    )
    pdf.ln(2)

    pdf.section_title(15, "Operational Boundaries & Legal Disclaimer")
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(
        0,
        4,
        "DISCLAIMER: This automated assurance report is generated for diagnostic, validation, and governance purposes. "
        "Metrics reflect empirical evaluation on the submitted test dataset and model file at execution time. "
        "Passing evaluation on this dataset confirms performance under observed data distributions, but does not guarantee "
        "flawless predictions or error-free execution on future, out-of-distribution, or adversarial production inputs. "
        "AI/ML Validation Platform 2026.",
    )

    # Return raw PDF bytes
    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()

