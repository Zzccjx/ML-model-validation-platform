import uuid
from datetime import datetime, timezone
import pytest

from app.models.validation_run import ValidationRun, RunStatus
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.models.finding import Finding, Severity
from app.services.report_service import generate_json_report, generate_pdf_report


@pytest.fixture
def sample_report_fixtures():
    run_id = uuid.uuid4()
    project_id = uuid.uuid4()
    model_id = uuid.uuid4()
    dataset_id = uuid.uuid4()

    model = MLModel(
        id=model_id,
        project_id=project_id,
        name="Credit_Risk_Classifier_v1",
        filename="model.joblib",
        file_path="dummy_path.joblib",
        file_size=102400,
        model_type="classification",
        framework="scikit-learn",
        feature_count=10,
        class_count=2,
        classes_json={"0": "Repaid", "1": "Default"},
    )

    dataset = Dataset(
        id=dataset_id,
        project_id=project_id,
        name="Credit_Validation_2026.csv",
        filename="Credit_Validation_2026.csv",
        file_path="dummy_dataset.csv",
        file_size=204800,
        row_count=1000,
        column_count=11,
        target_column="default_status",
        feature_columns_json=[f"feat_{i}" for i in range(10)],
        sensitive_columns_json=["gender", "age_group"],
    )

    run = ValidationRun(
        id=run_id,
        run_number="V-2026-0042",
        project_id=project_id,
        model_id=model_id,
        dataset_id=dataset_id,
        status=RunStatus.COMPLETED,
        overall_score=88.5,
        category_scores_json={
            "performance": 92.0,
            "data_quality": 95.0,
            "fairness": 78.0,
            "drift": 84.0,
            "robustness": 90.0,
            "explainability": 86.0,
            "reliability": 91.0,
        },
        config_json={
            "metrics": {
                "performance": {"accuracy": 0.88, "f1_macro": 0.85, "roc_auc": 0.91},
                "data_quality": {"missing_values_count": 0, "duplicate_rows_count": 0},
                "fairness": {"disparate_impact_ratio": 0.82, "demographic_parity_difference": 0.08},
                "drift": {"drifted_features_count": 1, "max_psi": 0.09, "prediction_drift_detected": False},
                "robustness": {"flip_rate_5pct": 0.03, "robustness_tier": "RESILIENT"},
                "explainability": {"top_feature_name": "feat_0", "top_feature_share_percentage": 25.4, "top_3_share_percentage": 58.0},
                "reliability": {"expected_calibration_error": 0.042, "brier_score": 0.118, "overconfident_errors_count": 2},
            }
        },
        duration_seconds=1.45,
    )

    findings = [
        Finding(
            id=uuid.uuid4(),
            validation_run_id=run_id,
            severity=Severity.HIGH,
            category="fairness",
            title="Subgroup Disparate Impact Near Boundary",
            description="Subgroup approval disparity is close to EEOC 0.80 boundary.",
            metric_name="disparate_impact_ratio",
            metric_value=0.82,
            threshold="0.80",
            recommendation="Review threshold assignment for underrepresented demographic subgroup.",
            validator_name="fairness",
            validator_version="1.0.0",
        ),
        Finding(
            id=uuid.uuid4(),
            validation_run_id=run_id,
            severity=Severity.INFO,
            category="reliability",
            title="Well-Calibrated Probability Distribution",
            description="ECE is 0.042, indicating trustworthy probabilistic estimates.",
            metric_name="expected_calibration_error",
            metric_value=0.042,
            threshold="0.08",
            recommendation="Retain probability calibration standards.",
            validator_name="reliability",
            validator_version="1.0.0",
        ),
    ]

    return run, model, dataset, findings


def test_json_report_generation(sample_report_fixtures):
    run, model, dataset, findings = sample_report_fixtures
    report = generate_json_report(run, model, dataset, findings)

    assert isinstance(report, dict)
    assert "report_metadata" in report
    assert report["report_metadata"]["run_number"] == "V-2026-0042"
    
    # Check 15 sections presence
    for sec_num in range(1, 16):
        matching_keys = [k for k in report.keys() if k.startswith(f"section_{sec_num}_")]
        assert len(matching_keys) == 1, f"Missing section {sec_num} in JSON report"

    assert report["section_1_executive_summary"]["overall_health_score"] == 88.5
    assert report["section_1_executive_summary"]["health_status"] == "HEALTHY"
    assert report["section_4_overall_health_score"]["category_scores"]["reliability"] == 91.0
    assert len(report["section_12_findings_inventory"]) == 2
    assert len(report["section_13_actionable_recommendations"]) >= 1


def test_pdf_report_generation(sample_report_fixtures):
    run, model, dataset, findings = sample_report_fixtures
    pdf_bytes = generate_pdf_report(run, model, dataset, findings)

    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert len(pdf_bytes) > 1000
    # PDF starts with magic header %PDF-
    assert pdf_bytes.startswith(b"%PDF-")
