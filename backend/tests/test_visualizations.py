"""Tests for the Interactive Visualization and Visual Analytics Engine."""
import uuid
import pytest
from app.models.validation_run import ValidationRun, RunStatus
from app.services.visualization_service import generate_validation_charts, chart_to_csv
from app.services.report_service import build_report_data
from app.models.ml_model import MLModel
from app.models.dataset import Dataset


def test_classification_charts_generation():
    """Test generating visualization charts from classification validation run."""
    run = ValidationRun(
        id=uuid.uuid4(),
        run_number="RUN-TEST-001",
        project_id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        status=RunStatus.COMPLETED,
        overall_score=88.5,
        config_json={
            "metrics": {
                "performance": {
                    "accuracy": 0.95,
                    "balanced_accuracy": 0.94,
                    "num_classes": 3,
                    "confusion_matrix": {
                        "labels": ["0", "1", "2"],
                        "matrix": [[10, 0, 0], [0, 9, 1], [0, 1, 9]],
                        "normalized": [[1.0, 0.0, 0.0], [0.0, 0.9, 0.1], [0.0, 0.1, 0.9]],
                    },
                    "per_class_metrics": {
                        "0": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "support": 10},
                        "1": {"precision": 0.9, "recall": 0.9, "f1_score": 0.9, "support": 10},
                        "2": {"precision": 0.9, "recall": 0.9, "f1_score": 0.9, "support": 10},
                    },
                },
                "data_quality": {
                    "row_count": 30,
                    "missing_values": {"sepal_length": {"count": 0, "percentage": 0.0}},
                    "feature_stats": {
                        "sepal_length": {"unique_count": 25, "variance": 0.68, "is_numeric": True},
                    },
                },
                "drift": {
                    "current_row_count": 30,
                    "is_sequential_partition": True,
                    "feature_drift": {
                        "sepal_length": {
                            "psi": 0.04,
                            "ks_statistic": 0.12,
                            "drift_status": "stable",
                            "drift_detected": False,
                        },
                        "petal_width": {
                            "psi": 0.18,
                            "ks_statistic": 0.28,
                            "drift_status": "moderate",
                            "drift_detected": True,
                        },
                    },
                },
                "explainability": {
                    "method": "tree_feature_importances",
                    "top_feature": "petal_length",
                    "top_feature_share": 44.5,
                    "features_for_80_percent": 2,
                    "features": [
                        {"feature": "petal_length", "relative_percentage": 44.5, "importance_score": 0.445, "cumulative_percentage": 44.5, "rank": 1},
                        {"feature": "petal_width", "relative_percentage": 42.0, "importance_score": 0.420, "cumulative_percentage": 86.5, "rank": 2},
                    ],
                },
                "robustness": {
                    "samples_evaluated": 30,
                    "overall_flip_rate": 0.05,
                    "perturbation_curve": [
                        {"noise_pct": 1, "flip_rate": 0.0, "metric_degradation": 0.0, "flip_count": 0},
                        {"noise_pct": 5, "flip_rate": 0.033, "metric_degradation": 0.033, "flip_count": 1},
                        {"noise_pct": 10, "flip_rate": 0.067, "metric_degradation": 0.067, "flip_count": 2},
                        {"noise_pct": 20, "flip_rate": 0.10, "metric_degradation": 0.10, "flip_count": 3},
                    ],
                },
            },
            "validator_statuses": {
                "performance": {"status": "success", "status_label": "PASSED", "score": 95.0},
                "fairness": {"status": "not_evaluated", "status_label": "NOT_EVALUATED", "explanation": "No sensitive attributes"},
                "reliability": {"status": "not_evaluated", "status_label": "NOT_EVALUATED", "explanation": "predict_proba unavailable"},
            },
        },
    )

    charts = generate_validation_charts(run)
    assert len(charts) > 0

    chart_ids = [c.chart_id for c in charts]
    assert "perf_confusion_matrix" in chart_ids
    assert "perf_per_class_metrics" in chart_ids
    assert "dq_missing_values" in chart_ids
    assert "drift_feature_psi" in chart_ids
    assert "explainability_feature_importance" in chart_ids
    assert "robustness_perturbation_curve" in chart_ids

    # Unevaluated modules return NOT_EVALUATED status
    fairness_chart = next(c for c in charts if c.category == "fairness")
    assert fairness_chart.status == "NOT_EVALUATED"

    reliability_chart = next(c for c in charts if c.category == "reliability")
    assert reliability_chart.status == "NOT_EVALUATED"

    # Verify PSI thresholds
    drift_chart = next(c for c in charts if c.chart_id == "drift_feature_psi")
    assert any(t.value == 0.10 for t in drift_chart.thresholds)
    assert any(t.value == 0.25 for t in drift_chart.thresholds)


def test_regression_charts_generation():
    """Test generating regression charts (actual vs predicted parity, residuals)."""
    run = ValidationRun(
        id=uuid.uuid4(),
        run_number="RUN-REG-001",
        project_id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        status=RunStatus.COMPLETED,
        config_json={
            "metrics": {
                "performance": {
                    "r2": 0.85,
                    "mae": 2.1,
                    "rmse": 3.4,
                    "predictions": {
                        "sample_rows": [
                            {"row_index": 0, "actual": 10.0, "predicted": 11.2, "abs_error": 1.2},
                            {"row_index": 1, "actual": 20.0, "predicted": 19.5, "abs_error": 0.5},
                            {"row_index": 2, "actual": 15.0, "predicted": 16.0, "abs_error": 1.0},
                        ],
                    },
                },
            },
        },
    )

    charts = generate_validation_charts(run)
    chart_ids = [c.chart_id for c in charts]
    assert "perf_actual_vs_predicted" in chart_ids
    assert "perf_residual_distribution" in chart_ids

    scatter_chart = next(c for c in charts if c.chart_id == "perf_actual_vs_predicted")
    assert len(scatter_chart.data) == 3
    assert scatter_chart.chart_type == "scatter"


def test_chart_to_csv_export():
    """Test converting chart data series into CSV format."""
    run = ValidationRun(
        id=uuid.uuid4(),
        run_number="RUN-CSV-001",
        project_id=uuid.uuid4(),
        model_id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        status=RunStatus.COMPLETED,
        config_json={
            "metrics": {
                "drift": {
                    "feature_drift": {
                        "f1": {"psi": 0.05, "drift_status": "stable", "drift_detected": False},
                        "f2": {"psi": 0.28, "drift_status": "significant", "drift_detected": True},
                    },
                },
            },
        },
    )
    charts = generate_validation_charts(run)
    psi_chart = next(c for c in charts if c.chart_id == "drift_feature_psi")

    csv_text = chart_to_csv(psi_chart)
    assert "feature" in csv_text
    assert "psi" in csv_text
    assert "f1" in csv_text
    assert "f2" in csv_text
    assert "0.28" in csv_text


def test_visual_analytics_embedded_in_report():
    """Test that build_report_data includes visual_analytics charts dictionary."""
    model = MLModel(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        name="Test Model",
        filename="model.joblib",
        file_path="dummy.joblib",
        file_size=1024,
        model_type="classification",
        framework="scikit-learn",
        feature_count=4,
    )
    dataset = Dataset(
        id=uuid.uuid4(),
        project_id=uuid.uuid4(),
        name="Test Dataset",
        filename="data.csv",
        file_path="dummy.csv",
        file_size=2048,
        row_count=100,
        column_count=5,
        target_column="target",
    )
    run = ValidationRun(
        id=uuid.uuid4(),
        run_number="RUN-REP-001",
        project_id=uuid.uuid4(),
        model_id=model.id,
        dataset_id=dataset.id,
        status=RunStatus.COMPLETED,
        overall_score=92.0,
        config_json={"metrics": {"performance": {"accuracy": 0.92}}},
    )

    report = build_report_data(run, model, dataset, [])
    assert "visual_analytics" in report
    assert report["visual_analytics"]["total_charts"] >= 0
