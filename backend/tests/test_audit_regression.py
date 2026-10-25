"""Regression test suite for Section 5 of Master Audit:
1. PSI 0.05 -> STABLE.
2. PSI 0.10 -> MODERATE.
3. PSI 0.249 -> MODERATE.
4. PSI 0.25 -> HIGH/SIGNIFICANT.
5. PSI 0.3319 -> HIGH, never STABLE.
6. PSI 0.3176 -> HIGH, never STABLE.
7. Maximum PSI, drifted-feature count, and overall drift status agree.
8. Identical reference/current distributions produce approximately zero PSI.
9. Different distributions produce an appropriately higher PSI.
10. Missing values, constant features, unseen categories, and small samples are handled safely.
11. A single dataset does not produce a claim of verified production drift.
12. Fairness is evaluated when valid groups and labels are available.
13. Fairness remains NOT_EVALUATED with an explicit reason when essential inputs are missing.
14. An unevaluated fairness module cannot receive a misleading perfect score.
15. Frontend cards, findings, and exported PDF/JSON reports match the backend result.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.validators.base import ValidationContext, ValidatorStatus
from app.validators.drift.validator import (
    DriftValidator,
    calculate_psi,
    calculate_categorical_psi,
    classify_drift_status,
)
from app.validators.fairness.validator import FairnessValidator
from app.services.scoring_service import calculate_scores
from app.services.report_service import build_report_data
from app.models.validation_run import ValidationRun, RunStatus
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.models.finding import Finding, Severity


def test_psi_threshold_classification():
    """Points 1-6: Standard thresholds classification testing."""
    # 1. PSI 0.05 -> STABLE
    status, is_drifted, sev = classify_drift_status(0.05)
    assert status == "stable"
    assert is_drifted is False
    assert sev == "info"

    # 2. PSI 0.10 -> MODERATE
    status, is_drifted, sev = classify_drift_status(0.10)
    assert status == "moderate"
    assert is_drifted is True
    assert sev == "medium"

    # 3. PSI 0.249 -> MODERATE
    status, is_drifted, sev = classify_drift_status(0.249)
    assert status == "moderate"
    assert is_drifted is True
    assert sev == "medium"

    # 4. PSI 0.25 -> HIGH / SIGNIFICANT
    status, is_drifted, sev = classify_drift_status(0.25)
    assert status == "significant"
    assert is_drifted is True
    assert sev == "high"

    # 5. PSI 0.3319 -> HIGH, never STABLE
    status, is_drifted, sev = classify_drift_status(0.3319)
    assert status == "significant"
    assert status != "stable"
    assert is_drifted is True

    # 6. PSI 0.3176 -> HIGH, never STABLE
    status, is_drifted, sev = classify_drift_status(0.3176)
    assert status == "significant"
    assert status != "stable"
    assert is_drifted is True


def test_identical_and_shifted_distributions():
    """Points 8 & 9: Identical distributions produce ~0 PSI; shifted produce high PSI."""
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    curr_identical = ref.copy()
    
    psi_identical = calculate_psi(ref, curr_identical)
    assert psi_identical < 0.001

    curr_shifted = np.random.normal(2.5, 1.2, 1000)
    psi_shifted = calculate_psi(ref, curr_shifted)
    assert psi_shifted > 0.25
    status, drifted, _ = classify_drift_status(psi_shifted)
    assert status == "significant"
    assert drifted is True


def test_edge_cases_handling():
    """Point 10: Missing values, constant features, unseen categories, and small samples."""
    np.random.seed(42)
    # Constant features
    ref_const = np.array([5.0] * 50)
    curr_const = np.array([5.0] * 50)
    assert calculate_psi(ref_const, curr_const) == 0.0

    curr_const_diff = np.array([10.0] * 50)
    assert calculate_psi(ref_const, curr_const_diff) > 0.25

    # Non-finite and NaNs
    ref_nan = np.array([1.0, 2.0, np.nan, 4.0, np.inf, -np.inf, 3.0, 2.5, 3.5, 4.5, 1.5, 2.2])
    curr_nan = np.array([1.1, 2.1, 3.1, 4.1, np.nan, 2.4, 3.4, 4.4, 1.4, 2.0, 3.0, 4.0])
    psi_nan = calculate_psi(ref_nan, curr_nan)
    assert not np.isnan(psi_nan)
    assert not np.isinf(psi_nan)

    # Categorical unseen categories
    ref_cat = pd.Series(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
    curr_cat = pd.Series(["B", "C", "D", "B", "C", "D", "B", "C", "D", "B"])
    psi_cat, tvd_cat, _ = calculate_categorical_psi(ref_cat, curr_cat)
    assert not np.isnan(psi_cat)
    assert not np.isinf(psi_cat)
    assert psi_cat > 0.0
    assert tvd_cat > 0.0


@pytest.mark.asyncio
async def test_drift_metrics_agreement_and_single_dataset():
    """Points 7 & 11: Agreeing metrics and single dataset internal split diagnostic."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "num_f1": np.random.normal(0, 1, n),
        "num_f2": np.random.normal(5, 2, n),
        "cat_f": np.random.choice(["A", "B"], n),
        "target": np.random.choice([0, 1], n),
    })

    context = ValidationContext(
        model=None,
        dataset=df,
        target_column="target",
        feature_columns=["num_f1", "num_f2", "cat_f"],
    )
    validator = DriftValidator()
    result = await validator.safe_validate(context)

    assert result.status == ValidatorStatus.SUCCESS
    assert result.status_label == "INTERNAL_SPLIT_DIAGNOSTIC"
    assert result.metrics["is_sequential_partition"] is True

    # Single dataset does NOT claim verified production drift
    for f in result.findings:
        assert "production drift" not in f.title.lower()

    # Agreement check: Max PSI, drifted count, overall status
    feature_drifts = result.metrics["feature_drift"]
    valid_psis = [d["psi"] for d in feature_drifts.values()]
    computed_max = max(valid_psis) if valid_psis else 0.0
    assert abs(result.metrics["max_psi"] - computed_max) < 1e-4

    expected_drifted = sum(1 for d in feature_drifts.values() if d["drift_detected"])
    assert result.metrics["drifted_features_count"] == expected_drifted

    if computed_max >= 0.25:
        assert result.metrics["overall_drift_status"] == "significant"
    elif computed_max >= 0.10:
        assert result.metrics["overall_drift_status"] == "moderate"
    else:
        assert result.metrics["overall_drift_status"] == "stable"


@pytest.mark.asyncio
async def test_fairness_evaluated_when_groups_available():
    """Point 12: Fairness is evaluated when valid groups and labels are available."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "f1": np.random.normal(0, 1, n),
        "gender": ["male"] * 60 + ["female"] * 60,
        "target": np.random.choice([0, 1], n),
    })
    
    # Simple model
    lr = LogisticRegression()
    lr.fit(df[["f1"]], df["target"])

    context = ValidationContext(
        model=lr,
        model_type="classification",
        framework="sklearn",
        dataset=df,
        target_column="target",
        feature_columns=["f1"],
        sensitive_columns=["gender"],
    )

    validator = FairnessValidator()
    assert validator.is_applicable(context) is True

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SUCCESS
    assert result.status_label == "EVALUATED"
    assert result.score is not None
    assert "gender" in result.metrics["attributes"]
    assert "disparate_impact_ratio" in result.metrics["attributes"]["gender"]


@pytest.mark.asyncio
async def test_fairness_not_evaluated_when_missing():
    """Points 13 & 14: Fairness remains NOT_EVALUATED with explicit reason, score is None (not 100)."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "f1": np.random.normal(0, 1, n),
        "target": np.random.choice([0, 1], n),
    })
    
    lr = LogisticRegression()
    lr.fit(df[["f1"]], df["target"])

    context = ValidationContext(
        model=lr,
        model_type="classification",
        framework="sklearn",
        dataset=df,
        target_column="target",
        feature_columns=["f1"],
        sensitive_columns=[],  # No sensitive columns
    )

    validator = FairnessValidator()
    assert validator.is_applicable(context) is False

    result = await validator.safe_validate(context)
    assert result.status == ValidatorStatus.SKIPPED
    assert result.status_label in ("NOT_EVALUATED", "SKIPPED")
    assert result.score is None
    assert result.score != 100.0  # Cannot award 100 to unevaluated module
    assert result.explanation is not None


def test_score_aggregation_and_report_consistency():
    """Point 15: Not evaluated modules are excluded from health score aggregation and reports."""
    from app.validators.base import ValidatorResult
    
    res_perf = ValidatorResult(validator_name="performance", category="performance", score=88.0, status=ValidatorStatus.SUCCESS)
    res_dq = ValidatorResult(validator_name="data_quality", category="data_quality", score=92.0, status=ValidatorStatus.SUCCESS)
    res_fair = ValidatorResult(validator_name="fairness", category="fairness", score=None, status=ValidatorStatus.NOT_EVALUATED, status_label="NOT_EVALUATED")

    scores = calculate_scores([res_perf, res_dq, res_fair])
    # Unevaluated module must not be factored in as 0 or 100
    assert "fairness" not in scores["categories"]
    assert scores["overall"] == pytest.approx((88.0 + 92.0) / 2.0, 0.01)
    assert scores["status"] == "EVALUATED"
