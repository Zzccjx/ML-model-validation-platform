"""Validation run API routes."""
import uuid
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.models.user import User
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.schemas.validation import (
    ValidationRunCreate,
    ValidationRunResponse,
    FindingResponse,
    FindingListResponse,
    CompatibilityCheckRequest,
    CompatibilityCheckResponse,
    ValidatorReadiness,
)
from app.services import validation_service
from app.services.compatibility_service import check_compatibility
from app.utils.model_utils import load_model_safely
from app.utils.dataset_utils import load_dataset
from app.api.deps import get_current_active_user

router = APIRouter(prefix='/validation', tags=['Validation'])


@router.post('/compatibility-check', response_model=CompatibilityCheckResponse)
async def compatibility_check(
    payload: CompatibilityCheckRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Run pre-flight compatibility and module readiness verification without executing run."""
    model = await db.get(MLModel, payload.model_id)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Model not found')
    
    dataset = await db.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')
    
    ref_dataset = None
    if payload.reference_dataset_id:
        ref_dataset = await db.get(Dataset, payload.reference_dataset_id)

    target_col = payload.target_column or dataset.target_column or ''

    # Run core compatibility verification
    compat_res = check_compatibility(
        model_path=model.file_path,
        dataset_path=dataset.file_path,
        target_column=target_col,
    )

    detected_target = compat_res.dataset_info.get('auto_detected_target') or compat_res.dataset_info.get('target_column')

    # Check validator module readiness
    readiness_list: list[ValidatorReadiness] = []
    
    # 1. Performance
    perf_ready = compat_res.compatible and bool(detected_target)
    readiness_list.append(ValidatorReadiness(
        name='performance',
        ready=perf_ready,
        status_label='Ready' if perf_ready else 'Not Configured (No Ground Truth Target)',
        reason=None if perf_ready else 'Performance evaluation requires ground-truth labels. Other validators will execute normally.'
    ))


    # 2. Data Quality
    readiness_list.append(ValidatorReadiness(
        name='data_quality',
        ready=True,
        status_label='Ready',
        reason='Evaluates data integrity and anomalies'
    ))

    # 3. Fairness
    fairness_ready = bool(payload.sensitive_column and payload.sensitive_column in compat_res.dataset_info.get('columns', []))
    readiness_list.append(ValidatorReadiness(
        name='fairness',
        ready=fairness_ready,
        status_label='Ready' if fairness_ready else 'Requires Sensitive Attribute',
        reason=None if fairness_ready else 'Select a sensitive attribute column (e.g. gender, age_group) to audit group disparities'
    ))

    # 4. Drift
    drift_ready = bool(ref_dataset is not None)
    readiness_list.append(ValidatorReadiness(
        name='drift',
        ready=drift_ready,
        status_label='Ready' if drift_ready else 'Requires Reference Baseline',
        reason=None if drift_ready else 'Select or upload a baseline reference dataset to compare population and prediction drift'
    ))

    # 5. Robustness
    readiness_list.append(ValidatorReadiness(
        name='robustness',
        ready=compat_res.compatible,
        status_label='Ready' if compat_res.compatible else 'Blocked by Incompatibility',
        reason=None if compat_res.compatible else 'Requires executable model predict function'
    ))

    # 6. Explainability
    readiness_list.append(ValidatorReadiness(
        name='explainability',
        ready=compat_res.compatible,
        status_label='Ready' if compat_res.compatible else 'Blocked by Incompatibility',
        reason=None if compat_res.compatible else 'Requires compatible numeric/encoded features'
    ))

    # 7. Reliability
    supports_proba = compat_res.model_info.get('supports_probabilities', False)
    readiness_list.append(ValidatorReadiness(
        name='reliability',
        ready=bool(supports_proba),
        status_label='Ready' if supports_proba else 'Requires Probabilistic Classifier',
        reason=None if supports_proba else 'Model estimator does not provide predict_proba() for probability calibration analysis'
    ))

    return CompatibilityCheckResponse(
        compatible=compat_res.compatible,
        errors=compat_res.messages if not compat_res.compatible else [],
        warnings=compat_res.warnings,
        model_info=compat_res.model_info,
        dataset_info=compat_res.dataset_info,
        validator_readiness=readiness_list,
    )


@router.post('/runs', response_model=ValidationRunResponse, status_code=status.HTTP_201_CREATED)
async def create_validation_run(
    payload: ValidationRunCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create and execute a new validation run."""
    # Verify model exists
    model = await db.get(MLModel, payload.model_id)
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Model not found')
    
    # Verify dataset exists
    dataset = await db.get(Dataset, payload.dataset_id)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')
    
    # Use model's project or fallback
    target_project_id = model.project_id
    
    run_config = payload.config or {}
    if payload.target_column:
        run_config['target_column'] = payload.target_column
    if payload.sensitive_column:
        run_config['sensitive_columns'] = [payload.sensitive_column]
    if payload.selected_validators:
        run_config['selected_validators'] = payload.selected_validators
    if payload.reference_dataset_id:
        run_config['reference_dataset_id'] = str(payload.reference_dataset_id)

    run = await validation_service.create_validation_run(
        db,
        project_id=model.project_id,
        model_id=payload.model_id,
        dataset_id=payload.dataset_id,
        config=run_config,
    )
    return run



@router.get('/runs', response_model=list[ValidationRunResponse])
async def list_validation_runs(
    project_id: uuid.UUID | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List validation runs."""
    return await validation_service.list_validation_runs(db, project_id)


@router.get('/runs/{run_id}', response_model=ValidationRunResponse)
async def get_validation_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a validation run by ID."""
    return await validation_service.get_validation_run(db, run_id)


@router.get('/runs/{run_id}/findings', response_model=FindingListResponse)
async def get_run_findings(
    run_id: uuid.UUID,
    severity: str | None = Query(None),
    category: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get findings for a validation run."""
    findings = await validation_service.get_run_findings(db, run_id, severity, category)
    return FindingListResponse(findings=findings, total=len(findings))


@router.get('/findings', response_model=FindingListResponse)
async def list_findings(
    project_id: uuid.UUID | None = Query(None),
    severity: str | None = Query(None),
    category: str | None = Query(None),
    limit: int = Query(100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List cross-run findings with optional filtering."""
    findings = await validation_service.list_findings(db, project_id, severity, category, limit)
    return FindingListResponse(findings=findings, total=len(findings))
