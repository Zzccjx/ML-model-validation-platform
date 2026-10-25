"""Validation run API routes."""
import uuid
from fastapi import APIRouter, Depends, Query, HTTPException, status, Response
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
    PredictionTableResponse,
    RowPredictionItem,
    ChartsResponse,
    ChartDefinition,
)
from app.services import validation_service
from app.services.visualization_service import generate_validation_charts, chart_to_csv
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


@router.get('/runs/{run_id}/predictions', response_model=PredictionTableResponse)
async def get_run_predictions(
    run_id: uuid.UUID,
    filter: str = Query('all', description="'all', 'incorrect', or 'correct'"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get row-level model prediction comparisons with pagination and filtering."""
    run = await validation_service.get_validation_run(db, run_id)
    cfg = run.config_json or {}
    metrics = cfg.get('metrics', {})
    perf = metrics.get('performance', {})
    pred_data = perf.get('predictions', {})

    task_type = 'regression' if 'r2' in perf else 'classification'
    all_sample_rows: list[dict] = pred_data.get('sample_rows', [])

    # Filter rows if requested
    filtered = []
    for r in all_sample_rows:
        if filter == 'incorrect':
            if task_type == 'classification' and r.get('correct') is False:
                filtered.append(r)
            elif task_type == 'regression' and r.get('abs_error', 0) > 0:
                filtered.append(r)
        elif filter == 'correct':
            if task_type == 'classification' and r.get('correct') is True:
                filtered.append(r)
            elif task_type == 'regression' and r.get('abs_error', 0) == 0:
                filtered.append(r)
        else:
            filtered.append(r)

    paged_items = filtered[offset : offset + limit]

    summary = {
        'total_evaluated': pred_data.get('total_rows', len(all_sample_rows)),
        'correct_count': pred_data.get('correct_rows'),
        'incorrect_count': pred_data.get('incorrect_rows'),
        'misclassification_rate': pred_data.get('misclassification_rate'),
        'mean_error': pred_data.get('mean_error'),
        'max_abs_error': pred_data.get('max_abs_error'),
    }

    return PredictionTableResponse(
        total_rows=len(filtered),
        returned_rows=len(paged_items),
        offset=offset,
        limit=limit,
        task_type=task_type,
        has_probabilities=bool(pred_data.get('has_probabilities', False)),
        summary=summary,
        items=[RowPredictionItem(**item) for item in paged_items],
    )


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


@router.get('/runs/{run_id}/charts', response_model=ChartsResponse)
async def get_run_charts(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Generate structured visual analytics charts for all evaluated validation modules."""
    run = await validation_service.get_validation_run(db, run_id)
    charts = generate_validation_charts(run)
    categories = sorted(list(set(c.category for c in charts)))
    return ChartsResponse(
        run_id=run.id,
        run_number=run.run_number,
        total_charts=len(charts),
        categories=categories,
        charts=charts,
    )


@router.get('/runs/{run_id}/charts/{chart_id}/export-csv')
async def export_chart_csv(
    run_id: uuid.UUID,
    chart_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Export tabular data backing a specific chart as CSV."""
    run = await validation_service.get_validation_run(db, run_id)
    charts = generate_validation_charts(run)
    matching = next((c for c in charts if c.chart_id == chart_id), None)
    if not matching:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chart '{chart_id}' not found for validation run {run_id}",
        )
    csv_content = chart_to_csv(matching)
    filename = f"{run.run_number}_{chart_id}.csv"
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
