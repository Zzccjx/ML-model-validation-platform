"""Validation run orchestration service."""
import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.validation_run import ValidationRun, RunStatus
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.models.finding import Finding, Severity
from app.services.compatibility_service import check_compatibility
from app.services.scoring_service import calculate_scores
from app.validators.registry import get_registry
from app.validators.base import ValidationContext, ValidationFinding
from app.utils.model_utils import load_model_safely, detect_model_type, detect_framework
from app.utils.dataset_utils import load_dataset

logger = logging.getLogger('ai_validation')


async def generate_run_number(db: AsyncSession) -> str:
    """Generate a unique run number like V-2026-0001."""
    year = datetime.now(timezone.utc).year
    prefix = f'V-{year}-'
    stmt = select(func.count()).where(ValidationRun.run_number.like(f'{prefix}%'))
    result = await db.execute(stmt)
    count = result.scalar() or 0
    return f'{prefix}{count + 1:04d}'


async def create_validation_run(
    db: AsyncSession,
    project_id: uuid.UUID,
    model_id: uuid.UUID,
    dataset_id: uuid.UUID,
    config: dict | None = None,
) -> ValidationRun:
    """Create and execute a validation run."""
    run_number = await generate_run_number(db)
    
    run = ValidationRun(
        run_number=run_number,
        project_id=project_id,
        model_id=model_id,
        dataset_id=dataset_id,
        status=RunStatus.PENDING,
        config_json=config,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    
    # Execute validation
    await execute_validation(db, run)
    return run


async def execute_validation(db: AsyncSession, run: ValidationRun) -> None:
    """Execute the validation run."""
    start_time = datetime.now(timezone.utc)
    run.status = RunStatus.RUNNING
    run.started_at = start_time
    await db.commit()
    
    try:
        # Get model and dataset records
        model_record = await db.get(MLModel, run.model_id)
        dataset_record = await db.get(Dataset, run.dataset_id)
        
        if not model_record or not dataset_record:
            run.status = RunStatus.FAILED
            run.error_message = 'Model or dataset not found'
            run.completed_at = datetime.now(timezone.utc)
            run.duration_seconds = (run.completed_at - start_time).total_seconds()
            await db.commit()
            return
        
        # Compatibility check
        feature_columns = dataset_record.feature_columns_json
        compat = check_compatibility(
            model_path=model_record.file_path,
            dataset_path=dataset_record.file_path,
            target_column=dataset_record.target_column or '',
            feature_columns=feature_columns,
        )
        
        if not compat.compatible:
            run.status = RunStatus.FAILED
            run.error_message = 'Compatibility check failed: ' + '; '.join(compat.messages)
            
            # Create finding for compatibility failure
            finding = Finding(
                validation_run_id=run.id,
                severity=Severity.CRITICAL,
                category='compatibility',
                title='Model-Dataset Compatibility Check Failed',
                description='; '.join(compat.messages),
                recommendation='Ensure the model and dataset are compatible. Check feature counts and column names.',
                validator_name='compatibility_check',
                validator_version='0.1.0',
            )
            db.add(finding)
            run.completed_at = datetime.now(timezone.utc)
            run.duration_seconds = (run.completed_at - start_time).total_seconds()
            await db.commit()
            return
        
        # Add compatibility warnings as findings
        for warning in compat.warnings:
            finding = Finding(
                validation_run_id=run.id,
                severity=Severity.LOW,
                category='compatibility',
                title='Compatibility Warning',
                description=warning,
                validator_name='compatibility_check',
                validator_version='0.1.0',
            )
            db.add(finding)
        
        # Run registered validators
        registry = get_registry()
        if not registry.get('performance'):
            from app.validators.performance import PerformanceValidator
            registry.register(PerformanceValidator())

        if not registry.get('data_quality'):
            from app.validators.data_quality import DataQualityValidator
            registry.register(DataQualityValidator())

        if not registry.get('fairness'):
            from app.validators.fairness import FairnessValidator
            registry.register(FairnessValidator())

        if not registry.get('drift'):
            from app.validators.drift import DriftValidator
            registry.register(DriftValidator())

        if not registry.get('robustness'):
            from app.validators.robustness import RobustnessValidator
            registry.register(RobustnessValidator())

        if not registry.get('explainability'):
            from app.validators.explainability import ExplainabilityValidator
            registry.register(ExplainabilityValidator())

        if not registry.get('reliability'):
            from app.validators.reliability import ReliabilityValidator
            registry.register(ReliabilityValidator())

        validators = registry.get_all()
        selected_module_names = (run.config_json or {}).get('selected_validators')
        if selected_module_names:
            validators = [v for v in validators if v.name in selected_module_names]

        # Load reference dataset if configured
        ref_dataset = None
        ref_id_str = (run.config_json or {}).get('reference_dataset_id')
        if ref_id_str:
            try:
                ref_record = await db.get(Dataset, uuid.UUID(ref_id_str))
                if ref_record:
                    ref_dataset = load_dataset(ref_record.file_path)
            except Exception as e:
                logger.warning('Failed to load reference dataset: %s', str(e))

        target_col = (run.config_json or {}).get('target_column') or dataset_record.target_column or ''
        sensitive_cols = (run.config_json or {}).get('sensitive_columns') or dataset_record.sensitive_columns_json or []

        if validators:
            # Build validation context
            model = load_model_safely(model_record.file_path)
            dataset = load_dataset(dataset_record.file_path)

            if not target_col:
                expected_names = getattr(model, 'feature_names_in_', None)
                if expected_names is not None:
                    candidates = [c for c in dataset.columns if c not in set(expected_names)]
                    if len(candidates) == 1:
                        target_col = candidates[0]
                elif 'target' in dataset.columns:
                    target_col = 'target'
                elif 'label' in dataset.columns:
                    target_col = 'label'
                elif 'y' in dataset.columns:
                    target_col = 'y'

            
            context = ValidationContext(
                model=model,
                model_type=model_record.model_type or detect_model_type(model),
                framework=model_record.framework or detect_framework(model),
                dataset=dataset,
                target_column=target_col,
                feature_columns=feature_columns or [],
                sensitive_columns=sensitive_cols,
                reference_dataset=ref_dataset,
                config=run.config_json or {},
            )
            
            results = []
            for v in validators:
                res = await v.safe_validate(context)
                results.append(res)
            
            # Persist computed validator metrics
            run_config = dict(run.config_json or {})
            run_config['metrics'] = {vr.validator_name: vr.metrics for vr in results}
            run.config_json = run_config


            # Process results into findings and scores
            has_errors = False
            for vr in results:
                if vr.status.value == 'error':
                    has_errors = True
                
                for vf in vr.findings:
                    sev = vf.severity.lower() if hasattr(vf.severity, 'lower') else str(vf.severity)
                    finding = Finding(
                        validation_run_id=run.id,
                        severity=Severity(sev),
                        category=vf.category,
                        title=vf.title,
                        description=vf.description,
                        metric_name=vf.metric_name,
                        metric_value=vf.metric_value,
                        threshold=vf.threshold,
                        recommendation=vf.recommendation,
                        validator_name=vr.validator_name,
                        validator_version=vr.validator_version,
                    )
                    db.add(finding)
            
            # Calculate scores
            scores = calculate_scores(results)
            run.overall_score = scores.get('overall')
            run.category_scores_json = scores.get('categories', {})
            run.status = RunStatus.PARTIAL if has_errors else RunStatus.COMPLETED
        else:
            run.status = RunStatus.COMPLETED
            run.category_scores_json = {}
            logger.info('No validators executed.')
        
        run.completed_at = datetime.now(timezone.utc)
        run.duration_seconds = (run.completed_at - start_time).total_seconds()
        await db.commit()
        await db.refresh(run)
        
    except Exception as e:
        logger.exception('Validation run failed: %s', str(e))
        run.status = RunStatus.FAILED
        run.error_message = f'Unexpected error: {str(e)}'
        run.completed_at = datetime.now(timezone.utc)
        run.duration_seconds = (run.completed_at - start_time).total_seconds()
        await db.commit()


async def get_validation_run(db: AsyncSession, run_id: uuid.UUID) -> ValidationRun:
    """Get a validation run by ID."""
    from fastapi import HTTPException, status
    stmt = select(ValidationRun).where(ValidationRun.id == run_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Validation run not found')
    return run


async def list_validation_runs(
    db: AsyncSession,
    project_id: uuid.UUID | None = None,
) -> list[ValidationRun]:
    """List validation runs, optionally filtered by project."""
    stmt = select(ValidationRun)
    if project_id:
        stmt = stmt.where(ValidationRun.project_id == project_id)
    stmt = stmt.order_by(ValidationRun.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_run_findings(
    db: AsyncSession,
    run_id: uuid.UUID,
    severity: str | None = None,
    category: str | None = None,
) -> list[Finding]:
    """Get findings for a validation run with optional filters."""
    stmt = select(Finding).where(Finding.validation_run_id == run_id)
    if severity:
        stmt = stmt.where(Finding.severity == Severity(severity.lower()))
    if category:
        stmt = stmt.where(Finding.category == category)
    stmt = stmt.order_by(Finding.severity, Finding.created_at)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def list_findings(
    db: AsyncSession,
    project_id: uuid.UUID | None = None,
    severity: str | None = None,
    category: str | None = None,
    limit: int = 100,
) -> list[Finding]:
    """Get all findings with optional filtering."""
    stmt = select(Finding)
    if project_id:
        stmt = stmt.join(ValidationRun).where(ValidationRun.project_id == project_id)
    if severity:
        stmt = stmt.where(Finding.severity == Severity(severity.lower()))
    if category:
        stmt = stmt.where(Finding.category == category)
    stmt = stmt.order_by(Finding.created_at.desc()).limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())
