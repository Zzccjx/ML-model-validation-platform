"""Report API routes for validation runs."""
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.user import User
from app.models.validation_run import ValidationRun
from app.models.ml_model import MLModel
from app.models.dataset import Dataset
from app.models.finding import Finding
from app.services.report_service import generate_json_report, generate_pdf_report
from app.api.deps import get_current_active_user

from app.models.project import Project
from app.schemas.validation import ReportListItemResponse

router = APIRouter(prefix='/reports', tags=['Reports'])


@router.get('', response_model=list[ReportListItemResponse])
async def list_reports(
    project_id: uuid.UUID | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List all validation runs formatted as audit reports."""
    stmt = (
        select(ValidationRun)
        .options(selectinload(ValidationRun.findings))
        .order_by(ValidationRun.created_at.desc())
    )
    if project_id:
        stmt = stmt.where(ValidationRun.project_id == project_id)
        
    result = await db.execute(stmt)
    runs = result.scalars().all()
    
    # Pre-fetch projects, models, datasets for names
    project_ids = {r.project_id for r in runs}
    model_ids = {r.model_id for r in runs}
    dataset_ids = {r.dataset_id for r in runs}
    
    p_map = {}
    if project_ids:
        p_res = await db.execute(select(Project).where(Project.id.in_(project_ids)))
        p_map = {p.id: p.name for p in p_res.scalars().all()}
        
    m_map = {}
    if model_ids:
        m_res = await db.execute(select(MLModel).where(MLModel.id.in_(model_ids)))
        m_map = {m.id: m.name for m in m_res.scalars().all()}
        
    d_map = {}
    if dataset_ids:
        d_res = await db.execute(select(Dataset).where(Dataset.id.in_(dataset_ids)))
        d_map = {d.id: d.name for d in d_res.scalars().all()}

    reports_list: list[ReportListItemResponse] = []
    for r in runs:
        sev_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        for f in r.findings:
            sev = f.severity.value if hasattr(f.severity, 'value') else str(f.severity).lower()
            if sev in sev_counts:
                sev_counts[sev] += 1
                
        reports_list.append(
            ReportListItemResponse(
                id=r.id,
                run_number=r.run_number,
                project_id=r.project_id,
                project_name=p_map.get(r.project_id, 'Unknown Project'),
                model_id=r.model_id,
                model_name=m_map.get(r.model_id, 'Unknown Model'),
                dataset_id=r.dataset_id,
                dataset_name=d_map.get(r.dataset_id, 'Unknown Dataset'),
                status=r.status.value if hasattr(r.status, 'value') else str(r.status),
                overall_score=r.overall_score,
                severity_counts=sev_counts,
                completed_at=r.completed_at,
                created_at=r.created_at,
            )
        )
    return reports_list


async def _load_report_context(run_id: uuid.UUID, db: AsyncSession) -> tuple[ValidationRun, MLModel, Dataset, list[Finding]]:
    """Helper to fetch run, model, dataset, and findings with verification."""
    stmt = (
        select(ValidationRun)
        .options(selectinload(ValidationRun.findings))
        .where(ValidationRun.id == run_id)
    )
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Validation run not found")

    # Fetch model
    m_stmt = select(MLModel).where(MLModel.id == run.model_id)
    m_res = await db.execute(m_stmt)
    model = m_res.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Associated model not found")

    # Fetch dataset
    d_stmt = select(Dataset).where(Dataset.id == run.dataset_id)
    d_res = await db.execute(d_stmt)
    dataset = d_res.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Associated dataset not found")

    findings = list(run.findings)
    return run, model, dataset, findings


@router.get('/{run_id}')
@router.get('/{run_id}/json')

async def get_json_report(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Generate and return structured 15-section JSON validation report."""
    run, model, dataset, findings = await _load_report_context(run_id, db)
    report_dict = generate_json_report(run, model, dataset, findings)
    
    filename = f"validation_report_{run.run_number}.json"
    return JSONResponse(
        content=report_dict,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get('/{run_id}/pdf')
async def get_pdf_report(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Generate and download audit-grade PDF validation report."""
    run, model, dataset, findings = await _load_report_context(run_id, db)
    pdf_bytes = generate_pdf_report(run, model, dataset, findings)
    
    filename = f"validation_report_{run.run_number}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "application/pdf",
        },
    )

