"""Dataset upload and management API routes."""
import uuid
import json
from fastapi import APIRouter, Depends, UploadFile, File, Form, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.schemas.dataset import DatasetUploadResponse
from app.services import dataset_service
from app.api.deps import get_current_active_user

router = APIRouter(prefix='/datasets', tags=['Datasets'])


@router.post('/upload', response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    project_id: uuid.UUID = Form(...),
    name: str = Form(None),
    target_column: str = Form(None),
    sensitive_columns: str = Form(None),  # JSON array string
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Upload a test dataset (CSV)."""
    dataset_name = name or file.filename or 'Unnamed Dataset'
    sensitive_list = None
    if sensitive_columns:
        try:
            sensitive_list = json.loads(sensitive_columns)
        except json.JSONDecodeError:
            sensitive_list = [s.strip() for s in sensitive_columns.split(',') if s.strip()]
    
    dataset = await dataset_service.upload_dataset(
        db, project_id, dataset_name, file, target_column, sensitive_list
    )
    return dataset


@router.get('', response_model=list[DatasetUploadResponse])
async def list_datasets(
    project_id: uuid.UUID | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List datasets."""
    return await dataset_service.list_datasets(db, project_id)


@router.get('/{dataset_id}', response_model=DatasetUploadResponse)
async def get_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get dataset details."""
    return await dataset_service.get_dataset(db, dataset_id)


@router.get('/{dataset_id}/preview', response_model=dict)
async def preview_dataset(
    dataset_id: uuid.UUID,
    max_rows: int = Query(15, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get top 15 preview rows and summary profiling for a dataset."""
    return await dataset_service.preview_dataset(db, dataset_id, max_rows=max_rows)



@router.delete('/{dataset_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a dataset."""
    await dataset_service.delete_dataset(db, dataset_id)
