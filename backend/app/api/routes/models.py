"""Model upload and management API routes."""
import uuid
from fastapi import APIRouter, Depends, UploadFile, File, Form, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.schemas.model import ModelUploadResponse
from app.services import model_service
from app.api.deps import get_current_active_user

router = APIRouter(prefix='/models', tags=['Models'])


@router.post('/upload', response_model=ModelUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_model(
    project_id: uuid.UUID = Form(...),
    name: str = Form(None),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Upload a trained ML model file."""
    model_name = name or file.filename or 'Unnamed Model'
    model = await model_service.upload_model(db, project_id, model_name, file)
    return model


@router.get('', response_model=list[ModelUploadResponse])
async def list_models(
    project_id: uuid.UUID | None = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List models."""
    return await model_service.list_models(db, project_id)


@router.get('/{model_id}', response_model=ModelUploadResponse)
async def get_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get model details."""
    return await model_service.get_model(db, model_id)


@router.get('/{model_id}/inspect', response_model=dict)
async def inspect_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Deep inspection of a trained model file."""
    return await model_service.inspect_model(db, model_id)



@router.delete('/{model_id}', status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a model."""
    await model_service.delete_model(db, model_id)
