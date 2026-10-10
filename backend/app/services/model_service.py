"""Service for managing ML model uploads and metadata extraction."""
import logging
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import UploadFile, HTTPException, status

from app.models.ml_model import MLModel
from app.utils.file_utils import save_upload_file, delete_file, validate_file_extension, ALLOWED_MODEL_EXTENSIONS
from app.adapters.factory import get_adapter

logger = logging.getLogger('ai_validation')


async def upload_model(
    db: AsyncSession,
    project_id: uuid.UUID,
    name: str,
    file: UploadFile,
) -> MLModel:
    """Upload and inspect a model file."""
    validate_file_extension(file.filename, ALLOWED_MODEL_EXTENSIONS)
    
    file_path, file_size = await save_upload_file(file, f'models/{project_id}')
    
    # Inspect the model using format adapter
    model_type = None
    framework = None
    feature_count = None
    class_count = None
    classes_json = None
    metadata_json = None
    
    try:
        adapter = get_adapter(file_path)
        metadata = adapter.inspect()
        model_type = adapter.model_type
        framework = adapter.framework_name
        feature_count = metadata.get('feature_count')
        class_count = metadata.get('class_count')
        classes_json = metadata.get('classes')
        metadata_json = metadata
    except Exception as e:
        logger.warning('Failed to inspect model %s: %s', file.filename, str(e))
        metadata_json = {'inspection_error': str(e)}
    
    db_model = MLModel(
        project_id=project_id,
        name=name,
        filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        model_type=model_type,
        framework=framework,
        feature_count=feature_count,
        class_count=class_count,
        classes_json=classes_json,
        metadata_json=metadata_json,
    )
    db.add(db_model)
    await db.commit()
    await db.refresh(db_model)
    return db_model


async def get_model(db: AsyncSession, model_id: uuid.UUID) -> MLModel:
    """Get a model by ID."""
    stmt = select(MLModel).where(MLModel.id == model_id)
    result = await db.execute(stmt)
    model = result.scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Model not found')
    return model


async def list_models(db: AsyncSession, project_id: uuid.UUID | None = None) -> list[MLModel]:
    """List models, optionally filtered by project."""
    stmt = select(MLModel)
    if project_id:
        stmt = stmt.where(MLModel.project_id == project_id)
    stmt = stmt.order_by(MLModel.uploaded_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def delete_model(db: AsyncSession, model_id: uuid.UUID) -> None:
    """Delete a model and its file."""
    model = await get_model(db, model_id)
    await delete_file(model.file_path)
    await db.delete(model)
    await db.commit()


async def inspect_model(db: AsyncSession, model_id: uuid.UUID) -> dict:
    """Load model file and return deep diagnostic metadata via format adapter."""
    model = await get_model(db, model_id)
    try:
        adapter = get_adapter(model.file_path)
        meta = adapter.inspect()
        
        return {
            'id': model.id,
            'name': model.name,
            'filename': model.filename,
            'file_size': model.file_size,
            'model_type': adapter.model_type,
            'framework': adapter.framework_name,
            'model_class': meta.get('model_class'),
            'feature_count': meta.get('feature_count'),
            'feature_names': meta.get('feature_names'),
            'class_count': meta.get('class_count'),
            'classes': meta.get('classes'),
            'has_predict_proba': meta.get('has_predict_proba', False),
            'has_decision_function': meta.get('has_decision_function', False),
            'has_predict': meta.get('has_predict', True),
            'input_shape': meta.get('input_shape'),
            'output_shape': meta.get('output_shape'),
            'total_parameters': meta.get('total_parameters'),
            'layer_count': meta.get('layer_count'),
            'params': meta.get('params', {}),
            'uploaded_at': model.uploaded_at.isoformat() if model.uploaded_at else None,
        }
    except Exception as e:
        logger.error('Failed to inspect model %s: %s', model_id, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to inspect model: {str(e)}'
        )
