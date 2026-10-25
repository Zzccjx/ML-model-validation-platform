"""Service for managing dataset uploads and metadata extraction."""
import logging
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import UploadFile, HTTPException, status

from app.models.dataset import Dataset
from app.utils.file_utils import save_upload_file, delete_file, validate_file_extension, ALLOWED_DATASET_EXTENSIONS
from app.utils.dataset_utils import load_dataset, get_dataset_summary, validate_target_column

logger = logging.getLogger('ai_validation')


async def upload_dataset(
    db: AsyncSession,
    project_id: uuid.UUID,
    name: str,
    file: UploadFile,
    target_column: str | None = None,
    sensitive_columns: list[str] | None = None,
) -> Dataset:
    """Upload and parse a dataset file."""
    validate_file_extension(file.filename, ALLOWED_DATASET_EXTENSIONS)
    
    file_path, file_size = await save_upload_file(file, f'datasets/{project_id}')
    
    row_count = None
    column_count = None
    feature_columns_json = None
    metadata_json = None
    
    try:
        df = load_dataset(file_path)
        summary = get_dataset_summary(df)
        row_count = summary['row_count']
        column_count = summary['column_count']
        metadata_json = summary
        
        if not target_column:
            # Auto-detect target column by standard naming conventions
            for candidate in ['target', 'label', 'y', 'class', 'price']:
                if candidate in df.columns:
                    target_column = candidate
                    break

        if not sensitive_columns:
            # Auto-detect demographic candidates with discrete cardinality (2 - 20 groups)
            candidate_keywords = ['gender', 'sex', 'race', 'ethnicity', 'age_group', 'region', 'country', 'marital_status']
            auto_sensitive = []
            for col in df.columns:
                if col == target_column:
                    continue
                c_lower = str(col).lower()
                if any(kw in c_lower for kw in candidate_keywords):
                    if 2 <= df[col].nunique(dropna=True) <= 20:
                        auto_sensitive.append(col)
            if auto_sensitive:
                sensitive_columns = auto_sensitive

        if target_column:
            target_result = validate_target_column(df, target_column)
            if not target_result['valid']:
                # Clean up the uploaded file
                await delete_file(file_path)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f'Target column validation failed: {"; ".join(target_result["messages"])}'
                )
            exclude_cols = {target_column} | set(sensitive_columns or [])
            feature_columns_json = [c for c in df.columns if c not in exclude_cols]
        else:
            exclude_cols = set(sensitive_columns or [])
            feature_columns_json = [c for c in df.columns if c not in exclude_cols]
    except HTTPException:
        raise
    except Exception as e:
        logger.warning('Failed to parse dataset: %s', str(e))
    
    db_dataset = Dataset(
        project_id=project_id,
        name=name,
        filename=file.filename,
        file_path=file_path,
        file_size=file_size,
        row_count=row_count,
        column_count=column_count,
        target_column=target_column,
        feature_columns_json=feature_columns_json,
        sensitive_columns_json=sensitive_columns,
        metadata_json=metadata_json,
    )
    db.add(db_dataset)
    await db.commit()
    await db.refresh(db_dataset)
    return db_dataset


async def get_dataset(db: AsyncSession, dataset_id: uuid.UUID) -> Dataset:
    """Get a dataset by ID."""
    stmt = select(Dataset).where(Dataset.id == dataset_id)
    result = await db.execute(stmt)
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Dataset not found')
    return dataset


async def list_datasets(db: AsyncSession, project_id: uuid.UUID | None = None) -> list[Dataset]:
    """List datasets, optionally filtered by project."""
    stmt = select(Dataset)
    if project_id:
        stmt = stmt.where(Dataset.project_id == project_id)
    stmt = stmt.order_by(Dataset.uploaded_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def delete_dataset(db: AsyncSession, dataset_id: uuid.UUID) -> None:
    """Delete a dataset and its file."""
    dataset = await get_dataset(db, dataset_id)
    await delete_file(dataset.file_path)
    await db.delete(dataset)
    await db.commit()


async def preview_dataset(db: AsyncSession, dataset_id: uuid.UUID, max_rows: int = 15) -> dict:
    """Load dataset file and return top rows, dtypes, null counts, duplicate count, and categorization."""
    from app.utils.dataset_utils import get_dataset_preview
    dataset = await get_dataset(db, dataset_id)
    try:
        df = load_dataset(dataset.file_path)
        preview_data = get_dataset_preview(df, max_rows=max_rows)
        preview_data['id'] = dataset.id
        preview_data['name'] = dataset.name
        preview_data['filename'] = dataset.filename
        preview_data['target_column'] = dataset.target_column
        preview_data['sensitive_columns'] = dataset.sensitive_columns_json
        return preview_data
    except Exception as e:
        logger.error('Failed to preview dataset %s: %s', dataset_id, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to preview dataset: {str(e)}'
        )

