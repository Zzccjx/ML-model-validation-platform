import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Any

class DatasetUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    filename: str
    file_size: int
    row_count: int | None
    column_count: int | None
    target_column: str | None
    feature_columns_json: Any | None
    sensitive_columns_json: Any | None
    metadata_json: dict | None
    uploaded_at: datetime

class DatasetPreviewResponse(BaseModel):
    id: uuid.UUID
    name: str
    filename: str
    total_rows: int
    total_columns: int
    columns: list[str]
    preview_rows: list[dict[str, Any]]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    duplicate_rows: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    target_column: str | None = None
    sensitive_columns: list[str] | None = None

