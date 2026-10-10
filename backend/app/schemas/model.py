import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import Any

class ModelUploadResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    filename: str
    file_size: int
    model_type: str | None
    framework: str | None
    feature_count: int | None
    class_count: int | None
    classes_json: Any | None
    metadata_json: dict | None
    uploaded_at: datetime

class ModelDetailResponse(ModelUploadResponse):
    validation_run_count: int = 0
