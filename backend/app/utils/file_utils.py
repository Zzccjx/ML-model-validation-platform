import os
import uuid
import aiofiles
from pathlib import Path
from typing import Protocol
from fastapi import UploadFile, HTTPException, status
from app.core.config import get_settings

ALLOWED_MODEL_EXTENSIONS = {'.pkl', '.joblib', '.pickle', '.keras', '.h5', '.onnx'}
ALLOWED_DATASET_EXTENSIONS = {'.csv'}

class FileStorageBackend(Protocol):
    """Protocol for file storage backends. Implement for S3/GCS/Azure."""
    async def save(self, file_path: str, content: bytes) -> str: ...
    async def delete(self, file_path: str) -> None: ...
    async def exists(self, file_path: str) -> bool: ...

def validate_file_extension(filename: str, allowed: set[str]) -> str:
    """Validate file extension against whitelist. Returns the extension."""
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'File type {ext} not allowed. Allowed: {allowed}'
        )
    return ext

async def save_upload_file(upload_file: UploadFile, subdirectory: str) -> tuple[str, int]:
    """Save uploaded file to disk. Returns (file_path, file_size)."""
    settings = get_settings()
    upload_dir = Path(settings.UPLOAD_DIRECTORY) / subdirectory
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    unique_name = f'{uuid.uuid4().hex}_{upload_file.filename}'
    file_path = upload_dir / unique_name
    
    content = await upload_file.read()
    file_size = len(content)
    
    max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f'File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB'
        )
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    return str(file_path), file_size

async def delete_file(file_path: str) -> None:
    """Safely delete a file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass  # Log but don't crash

def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)
