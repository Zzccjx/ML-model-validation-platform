from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import String, Text, Integer, BigInteger, ForeignKey, DateTime, JSON, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.validation_run import ValidationRun

class MLModel(Base):
    __tablename__ = 'ml_models'
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    model_type: Mapped[str | None] = mapped_column(String(50), nullable=True)  # classification, regression, unknown
    framework: Mapped[str | None] = mapped_column(String(100), nullable=True)  # sklearn, etc.
    feature_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    class_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    classes_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    project: Mapped['Project'] = relationship(back_populates='models')
    validation_runs: Mapped[list['ValidationRun']] = relationship(back_populates='model', cascade='all, delete-orphan')
