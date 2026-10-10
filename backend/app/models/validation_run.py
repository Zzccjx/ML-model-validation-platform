from __future__ import annotations
import uuid
import enum
from datetime import datetime
from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, JSON, Text, Enum as SAEnum, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.project import Project
    from app.models.ml_model import MLModel
    from app.models.dataset import Dataset
    from app.models.finding import Finding

class RunStatus(str, enum.Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    PARTIAL = 'partial'

class ValidationRun(Base):
    __tablename__ = 'validation_runs'
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_number: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    project_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'), nullable=False)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('ml_models.id', ondelete='CASCADE'), nullable=False)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False)
    status: Mapped[RunStatus] = mapped_column(SAEnum(RunStatus), default=RunStatus.PENDING, nullable=False)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    category_scores_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    project: Mapped['Project'] = relationship(back_populates='validation_runs')
    model: Mapped['MLModel'] = relationship(back_populates='validation_runs')
    dataset: Mapped['Dataset'] = relationship(back_populates='validation_runs')
    findings: Mapped[list['Finding']] = relationship(back_populates='validation_run', cascade='all, delete-orphan')
