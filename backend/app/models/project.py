from __future__ import annotations
import uuid
from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.core.database import Base
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.user import User
    from app.models.ml_model import MLModel
    from app.models.dataset import Dataset
    from app.models.validation_run import ValidationRun

class Project(Base):
    __tablename__ = 'projects'
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    owner: Mapped['User'] = relationship(back_populates='projects')
    models: Mapped[list['MLModel']] = relationship(back_populates='project', cascade='all, delete-orphan')
    datasets: Mapped[list['Dataset']] = relationship(back_populates='project', cascade='all, delete-orphan')
    validation_runs: Mapped[list['ValidationRun']] = relationship(back_populates='project', cascade='all, delete-orphan')
