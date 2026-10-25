import uuid
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from typing import Any

class ValidationRunCreate(BaseModel):
    model_id: uuid.UUID
    dataset_id: uuid.UUID
    reference_dataset_id: uuid.UUID | None = None
    target_column: str | None = None
    sensitive_column: str | None = None
    selected_validators: list[str] | None = None
    config: dict | None = None

class CompatibilityCheckRequest(BaseModel):
    model_id: uuid.UUID
    dataset_id: uuid.UUID
    target_column: str | None = None
    reference_dataset_id: uuid.UUID | None = None
    sensitive_column: str | None = None
    selected_validators: list[str] | None = None


class ValidatorReadiness(BaseModel):
    name: str
    ready: bool
    status_label: str  # e.g., 'Ready', 'Requires Reference Dataset', 'Requires Sensitive Column'
    reason: str | None = None

class CompatibilityCheckResponse(BaseModel):
    compatible: bool
    errors: list[str]
    warnings: list[str]
    model_info: dict[str, Any]
    dataset_info: dict[str, Any]
    validator_readiness: list[ValidatorReadiness]

class ReportListItemResponse(BaseModel):
    id: uuid.UUID
    run_number: str
    project_id: uuid.UUID
    project_name: str | None = None
    model_id: uuid.UUID
    model_name: str | None = None
    dataset_id: uuid.UUID
    dataset_name: str | None = None
    status: str
    overall_score: float | None = None
    severity_counts: dict[str, int]
    completed_at: datetime | None = None
    created_at: datetime


class ValidationRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    run_number: str
    project_id: uuid.UUID
    model_id: uuid.UUID
    dataset_id: uuid.UUID
    status: str
    overall_score: float | None
    category_scores_json: dict | None
    config_json: dict | None
    started_at: datetime | None
    completed_at: datetime | None
    duration_seconds: float | None
    error_message: str | None
    created_at: datetime

class FindingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    validation_run_id: uuid.UUID
    severity: str
    category: str
    title: str
    description: str
    metric_name: str | None
    metric_value: float | None
    threshold: str | None
    recommendation: str | None
    validator_name: str
    validator_version: str
    created_at: datetime

class FindingListResponse(BaseModel):
    findings: list[FindingResponse]
    total: int


class RowPredictionItem(BaseModel):
    row_index: int
    actual: Any
    predicted: Any
    correct: bool | None = None
    error: float | None = None
    abs_error: float | None = None
    pct_error: float | None = None
    probabilities: dict[str, float] | None = None


class PredictionTableResponse(BaseModel):
    total_rows: int
    returned_rows: int
    offset: int
    limit: int
    task_type: str  # 'classification' or 'regression'
    has_probabilities: bool
    summary: dict[str, Any]
    items: list[RowPredictionItem]


class ChartAxis(BaseModel):
    key: str
    label: str
    type: str = "category"
    unit: str | None = None


class ChartSeries(BaseModel):
    key: str
    label: str
    color: str | None = None
    type: str | None = None


class ChartThreshold(BaseModel):
    value: float
    label: str
    color: str = "#ef4444"
    style: str = "dashed"


class ChartProvenance(BaseModel):
    source_validator: str
    status: str = "EVALUATED"
    sample_count: int | None = None
    note: str | None = None


class ChartDefinition(BaseModel):
    chart_id: str
    title: str
    description: str
    category: str
    chart_type: str
    status: str = "EVALUATED"
    x_axis: ChartAxis
    y_axis: ChartAxis
    series: list[ChartSeries] = []
    data: list[dict[str, Any]] = []
    matrix: dict[str, Any] | None = None
    thresholds: list[ChartThreshold] = []
    interpretation: str | None = None
    provenance: ChartProvenance


class ChartsResponse(BaseModel):
    run_id: uuid.UUID
    run_number: str
    total_charts: int
    categories: list[str]
    charts: list[ChartDefinition]

