from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.task_types import TaskType

ModelSource = Literal["training_task", "exported"]
ModelStatus = Literal["active", "archived"]
ModelArtifactType = Literal["best", "last", "onnx"]
ModelFormat = Literal["pt", "onnx"]


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    version: str
    dataset_id: str | None
    training_task_id: str | None
    source_model_id: str | None
    source: ModelSource
    artifact_type: ModelArtifactType
    format: ModelFormat
    task_type: TaskType
    engine_type: str
    model_path: str
    base_model: str | None
    status: ModelStatus
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    metrics_json: dict
    notes: str
    archived_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ModelVersionList(BaseModel):
    items: list[ModelVersionResponse]
    total: int


class ModelVersionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    version: str | None = Field(default=None, min_length=1, max_length=64)
    notes: str | None = Field(default=None, max_length=4000)
