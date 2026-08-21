from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.task_types import TaskType


class AutoAnnotationCreateRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=64)
    confidence: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.45, ge=0, le=1)
    clean_old_annotations: bool = False
    skip_annotated_images: bool = True
    class_mapping: dict[str, str] = Field(default_factory=dict, max_length=100)


class AutoAnnotationTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    model_id: str
    task_type: TaskType
    status: Literal["pending", "running", "completed", "failed", "stopped"]
    clean_old_annotations: bool
    skip_annotated_images: bool
    confidence: float
    iou: float
    class_mapping: dict[str, str]
    total_images: int
    processed_images: int
    created_annotations: int
    skipped_images: int
    progress_percent: float
    logs_path: str | None
    error_message: str | None
    stop_requested: bool
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class AutoAnnotationLogResponse(BaseModel):
    task_id: str
    logs: str
    line_count: int
