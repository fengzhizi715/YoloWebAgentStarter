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


class InferenceDetection(BaseModel):
    class_index: int
    class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    polygon: list[list[float]] | None = None
    obb_points: list[list[float]] | None = None


class InferenceResult(BaseModel):
    model_id: str
    task_type: TaskType
    detections: list[InferenceDetection]
    inference_time_ms: float


class ModelTestRecordResponse(BaseModel):
    id: str
    model_id: str
    file_name: str
    result_json: dict
    created_at: datetime


class ModelEvaluationRequest(BaseModel):
    split: Literal["train", "val", "test"] = "val"
    confidence: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.5, ge=0, le=1)


class ModelEvaluationRecordResponse(BaseModel):
    id: str
    model_id: str
    dataset_id: str
    split: Literal["train", "val", "test"]
    confidence: float
    iou: float
    result_json: dict
    created_at: datetime


class PreAnnotationRequest(BaseModel):
    dataset_id: str = Field(min_length=1, max_length=64)
    image_ids: list[str] = Field(min_length=1, max_length=100)
    confidence: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.45, ge=0, le=1)


class PreAnnotationImage(BaseModel):
    image_id: str
    annotations: list[dict]


class PreAnnotationResponse(BaseModel):
    model_id: str
    dataset_id: str
    images: list[PreAnnotationImage]


class ModelCompareRequest(BaseModel):
    baseline_model_id: str
    candidate_model_id: str


class ModelCompareResponse(BaseModel):
    dataset_id: str
    baseline: dict
    candidate: dict
    deltas: dict[str, float | None]
    suggestions: list[str]
