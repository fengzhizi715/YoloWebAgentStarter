from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.task_types import TaskType

TrainingStatus = Literal["pending", "running", "completed", "failed", "stopped"]


class TrainingTaskCreate(BaseModel):
    dataset_id: str
    name: str = Field(default="local-yolo-training", min_length=1, max_length=255)
    model: str = Field(default="yolo11n.pt", min_length=1, max_length=255)
    task_type: TaskType | None = None
    epochs: int = Field(default=50, ge=1, le=10000)
    img_size: int = Field(default=640, ge=32, le=4096)
    batch_size: int = Field(default=16, ge=1, le=4096)
    device: str = Field(default="auto", min_length=1, max_length=64)
    workers: int = Field(default=2, ge=0, le=128)
    val_ratio: float = Field(default=0.2, ge=0.0, le=0.9)
    seed: int = Field(default=42, ge=0)
    optimizer: str | None = Field(default=None, max_length=64)
    lr0: float | None = Field(default=None, gt=0)
    patience: int | None = Field(default=None, ge=0)


class TrainingTaskResumeRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    epochs: int | None = Field(default=None, ge=1, le=10000)
    resume_epoch: bool = True


class TrainingTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    profile_id: str | None
    name: str
    status: TrainingStatus
    task_type: TaskType
    model_name: str
    model_path: str
    epochs: int
    img_size: int
    batch_size: int
    device: str
    workers: int
    val_ratio: float
    seed: int
    optimizer: str | None
    lr0: float | None
    patience: int | None
    command_preview: str | None
    export_path: str | None
    data_yaml_path: str | None
    run_dir: str | None
    logs_path: str | None
    summary_path: str | None
    best_model_path: str | None
    last_model_path: str | None
    progress_epoch: int
    progress_total_epochs: int
    progress_percent: float
    metrics_json: dict
    error_message: str | None
    stop_requested: bool
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class TrainingTaskList(BaseModel):
    items: list[TrainingTaskResponse]


class TrainingLogResponse(BaseModel):
    task_id: str
    logs: str
    line_count: int


class TrainingDeviceResponse(BaseModel):
    id: str
    type: Literal["cpu", "mps", "cuda"]
    name: str
    index: int | None = None
    memory_total_mb: int | None = None
    memory_free_mb: int | None = None
    status: Literal["available", "idle", "busy", "unavailable", "unknown"] = "unknown"


class TrainingDeviceListResponse(BaseModel):
    items: list[TrainingDeviceResponse]


class TrainingSummaryResponse(BaseModel):
    task_id: str
    status: TrainingStatus
    training_config: dict
    dataset: dict
    progress: dict
    metrics: dict
    checkpoints: dict
    log_summary: dict
    risks: list[str]
    next_steps: list[str]


class TrainingProfileCreate(BaseModel):
    dataset_id: str
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=2000)
    model: str = Field(default="yolo11n.pt", min_length=1, max_length=255)
    task_type: TaskType
    epochs: int = Field(default=50, ge=1, le=10000)
    img_size: int = Field(default=640, ge=32, le=4096)
    batch_size: int = Field(default=16, ge=1, le=4096)
    device: str = Field(default="auto", min_length=1, max_length=64)
    workers: int = Field(default=2, ge=0, le=128)
    val_ratio: float = Field(default=0.2, ge=0.0, le=0.9)
    seed: int = Field(default=42, ge=0)
    optimizer: str | None = Field(default=None, max_length=64)
    lr0: float | None = Field(default=None, gt=0)
    patience: int | None = Field(default=None, ge=0)


class TrainingProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    model: str | None = Field(default=None, min_length=1, max_length=255)
    task_type: TaskType | None = None
    epochs: int | None = Field(default=None, ge=1, le=10000)
    img_size: int | None = Field(default=None, ge=32, le=4096)
    batch_size: int | None = Field(default=None, ge=1, le=4096)
    device: str | None = Field(default=None, min_length=1, max_length=64)
    workers: int | None = Field(default=None, ge=0, le=128)
    val_ratio: float | None = Field(default=None, ge=0.0, le=0.9)
    seed: int | None = Field(default=None, ge=0)
    optimizer: str | None = Field(default=None, max_length=64)
    lr0: float | None = Field(default=None, gt=0)
    patience: int | None = Field(default=None, ge=0)


class TrainingProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    name: str
    description: str
    model_name: str
    task_type: TaskType
    epochs: int
    img_size: int
    batch_size: int
    device: str
    workers: int
    val_ratio: float
    seed: int
    optimizer: str | None
    lr0: float | None
    patience: int | None
    created_at: datetime
    updated_at: datetime
