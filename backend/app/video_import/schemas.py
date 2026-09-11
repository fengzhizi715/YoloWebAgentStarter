from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.core.task_types import TaskType


SamplingMode = Literal["fps", "interval_seconds", "frame_interval"]


class VideoImportTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str | None
    name: str
    task_type: TaskType
    split: Literal["train", "val", "test"]
    status: Literal["pending", "running", "completed", "failed"]
    start_requested: bool
    checkpoint_next_frame_index: int
    output_bytes: int
    config_json: dict
    source_file_name: str
    source_checksum: str
    video_info_json: dict
    total_images: int
    generated_images: int
    progress_percent: float
    error_message: str | None
    created_at: datetime
    updated_at: datetime
