from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QualityIssue(BaseModel):
    level: Literal["error", "warning", "info"]
    type: str
    message: str
    image_id: str | None = None
    annotation_ids: list[str] = Field(default_factory=list)
    class_id: str | None = None
    iou: float | None = None
    value: float | None = None


class QualitySummary(BaseModel):
    image_count: int
    annotated_image_count: int
    unannotated_image_count: int
    coverage: float
    annotation_count: int
    bbox_count: int
    polygon_count: int
    obb_count: int
    classify_count: int
    small_object_count: int
    small_object_ratio: float


class ClassDistributionItem(BaseModel):
    class_id: str
    class_index: int
    name: str
    count: int
    ratio: float


class DatasetQualityReport(BaseModel):
    dataset_id: str
    task_type: str
    summary: QualitySummary
    class_distribution: list[ClassDistributionItem]
    issues: list[QualityIssue]
